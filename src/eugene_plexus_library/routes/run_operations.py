"""Operator run intents and narrowly scoped execution grants for assigned agents."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel

from .. import tokens
from .._generated.models import DownloadSpec, LibraryModel, ModelProfileSpec
from ..dependencies import require_authorized, require_operator
from ..run_operations import (
    TERMINAL,
    Checkpoint,
    ClaimedOperation,
    Intent,
    Journal,
    Operation,
    OperationList,
    ProfileRequest,
    public,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/run-operations", tags=["run operations"])


def journal(request: Request) -> Journal:
    return request.app.state.run_operations  # type: ignore[no-any-return]


def assigned_agent(
    request: Request,
    claims: Annotated[tokens.Claims | None, Depends(require_authorized)],
    node: str | None = None,
) -> str | None:
    if claims is None:  # explicit unauthenticated development mode
        return node
    if not claims.is_service or claims.sub != tokens.SUB_AGENT:
        raise HTTPException(403, "An assigned node agent is required")
    return claims.issuer_node


@router.get("", response_model=OperationList, dependencies=[Depends(require_operator)])
def list_operations(request: Request) -> dict[str, Any]:
    return {"operations": [public(r) for r in journal(request).list()]}


@router.get("/assigned", response_model=OperationList)
def assigned(request: Request, node: str | None = Depends(assigned_agent)) -> dict[str, Any]:
    return {
        "operations": [
            public(r)
            for r in journal(request).list(pending_only=True)
            if r["node"] == node and r["step"] not in TERMINAL | {"downloading"}
        ]
    }


@router.put(
    "/{id}", status_code=202, response_model=Operation, dependencies=[Depends(require_operator)]
)
def submit(
    request: Request, intent: Intent, id: str = Path(pattern="^[a-zA-Z0-9_-]{1,100}$")
) -> dict[str, Any]:
    def resolve() -> dict[str, Any] | None:
        if intent.modelId is not None:
            found = request.app.state.state_store.get_model(intent.modelId)
            if found is None:
                raise HTTPException(404, "Model is not in the Library")
            return found.model_dump(mode="json")  # type: ignore[no-any-return]
        if (
            intent.downloadId is not None
            and request.app.state.download_manager.get(intent.downloadId) is None
        ):
            raise HTTPException(404, "Download is not in the Library")
        return None

    return public(journal(request).create(id, intent, resolve))


class Answer(BaseModel):
    answer: Literal["install", "skip"]


@router.get("/{id}", response_model=Operation, dependencies=[Depends(require_operator)])
def get_operation(request: Request, id: str) -> dict[str, Any]:
    return public(journal(request).get(id))


@router.post("/{id}/answer", response_model=Operation, dependencies=[Depends(require_operator)])
def answer(request: Request, id: str, body: Answer) -> dict[str, Any]:
    def update(record: dict[str, Any]) -> None:
        if record["answer"] is not None:
            if record["answer"] == body.answer:
                return
            raise HTTPException(409, "This run already has an install decision")
        if record["step"] != "awaiting-install":
            if record["answer"] == body.answer:
                return
            raise HTTPException(409, "Run is not waiting for an install decision")
        record.update(
            answer=body.answer,
            step="installing" if body.answer == "install" else "settings",
            lease=None,
            leaseUntil=0,
        )

    return public(journal(request).change(id, update))


@router.post("/{id}/cancel", response_model=Operation, dependencies=[Depends(require_operator)])
def cancel(request: Request, id: str) -> dict[str, Any]:
    def update(record: dict[str, Any]) -> None:
        if record["step"] not in TERMINAL:
            record.update(step="cancelled", lease=None, leaseUntil=0)

    return public(journal(request).change(id, update))


@router.delete("/{id}", status_code=204, dependencies=[Depends(require_operator)])
def dismiss(request: Request, id: str) -> None:
    def update(record: dict[str, Any]) -> None:
        if record["step"] not in TERMINAL:
            raise HTTPException(409, "Cancel the run before dismissing it")
        record["dismissed"] = True

    journal(request).change(id, update)


@router.post("/{id}/claim", response_model=ClaimedOperation)
def claim(request: Request, id: str, node: str | None = Depends(assigned_agent)) -> dict[str, Any]:
    return journal(request).claim(id, node)


@router.post("/{id}/checkpoint", response_model=Operation)
def checkpoint(
    request: Request, id: str, patch: Checkpoint, node: str | None = Depends(assigned_agent)
) -> dict[str, Any]:
    return public(journal(request).checkpoint(id, node, patch))


@router.post("/{id}/profile", response_model=Operation)
def profile(
    request: Request, id: str, body: ProfileRequest, node: str | None = Depends(assigned_agent)
) -> dict[str, Any]:
    store = request.app.state.state_store

    def ensure(record: dict[str, Any]) -> None:
        journal(request).verify_lease(record, node, body.lease)
        if record["step"] != "settings" or record["engine"] != body.engine:
            raise HTTPException(409, "Run is not choosing a profile for this engine")
        if record.get("profile") is not None:
            return
        model_id = record["model"]["id"]
        profiles = store.list_profiles(model_id)
        matching = [p for p in profiles if p.engine.value == body.engine]
        chosen = next((p for p in matching if p.default), matching[0] if matching else None)
        if chosen is None:
            name = (
                "default"
                if not any(p.name == "default" for p in profiles)
                else f"default-{body.engine}"
            )
            spec = ModelProfileSpec.model_validate(
                {
                    "name": name,
                    "engine": body.engine,
                    "default": not profiles,
                    "flags": {"contextSize": body.contextSize} if body.contextSize else {},
                    "extraArgs": [],
                    "env": {},
                }
            )
            chosen = store.create_profile(model_id, spec)
        record["profile"] = chosen.model_dump(mode="json")

    return public(journal(request).change(id, ensure))


async def advance_downloads(app: Any) -> None:
    """Persist intent before starting downloads; reconcile after every restart."""
    jobs: Journal = app.state.run_operations
    manager = app.state.download_manager
    while True:
        for record in await asyncio.to_thread(jobs.list, all_records=True, pending_only=True):
            try:
                intent = record["intent"]
                download_id = intent.get("downloadId") or record["id"]
                if (
                    record["step"] == "cancelled"
                    and intent.get("download")
                    and not record.get("downloadCancelled")
                ):
                    if manager.get(download_id) is not None:
                        await manager.cancel(download_id)
                    await asyncio.to_thread(
                        jobs.change, record["id"], lambda r: r.update(downloadCancelled=True)
                    )
                    continue
                if record["step"] != "downloading":
                    continue
                download = manager.get(download_id)
                if download is None and intent.get("download"):
                    download = await manager.start(
                        DownloadSpec.model_validate(intent["download"]), download_id=download_id
                    )
                if download is None:
                    raise ValueError("The download was removed")
                # Only an operation's own transfer is resumed automatically.
                if (
                    download.state.value == "paused"
                    and intent.get("download")
                    and not record.get("downloadPaused")
                ):
                    # The listing can predate an operator's pause. Serialize
                    # recovery with pause/resume, and consult the current intent.
                    async with app.state.run_download_lock:
                        current = await asyncio.to_thread(jobs.get, record["id"])
                        download = manager.get(download_id)
                        if download is None:
                            raise ValueError("The download was removed")
                        if (
                            current["step"] == "downloading"
                            and not current.get("downloadPaused")
                            and download.state.value == "paused"
                        ):
                            download = await manager.resume(download_id)
                data = download.model_dump(mode="json")
                model = (
                    app.state.state_store.get_model(download.modelId) if download.modelId else None
                )

                def update(
                    r: dict[str, Any],
                    data: dict[str, Any] = data,
                    model: LibraryModel | None = model,
                ) -> None:
                    if r["step"] != "downloading":
                        return
                    r["download"] = data
                    if data["state"] in {"failed", "cancelled"}:
                        r.update(
                            step="failed",
                            failedStep="download",
                            error=data.get("message") or "Download failed",
                        )
                    elif model is not None and data["state"] == "done":
                        r.update(model=model.model_dump(mode="json"), step="checking")
                    elif data["state"] == "done":
                        since = r.setdefault("catalogueSince", jobs.clock())
                        if jobs.clock() - since > 120:
                            r.update(
                                step="failed",
                                failedStep="download",
                                error=(
                                    "The file downloaded but Library could not catalogue it. "
                                    "Scan the folder, then run it from Library."
                                ),
                            )

                await asyncio.to_thread(jobs.change, record["id"], update)
                # Legacy browser intent clears only AFTER the durable job exists.
                if download.runWhenReady:
                    await manager.claim(download_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("run download %s: %s", record["id"], exc)
                message = str(exc)

                def failed(r: dict[str, Any], message: str = message) -> None:
                    if r["step"] == "downloading":
                        r.update(step="failed", failedStep="download", error=message)

                with suppress(Exception):
                    await asyncio.to_thread(
                        jobs.change,
                        record["id"],
                        failed,
                    )
        await asyncio.sleep(2)


async def set_download_pause(request: Request, download_id: str, *, paused: bool) -> None:
    """An explicit pause must survive restarts and override automatic recovery."""
    jobs = journal(request)
    for record in await asyncio.to_thread(jobs.list, pending_only=True):
        if (
            record["step"] == "downloading"
            and record["id"] == download_id
            and record["intent"].get("download")
        ):
            await asyncio.to_thread(
                jobs.change, record["id"], lambda r: r.update(downloadPaused=paused)
            )
