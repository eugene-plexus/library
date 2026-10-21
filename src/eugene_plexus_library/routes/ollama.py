"""Keep actions and source-bound Ollama metadata and file transfers."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, cast
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import ValidationError

from .. import tokens
from .._generated.models import (
    KeepFileRequest,
    KeepFileState,
    KeepJob,
    KeepRequest,
    NodeOllamaKeepRequest,
    NodeOllamaList,
    OllamaReport,
)
from .._http import shared_internal_client
from ..dependencies import require_authorized, require_operator
from ..keep import CHUNK, File, Job, KeepError, Keeps
from ..node_ollama import NodeStores

router = APIRouter(tags=["ollama"])


def manager(request: Request) -> Keeps:
    keeps: Keeps = request.app.state.keeps
    return keeps


def own(node: str, claims: tokens.Claims | None) -> None:
    if claims is None or not (
        claims.is_service and claims.sub == tokens.SUB_AGENT and claims.issuer_node == node
    ):
        raise HTTPException(403, "A machine reports only its own Ollama store.")


async def limited(request: Request, limit: int) -> bytes:
    chunks, count = [], 0
    async for chunk in request.stream():
        count += len(chunk)
        if count > limit:
            raise HTTPException(413, f"The request exceeds {limit} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/v1/models/{id}/keep",
    status_code=202,
    response_model=KeepJob,
    dependencies=[Depends(require_operator)],
)
async def keep_model(id: str, body: KeepRequest, request: Request) -> KeepJob:
    keeps = manager(request)
    model = keeps.store.get_model(id)
    if model is None:
        raise HTTPException(404, "The model is not listed.")
    job = await asyncio.to_thread(keeps.local, model, body.folder)
    keeps.launch(keeps.run_local(job))
    return job.public


@router.get(
    "/v1/keep-jobs/{id}", response_model=KeepJob, dependencies=[Depends(require_authorized)]
)
async def get_job(id: str, request: Request) -> KeepJob:
    return manager(request).get(id).public


@router.delete(
    "/v1/keep-jobs/{id}", response_model=KeepJob, dependencies=[Depends(require_operator)]
)
async def cancel_job(id: str, request: Request) -> KeepJob:
    keeps = manager(request)
    return await asyncio.to_thread(keeps.cancel, keeps.get(id))


@router.put("/v1/nodes/{node}/ollama", status_code=204)
async def report(
    node: str,
    request: Request,
    claims: Annotated[tokens.Claims | None, Depends(require_authorized)],
) -> Response:
    own(node, claims)
    raw = await limited(request, 1024 * 1024)
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and isinstance(decoded.get("stores"), list):
            total = sum(
                len(s["models"])
                for s in decoded["stores"]
                if isinstance(s, dict) and isinstance(s.get("models"), list)
            )
            if total > 2000:
                raise HTTPException(413, "The report exceeds 2,000 models.")
        body = OllamaReport.model_validate_json(raw)
    except ValueError as exc:
        if isinstance(exc, ValidationError):
            raise HTTPException(422, json.loads(exc.json())) from exc
        raise HTTPException(422, "The report is not JSON.") from exc
    if sum(len(s.models) for s in body.stores) > 2000:
        raise HTTPException(413, "The report exceeds 2,000 models.")
    reports: NodeStores = request.app.state.node_ollama
    await asyncio.to_thread(reports.replace, node, body.model_dump(mode="json", exclude_none=True))
    return Response(status_code=204)


@router.get(
    "/v1/nodes/ollama",
    response_model=NodeOllamaList,
    response_model_exclude_none=True,
    dependencies=[Depends(require_authorized)],
)
async def list_reports(request: Request) -> dict[str, Any]:
    reports: NodeStores = request.app.state.node_ollama
    return reports.listing(manager(request).store.list_models())


async def send_from_node(request: Request, job: Job) -> None:
    keeps = manager(request)
    try:
        # The local agent owns peer lookup and machine credentials. Its address
        # is supplied by the supervisor, never by a worker's report or the caller.
        agent = request.app.state.settings.agent_url
        if not agent:
            raise KeepError(503, "The Library's agent address is unavailable.")
        client = shared_internal_client("ollama-keep", timeout=httpx.Timeout(3600, connect=10))
        response = await client.post(
            agent.rstrip("/")
            + "/api/proxy/node:"
            + quote(cast(str, job.public.node), safe="")
            + "/v1/ollama/send",
            headers={"Authorization": "Bearer " + request.app.state.auth_state.service_token}
            if request.app.state.auth_state.service_token
            else {},
            json={"job": job.public.id, "digest": job.public.digest},
        )
        response.raise_for_status()
        if job.public.state == "running":
            raise KeepError(502, "The machine finished sending before every file was verified.")
    except Exception as exc:
        with job.lock:
            if job.public.state == "running":
                keeps.fail(job, f"The machine could not send the model: {exc}")


@router.post(
    "/v1/nodes/{node}/ollama/keep",
    status_code=202,
    response_model=KeepJob,
    dependencies=[Depends(require_operator)],
)
async def keep_remote(node: str, body: NodeOllamaKeepRequest, request: Request) -> KeepJob:
    reports = request.app.state.node_ollama.reports
    for store in reports.get(node, {}).get("stores", []):
        for model in store["models"]:
            if model["digest"] != body.digest:
                continue
            if bool(model.get("projectorDigest")) != (model.get("projectorSizeBytes") is not None):
                raise HTTPException(409, "The report did not give the projector's digest and size.")
            keeps = manager(request)
            job = await asyncio.to_thread(
                keeps.reserve,
                name=model["tags"][0],
                tags=model["tags"],
                digest=body.digest,
                size=model["sizeBytes"],
                store=store["path"],
                folder=body.folder,
                node=node,
                projector_digest=model.get("projectorDigest"),
                projector_size=model.get("projectorSizeBytes", 0),
            )
            keeps.launch(send_from_node(request, job))
            return job.public
    raise HTTPException(404, "This machine has not reported that Ollama digest.")


def assigned(
    request: Request, id: str, role: str, claims: tokens.Claims | None
) -> tuple[Keeps, Job, File]:
    keeps = manager(request)
    job = keeps.get(id)
    if not job.public.node:
        raise HTTPException(403, "Only the source agent of a remote Keep may send files.")
    own(job.public.node, claims)
    return keeps, job, keeps.file(job, role)


@router.post("/v1/keep-jobs/{id}/files/state", response_model=KeepFileState)
async def file_state(
    id: str,
    body: KeepFileRequest,
    request: Request,
    claims: Annotated[tokens.Claims | None, Depends(require_authorized)],
) -> dict[str, Any]:
    keeps, job, file = assigned(request, id, body.role, claims)
    if job.public.state not in ("running", "done"):
        raise HTTPException(409, "This Keep is no longer running.")
    return await asyncio.to_thread(keeps.state, job, file)


@router.put("/v1/keep-jobs/{id}/files", response_model=KeepFileState)
async def file_chunk(
    id: str,
    request: Request,
    claims: Annotated[tokens.Claims | None, Depends(require_authorized)],
    role: str = Query(pattern="^(weights|projector)$"),
    offset: int = Query(ge=0),
) -> dict[str, Any]:
    keeps, job, file = assigned(request, id, role, claims)
    data = await limited(request, CHUNK)
    return await asyncio.to_thread(keeps.write, job, file, offset, data)


@router.post("/v1/keep-jobs/{id}/files/complete", response_model=KeepFileState)
async def file_complete(
    id: str,
    body: KeepFileRequest,
    request: Request,
    claims: Annotated[tokens.Claims | None, Depends(require_authorized)],
) -> dict[str, Any]:
    keeps, job, file = assigned(request, id, body.role, claims)
    return await asyncio.to_thread(keeps.complete, job, file)
