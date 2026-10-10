"""Model routes: list, read, and forget."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from .. import deletion, eligibility, engine_fit, hardware, prepared
from .._generated.models import (
    EligibilityList,
    EligibilityRequest,
    FitQuestion,
    LibraryFolderList,
    LibraryModel,
    LibraryModelList,
    ModelDeleted,
    ModelDeleteRequest,
    ModelDeletion,
    ModelStatus,
    PreparedModelRequest,
    Problem,
)
from ..config import ConfigStore
from ..dependencies import require_operator
from ..paths import is_within, same_path
from ..scan_manager import ScanManager
from ..store import StateStore

router = APIRouter(tags=["models"])


@router.get("/v1/folders", response_model=LibraryFolderList)
async def list_folders(request: Request) -> LibraryFolderList:
    """`modelRoots` as a resource, readable with a service token.

    The reader that matters is a node's agent inheriting its path rules
    (2026-09-14): a worker reaches this through the install with a
    `service:agent` token at every launch and keeps a copy for the
    spawns the library is not around for. The config trio it mirrors is
    operator-only, which is why this exists; nothing here is secret --
    the same paths are on every model below.
    """
    config: ConfigStore = request.app.state.config_store
    return LibraryFolderList(folders=config.library_folders())


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/library#{slug}",
            title=title,
            status=status_code,
            detail=detail,
            component="library",
        ).model_dump(exclude_none=True),
    )


@router.get("/v1/models", response_model=LibraryModelList)
async def list_models(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Reverse lookup: the single entry at this absolute path, or nothing.",
    ),
) -> LibraryModelList:
    """Every model, including ones whose files are gone.

    `missing` entries are listed rather than hidden: they carry the
    profiles someone spent an afternoon tuning, and those are the one
    thing here that cannot be recovered by looking at the disk again.
    """
    store: StateStore = request.app.state.state_store

    if path is not None:
        found = store.find_by_path(path)
        # Normalization happens store-side, which is the entire point of
        # this parameter: a caller holding a path should not have to
        # reproduce the id derivation, case-folding and all.
        models = [found] if found else []
    else:
        models = sorted(store.list_models(), key=lambda m: m.name.lower())

    return LibraryModelList(models=models, lastScanAt=store.last_scan_at)


@router.post("/v1/eligibility", response_model=EligibilityList)
async def judge_eligibility(request: Request, body: EligibilityRequest) -> EligibilityList:
    """Which engines can run each model, and the one dot it carries (LS1),
    and each candidate not downloaded yet, by its facts (LS2).

    A read, so a service token may ask: the agent's Run does, with its
    own engines. An id the library does not know is left out rather than
    failing the rest.
    """
    store: StateStore = request.app.state.state_store
    candidates = body.candidates or []
    if body.models is None and not candidates:
        models = sorted(store.list_models(), key=lambda m: m.name.lower())
    else:
        models = [m for m in (store.get_model(i) for i in body.models or []) if m is not None]
    question = await _fit_question(request, body.fit) if body.fit is not None else None
    return EligibilityList(
        models=[
            *(eligibility.judge(m, body.engines, question) for m in models),
            # Not downloaded yet (LS2): Discover's rows, versions and starters.
            *(eligibility.judge_candidate(c, body.engines, question) for c in candidates),
        ]
    )


async def _fit_question(request: Request, asked: FitQuestion) -> engine_fit.Question:
    """The node's memory as the caller sent it, or this host's when it sent
    no numbers, and the context to score at (LS6)."""
    config: ConfigStore = request.app.state.config_store
    detected = None if engine_fit.has_numbers(asked) else await asyncio.to_thread(hardware.detect)
    return engine_fit.Question(
        budget=engine_fit.budget_of(asked, detected),
        context_length=asked.contextLength or config.guidance_context_length(),
    )


def _write_provenance(body: PreparedModelRequest, roots: list[Path]) -> tuple[Path, Path]:
    """Choose the folder, check the entry where it can be checked, and
    create the file: `(provenance file, its Library folder)`. Blocking,
    so the route runs it in a thread: a folder can be on a dead share."""
    entry = body.provenance.entry
    if body.root is None and body.subdirectory is None:
        # Beside what it describes, when that is in a Library folder.
        holder = next(
            (r for r in roots if prepared.is_absolute(entry) and is_within(entry, r)), None
        )
        root = holder or roots[0]
        folder = Path(entry).parent if holder is not None else root
    else:
        if body.root is None:
            root = roots[0]
        else:
            match = next((r for r in roots if same_path(r, body.root)), None)
            if match is None:
                raise _problem(
                    status.HTTP_400_BAD_REQUEST,
                    "Not a Library folder",
                    f"{body.root!r} is not one of this library's folders "
                    f"({', '.join(str(r) for r in roots)}).",
                )
            root = match
        folder = (root / body.subdirectory).expanduser() if body.subdirectory else root
        if not is_within(folder, root):
            raise _problem(
                status.HTTP_400_BAD_REQUEST,
                "Outside the Library folder",
                f"{body.subdirectory!r} resolves outside {root}.",
            )
    target = folder / f"{body.name}{prepared.SUFFIX}"
    written = prepared.written_entry(entry, folder)
    resolved = prepared.entry_path(target, written)
    # The Library is the home of a model's files (Troy, LS7: B19/B20 amended),
    # so an entry is in the Library folder its provenance goes into, never a
    # path on one node's own disk; and being here, it is checked now.
    if prepared.is_absolute(written) and not is_within(written, root):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Entry outside the Library",
            f"{written} is not in the Library folder {root}. {prepared.OUTSIDE_LIBRARY}",
        )
    if not os.path.isfile(resolved):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Entry file not found",
            f"The entry file {resolved} is not there. Nothing was written.",
        )
    provenance = prepared.rebased(body.provenance, entry, folder).model_copy(
        update={
            "formatVersion": body.provenance.formatVersion or prepared.FORMAT_VERSION,
            "entry": written,
            "preparedAt": body.provenance.preparedAt or datetime.now(tz=UTC),
        }
    )
    try:
        folder.mkdir(parents=True, exist_ok=True)
        prepared.write(target, provenance)
    except FileExistsError:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Already there",
            f"{target} already exists. Nothing was written; choose another name.",
        ) from None
    except OSError as exc:
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Cannot write there",
            f"Could not write {target}: {exc}. Nothing was written.",
        ) from exc
    return target, root


@router.post(
    "/v1/models/prepared",
    response_model=LibraryModel,
    status_code=201,
    dependencies=[Depends(require_operator)],
)
async def add_prepared_model(request: Request, body: PreparedModelRequest) -> LibraryModel:
    """Adopt a model an engine prepared, by writing its provenance file (LS3).

    One small file in a Library folder; the engine's own files are never
    read, copied or moved. Listed at once rather than at the next walk,
    after any walk in flight, which could otherwise drop it from the list
    it installs.
    """
    config: ConfigStore = request.app.state.config_store
    store: StateStore = request.app.state.state_store
    roots = config.model_roots()
    if not roots:
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "No Library folder",
            "No Library folders are configured, so there is nowhere to keep the provenance "
            "file. Add one under Library → Folders.",
        )
    target, root = await asyncio.to_thread(_write_provenance, body, roots)
    scans: ScanManager | None = getattr(request.app.state, "scan_manager", None)
    if scans is not None:
        await scans.wait()
    found = await asyncio.to_thread(prepared.model_from, target, root)
    model = prepared.link_sources([*store.list_models(), found])[-1]
    return store.add_model(model, seen_at=datetime.now(tz=UTC))


def _deletion_context(request: Request) -> dict[str, Any]:
    """What can hold a model back from deletion: the downloads writing
    files, and the run operations still working on a model."""
    manager = getattr(request.app.state, "download_manager", None)
    journal = getattr(request.app.state, "run_operations", None)
    return {
        "downloads": manager.records() if manager is not None else [],
        "operations": journal.list(pending_only=True) if journal is not None else [],
    }


def _present(store: StateStore, model_id: str) -> LibraryModel:
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    return model


@router.get(
    "/v1/models/{model_id}/deletion",
    response_model=ModelDeletion,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator)],
)
async def plan_model_deletion(request: Request, model_id: str) -> ModelDeletion:
    """What Delete would remove, keep and refuse, read now (LS8). Nothing
    is removed."""
    store: StateStore = request.app.state.state_store
    model = _present(store, model_id)
    listed = store.list_models()
    context = _deletion_context(request)

    def answer() -> ModelDeletion:
        return deletion.as_answer(deletion.plan(model, listed, **context), listed, **context)

    return await asyncio.to_thread(answer)


@router.post(
    "/v1/models/{model_id}/delete",
    response_model=ModelDeleted,
    dependencies=[Depends(require_operator)],
)
async def delete_model(request: Request, model_id: str, body: ModelDeleteRequest) -> ModelDeleted:
    """Delete a model's files and the model, all or nothing (LS8): what
    the confirmed plan named, and the prepared models made from it the
    person ticked."""
    config: ConfigStore = request.app.state.config_store
    store: StateStore = request.app.state.state_store
    scans: ScanManager | None = getattr(request.app.state, "scan_manager", None)
    if scans is not None:
        # A walk in flight could list the files again after they go.
        await scans.wait()
    model = _present(store, model_id)
    listed = store.list_models()
    context = _deletion_context(request)
    also = list(dict.fromkeys(body.alsoDelete or []))

    def run() -> ModelDeleted:
        shown = deletion.plan(model, listed, **context)
        if shown.token != body.token:
            raise _problem(
                status.HTTP_409_CONFLICT,
                "The model changed",
                "Its files changed since the confirmation was read. Nothing was deleted: "
                "open Delete again to see what it would remove now.",
            )
        if shown.refusal:
            raise _problem(status.HTTP_409_CONFLICT, "Cannot delete it now", shown.refusal)
        dependents = {m.id: m for m in shown.prepared_from}
        unknown = [a for a in also if a not in dependents]
        if unknown:
            raise _problem(
                status.HTTP_409_CONFLICT,
                "Not made from it",
                f"{', '.join(unknown)} is not prepared from {model.name}. Nothing was deleted.",
            )
        together = {model.id, *also}
        plans = [deletion.plan(model, listed, also=also, **context)]
        for other in also:
            theirs = deletion.plan(dependents[other], listed, also=together, **context)
            if theirs.refusal:
                raise _problem(
                    status.HTTP_409_CONFLICT,
                    "Cannot delete it now",
                    f"{dependents[other].name}: {theirs.refusal}",
                )
            plans.append(theirs)
        paths = list(dict.fromkeys(f.path for p in plans for f in p.files))
        try:
            deleted = deletion.remove(paths, config.model_roots())
        except deletion.DeleteRefused as exc:
            raise _problem(status.HTTP_409_CONFLICT, "A file is in use", str(exc)) from exc
        for p in plans:
            store.forget_model(p.model.id)
        kept = list(dict.fromkeys(k.path for p in plans for k in p.kept))
        return ModelDeleted(
            models=[p.model.id for p in plans],
            deleted=deleted,
            kept=kept,
            bytesFreed=sum(p.bytes_freed for p in plans),
        )

    return await asyncio.to_thread(run)


@router.get("/v1/models/{model_id}", response_model=LibraryModel)
async def get_model(request: Request, model_id: str) -> LibraryModel:
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    return model


@router.delete(
    "/v1/models/{model_id}",
    status_code=204,
    dependencies=[Depends(require_operator)],
)
async def forget_model(request: Request, model_id: str) -> None:
    """Forget a missing entry and the profiles saved against it.

    **No file is ever touched.** What this deletes is the library's own
    record, which is to say the profiles — the only thing this component
    persists.

    Refused for a model still on disk: the next scan would find it
    again, so a delete that appeared to work would be a lie. That makes
    this precisely the "yes, I moved that model and I don't want its old
    settings back" button, and nothing else.
    """
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    if model.status == ModelStatus.present:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Model is present",
            (
                f"{model.path} is still on disk, so forgetting it would only last until the "
                "next scan. Nothing was deleted. Delete the file yourself if that is what "
                "you meant; this endpoint only forgets entries whose files are already gone."
            ),
        )
    store.forget_model(model_id)
