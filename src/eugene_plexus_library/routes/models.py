"""Model routes: list, read, and forget."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from .. import eligibility, prepared
from .._generated.models import (
    EligibilityList,
    EligibilityRequest,
    LibraryFolderList,
    LibraryModel,
    LibraryModelList,
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
    return EligibilityList(
        models=[
            *(eligibility.judge(m, body.engines) for m in models),
            # Not downloaded yet (LS2): Discover's rows, versions and starters.
            *(eligibility.judge_candidate(c, body.engines) for c in candidates),
        ]
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
    # An entry in a Library folder is on this host, so it can be checked
    # now. Any other absolute entry is a path on the node that runs it.
    here = not prepared.is_absolute(written) or any(is_within(written, r) for r in roots)
    if here and not os.path.isfile(resolved):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Entry file not found",
            f"The entry file {resolved} is not there. Nothing was written.",
        )
    provenance = body.provenance.model_copy(
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
