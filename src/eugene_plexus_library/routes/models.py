"""Model routes: list, read, forget, and re-point a moved file's profiles."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .. import deletion, eligibility, engine_fit, fingerprint, hardware, prepared
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
    ModelFormat,
    ModelRelocated,
    ModelStatus,
    PreparedModelRequest,
    Problem,
    ProfileRename,
    RelocateCandidate,
    RelocateCandidateList,
    RelocateRequest,
    SpecialTokens,
)
from ..config import ConfigStore
from ..dependencies import require_operator
from ..formats import gguf
from ..model_pages import ModelPages
from ..paths import is_within, real_within, same_path
from ..scan_manager import ScanManager
from ..store import RelocateRefused, StateStore

router = APIRouter(tags=["models"])


@router.get("/v1/folders", response_model=LibraryFolderList, response_model_exclude_none=True)
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
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> JSONResponse:
    """Every model, including ones whose files are gone.

    `missing` entries are listed rather than hidden: they carry the
    profiles someone spent an afternoon tuning, and those are the one
    thing here that cannot be recovered by looking at the disk again.
    """
    store: StateStore = request.app.state.state_store
    next_cursor = None

    if path is not None:
        found = store.find_by_path(path)
        # Normalization happens store-side, which is the entire point of
        # this parameter: a caller holding a path should not have to
        # reproduce the id derivation, case-folding and all.
        models = [found] if found else []
    else:
        # Validate here, after the reverse lookup, so path ignores even bad
        # paging parameters and invalid limits answer the contracted 400.
        page_limit = None
        if limit is not None:
            try:
                page_limit = int(limit) if limit.isascii() and limit.isdecimal() else 0
            except ValueError:
                page_limit = 0
            if not 1 <= page_limit <= 500:
                raise _problem(
                    400, "Invalid page limit", "The limit must be an integer between 1 and 500."
                )
        pager: ModelPages = request.app.state.model_pages
        try:
            models, next_cursor = pager.page(store.list_models(), page_limit, cursor)
        except ValueError as exc:
            raise _problem(400, "Invalid model cursor", str(exc)) from exc

    result = LibraryModelList(models=models, lastScanAt=store.last_scan_at, nextCursor=next_cursor)
    # Omit only this new field on a last/unpaginated/path response. Recursive
    # exclude_none/unset would also change the legacy model and scan fields.
    return JSONResponse(
        result.model_dump(mode="json", exclude={"nextCursor"} if next_cursor is None else set())
    )


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


def _read_only_folder(path: str, stores: list[Path]) -> HTTPException | None:
    """The refusal for a path in an Ollama store: the Library never writes there."""
    if any(is_within(path, store) or real_within(path, store) for store in stores):
        return _problem(
            status.HTTP_409_CONFLICT,
            "Ollama manages this folder",
            f"{path} is in an Ollama model store. The Library reads it and never writes there, "
            "so no provenance file goes into it.",
        )
    return None


def _write_provenance(
    body: PreparedModelRequest, roots: list[Path], stores: list[Path]
) -> tuple[Path, Path]:
    """Choose the folder, check the entry where it can be checked, and
    create the file: `(provenance file, its Library folder)`. Blocking,
    so the route runs it in a thread: a folder can be on a dead share.

    `roots` are the folders that may be written; `stores` the Ollama stores,
    which never are (CX9)."""
    entry = body.provenance.entry
    if body.root is not None and (refused := _read_only_folder(body.root, stores)):
        raise refused
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
    # Where it comes to, not the folder named: a store can sit inside a writable
    # folder, and the entry's own folder is where the file goes.
    if refused := _read_only_folder(str(folder), stores):
        raise refused
    target = folder / f"{body.name}{prepared.SUFFIX}"
    for candidate in (target, target.with_name(target.name + ".partial")):
        if refused := _read_only_folder(str(candidate), stores):
            raise refused
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
    roots = config.writable_roots()
    if not roots:
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "No Library folder",
            "No Library folders are configured, so there is nowhere to keep the provenance "
            "file. Add one under Library → Folders.",
        )
    target, root = await asyncio.to_thread(_write_provenance, body, roots, config.ollama_roots())
    scans: ScanManager | None = getattr(request.app.state, "scan_manager", None)
    if scans is not None:
        await scans.wait()
    found = await asyncio.to_thread(prepared.model_from, target, root)
    model = prepared.link_sources([*store.list_models(), found])[-1]
    # Fingerprinted here as a scan would, so its files can be recognised if they move.
    model = await asyncio.to_thread(fingerprint.stamp, model)
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


def _refuse_ollama(model: LibraryModel, stores: list[Path]) -> None:
    """An Ollama model is Ollama's file: the Library lists it and never deletes it.

    By where its files are, not only by what the scan called it: a store can sit
    inside a writable folder."""
    paths = [model.path, *(f.path for f in model.files or [])]
    if model.ollama is not None or any(
        is_within(p, s) or real_within(p, s) for p in paths for s in stores
    ):
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Ollama manages this model",
            f"{model.name} is Ollama's file, in Ollama's own store, and the Library does not "
            "delete it. Nothing was deleted.",
        )


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
    _refuse_ollama(model, request.app.state.config_store.ollama_roots())
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
    _refuse_ollama(model, config.ollama_roots())
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
            deleted = deletion.remove(
                paths, config.writable_roots(), protected=config.ollama_roots()
            )
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


def _refused(exc: RelocateRefused) -> HTTPException:
    return _problem(exc.status, exc.title, exc.detail)


@router.get(
    "/v1/models/{model_id}/relocate-candidates",
    response_model=RelocateCandidateList,
    response_model_exclude_none=True,
)
async def list_relocate_candidates(request: Request, model_id: str) -> RelocateCandidateList:
    """Where a missing model's file may have gone (row 22): the present models
    that have its fingerprint (`same`), or, for an entry from before
    fingerprints, that match its size and details (`likely`). A read: nothing
    is moved."""
    store: StateStore = request.app.state.state_store
    _present(store, model_id)
    return RelocateCandidateList(
        candidates=[
            RelocateCandidate(model=model, match=match, differences=differences)
            for model, match, differences in store.relocate_candidates(model_id)
        ]
    )


@router.post(
    "/v1/models/{model_id}/relocate",
    response_model=ModelRelocated,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator)],
)
async def relocate_model(request: Request, model_id: str, body: RelocateRequest) -> ModelRelocated:
    """Move a missing model's profiles to the same file at its new path (row 22).

    Only on a click: nothing here is called by a scan. The profiles move to a
    file with the same contents, which the new file is read for **now**, or, for
    an entry from before fingerprints, after `confirmLikely`. The missing entry
    is forgotten in the same transaction. No file is touched.
    """
    store: StateStore = request.app.state.state_store
    try:
        # Asked before a byte is read, so a call that cannot work costs nothing.
        _, target = store.check_relocation(model_id, body.to)
        try:
            found = await asyncio.to_thread(fingerprint.of, target)
        except fingerprint.FingerprintError as exc:
            if exc.absent:
                raise RelocateRefused("File not there", f"{exc.path} is no longer there.") from exc
            raise RelocateRefused(
                "File unreadable",
                f"Eugene could not read {exc.path} to check it: {exc.cause}.",
            ) from exc
        if found is None:
            raise RelocateRefused(
                "File unreadable",
                f"Eugene could not read {target.path} to check it: it lists no files.",
            )
        done = store.relocate(
            model_id, body.to, found=found, confirm_likely=bool(body.confirmLikely)
        )
    except RelocateRefused as exc:
        raise _refused(exc) from exc
    return ModelRelocated(
        model=done.model,
        moved=done.moved,
        renamed=[ProfileRename(name=was, renamedTo=now) for was, now in done.renamed],
        warnings=done.warnings,
    )


@router.get("/v1/models/{model_id}", response_model=LibraryModel)
async def get_model(request: Request, model_id: str) -> LibraryModel:
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    return model


#: (path, mtime, size) -> spellings. A vocabulary is read whole for this, so
#: it is read once per file; a model replaced on disk is read again.
_SPECIAL_CACHE: dict[tuple[str, int, int], list[str]] = {}
_SPECIAL_CACHE_MAX = 64
_SPECIAL_LOCK = threading.Lock()


@router.get("/v1/models/{model_id}/special-tokens", response_model=SpecialTokens)
async def get_model_special_tokens(request: Request, model_id: str) -> SpecialTokens:
    """The spellings this model's engine reads as special tokens in text (DI4).

    The gateway breaks each one in user and tool content, so text in a
    message cannot act as an end of turn or a new system turn."""
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    if model.format != ModelFormat.gguf or not model.path:
        return SpecialTokens(tokens=[])
    path = Path(model.path)

    def read() -> list[str]:
        # A file that cannot be read is an error, never an empty list: the
        # gateway would take "no spellings" as an answer and send this
        # model's requests unescaped without a word (DI4 review).
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with _SPECIAL_LOCK:
            cached = _SPECIAL_CACHE.get(key)
        if cached is not None:
            return cached
        found = gguf.special_tokens(path)
        with _SPECIAL_LOCK:
            while len(_SPECIAL_CACHE) >= _SPECIAL_CACHE_MAX:
                _SPECIAL_CACHE.pop(next(iter(_SPECIAL_CACHE)))
            _SPECIAL_CACHE[key] = found
        return found

    try:
        return SpecialTokens(tokens=await asyncio.to_thread(read))
    except (OSError, gguf.GgufError) as exc:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Vocabulary unreadable",
            f"Could not read the vocabulary of {path.name}: {exc}",
        ) from exc


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
        _refuse_ollama(model, request.app.state.config_store.ollama_roots())
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
