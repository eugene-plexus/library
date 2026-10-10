"""Delete a model from the Library, any format (LS8).

library-sources-and-engines.md §6.8 (Troy: a Delete for every model) and
§6.11. Today's `DELETE /v1/models/{id}` forgets a missing entry and touches
no file; this removes a present model's files from disk.

**One rule for shared parts.** A file a model is made of is kept when
another listed model names the same file (a projector several quants share,
Strata's MTP helper, and later an image model's VAE or text encoder): no
special case per format or engine. What a model is made of is its
`LibraryModel.files`, which the scan fills per format (a GGUF's shards and
projector, a safetensors folder's files, a prepared model's provenance,
entry and recorded files).

**All or nothing per delete.** Each file is first moved aside under a hidden
name in its own folder; if one cannot be (Windows refuses to move a file a
running engine holds), the ones already moved are put back and nothing is
deleted. Then the moved files are removed, and folders left empty are
removed up to the Library folder.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._generated.models import (
    DownloadState,
    KeptModelFile,
    LibraryModel,
    ModelDeletion,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelRef,
    ModelStatus,
    PreparedDependent,
)
from .paths import is_within

#: Downloads still writing files.
WRITING = {
    DownloadState.queued,
    DownloadState.resolving,
    DownloadState.downloading,
    DownloadState.verifying,
    DownloadState.paused,
}
#: Run operation steps after which nothing more is done to the model.
FINISHED_STEPS = {"ready", "skipped", "failed", "cancelled"}
MOVED_ASIDE = ".eugene-deleting-"


class DeleteRefused(Exception):
    """Nothing was deleted, and why."""


def _key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def own_files(model: LibraryModel) -> list[ModelFile]:
    """Every file the model is made of, as the scan listed it; a folder
    model (safetensors) by every file in its folder too, so nothing of it
    is left behind."""
    files = list(model.files or [])
    seen = {_key(f.path) for f in files}
    if model.format is ModelFormat.safetensors and os.path.isdir(model.path):
        for directory, _dirs, names in os.walk(model.path):
            for name in sorted(names):
                path = os.path.join(directory, name)
                if _key(path) in seen:
                    continue
                seen.add(_key(path))
                try:
                    size: int | None = os.stat(path).st_size
                except OSError:
                    size = None
                files.append(ModelFile(path=path, role=ModelFileRole.other, sizeBytes=size))
    return files


def _users(listed: Sequence[LibraryModel], skip: set[str]) -> dict[str, list[LibraryModel]]:
    """Each file another model (not in `skip`) names, and who names it."""
    out: dict[str, list[LibraryModel]] = {}
    for other in listed:
        if other.id in skip:
            continue
        for f in own_files(other):
            out.setdefault(_key(f.path), []).append(other)
    return out


def refusal(
    model: LibraryModel,
    files: Iterable[ModelFile],
    *,
    downloads: Iterable[Any],
    operations: Iterable[dict[str, Any]],
) -> str | None:
    """Why it cannot be deleted now: a download writing into its files or
    its folder, or a run operation still working on it."""
    keys = {_key(f.path) for f in files}
    folder = model.path if model.format is ModelFormat.safetensors else None
    for download in downloads:
        if download.state not in WRITING:
            continue
        for item in download.files or []:
            target = item.destinationPath
            if target and (
                _key(target) in keys or (folder is not None and is_within(target, folder))
            ):
                return (
                    f"A download is writing into it ({download.repo}): let it finish or cancel "
                    "it in Downloads first."
                )
        if download.modelId == model.id:
            return (
                f"A download is writing into it ({download.repo}): let it finish or cancel it "
                "in Downloads first."
            )
    for record in operations:
        if record.get("step") in FINISHED_STEPS:
            continue
        ids = {
            (record.get("model") or {}).get("id"),
            (record.get("preparedFrom") or {}).get("id"),
        }
        if model.id in ids:
            return (
                f"Run is {record.get('step') or 'working'} on it: let it finish or cancel it first."
            )
    return None


@dataclass
class Plan:
    model: LibraryModel
    files: list[ModelFile]
    kept: list[KeptModelFile]
    refusal: str | None
    profiles: int
    prepared_from: list[LibraryModel] = field(default_factory=list)

    @property
    def bytes_freed(self) -> int:
        return sum(f.sizeBytes or 0 for f in self.files)

    @property
    def token(self) -> str:
        body = json.dumps(
            {
                "files": sorted((f.path, f.sizeBytes) for f in self.files),
                "kept": sorted(k.path for k in self.kept),
                "prepared": sorted(m.id for m in self.prepared_from),
            }
        )
        return hashlib.sha256(body.encode()).hexdigest()[:32]


def plan(
    model: LibraryModel,
    listed: Sequence[LibraryModel],
    *,
    downloads: Iterable[Any] = (),
    operations: Iterable[dict[str, Any]] = (),
    also: Iterable[str] = (),
) -> Plan:
    """What deleting `model` (and the models in `also` with it) would
    remove and keep. A file is kept when a model not being deleted names it."""
    downloads = list(downloads)
    operations = list(operations)
    skip = {model.id, *also}
    users = _users(listed, skip)
    files: list[ModelFile] = []
    kept: list[KeptModelFile] = []
    for f in own_files(model):
        others = users.get(_key(f.path))
        if others:
            kept.append(
                KeptModelFile(
                    path=f.path,
                    sizeBytes=f.sizeBytes,
                    usedBy=[ModelRef(id=o.id, name=o.name) for o in others],
                )
            )
        else:
            files.append(f)
    prepared_from = [
        m
        for m in listed
        if m.id != model.id and m.prepared is not None and m.prepared.sourceModelId == model.id
    ]
    why = refusal(model, files, downloads=downloads, operations=operations)
    if why is None and model.status is not ModelStatus.present:
        why = "Its files are not there: Forget removes it from the Library."
    return Plan(
        model=model,
        files=files,
        kept=kept,
        refusal=why,
        profiles=model.profileCount or 0,
        prepared_from=prepared_from,
    )


def as_answer(p: Plan, listed: Sequence[LibraryModel], **context: Any) -> ModelDeletion:
    dependents = []
    for m in p.prepared_from:
        theirs = plan(m, listed, **context)
        dependents.append(
            PreparedDependent(
                id=m.id, name=m.name, bytesFreed=theirs.bytes_freed, refusal=theirs.refusal
            )
        )
    return ModelDeletion(
        modelId=p.model.id,
        files=p.files,
        kept=p.kept,
        bytesFreed=p.bytes_freed,
        profiles=p.profiles,
        preparedFrom=dependents,
        refusal=p.refusal,
        token=p.token,
    )


def remove(paths: Sequence[str], roots: Sequence[Path]) -> list[str]:
    """Move every file aside, then remove them; or move them back and
    refuse, naming the file that could not be moved. Folders left empty go,
    up to the Library folder holding them."""
    moved: list[tuple[str, str]] = []
    tag = secrets.token_hex(4)
    for path in paths:
        if not os.path.exists(path):
            continue
        aside = os.path.join(os.path.dirname(path), f"{MOVED_ASIDE}{tag}-{os.path.basename(path)}")
        try:
            os.replace(path, aside)
        except OSError as exc:
            for original, hidden in reversed(moved):
                with contextlib.suppress(OSError):
                    os.replace(hidden, original)
            raise DeleteRefused(
                f"{path} is in use ({exc.strerror or exc}): stop the model on every machine "
                "running it, then delete it again. Nothing was deleted."
            ) from exc
        moved.append((path, aside))
    deleted = []
    for original, hidden in moved:
        try:
            os.remove(hidden)
        except OSError:
            # Moved aside already: the model is gone from its place; a
            # leftover hidden file is reported by the next scan's size.
            continue
        deleted.append(original)
    for original, _hidden in moved:
        _prune(Path(original).parent, roots)
    return deleted


def _prune(folder: Path, roots: Sequence[Path]) -> None:
    root = next((r for r in roots if is_within(str(folder), r)), None)
    if root is None:
        return
    current = folder
    while _key(str(current)) != _key(str(root)) and is_within(str(current), root):
        try:
            current.rmdir()  # only when empty
        except OSError:
            return
        current = current.parent
