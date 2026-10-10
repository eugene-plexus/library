"""Prepared models: an engine's own files, listed through a provenance file.

A model an engine prepared for itself (Strata's expert pack, lookup table
and MTP helper, with its JSON configuration) is in the engine's own format,
which this library does not read (experimental-engines.md: prepared files
stay usable without the library parsing them). What makes it a library
model is one small file beside it, `<name>.eugene-prepared.json`
(`PreparedProvenance`): which engine, its entry file, what it was made
from (library-sources-and-engines.md §4.5, Troy's L6). That file is the
model's path, so profiles, Run and switching key on it like any other.

What the Library says of it (LS7, B22 replaced: the Library imparts what
is known and names what is not): what the engine's adapter read off its
own files when it was prepared or added (title, context, mode, its files),
measured here; what it inherits from the model it was made from, when the
Library lists that; and each fact it cannot give, with why.

The scan reads it here; `POST /v1/models/prepared` writes it here. The
agent reads the same file at every launch to find the entry it hands the
engine.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import ValidationError

from ._generated.models import (
    LibraryModel,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    PreparedDetail,
    PreparedFactMissing,
    PreparedFile,
    PreparedProvenance,
)
from .paths import is_within, model_id

SUFFIX = ".eugene-prepared.json"
#: A folder holding this file is an engine's own (LS5, B44): a preparation
#: writes it before the engine's tools start. The scan lists the provenance
#: files in that folder and reads nothing else there, nor below it: Strata's
#: MTP helper is itself a GGUF, and is not a model.
ENGINE_FILES_MARKER = ".eugene-engine-files"
#: The provenance layout this library reads and writes.
FORMAT_VERSION = 1


class PreparedError(Exception):
    """A provenance file that cannot be listed as a model, and why."""


def is_provenance(name: str) -> bool:
    return name.lower().endswith(SUFFIX)


def name_of(path: Path) -> str:
    """`qwen.eugene-prepared.json` is called `qwen`, as a GGUF is called by its stem."""
    return path.name[: -len(SUFFIX)] if is_provenance(path.name) else path.stem


def is_absolute(entry: str) -> bool:
    """Absolute under either convention: the library may run in a Linux
    container while the entry is a path on a Windows node (`C:\\Strata\\x.json`)."""
    return PurePosixPath(entry).is_absolute() or PureWindowsPath(entry).is_absolute()


def entry_path(provenance: Path, entry: str) -> str:
    """The entry on this host: as written when absolute, else beside the provenance."""
    if is_absolute(entry):
        return entry
    return os.path.normpath(provenance.parent / entry)


def read(path: Path) -> PreparedProvenance:
    """Parse a provenance file, or say what is wrong with it."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise PreparedError(f"cannot read: {exc}") from exc
    except ValueError as exc:
        raise PreparedError(f"not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PreparedError("expected a JSON object")
    version = raw.get("formatVersion", FORMAT_VERSION)
    if isinstance(version, int) and not isinstance(version, bool) and version > FORMAT_VERSION:
        raise PreparedError(
            f"written by a newer Eugene (formatVersion {version}; this library reads "
            f"{FORMAT_VERSION}). Update the library to list it."
        )
    try:
        return PreparedProvenance.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'file'}: {e['msg']}" for e in exc.errors()
        )
        raise PreparedError(problems) from exc


#: Why an entry outside the Library folders is refused, or listed unreadable
#: (Troy, LS7: B19/B20 amended).
OUTSIDE_LIBRARY = (
    "A prepared model's files live in the Library, so any node can run it and nothing is "
    "lost with a node or an engine: move the engine's folder into a Library folder, then "
    "add it from there."
)


def model_from(path: Path, root: Path) -> LibraryModel:
    """One prepared model from its provenance file, or an unreadable entry
    naming why. Never cached: the file is a few hundred bytes, and whether
    its entry is there can change without the provenance changing."""
    try:
        stat = path.stat()
        provenance = read(path)
    except (OSError, PreparedError) as exc:
        return unreadable(path, root, str(exc))
    entry = provenance.entry
    resolved = entry_path(path, entry)
    if is_absolute(entry) and not is_within(resolved, root):
        return unreadable(
            path,
            root,
            f"its entry file {resolved} is outside the Library folder {root}. {OUTSIDE_LIBRARY}",
            provenance=provenance,
        )
    found = os.path.isfile(resolved)
    if not found and not is_absolute(entry):
        # Relative means it travels with this folder, so it should be here.
        return unreadable(
            path, root, f"its entry file {resolved} is missing", provenance=provenance
        )
    files = [ModelFile(path=str(path), role=ModelFileRole.index, sizeBytes=stat.st_size)]
    if found:
        files.append(ModelFile(path=resolved, role=ModelFileRole.config, sizeBytes=_size(resolved)))
    about = detail(provenance, resolved, found=found, provenance_path=path)
    seen = {os.path.normcase(f.path) for f in files}
    for item in about.files or []:
        on_disk = os.path.normpath(path.parent / item.path)
        if os.path.normcase(on_disk) in seen:
            continue
        seen.add(os.path.normcase(on_disk))
        role = ModelFileRole.tokenizer if "tokenizer" in item.path.lower() else ModelFileRole.other
        files.append(ModelFile(path=on_disk, role=role, sizeBytes=item.sizeBytes))
    return LibraryModel(
        id=model_id(path),
        path=str(path),
        root=str(root),
        format=ModelFormat.prepared,
        name=name_of(path),
        displayName=provenance.title,
        status=ModelStatus.present,
        sizeBytes=about.diskBytes,
        fileCount=len(files),
        architecture=provenance.architecture,
        contextLength=provenance.contextLength,
        files=files,
        prepared=about,
        modifiedAt=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def detail(
    provenance: PreparedProvenance,
    resolved: str,
    *,
    found: bool,
    provenance_path: Path | None = None,
) -> PreparedDetail:
    """The provenance as the file says it, its files measured here, and
    each fact the Library cannot give, with why."""
    missing: list[PreparedFactMissing] = []
    files: list[PreparedFile] | None = None
    disk: int | None = None
    if provenance.files is None:
        missing.append(
            PreparedFactMissing(
                fact="files",
                reason="its provenance file does not list the files it is made of: add it again "
                "or prepare it again, and the engine's adapter lists them",
            )
        )
    elif provenance_path is not None:
        files, disk, absent = _measured(provenance.files, provenance_path)
        if absent:
            missing.append(
                PreparedFactMissing(
                    fact="diskBytes",
                    reason=f"{len(absent)} of its files are not where its provenance file "
                    f"says, on the Library's machine (first: {absent[0]})",
                )
            )
            disk = None
    else:
        files = list(provenance.files)
    source = provenance.source
    if source is None or not (source.path or source.repoId or source.file):
        missing.append(
            PreparedFactMissing(
                fact="source",
                reason="it was added without naming the model it was made from",
            )
        )
    if provenance.architecture is None:
        missing.append(
            PreparedFactMissing(
                fact="architecture",
                reason="its provenance file does not record it, and the model it was made from "
                "is not in the Library",
            )
        )
    if provenance.contextLength is None:
        missing.append(
            PreparedFactMissing(
                fact="contextLength",
                reason="its provenance file does not record the context it was prepared for",
            )
        )
    return PreparedDetail(
        engine=provenance.engine,
        entry=provenance.entry,
        entryPath=resolved,
        entryFound=found,
        recipe=provenance.recipe,
        recipeVersion=provenance.recipeVersion,
        source=provenance.source,
        preparedAt=provenance.preparedAt,
        title=provenance.title,
        contextLength=provenance.contextLength,
        mode=provenance.mode,
        files=files,
        diskBytes=disk,
        missing=missing or None,
    )


def _measured(
    listed: list[PreparedFile], provenance_path: Path
) -> tuple[list[PreparedFile], int, list[str]]:
    """Each listed file sized on this host, the provenance file and the
    listed ones summed, and the ones not found."""
    out: list[PreparedFile] = []
    absent: list[str] = []
    total = _size(str(provenance_path)) or 0
    for item in listed:
        size = _size(os.path.normpath(provenance_path.parent / item.path))
        if size is None:
            absent.append(item.path)
            out.append(item)
            continue
        total += size
        out.append(item.model_copy(update={"sizeBytes": size}))
    return out, total, absent


def unreadable(
    path: Path, root: Path, error: str, *, provenance: PreparedProvenance | None = None
) -> LibraryModel:
    return LibraryModel(
        id=model_id(path),
        path=str(path),
        root=str(root),
        format=ModelFormat.prepared,
        name=name_of(path),
        status=ModelStatus.unreadable,
        error=error,
        prepared=(
            detail(provenance, entry_path(path, provenance.entry), found=False)
            if provenance is not None
            else None
        ),
        displayName=provenance.title if provenance is not None else None,
    )


def link_sources(models: list[LibraryModel]) -> list[LibraryModel]:
    """Name each prepared model's source by id, when the library lists it,
    and give it what it inherits from that model (LS7, B22 replaced): its
    architecture when its own provenance does not record one, its
    parameters and size label.

    By `source.path` only, and only to a model listed in the same answer:
    a guess from a repo name could tie a model to the wrong quant of it.
    """
    listed = {m.id: m for m in models}
    out = []
    for model in models:
        prepared = model.prepared
        if prepared is not None and prepared.source is not None and prepared.source.path:
            wanted = model_id(prepared.source.path)
            source = listed.get(wanted) if wanted != model.id else None
            linked = source.id if source is not None else None
            update: dict[str, object] = {}
            if linked != prepared.sourceModelId:
                update["prepared"] = prepared.model_copy(update={"sourceModelId": linked})
            if source is not None:
                inherited = {
                    "architecture": model.architecture or source.architecture,
                    "parameters": model.parameters or source.parameters,
                    "sizeLabel": model.sizeLabel or source.sizeLabel,
                }
                update.update({k: v for k, v in inherited.items() if v is not None})
                if inherited["architecture"] is not None and prepared.missing:
                    kept = [m for m in prepared.missing if m.fact != "architecture"]
                    base = update.get("prepared", prepared)
                    assert isinstance(base, PreparedDetail)
                    update["prepared"] = base.model_copy(update={"missing": kept or None})
            if update:
                model = model.model_copy(update=update)
        out.append(model)
    return out


def _size(path: str) -> int | None:
    try:
        return os.stat(path).st_size
    except OSError:
        return None


def written_entry(entry: str, folder: Path) -> str:
    """What the provenance file says: relative when the entry lies inside its
    folder, so the two move together; otherwise as the person gave it."""
    if is_absolute(entry) and is_within(entry, folder):
        relative = Path(os.path.relpath(entry, folder))
        return relative.as_posix()
    return entry


def rebased(provenance: PreparedProvenance, entry: str, folder: Path) -> PreparedProvenance:
    """Its files relative to `folder`, where its provenance file goes: they
    come relative to the entry's folder (the agent's inspect answers so,
    LS7), which is the same folder unless the person chose another."""
    if not provenance.files or not is_absolute(entry):
        return provenance
    base = Path(entry).parent
    if os.path.normcase(os.path.normpath(base)) == os.path.normcase(os.path.normpath(folder)):
        return provenance
    files = []
    for item in provenance.files:
        try:
            moved = Path(os.path.relpath(base / item.path, folder)).as_posix()
        except ValueError:  # another drive: as absolute as it is
            moved = (base / item.path).as_posix()
        files.append(item.model_copy(update={"path": moved}))
    return provenance.model_copy(update={"files": files})


def write(path: Path, provenance: PreparedProvenance, *, replace: bool = False) -> None:
    """Create the file, refusing to replace one: `x` mode, so two adoptions
    racing for one name cannot both win. `replace` is for a preparation
    making its own file again (`replaces`), written whole or not at all."""
    body = provenance.model_dump(mode="json", exclude_none=True)
    text = json.dumps(body, indent=2) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(partial, path)


def replaces(existing: Path, provenance: PreparedProvenance, written: str) -> bool:
    """A preparation may write over a provenance file only when it is its own:
    the same engine and recipe, for the same entry (LS5, B55). Preparing a
    model again is how its configuration changes; anything else of that name
    is someone else's and stays."""
    try:
        found = read(existing)
    except PreparedError:
        return False
    return (
        provenance.recipe is not None
        and found.recipe == provenance.recipe
        and found.engine == provenance.engine
        and found.entry == written
    )
