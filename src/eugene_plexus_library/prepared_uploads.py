"""A preparation's files, written into the Library folder by the Library (LS10).

library-sources-and-engines.md §6.13. An engine prepares a model on its own
node, in a folder of that node's (setup writes there, marks and all), and
sends the files the prepared model is made of here, a chunk at a time, under
the run's lease. The Library writes them itself, as its own account, so a
node never needs to be able to write in a Library folder: on a NAS share,
whatever account a node reaches the share as (a Windows service reaches it
as the machine, which a workgroup NAS treats as its guest) needs only read.

Where it writes (B105): only inside an engine's own folder at the top of the
run's Library folder, `<Engine>-data` holding `.eugene-engine-files` (B44),
which it makes and marks when there is none. A folder of that name without
the marker is the person's own, and nothing is written in it. Never the
provenance file (`POST /{id}/prepared` writes that), never a name of its own
partial files.

How (B104): bytes land in `<file>.eugene-upload` at the offset the node
names, which must be what has arrived so far (or 0, which starts it again);
completing checks the size and SHA-256 the node states, then renames, so a
file under its own name is only ever whole. A send cut off carries on from
what arrived.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import prepared

#: Bytes of a file still arriving, beside it. Not a name any engine opens.
PARTIAL_SUFFIX = ".eugene-upload"

#: The most one chunk may carry: what an agent's proxy passes on (32 MiB).
MAX_CHUNK_BYTES = 32 * 1024 * 1024

#: An engine's own folder at the top of a Library folder (B43: `Tom-data`).
ENGINE_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,62}-data$")

#: Folders below the engine's own folder, at most.
MAX_DEPTH = 12

MARKER_TEXT = (
    "An engine's own files, written by Eugene Plexus's Library for models it prepared.\n"
    "The Library lists the *.eugene-prepared.json files in this folder and reads nothing else\n"
    "here. Deleting a model's .eugene-prepared.json removes it from the Library.\n"
)

#: Names Windows will not make, whatever their extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10))}
_RESERVED |= {f"LPT{i}" for i in range(1, 10)}

_HASH_CHUNK = 4 * 1024 * 1024


class UploadRefused(Exception):
    """Nothing was written, and why, with the HTTP status that says so."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Target:
    """Where one sent file goes, checked."""

    #: The engine's own folder it is inside.
    own: Path
    #: The file under its own name.
    path: Path

    @property
    def partial(self) -> Path:
        return self.path.with_name(self.path.name + PARTIAL_SUFFIX)


def target(folder: Path, path: str) -> Target:
    """`path` (relative to the Library folder `folder`, `/` between folders)
    if it is a place a preparation may write, or why not (B105)."""
    if not path or "\\" in path or ":" in path or path.startswith("/"):
        raise UploadRefused(
            400,
            f"{path!r} is not a path inside the Library folder: give it relative to the "
            "folder, with / between folders.",
        )
    parts = path.split("/")
    for part in parts:
        if part in ("", ".", "..") or part != part.strip() or len(part) > 255:
            raise UploadRefused(400, f"{path!r} has a part the Library will not write: {part!r}.")
        if any(ord(c) < 32 for c in part) or part.split(".")[0].upper() in _RESERVED:
            raise UploadRefused(400, f"{path!r} has a part the Library will not write: {part!r}.")
    if len(parts) < 2 or not ENGINE_FOLDER.match(parts[0]):
        raise UploadRefused(
            400,
            f"A preparation writes only inside an engine's own folder at the top of the "
            f"Library folder (a name ending -data, such as Strata-data), not at {path!r}.",
        )
    if len(parts) > MAX_DEPTH + 1:
        raise UploadRefused(400, f"{path!r} is deeper than the Library writes.")
    last = parts[-1]
    if last.endswith(prepared.SUFFIX):
        raise UploadRefused(
            400, f"{last} is a provenance file: the Library writes it when the run lists the model."
        )
    if last.endswith(PARTIAL_SUFFIX) or last == prepared.ENGINE_FILES_MARKER:
        raise UploadRefused(400, f"{last} is a name the Library keeps for itself.")
    own = folder / parts[0]
    return Target(own=own, path=own.joinpath(*parts[1:]))


def _own_folder(found: Target) -> None:
    """The engine's own folder, made and marked if there is none (B44); a
    folder of that name the person made is left alone."""
    own = found.own
    if _linked(own):
        raise UploadRefused(
            409, f"{own} is a link, not an engine's own folder: nothing was written in it."
        )
    if own.exists():
        if not own.is_dir() or not (own / prepared.ENGINE_FILES_MARKER).is_file():
            raise UploadRefused(
                409,
                f"{own} is not an engine's own folder (it has no {prepared.ENGINE_FILES_MARKER}), "
                "so it is yours and nothing was written in it. Rename it, then prepare the "
                "model again.",
            )
        return
    own.mkdir()
    marker = own / prepared.ENGINE_FILES_MARKER
    with open(marker, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(MARKER_TEXT)


def _linked(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _inside(found: Target) -> None:
    """The file's folders, made as needed, one at a time below the engine's
    own folder, none of them a link, so nothing is written outside it."""
    here = found.own
    for part in found.path.relative_to(found.own).parts[:-1]:
        here = here / part
        if _linked(here):
            raise UploadRefused(409, f"{here} is a link: nothing was written through it.")
        if not here.exists():
            here.mkdir()
        elif not here.is_dir():
            raise UploadRefused(409, f"{here} is a file, not a folder: nothing was written.")
    if _linked(found.path) or _linked(found.partial):
        raise UploadRefused(409, f"{found.path} is a link: nothing was written.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def state(found: Target) -> dict[str, object]:
    """What the Library holds at `found`: the whole file (its size and
    SHA-256), and what has arrived of a send not yet complete."""
    size: int | None = None
    sha: str | None = None
    if found.path.is_file() and not _linked(found.path):
        size = found.path.stat().st_size
        sha = sha256_of(found.path)
    received = found.partial.stat().st_size if found.partial.is_file() else 0
    return {"sizeBytes": size, "sha256": sha, "receivedBytes": received}


def _full(found: Target, exc: OSError) -> UploadRefused:
    try:
        free = shutil.disk_usage(found.own.parent).free
        room = f" ({free / 1e9:.1f} GB free)"
    except OSError:
        room = ""
    return UploadRefused(
        507,
        f"The drive holding the Library folder {found.own.parent} is full{room}, so "
        f"{found.path.name} could not be written: {exc.strerror or exc}. Make room, then "
        "prepare the model again.",
    )


def write(found: Target, offset: int, data: bytes) -> int:
    """`data` at `offset` of the file arriving; what has arrived after it.
    `offset` is what has arrived so far, or 0 to start the file again."""
    _own_folder(found)
    _inside(found)
    partial = found.partial
    have = partial.stat().st_size if partial.is_file() else 0
    if offset not in (0, have):
        raise UploadRefused(
            409,
            f"{have} bytes of {found.path.name} have arrived, not {offset}: send from {have}.",
        )
    try:
        with open(partial, "wb" if offset == 0 else "r+b") as handle:
            handle.seek(offset)
            handle.write(data)
            return handle.tell()
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise _full(found, exc) from exc
        raise UploadRefused(
            409, f"The Library could not write {partial}: {exc.strerror or exc}"
        ) from exc


def complete(found: Target, size: int, sha256: str) -> dict[str, object]:
    """The file arrived whole, as the node says it is: under its own name.
    Said again for a file already in place, it changes nothing."""
    _own_folder(found)
    _inside(found)
    partial = found.partial
    if not partial.is_file():
        held = state(found)
        if held["sizeBytes"] == size and held["sha256"] == sha256:
            return held
        raise UploadRefused(409, f"Nothing of {found.path.name} has arrived: send it first.")
    have = partial.stat().st_size
    if have != size:
        raise UploadRefused(
            409, f"{have} of {size} bytes of {found.path.name} have arrived: send the rest."
        )
    if sha256_of(partial) != sha256:
        partial.unlink(missing_ok=True)
        raise UploadRefused(
            409,
            f"{found.path.name} arrived damaged (its SHA-256 is not the one the node "
            "stated): send it again.",
        )
    try:
        with open(partial, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, found.path)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise _full(found, exc) from exc
        raise UploadRefused(
            409, f"The Library could not write {found.path}: {exc.strerror or exc}"
        ) from exc
    return {"sizeBytes": size, "sha256": sha256, "receivedBytes": 0}
