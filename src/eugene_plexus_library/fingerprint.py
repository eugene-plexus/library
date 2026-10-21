"""Content fingerprints: the proof that two files are the same file (row 22).

A model's `id` is the hash of its **path**, so a file that moved is a new
entry, and the profiles stay on the old one. Before they may move, Eugene
has to know the new file is the same file, and neither the name nor the size
is proof: two quants of one model can have the same size, and two downloads
can have the same name.

A fingerprint is `v1:` and a SHA-256 over:

* the format;
* every file's role and size;
* the first 1 MiB of each file (a safetensors file's whole JSON header
  instead when that is longer, up to 100 MiB);
* eight evenly spaced 64 KiB blocks;
* the last 64 KiB.

About 1.6 MiB is read per file, never the whole of a 40 GB model. A file
small enough that those regions would overlap is read whole.

It names **no path and no date**, so a file that moved or was copied has the
same fingerprint at its new place. A sampled fingerprint can only collide
between files of the same size with the same header and the same samples.
Version 1 is written in the string, so a later version can change the
sampling without a fingerprint of this one ever comparing equal to it.

A file that cannot be read gives no fingerprint (`stamp`: a warning in the
log), and an entry without one can still be moved, but only after the person
confirms, because Eugene cannot prove anything about it.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from ._causes import plain_cause
from ._generated.models import LibraryModel, ModelFile, ModelFileRole, ModelStatus

log = logging.getLogger(__name__)

VERSION = "v1"
HEAD_BYTES = 1024 * 1024
BLOCK_BYTES = 64 * 1024
BLOCK_COUNT = 8
TAIL_BYTES = 64 * 1024
HEADER_CAP_BYTES = 100 * 1024 * 1024
#: At or under this, the sampled regions would overlap, so the file is read whole.
WHOLE_FILE_BYTES = HEAD_BYTES + BLOCK_COUNT * BLOCK_BYTES + TAIL_BYTES

#: A file that only sits beside the model (a README, a licence) is not part of
#: what the model is, and a copy of the model need not carry it.
_IGNORED_ROLES = frozenset({ModelFileRole.other})


class FingerprintError(Exception):
    """A file could not be read. `path` is the file; `absent` says it is no longer there."""

    def __init__(self, path: str, cause: str, *, absent: bool = False) -> None:
        super().__init__(f"{path}: {cause}")
        self.path = path
        self.cause = cause
        self.absent = absent


def counted_files(model: LibraryModel) -> list[ModelFile]:
    """The files a fingerprint, and a size comparison, are made of."""
    return [f for f in model.files or [] if f.role not in _IGNORED_ROLES]


def _sample(handle: BinaryIO, offset: int, length: int) -> bytes:
    """Read `length` bytes at `offset`. The one place bytes leave a file, so a
    test can count them."""
    handle.seek(offset)
    return handle.read(length)


def _safetensors_header_end(handle: BinaryIO, size: int) -> int:
    """Where a safetensors file's JSON header ends: 8 bytes of length and the
    header. 0 when the length is not plausible (the file is then sampled like
    any other, and the scan reports it as unreadable)."""
    raw = _sample(handle, 0, 8)
    if len(raw) != 8:
        return 0
    (length,) = struct.unpack("<Q", raw)
    end = 8 + int(length)
    if length == 0 or length > HEADER_CAP_BYTES or end > size:
        return 0
    return end


def _digest_file(path: str, role: ModelFileRole) -> tuple[int, str]:
    """`(size, hex digest)` of one file's role, size and sampled contents."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            digest = hashlib.sha256()
            digest.update(f"{role.value}\0{size}\0".encode())
            if size <= WHOLE_FILE_BYTES:
                parts = [_sample(handle, 0, size)]
            else:
                head = HEAD_BYTES
                if path.lower().endswith(".safetensors"):
                    head = max(head, _safetensors_header_end(handle, size))
                parts = [_sample(handle, 0, head)]
                span = size - BLOCK_BYTES
                for index in range(1, BLOCK_COUNT + 1):
                    parts.append(_sample(handle, span * index // (BLOCK_COUNT + 1), BLOCK_BYTES))
                parts.append(_sample(handle, size - TAIL_BYTES, TAIL_BYTES))
    except FileNotFoundError as exc:
        raise FingerprintError(path, "it is no longer there", absent=True) from exc
    except OSError as exc:
        raise FingerprintError(path, plain_cause(exc, Path(path))) from exc
    for part in parts:
        # Length-prefixed, so one part cannot slide bytes into the next.
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return size, digest.hexdigest()


def of(model: LibraryModel) -> str | None:
    """The fingerprint of a model's files as they are on disk now.

    `None` when the model lists no file to read. Raises `FingerprintError` for
    the first file that cannot be read: a fingerprint of some of a model is
    never a fingerprint of the model.
    """
    files = counted_files(model)
    if not files:
        return None
    entries = sorted((f.role.value, *_digest_file(f.path, f.role)) for f in files)
    outer = hashlib.sha256()
    outer.update(f"{VERSION}\0{model.format.value}\0{len(entries)}\0".encode())
    for role, size, digest in entries:
        outer.update(f"{role}\0{size}\0{digest}\0".encode())
    return f"{VERSION}:{outer.hexdigest()}"


def stamp(model: LibraryModel) -> LibraryModel:
    """The model with its fingerprint, or as it was when it cannot be read
    (a warning says which file). Blocking disk reads: run it in a thread."""
    if model.status is not ModelStatus.present:
        return model
    try:
        found = of(model)
    except FingerprintError as exc:
        log.warning("no fingerprint for %s: %s", model.path, exc)
        return model.model_copy(update={"fingerprint": None})
    return model.model_copy(update={"fingerprint": found})


def stamp_all(
    models: list[LibraryModel], *, should_cancel: Callable[[], bool] = lambda: False
) -> list[LibraryModel]:
    """Fingerprint every present model that has none, one file set at a time.

    Stops between models when cancelled: what is read is kept, and the next
    scan reads the rest. An entry that already has a fingerprint (a cached one,
    unchanged on disk) is left alone, so a rescan reads nothing for it.
    """
    out: list[LibraryModel] = []
    for model in models:
        if model.fingerprint is None and not should_cancel():
            out.append(stamp(model))
        else:
            out.append(model)
    return out


# -- comparing ----------------------------------------------------------------


def _size_list(model: LibraryModel) -> list[tuple[str, int | None]]:
    return sorted(
        ((f.role.value, f.sizeBytes) for f in counted_files(model)),
        key=lambda pair: (pair[0], -1 if pair[1] is None else pair[1]),
    )


def _quantization(model: LibraryModel) -> str | None:
    if model.gguf is not None:
        return model.gguf.quantization
    if model.safetensors is not None:
        mlx = model.safetensors.mlxQuantization
        return (
            f"{model.safetensors.dtype}; {mlx.model_dump(mode='json')}"
            if mlx
            else (model.safetensors.dtype)
        )
    return None


def difference(missing: LibraryModel, found: LibraryModel) -> str | None:
    """The first plain difference between the file that went missing and one
    that is there, from what the entries already say. `None` when none shows.

    Used when no fingerprint can say: the entry went missing before they
    existed, and to say *what* differs when they disagree.
    """
    if missing.format != found.format:
        return f"its format is {found.format.value}, not {missing.format.value}"
    before, after = _size_list(missing), _size_list(found)
    if len(before) != len(after):
        return f"it has {len(after)} files, and the missing model had {len(before)}"
    for (role, was), (_, now) in zip(before, after, strict=True):
        if was != now:
            return f"its {role} file is {_bytes(now)}, and the missing model's was {_bytes(was)}"
    details: list[tuple[str, object, object]] = [
        ("architecture", missing.architecture, found.architecture),
        ("parameters", missing.parameters, found.parameters),
        ("quantization", _quantization(missing), _quantization(found)),
        ("context length", missing.contextLength, found.contextLength),
    ]
    for what, had, has in details:
        if had != has:
            return f"its {what} is {has if has is not None else 'not given'}, not {had}"
    return None


def _bytes(count: int | None) -> str:
    return "of unknown size" if count is None else f"{count:,} bytes"
