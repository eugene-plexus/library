"""Path normalization and the model `id` derived from it.

A model's identity is its path. This module is the single place that
decides what "the same path" means, because getting it wrong twice in
two different ways produces two library entries for one file, each with
half the operator's profiles.

The `id` is the first 16 hex characters of the SHA-256 of the normalized
absolute path. Three properties matter and each is load-bearing:

* **Derived, not random** — profiles are keyed to it, so it has to
  survive a restart and a rebuilt state file.
* **URL-safe** — a Windows path cannot go in a path segment, and
  percent-encoding `C:\\Users\\...` produces URLs nobody can read in a
  log line.
* **Not content addressing** — no byte of the model is read. Hashing
  40 GB on every scan is not on the table, and a content-derived
  identity would change when nothing the operator did changed.

Symlinks are deliberately **not** resolved. Resolving them turns a
HuggingFace cache snapshot into `blobs/<sha256>`, which is content
addressing arriving through the back door, and would also collapse two
revisions of one model into one entry on the platforms where the cache
uses symlinks at all (Linux and macOS; on Windows the snapshots are
real copies).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path, PurePath

ID_LENGTH = 16
"""Hex characters kept from the SHA-256. 64 bits of a cryptographic hash
over a path space of at most a few thousand entries: a collision is not
a thing that happens, and a short id is a thing an operator can read out
of a URL."""


def _case_fold(text: str) -> str:
    """Windows and macOS filesystems are conventionally case-insensitive,
    Linux's are not. Fold on the platforms where two spellings name one
    file, and leave them distinct where they name two.

    macOS is the awkward one: APFS *can* be case-sensitive but almost
    never is. Folding there matches what users actually have; the cost
    of being wrong is two entries for one model on a rare setup, which
    is visible and fixable, where the cost of not folding on the common
    setup is the same bug for everyone.
    """
    return text.casefold() if sys.platform in ("win32", "darwin") else text


def normalize(path: str | os.PathLike[str]) -> str:
    """Canonical string form of a path, for identity purposes.

    Absolute, `..`/`.` collapsed, separators canonical for the platform,
    trailing separator removed, case-folded where the filesystem is.
    Symlinks are left alone — see the module docstring.

    Note `os.path.abspath` rather than `Path.resolve()`: resolve()
    follows symlinks, which is exactly what must not happen here.
    """
    text = os.path.abspath(os.fspath(path))
    # abspath already normalizes separators and collapses `..`; strip a
    # trailing separator so `D:\models` and `D:\models\` are one path.
    # Guard the root itself (`C:\`, `/`), where the separator is content.
    if len(text) > 1 and text.endswith((os.sep, os.altsep or os.sep)):
        stripped = text.rstrip(os.sep + (os.altsep or ""))
        # A bare drive letter would be left as `D:`, which is a
        # *relative* path on Windows ("current dir on D:"). Keep the sep.
        if stripped and not stripped.endswith(":"):
            text = stripped
    return _case_fold(text)


def model_id(path: str | os.PathLike[str]) -> str:
    """Stable, URL-safe handle for the model at `path`."""
    digest = hashlib.sha256(normalize(path).encode("utf-8")).hexdigest()
    return digest[:ID_LENGTH]


def same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    """Do these two spellings name the same path?"""
    return normalize(left) == normalize(right)


def is_within(child: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    """Is `child` at or below `parent`?

    Compares normalized forms component-wise rather than with a string
    prefix test, so `/models2` is not treated as living inside
    `/models`. Used to attribute a found model to the root it came from
    when roots overlap.
    """
    child_parts = PurePath(normalize(child)).parts
    parent_parts = PurePath(normalize(parent)).parts
    return child_parts[: len(parent_parts)] == parent_parts


def display_name(path: Path) -> str:
    """The name an operator calls this model.

    The filename with its extension stripped, or the directory name for
    a multi-file model. Deliberately *not* taken from metadata: the
    file's own name is what they downloaded and what they will look for,
    and `Runtime.modelAlias` defaults to the same thing — so plainly
    named files mean the obvious name is already the right one.

    For a sharded GGUF the shard suffix comes off too, so five parts of
    one model do not read as `…-00001-of-00005`.
    """
    from .formats.gguf import strip_shard_suffix

    if path.is_dir():
        return path.name
    return strip_shard_suffix(path.stem)
