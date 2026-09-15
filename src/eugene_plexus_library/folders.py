"""Library folders: the directories this component catalogues, and where
other machines find them (2026-09-14).

`modelRoots` was a `path_list` from M2 until now: the directories, as
this host spells them, and nothing else. Where a *node* found the same
directory lived on that node's agent, one row per node per folder --
which grew as nodes x folders, every row typed by hand. The change here
is the unit: a folder is one exported share, mounted the same way on
every node of one OS, so the folder record carries its `mounts` and
every node inherits them. The library still never opens a mount; it
states them.

Two rules this module enforces, and one it deliberately does not:

* **A bare string is a folder with no mounts.** Every config file, PATCH
  body, wizard and default variable written for `path_list` keeps
  working; `GET /v1/config` always answers in the object form, so a
  reader sees one shape.
* **A mount is absolute and shaped.** A drive letter or UNC prefix means
  Windows nodes, a leading `/` means POSIX nodes; that shape is how a
  node picks its entry, so an unshaped mount could never be picked and
  is refused at PATCH rather than silently ignored at spawn. The same
  classifier the agent uses (`is_windows_shaped` in its `model_paths`),
  restated here because components share schemas, not code.
* **Existence is not checked here.** `POST /v1/config/test` stats the
  folder's own path; a mount is a path on another machine and only that
  machine can stat it (`POST agent/v1/library/folders/check`).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ._generated.models import LibraryFolder
from .paths import normalize

log = logging.getLogger(__name__)

# `X:` followed by a separator or the end of the string. A bare `X:` is
# drive-relative on Windows and is accepted as a root here because an
# operator who types `D:` means the drive.
_DRIVE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")


def is_windows_shaped(path: str) -> bool:
    """Drive letter or UNC prefix -- the string is for a Windows host."""
    return bool(_DRIVE.match(path)) or path.startswith(("\\\\", "//"))


def is_absolute_path(path: str) -> bool:
    return is_windows_shaped(path) or path.startswith("/") or path.startswith("~")


def coerce_folders(value: Any) -> list[dict[str, Any]]:
    """The object form of a folder list, leniently.

    Strings become `{path, mounts: []}`; objects keep their `path` and a
    cleaned `mounts`; anything else is skipped with a log line rather
    than raised, because a hand-edited config file must not stop the
    component from starting -- `validate_folders` is the strict half,
    applied at PATCH time.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        log.warning("modelRoots is %s, expected a list; treating as empty", type(value).__name__)
        return []
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            if item.strip():
                out.append({"path": item.strip(), "mounts": []})
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"].strip():
            raw_mounts = item.get("mounts")
            mounts = (
                [m.strip() for m in raw_mounts if isinstance(m, str) and m.strip()]
                if isinstance(raw_mounts, list)
                else []
            )
            out.append({"path": item["path"].strip(), "mounts": mounts})
            continue
        log.warning("modelRoots entry %d is malformed (%r); ignoring it", index, item)
    return out


def folder_paths(value: Any) -> list[str]:
    """Just the directories, as this host spells them."""
    return [folder["path"] for folder in coerce_folders(value)]


def as_models(value: Any) -> list[LibraryFolder]:
    return [LibraryFolder.model_validate(folder) for folder in coerce_folders(value)]


def validate_folders(value: Any) -> str | None:
    """None if `value` is a well-formed folder list, else the reason."""
    if not isinstance(value, list):
        return f"expected a list of folders, got {type(value).__name__}"
    seen: dict[str, int] = {}
    for index, item in enumerate(value):
        path: Any
        mounts: list[Any]
        if isinstance(item, str):
            path, mounts = item, []
        elif isinstance(item, dict):
            extra = sorted(set(item) - {"path", "mounts"})
            if extra:
                return f"entry {index} has unknown key(s) {extra}; a folder is `path` and `mounts`"
            path = item.get("path")
            raw_mounts = item.get("mounts", [])
            if raw_mounts is None:
                raw_mounts = []
            if not isinstance(raw_mounts, list):
                return f"entry {index}: `mounts` must be a list of paths"
            mounts = raw_mounts
        else:
            return (
                f"entry {index} is {type(item).__name__}, expected a path or an object "
                f"with `path` and `mounts`"
            )
        if not isinstance(path, str) or not path.strip():
            return f"entry {index}: `path` must be a non-empty directory path"
        shapes: dict[bool, int] = {}
        for m_index, mount in enumerate(mounts):
            if not isinstance(mount, str) or not mount.strip():
                return f"entry {index}: mount {m_index} must be a non-empty path"
            if not is_absolute_path(mount.strip()):
                return (
                    f"entry {index}: mount {m_index} must be an absolute path as the nodes "
                    f"spell it (`/mnt/models`, `Z:\\models`, `\\\\nas\\models`), got {mount!r}"
                )
            shape = is_windows_shaped(mount.strip())
            if shape in shapes:
                return (
                    f"entry {index}: mounts {shapes[shape]} and {m_index} are both "
                    f"{'Windows' if shape else 'POSIX'}-shaped; a node takes the first of its "
                    f"shape, so the second would never be used"
                )
            shapes[shape] = m_index
        # Duplicates are rejected rather than de-duplicated. Two spellings
        # of one directory would scan it twice and attribute its models to
        # whichever folder won, and silently dropping one of the operator's
        # entries is worse than telling them.
        key = normalize(path.strip())
        if key in seen:
            return f"entry {index} duplicates entry {seen[key]} ({path!r})"
        seen[key] = index
    return None


__all__ = [
    "as_models",
    "coerce_folders",
    "folder_paths",
    "is_absolute_path",
    "is_windows_shaped",
    "validate_folders",
]
