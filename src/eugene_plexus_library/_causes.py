"""Plain words for a failure, a backup kept before a repair, and the v0.1 notice.

Three helpers that more than one module needs and that belong to none of them:
the state store and the config routes both name a cause without a path, and
the config repair and the app start both say what v0.1 left behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ._private_files import write_private_bytes

#: Appended to a file's name for the copy kept before a broken config is repaired.
BACKUP_SUFFIX = ".bak"

_MOST_BACKUPS = 100


def v01_data(file: str, what: str) -> str:
    """The sentence for data v0.1 wrote that this version no longer converts.

    v0.3.0 converted it once, with a backup; from v0.3.1 on nothing here knows
    the old shape, so the file is named with what was found and the way out.
    """
    return (
        f"{file} holds {what}, which v0.1 wrote and v0.3.0 converted. "
        "This version no longer converts it: start Eugene v0.3.0 once to convert it, "
        "then update."
    )


def keep_backup(path: Path, data: bytes) -> Path:
    """Keep `data`, what `path` held before it was repaired, beside it.

    A copy that is already there and says the same thing is the one kept (a
    repair that stopped after this and began again). One that says something
    else is never written over, because it may be the only copy of what
    that was: the next free name, `.bak.1` and so on, holds this one.
    """
    for number in range(_MOST_BACKUPS):
        candidate = path.with_name(path.name + BACKUP_SUFFIX + (f".{number}" if number else ""))
        try:
            if candidate.read_bytes() == data:
                return candidate
        except FileNotFoundError:
            write_private_bytes(candidate, data)
            return candidate
    raise OSError(f"{_MOST_BACKUPS} different copies of {path.name} are already kept beside it")


def plain_cause(exc: Exception, *paths: Path) -> str:
    """What `exc` says, without a path: for an `OSError` the system's words and
    the names (not the paths) of the files it was about; for anything else its
    type and message, which name no file."""
    if isinstance(exc, yaml.YAMLError):
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return f"invalid YAML ({type(exc).__name__}){location}"
    if isinstance(exc, UnicodeError):
        return f"invalid text encoding ({type(exc).__name__})"
    if isinstance(exc, OSError) and exc.strerror:
        names = ", ".join(
            Path(os.fsdecode(named)).name
            for named in (exc.filename, exc.filename2)
            if isinstance(named, str | bytes | os.PathLike)
        )
        return f"{type(exc).__name__}: {exc.strerror}" + (f" ({names})" if names else "")
    message = f"{type(exc).__name__}: {exc}"
    for path in paths:
        message = message.replace(repr(str(path))[1:-1], path.name).replace(str(path), path.name)
    return message
