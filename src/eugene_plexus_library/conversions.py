"""What v0.1 saved, converted once.

Old saved data is rewritten in the current format ONCE. The bytes it held are
kept beside it (`<name>.v0.1.bak`), the rewrite carries the new version, and
from then on only the current format is read: nothing else in this component
knows a v0.1 shape. The release after v0.2 deletes this module and its two
calls in `app.py`.

A conversion that cannot finish raises `ConversionFailed`, whose message names
the cause and says the file was left as it was. Its caller degrades instead of
crashing (a store that refuses writes, a config that runs on defaults). Every
rewrite is a temp file and a rename, so a crash leaves the old bytes or the new
ones, and the next start begins again from whichever it finds. The rewrite is
also the record: a converted file no longer has the old version (or the old
shape), so a second start finds nothing to do and writes nothing.

Three things are converted:

* **The state** (`library-state.json`), version 1 to 2. v0.1 kept the profiles
  inside it; they now live in the profile document beside it, so a scan can
  never cost anyone their tuning work. The profile document is written first
  and the rewritten state (the version 2 stamp) is the commit.
* **The profile document** (`library-state.profiles.json`), version 1 to 2.
  Flash attention was a checkbox and is now On, Off or not set. `true` always
  sent `on`, so it becomes `"on"`. `false` always sent nothing, which left
  llama.cpp's own choice (auto), so the key is dropped, never turned into
  `"off"`: that would switch it off in every profile saved before. The
  settings builder's record of what it set (`builtBy.flags`) is converted the
  same way, because the console compares it with the profile's own flags. The
  agent converts its saved runtimes in its own file.
* **The config** (`config.yaml`). A `modelRoots` entry that is a bare string,
  which is how `modelRoots` was written before folders carried their mounts,
  becomes `{path, mounts: []}`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from . import profile_storage
from ._private_files import write_private_bytes, write_private_text

#: Appended to a file's name for the copy kept before it was converted.
BACKUP_SUFFIX = ".v0.1.bak"

#: The versions the conversions write: what this release reads. A test holds them to it.
STATE_VERSION = 2
PROFILES_VERSION = 2

_MOST_BACKUPS = 100


class ConversionFailed(Exception):
    """Saved data from v0.1 could not be converted; the file is as it was."""


def keep_backup(path: Path, data: bytes) -> Path:
    """Keep `data`, what `path` held before it was converted, beside it.

    A copy that is already there and says the same thing is the one kept (a
    start that stopped after this and began again). One that says something
    else is never written over, because it may be the only copy of what
    that was: the next free name, `.v0.1.bak.1` and so on, holds this one.
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


def _failed(path: Path, cause: str, backup: Path | None = None) -> ConversionFailed:
    kept = f" The copy from before is {backup}." if backup is not None else ""
    return ConversionFailed(
        f"{path}: what v0.1 saved could not be converted ({cause}). It is left as it was.{kept} "
        "Fix the cause and restart."
    )


def _guarded(path: Path, step: Callable[..., None], *args: Path) -> None:
    """Run one step; whatever it raises that does not already say why is named here."""
    try:
        step(*args)
    except ConversionFailed:
        raise
    except Exception as exc:
        raise _failed(path, f"{type(exc).__name__}: {exc}") from exc


@contextmanager
def _after_backup(path: Path, backup: Path) -> Iterator[None]:
    """Like `_guarded`, for the part of a step that comes after its backup."""
    try:
        yield
    except ConversionFailed:
        raise
    except Exception as exc:
        raise _failed(path, f"{type(exc).__name__}: {exc}", backup) from exc


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _is_version(document: Any, version: int) -> bool:
    return (
        isinstance(document, dict)
        and type(document.get("version")) is int
        and document["version"] == version
    )


def _json(data: bytes) -> Any:
    """The JSON in `data`, or None when it is not readable JSON: that is `load()`'s to name."""
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        return None


# --- flash attention ---------------------------------------------------------


def _flash_attention_three_states(settings: dict[str, Any]) -> None:
    """A profile's `flags`, and the builder's record of them, in place."""
    built = settings.get("builtBy")
    for flags in (settings.get("flags"), built.get("flags") if isinstance(built, dict) else None):
        if not isinstance(flags, dict):
            continue
        value = flags.get("flashAttention")
        if value is True:
            flags["flashAttention"] = "on"
        elif value is False:
            del flags["flashAttention"]


# --- the state and its profile document ----------------------------------------


def convert_state(state_file: Path) -> None:
    """Version 1 of the state and of the profile document, as version 2 of each."""
    profiles_file = profile_storage.path_for(state_file)
    _guarded(profiles_file, _profile_document, profiles_file)
    _guarded(state_file, _state, state_file, profiles_file)


def _profile_document(path: Path) -> None:
    data = _read(path)
    if data is None:
        return
    document = _json(data)
    if not _is_version(document, 1):
        return  # current, newer or unreadable: `StateStore.load` says which
    backup = keep_backup(path, data)
    with _after_backup(path, backup):
        profiles = document.get("profiles")
        if not isinstance(profiles, dict):
            raise _failed(path, "its profiles are not an object", backup)
        for model_id, records in profiles.items():
            if not isinstance(records, list) or not all(
                isinstance(r, dict) and isinstance(r.get("settings"), dict) for r in records
            ):
                raise _failed(path, f"the profiles of {model_id!r} are not records", backup)
            for record in records:
                _flash_attention_three_states(record["settings"])
        document["version"] = PROFILES_VERSION
        write_private_text(path, json.dumps(document, indent=2))


def _state(path: Path, profiles_file: Path) -> None:
    data = _read(path)
    if data is None:
        return
    raw = _json(data)
    if not _is_version(raw, 1):
        return  # current, newer or unreadable: `StateStore.load` says which
    backup = keep_backup(path, data)
    with _after_backup(path, backup):
        embedded = raw.pop("profiles", None) or {}
        if not isinstance(embedded, dict):
            raise _failed(path, f"its profiles are a {type(embedded).__name__}", backup)
        if not profiles_file.exists():
            # A document that is there is one an earlier start made before it stopped,
            # and it is never replaced with what is embedded.
            records: dict[str, list[dict[str, Any]]] = {}
            for model_id, profiles in embedded.items():
                if not profiles:
                    continue
                if not isinstance(profiles, list) or not all(
                    isinstance(p, dict) and isinstance(p.get("id"), str) for p in profiles
                ):
                    raise _failed(path, f"the profiles of {model_id!r} are not records", backup)
                for profile in profiles:
                    _flash_attention_three_states(profile)
                records[model_id] = profiles
            if records:
                models = [
                    m
                    for m in raw.get("models") or []
                    if isinstance(m, dict) and m.get("id") in records
                ]
                profile_storage.write(profiles_file, records, models)
        raw["version"] = STATE_VERSION
        write_private_text(path, json.dumps(raw, indent=2))


# --- the config ------------------------------------------------------------------


def convert_config(config_file: Path) -> None:
    """A `modelRoots` written as bare strings, as folders with no mounts."""
    _guarded(config_file, _config, config_file)


def _config(path: Path) -> None:
    data = _read(path)
    if data is None:
        return
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return  # `ConfigStore.load` says what is wrong with it
    roots = raw.get("modelRoots") if isinstance(raw, dict) else None
    if not isinstance(roots, list) or not any(isinstance(item, str) for item in roots):
        return
    backup = keep_backup(path, data)
    with _after_backup(path, backup):
        folders: list[Any] = []
        for item in roots:
            if not isinstance(item, str):
                folders.append(item)  # a folder is as it should be; junk is `load`'s to skip
            elif item.strip():
                folders.append({"path": item.strip(), "mounts": []})
        raw["modelRoots"] = folders
        write_private_text(path, yaml.safe_dump(raw, sort_keys=True, default_flow_style=False))
