"""Versioned, user-owned profile documents, independent of the scan cache.

The persistence envelope owns its version and record shape. HTTP model
validation belongs to its caller; changing generated API models does not
implicitly change this format.

Only the current version is read. Version 1 (flash attention saved as a
checkbox) is what v0.1 and the first v0.2 builds wrote and v0.3.0 converted;
this version no longer converts it, so it is refused by name (`V01Data`) with
the way out, and never read as if it were version 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._causes import v01_data
from ._private_files import write_private_text

VERSION = 2


class StorageUnavailable(RuntimeError):
    """Stored user data cannot safely be changed by this version."""


class V01Data(StorageUnavailable):
    """The data is what v0.1 wrote. The message is already the whole sentence
    for the health report: it names the file by its name and the fix."""


def path_for(state_file: Path) -> Path:
    """Where the profile document lives: beside the state file it belongs to."""
    return state_file.with_suffix(".profiles.json")


def read(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and type(raw.get("version")) is int and raw["version"] == 1:
        raise V01Data(
            v01_data(path.name, "version 1 profiles (flash attention saved as a checkbox)")
        )
    if (
        not isinstance(raw, dict)
        or type(raw.get("version")) is not int
        or raw["version"] != VERSION
    ):
        found = raw.get("version") if isinstance(raw, dict) else None
        raise StorageUnavailable(
            f"{path}: unsupported profile storage version {found!r}; this Library reads "
            f"version {VERSION}"
        )
    profiles: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw.get("profiles"), dict) or not isinstance(raw.get("models", []), list):
        raise StorageUnavailable(f"{path}: invalid profile storage document")
    for model_id, records in raw["profiles"].items():
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise StorageUnavailable(f"{path}: invalid profile records")
        profiles[model_id] = [
            {
                **r["settings"],
                "id": r["id"],
                "createdAt": r["createdAt"],
                "updatedAt": r["updatedAt"],
            }
            for r in records
        ]
    return profiles, raw.get("models", [])


def write(
    path: Path, profiles: dict[str, list[dict[str, Any]]], models: list[dict[str, Any]]
) -> None:
    identity = {"id", "createdAt", "updatedAt"}
    document = {
        "version": VERSION,
        "profiles": {
            model: [
                {
                    "id": p["id"],
                    "createdAt": p.get("createdAt"),
                    "updatedAt": p.get("updatedAt"),
                    "settings": {k: v for k, v in p.items() if k not in identity},
                }
                for p in records
            ]
            for model, records in profiles.items()
        },
        # Retain the identity of a missing model even if its scan cache is lost.
        "models": models,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(path, json.dumps(document, indent=2))
