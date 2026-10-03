"""Versioned, user-owned profile documents, independent of the scan cache.

The persistence envelope owns its version and record shape. HTTP model
validation belongs to its caller; changing generated API models does not
implicitly change this format. Keep a legacy backup before migrating.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._private_files import write_private_text

VERSION = 1


class StorageUnavailable(RuntimeError):
    """Stored user data cannot safely be changed by this version."""


def read(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or type(raw.get("version")) is not int
        or raw["version"] != VERSION
    ):
        raise StorageUnavailable(f"{path}: unsupported profile storage version")
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
