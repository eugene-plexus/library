"""Finding Ollama's model store on the Library's own machine (CX9-C2, C3).

Only what this machine can see: `OLLAMA_MODELS` when it is set (Ollama itself
uses nothing else then), else `~/.ollama/models` (Windows and macOS, and a
Linux user's own Ollama), else the Linux service's
`/usr/share/ollama/.ollama/models`. A store is a directory holding `blobs/`
and `manifests/` or `manifests-v2/`. Another machine's store is not looked
for: its agent would have to read it.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .formats.ollama import is_store

log = logging.getLogger(__name__)

LINUX_SERVICE_STORE = "/usr/share/ollama/.ollama/models"


def candidates(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> list[Path]:
    """Where Ollama's store would be, in the order Ollama itself decides."""
    env = os.environ if environ is None else environ
    chosen = (env.get("OLLAMA_MODELS") or "").strip()
    if chosen:
        return [Path(chosen).expanduser()]
    found = [(home if home is not None else Path.home()) / ".ollama" / "models"]
    if (platform or sys.platform).startswith("linux"):
        found.append(Path(LINUX_SERVICE_STORE))
    return found


def find_store(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> Path | None:
    """The first candidate that is a store, or None.

    Never raises: the Library starts, scans and answers `/v1/folders` through
    this, and a machine with no home folder (a container under an arbitrary uid)
    or a share that errors is a store not found, said once in the log.
    """
    try:
        for path in candidates(environ=environ, home=home, platform=platform):
            if is_store(path):
                return path
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("could not look for Ollama's model store: %s", exc)
    return None
