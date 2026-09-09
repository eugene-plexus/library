"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    """Editing the roots does **not** start a walk.

    `POST /v1/scan` is a separate, explicit act. A PATCH that quietly
    kicks off minutes of disk work is a PATCH that appears to hang, and
    an operator adding three directories one at a time would start three
    scans they didn't ask for.
    """
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
def test_config(request: Request, body: ConfigTestRequest | None = None) -> ConfigTestResult:
    """Stat the configured roots and report which cannot be read.

    Deliberately does not scan. This answers "is this path right" in
    milliseconds, which is what belongs behind a Test button next to a
    directory picker; finding zero models in a readable directory is not
    a failure and must not read as one.

    Declared `def` rather than `async def` so FastAPI runs it in a
    threadpool: these stats are real filesystem calls and a dead network
    share blocks for as long as the OS takes to give up. That is
    acceptable for an endpoint someone pressed a button to reach, and
    not acceptable on the event loop.
    """
    start = time.perf_counter()
    store: ConfigStore = request.app.state.config_store

    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = body.overrides.model_dump(exclude_none=True)

    raw = overrides.get("modelRoots", store.get("modelRoots")) or []
    if not isinstance(raw, list):
        return ConfigTestResult(
            ok=False,
            component="library",
            latencyMs=int((time.perf_counter() - start) * 1000),
            error=f"modelRoots must be a list of paths, got {type(raw).__name__}",
        )

    roots = [Path(str(item)).expanduser() for item in raw if str(item).strip()]
    problems: list[str] = []
    readable: list[str] = []

    for root in roots:
        try:
            if not root.exists():
                problems.append(f"{root}: does not exist")
            elif not root.is_dir():
                problems.append(f"{root}: not a directory")
            else:
                with os.scandir(root) as probe:
                    next(iter(probe), None)
                readable.append(str(root))
        except PermissionError:
            problems.append(f"{root}: permission denied")
        except OSError as exc:
            problems.append(f"{root}: {exc}")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if not roots:
        # Nothing configured is not a failure — it is a fresh install,
        # and the operator is most likely mid-way through adding the
        # first one. Say what is true and let them get on with it.
        return ConfigTestResult(
            ok=True,
            component="library",
            latencyMs=elapsed_ms,
            summary="No model directories configured yet.",
        )

    if problems:
        return ConfigTestResult(
            ok=False,
            component="library",
            latencyMs=elapsed_ms,
            summary=f"{len(readable)} of {len(roots)} directories readable.",
            error="; ".join(problems),
        )

    return ConfigTestResult(
        ok=True,
        component="library",
        latencyMs=elapsed_ms,
        summary=f"All {len(roots)} model {'directory is' if len(roots) == 1 else 'directories are'}"
        " readable.",
        sampleOutput="\n".join(readable),
    )
