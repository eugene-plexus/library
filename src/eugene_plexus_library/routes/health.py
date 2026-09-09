"""GET /healthz — liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Health, ScanState, Status, Status1
from ..config import ConfigStore
from ..scan_manager import ScanManager
from ..store import StateStore

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    """Degraded when a *configured* root could not be read.

    The distinction that matters: having **no** roots configured is
    `ok`. That is a fresh install waiting for the wizard, not a fault,
    and a health surface that goes yellow on day one teaches people to
    ignore it. A root the operator did configure and that cannot be read
    — an unplugged drive, a dead network share, a typo — is exactly what
    `degraded` is for.

    **No filesystem call happens here.** The answer comes from the last
    scan's per-root results, not from stat'ing the roots on this
    request. The watchdog polls this endpoint continuously, and a dead
    network share is precisely the case where a stat blocks for tens of
    seconds — turning the health probe into the thing that hangs. The
    reported state is therefore "as of the last scan", which is also the
    honest answer: whether a directory was readable when we last tried
    to read it is the fact that matters.

    Neither case ever prevents startup. Bad config never crashes a
    component; the config endpoints stay reachable so the UI can fix it.
    """
    config: ConfigStore = request.app.state.config_store
    state: StateStore = request.app.state.state_store
    manager: ScanManager = request.app.state.scan_manager
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))

    scan = manager.snapshot()
    unreadable = [
        root.path
        for root in (scan.roots or [])
        if root.status in (Status1.missing, Status1.unreadable)
    ]

    details: dict[str, object] = {
        "rootsConfigured": len(config.model_roots()),
        "modelsKnown": len(state.list_models()),
        "scanState": scan.state.value,
    }
    if unreadable:
        details["unreadableRoots"] = unreadable
    if scan.state == ScanState.failed and scan.error:
        details["scanError"] = scan.error

    degraded = safe_mode or bool(unreadable)
    return Health(
        status=Status.degraded if degraded else Status.ok,
        version=__version__,
        component="library",
        safeMode=safe_mode,
        details=details,
    )
