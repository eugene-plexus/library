"""POST /v1/admin/restart — schedule a process exit so the supervisor relaunches us."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter

from .._generated.models import RestartResult

router = APIRouter(tags=["meta"])

# Long enough for the 202 body to flush over a slow LAN, short enough
# that the operator isn't left watching a "restarting…" dialog.
_EXIT_DELAY_MS = 500


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart() -> RestartResult:
    """A scan in flight is abandoned. Its findings so far are already
    persisted, and the next scan is incremental over them, so the cost
    of restarting mid-walk is the files it hadn't reached yet."""
    log = logging.getLogger(__name__)
    log.warning("restart requested via /v1/admin/restart; exiting in %dms", _EXIT_DELAY_MS)

    loop = asyncio.get_event_loop()
    loop.call_later(_EXIT_DELAY_MS / 1000.0, lambda: os._exit(0))

    return RestartResult(
        scheduled=True,
        delayMs=_EXIT_DELAY_MS,
        message=(
            f"Process exiting in {_EXIT_DELAY_MS}ms. The agent is expected to "
            "relaunch it; running standalone, relaunch manually."
        ),
    )
