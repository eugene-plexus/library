"""Scan routes: start, poll, cancel. One walk at a time."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .._generated.models import Problem, Scan, ScanRequest
from ..dependencies import require_operator
from ..scan_manager import ScanManager

router = APIRouter(tags=["scan"])


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/library#{slug}",
            title=title,
            status=status_code,
            detail=detail,
            component="library",
        ).model_dump(exclude_none=True),
    )


@router.get("/v1/scan", response_model=Scan)
async def get_scan(request: Request) -> Scan:
    """State of the current or most recent walk.

    When a model the operator can see in a file browser is not in the
    library, this is the first place to look: `skipped[]` says what was
    passed over and why, and `roots[]` says which directories could not
    be read at all.
    """
    manager: ScanManager = request.app.state.scan_manager
    return manager.snapshot()


@router.post(
    "/v1/scan",
    response_model=Scan,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def start_scan(request: Request, body: ScanRequest | None = None) -> Scan:
    """Start a walk and return immediately; poll `GET /v1/scan`.

    Incremental by default — a file whose `(path, size, mtime)` is
    unchanged keeps its cached metadata and is never opened. `full`
    re-reads everything, which is the escape hatch for when the metadata
    reader has changed rather than the files.
    """
    manager: ScanManager = request.app.state.scan_manager
    try:
        return await manager.start(full=bool(body.full) if body else False)
    except RuntimeError as exc:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Scan already running",
            (
                f"{exc}. There is only ever one scan; starting a second would double "
                "the disk cost to reach the same answer. Poll GET /v1/scan, or cancel "
                "with DELETE /v1/scan."
            ),
        ) from exc


@router.delete(
    "/v1/scan",
    response_model=Scan,
    dependencies=[Depends(require_operator)],
)
async def cancel_scan(request: Request) -> Scan:
    """Stop the walk.

    Models already found keep their entries — a partial scan learned
    true things and discarding them would lose work the operator waited
    for. `state: cancelled` is what says the list is not complete.
    """
    manager: ScanManager = request.app.state.scan_manager
    try:
        return await manager.cancel()
    except RuntimeError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "No scan running", str(exc)) from exc
