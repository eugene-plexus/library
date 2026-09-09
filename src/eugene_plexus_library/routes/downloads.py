"""Download routes: start, poll, pause, resume, cancel."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .._generated.models import Download, DownloadList, DownloadSpec, Problem
from ..dependencies import require_operator
from ..downloads import DownloadError, DownloadManager
from ..hub import HubError

router = APIRouter(tags=["downloads"])


def _problem(
    status_code: int, title: str, detail: str, *, code: str | None = None
) -> HTTPException:
    slug = (code or title).replace(" ", "-").lower()
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


def _translate(exc: DownloadError | HubError) -> HTTPException:
    titles = {
        400: "Bad download request",
        403: "Access restricted upstream",
        404: "Not found",
        409: "Download conflict",
        429: "Upstream rate limit",
        502: "Catalogue error",
        503: "Catalogue unreachable",
        507: "Not enough disk space",
    }
    return _problem(exc.status, titles.get(exc.status, "Download failed"), str(exc), code=exc.code)


def _manager(request: Request) -> DownloadManager:
    manager: DownloadManager | None = getattr(request.app.state, "download_manager", None)
    if manager is None:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Downloads unavailable",
            "The download manager did not start. Check this component's logs.",
        )
    return manager


@router.get("/v1/downloads", response_model=DownloadList)
async def list_downloads(request: Request) -> DownloadList:
    """Every download, in flight or finished.

    Records persist — across restarts too, which is what makes a 40 GB
    transfer interrupted by one resumable rather than lost. Anything
    that was transferring when the process stopped comes back `paused`.
    """
    return DownloadList(downloads=_manager(request).records())


@router.post(
    "/v1/downloads",
    response_model=Download,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def start_download(request: Request, spec: DownloadSpec) -> Download:
    """Fetch a model into a directory you chose.

    The destination is resolved and reported **before a byte moves**, so
    you can see where the model is going while there is still time to
    change it. The root has to be one of the configured model
    directories: this component will not write somewhere you did not
    nominate.

    Returns 202 with the record in `queued`; poll
    `GET /v1/downloads/{id}`.
    """
    manager = _manager(request)
    try:
        return await manager.start(spec)
    except (DownloadError, HubError) as exc:
        raise _translate(exc) from exc


@router.get("/v1/downloads/{download_id}", response_model=Download)
async def get_download(request: Request, download_id: str) -> Download:
    record = _manager(request).get(download_id)
    if record is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown download", f"No download with id {download_id!r}."
        )
    return record


@router.delete(
    "/v1/downloads/{download_id}",
    status_code=204,
    dependencies=[Depends(require_operator)],
)
async def cancel_download(request: Request, download_id: str) -> None:
    """Cancel in flight, or forget a finished record.

    Cancelling removes the `.part` files. That is the one thing on disk
    this component will delete, and the reason it may: a `.part` is
    ours, created as a side effect of a job you just abandoned. Files
    that already finished and were renamed into place are yours and stay
    exactly where they are — the same guarantee `DELETE
    /v1/models/{id}` makes.

    To keep the partial bytes for later, `pause` instead.
    """
    try:
        await _manager(request).cancel(download_id)
    except DownloadError as exc:
        raise _translate(exc) from exc


@router.post(
    "/v1/downloads/{download_id}/pause",
    response_model=Download,
    dependencies=[Depends(require_operator)],
)
async def pause_download(request: Request, download_id: str) -> Download:
    """Stop transferring; keep the partial file.

    The `.part` stays on disk and shows up in the next scan as
    `incomplete_download`. `resume` continues from exactly those bytes.
    """
    try:
        return await _manager(request).pause(download_id)
    except DownloadError as exc:
        raise _translate(exc) from exc


@router.post(
    "/v1/downloads/{download_id}/resume",
    response_model=Download,
    dependencies=[Depends(require_operator)],
)
async def resume_download(request: Request, download_id: str) -> Download:
    """Continue a paused or failed download.

    Runs the same loop the automatic retry runs: re-resolve upstream,
    compare the remote digest and size against what the record stored,
    and continue from the byte count on disk with a `Range` request.

    That comparison is the load-bearing step. When the digest disagrees
    the file was replaced upstream — publishers requantize under the
    same filename — and the partial is discarded rather than appended
    to, which the record reports as `restartedFromZero`. Appending the
    tail of a new file to the head of an old one produces exactly the
    right number of bytes and a corrupt model.
    """
    try:
        return await _manager(request).resume(download_id)
    except DownloadError as exc:
        raise _translate(exc) from exc
