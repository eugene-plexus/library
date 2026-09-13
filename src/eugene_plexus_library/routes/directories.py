"""`GET /v1/directories`: the picker behind the model-roots field (M11)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .._generated.models import DirectoryListing, Problem
from ..directory_listing import ListingError, list_directory

router = APIRouter(tags=["config"])


@router.get(
    "/v1/directories",
    response_model=DirectoryListing,
    response_model_exclude_none=True,
)
def list_directories(
    path: str | None = Query(default=None),
    include_files: bool = Query(default=False, alias="includeFiles"),
    show_hidden: bool = Query(default=False, alias="showHidden"),
) -> DirectoryListing:
    """List one directory on this host, for the config editor's picker.

    This is the library's own host -- the machine whose directories
    `modelRoots` names, inside its container if it runs in one, which on
    a multi-host install is frequently not the machine the browser is
    on. `host` on the response says which.

    Operator-only (mounted that way in `app.py`, beside the config trio
    it serves), and unrestricted: an operator can already type any path
    into `modelRoots`, and `POST /v1/config/test` already stats whatever
    they name. A plain `def` for the same reason that endpoint is one --
    these are real filesystem calls, and a dead network share blocks for
    as long as the OS takes to give up.
    """
    try:
        return list_directory(path, include_files=include_files, show_hidden=show_hidden)
    except ListingError as exc:
        raise HTTPException(
            status_code=exc.status,
            detail=Problem(
                type="https://github.com/eugene-plexus/library#directory-listing",
                title=exc.title,
                status=exc.status,
                detail=exc.detail,
                component="library",
            ).model_dump(exclude_none=True),
        ) from exc
