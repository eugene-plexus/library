"""Model routes: list, read, and forget."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from .._generated.models import LibraryModel, LibraryModelList, ModelStatus, Problem
from ..dependencies import require_operator
from ..store import StateStore

router = APIRouter(tags=["models"])


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


@router.get("/v1/models", response_model=LibraryModelList)
async def list_models(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Reverse lookup: the single entry at this absolute path, or nothing.",
    ),
) -> LibraryModelList:
    """Every model, including ones whose files are gone.

    `missing` entries are listed rather than hidden: they carry the
    profiles someone spent an afternoon tuning, and those are the one
    thing here that cannot be recovered by looking at the disk again.
    """
    store: StateStore = request.app.state.state_store

    if path is not None:
        found = store.find_by_path(path)
        # Normalization happens store-side, which is the entire point of
        # this parameter: a caller holding a path should not have to
        # reproduce the id derivation, case-folding and all.
        models = [found] if found else []
    else:
        models = sorted(store.list_models(), key=lambda m: m.name.lower())

    return LibraryModelList(models=models, lastScanAt=store.last_scan_at)


@router.get("/v1/models/{model_id}", response_model=LibraryModel)
async def get_model(request: Request, model_id: str) -> LibraryModel:
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    return model


@router.delete(
    "/v1/models/{model_id}",
    status_code=204,
    dependencies=[Depends(require_operator)],
)
async def forget_model(request: Request, model_id: str) -> None:
    """Forget a missing entry and the profiles saved against it.

    **No file is ever touched.** What this deletes is the library's own
    record, which is to say the profiles — the only thing this component
    persists.

    Refused for a model still on disk: the next scan would find it
    again, so a delete that appeared to work would be a lie. That makes
    this precisely the "yes, I moved that model and I don't want its old
    settings back" button, and nothing else.
    """
    store: StateStore = request.app.state.state_store
    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    if model.status == ModelStatus.present:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Model is present",
            (
                f"{model.path} is still on disk, so forgetting it would only last until the "
                "next scan. Nothing was deleted. Delete the file yourself if that is what "
                "you meant; this endpoint only forgets entries whose files are already gone."
            ),
        )
    store.forget_model(model_id)
