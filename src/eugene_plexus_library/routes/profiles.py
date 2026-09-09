"""Profile routes: the per-model launch settings.

Flags are **stored, not validated**. The validator is the agent's
engine adapter `flagSchema`, and it runs when a runtime is actually
created. A profile is allowed to be wrong; the runtime that uses one is
not, and that is where an unknown flag becomes a 400. Validating in both
places would mean two copies of engine knowledge, and the copy in the
component that never launches anything is the one that would go stale.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .._generated.models import ModelProfile, ModelProfileList, ModelProfileSpec, Problem
from ..dependencies import require_operator
from ..store import StateStore

router = APIRouter(tags=["profiles"])


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


def _require_model(request: Request, model_id: str) -> StateStore:
    store: StateStore = request.app.state.state_store
    if store.get_model(model_id) is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    return store


@router.get("/v1/models/{model_id}/profiles", response_model=ModelProfileList)
async def list_profiles(request: Request, model_id: str) -> ModelProfileList:
    store = _require_model(request, model_id)
    return ModelProfileList(profiles=store.list_profiles(model_id))


@router.post(
    "/v1/models/{model_id}/profiles",
    response_model=ModelProfile,
    status_code=201,
    dependencies=[Depends(require_operator)],
)
async def create_profile(request: Request, model_id: str, body: ModelProfileSpec) -> ModelProfile:
    store = _require_model(request, model_id)
    try:
        return store.create_profile(model_id, body)
    except ValueError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "Duplicate profile name", str(exc)) from exc


@router.get("/v1/models/{model_id}/profiles/{profile_id}", response_model=ModelProfile)
async def get_profile(request: Request, model_id: str, profile_id: str) -> ModelProfile:
    store = _require_model(request, model_id)
    profile = store.get_profile(model_id, profile_id)
    if profile is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "Unknown profile",
            f"No profile {profile_id!r} for model {model_id!r}.",
        )
    return profile


@router.put(
    "/v1/models/{model_id}/profiles/{profile_id}",
    response_model=ModelProfile,
    dependencies=[Depends(require_operator)],
)
async def replace_profile(
    request: Request, model_id: str, profile_id: str, body: ModelProfileSpec
) -> ModelProfile:
    """Whole-document replace, not a merge.

    `flags` is a document, and merge semantics give no way to express
    *removing* a flag — "why is `--n-gpu-layers` still on the command
    line after I deleted it" is a bug report nobody should have to file.
    """
    store = _require_model(request, model_id)
    try:
        updated = store.replace_profile(model_id, profile_id, body)
    except ValueError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "Duplicate profile name", str(exc)) from exc
    if updated is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "Unknown profile",
            f"No profile {profile_id!r} for model {model_id!r}.",
        )
    return updated


@router.delete(
    "/v1/models/{model_id}/profiles/{profile_id}",
    status_code=204,
    dependencies=[Depends(require_operator)],
)
async def delete_profile(request: Request, model_id: str, profile_id: str) -> None:
    """Deleting the default promotes the oldest survivor.

    Running runtimes are unaffected — a profile is a template, and once
    a runtime is declared nothing holds a reference back to it.
    """
    store = _require_model(request, model_id)
    if not store.delete_profile(model_id, profile_id):
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "Unknown profile",
            f"No profile {profile_id!r} for model {model_id!r}.",
        )
