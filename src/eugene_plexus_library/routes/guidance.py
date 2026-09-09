"""Guidance routes: detected hardware, the fit of a local model, the quant table."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from .. import fit as fit_mod
from .. import hardware, quants
from .._generated.models import (
    HostHardware,
    KvCacheType,
    ModelFit,
    ModelFormat,
    ModelStatus,
    Problem,
    QuantTable,
)
from ..config import ConfigStore
from ..store import StateStore

router = APIRouter(tags=["guidance"])

# Repeated across the endpoints that score a fit, and long enough that
# inlining them three times would be unreadable.
CONTEXT_DESCRIPTION = (
    "Context length to score against. Defaults to the configured "
    "`guidanceContextLength`; the KV cache grows linearly with it, so this is the "
    "number that decides which quant is recommended."
)
KV_CACHE_DESCRIPTION = (
    "KV cache element type to assume. `f16` is llama.cpp's own default; the "
    "quantized types shrink the cache term and are a launch flag, which is why "
    '"does this fit" has no answer independent of how it will be launched.'
)
VRAM_DESCRIPTION = (
    "Override the detected VRAM budget, in bytes. How a model gets scored against "
    "*another* host's memory, which is the only honest answer when the GPU is in a "
    "different building."
)
RAM_DESCRIPTION = "Override the detected host-memory budget, in bytes."


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


@router.get("/v1/hardware", response_model=HostHardware)
async def get_hardware(request: Request) -> HostHardware:
    """What this host has to spend on a model.

    Detected fresh on every call rather than cached: free memory moves
    constantly, and a verdict computed from a reading taken at startup
    would be about a machine that no longer exists. The probe is a
    couple of subprocess calls.

    `hostname` is on the response because in a multi-host deployment the
    GPU that will load the model may be somewhere else entirely — a
    driver lives next to its engine. `warnings` says what could not be
    detected, which matters: three of the detection paths here are
    untested against real hardware.
    """
    return await _detect(request)


async def _detect(request: Request) -> HostHardware:
    import asyncio

    # Vendor CLIs are blocking subprocess calls; off the event loop so a
    # wedged driver cannot stall every other request for ten seconds.
    return await asyncio.to_thread(hardware.detect)


@router.get("/v1/quants", response_model=QuantTable)
async def get_quant_table() -> QuantTable:
    """What the quant tiers mean.

    Static reference content, identical for every model. It explains the
    *schemes* — how the bits are allocated, what `K` / `IQ` / `UD` mean,
    where quality starts to go — and contains no per-model judgement,
    because the relative quality of one family against another is
    upstream research and would have to be invented.
    """
    return quants.table()


@router.get("/v1/models/{model_id}/fit", response_model=ModelFit)
async def get_model_fit(
    request: Request,
    model_id: str,
    contextLength: int | None = Query(
        default=None,
        ge=1,
        description=(
            "Context to measure at. Defaults to the configured "
            "`guidanceContextLength`. Pass the model's own `contextLength` to ask "
            "whether this host can serve what the model was trained for."
        ),
    ),
    kvCacheType: KvCacheType = Query(
        default=KvCacheType.f16,
        description=KV_CACHE_DESCRIPTION,
    ),
    vramBytes: int | None = Query(
        default=None,
        ge=0,
        description=VRAM_DESCRIPTION,
    ),
    ramBytes: int | None = Query(
        default=None, ge=0, description="Override the detected host-memory budget, in bytes."
    ),
) -> ModelFit:
    """Will a model already on this disk run here, and at what context?

    The same arithmetic the catalogue applies to a download candidate,
    pointed at something already owned — where the scan has already read
    the real metadata, so `basis` is `metadata` rather than `estimate`.

    `maxContextLength` is the more useful half of the answer: it is the
    number that goes in a profile's `-c`, and it is frequently far below
    what the model declares. A current 27B says 262144 and almost nobody
    can hold that.
    """
    store: StateStore = request.app.state.state_store
    config: ConfigStore = request.app.state.config_store

    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    if model.status is ModelStatus.missing:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Model is missing",
            (
                f"{model.path} was not found by the last scan, so there is nothing to "
                "measure. The entry survives because it carries saved profiles."
            ),
        )

    detected = await _detect(request)
    budget = fit_mod.budget_from_hardware(detected, vram_override=vramBytes, ram_override=ramBytes)
    context = contextLength or config.guidance_context_length()
    shape = _shape_for(model)

    result = fit_mod.compute(
        weights_bytes=model.sizeBytes or 0,
        budget=budget,
        context_length=context,
        shape=shape,
        kv_cache_type=kvCacheType,
    )
    return ModelFit(
        modelId=model.id,
        path=model.path,
        fit=result,
        maxContextLength=fit_mod.max_context_that_fits(
            weights_bytes=model.sizeBytes or 0,
            budget=budget,
            shape=shape,
            kv_cache_type=kvCacheType,
        ),
        modelContextLength=model.contextLength,
    )


def _shape_for(model: object) -> fit_mod.ModelShape:
    """Recover the KV-cache terms from a stored library entry.

    The scan keeps selected raw KV pairs on `gguf.metadata` as an escape
    hatch for the long tail, and this is one of the things that pays
    for: the layer and head counts are architecture-prefixed, so they
    are read back by suffix rather than by a fixed key set.

    A safetensors entry has none of this — the shape lives in
    `config.json`, which the scan reads for architecture and context and
    not for head counts — so its fit stays an estimate and says so.
    """
    from .._generated.models import LibraryModel

    assert isinstance(model, LibraryModel)
    if model.format is not ModelFormat.gguf or model.gguf is None:
        return fit_mod.ModelShape(context_length=model.contextLength, parameters=model.parameters)

    raw = model.gguf.metadata or {}

    def by_suffix(suffix: str) -> int | None:
        for key, value in raw.items():
            if key.endswith(f".{suffix}") and isinstance(value, int):
                return value
        return None

    blocks = by_suffix("block_count")
    interval = by_suffix("full_attention_interval")
    attention = max(blocks // interval, 1) if blocks and interval and interval > 1 else blocks

    return fit_mod.ModelShape(
        block_count=blocks,
        attention_layers=attention,
        head_count_kv=by_suffix("attention.head_count_kv"),
        key_length=by_suffix("attention.key_length"),
        value_length=by_suffix("attention.value_length"),
        embedding_length=by_suffix("embedding_length"),
        head_count=by_suffix("attention.head_count"),
        context_length=model.contextLength,
        parameters=model.parameters,
    )
