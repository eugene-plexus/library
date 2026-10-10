"""Guidance routes: detected hardware, the fit of a local model, the quant table."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from .. import fit as fit_mod
from .. import hardware, model_fit, quants
from .._generated.models import (
    FitModelKind,
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
GPU_COUNT_DESCRIPTION = (
    "How many cards `vramBytes` is the combined free memory of, for a launch that "
    "spreads one model across them. Each card holds its own compute buffers, so the "
    "overhead allowance is counted once per card. Omitted, `vramBytes` is one card."
)
FIT_MODEL_DESCRIPTION = (
    "The engine's own fit model (`EngineDescriptor.fit.kind`, LS6). Absent: `spill`, "
    "llama.cpp's arithmetic. `engine_table` is answered by the judge, not here."
)
SHARE_DESCRIPTION = (
    "With `fitModel=reserved_share`, required: the share of each card's total memory "
    "the engine takes."
)
VRAM_TOTAL_DESCRIPTION = (
    "The total memory of the cards `vramBytes` is the free memory of, summed: a "
    "share-taking engine's share is of this. Absent: taken as `vramBytes`."
)
UNIFIED_DESCRIPTION = (
    "Score against one pool shared with host memory: an integrated GPU, Apple silicon "
    "or a GB10 on a host this library did not measure. Pass the device's own "
    "`sharedMemory`. Without it a `vramBytes` override reads as a card with memory of "
    "its own, and a partial offload is scored as spilling into the same RAM."
)


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
    unifiedMemory: bool | None = Query(default=None, description=UNIFIED_DESCRIPTION),
    gpuCount: int | None = Query(default=None, ge=1, description=GPU_COUNT_DESCRIPTION),
    fitModel: FitModelKind | None = Query(default=None, description=FIT_MODEL_DESCRIPTION),
    gpuMemoryUtilization: float | None = Query(
        default=None, gt=0, le=1, description=SHARE_DESCRIPTION
    ),
    vramTotalBytes: int | None = Query(default=None, ge=0, description=VRAM_TOTAL_DESCRIPTION),
) -> ModelFit:
    """Will a model already on this disk run here, and at what context?

    The same arithmetic the catalogue applies to a download candidate,
    pointed at something already owned — where the scan has already read
    the real metadata, so `basis` is `metadata` rather than `estimate`.

    `maxContextLength` is the more useful half of the answer: it is the
    number that goes in a profile's `-c`, and it is frequently far below
    what the model declares. A current 27B says 262144 and almost nobody
    can hold that.

    Whose fit (LS6): `fitModel` is the engine's own fit model. Absent, it
    is `spill`, llama.cpp's, which is what every caller before LS6 meant.
    """
    store: StateStore = request.app.state.state_store
    config: ConfigStore = request.app.state.config_store
    kind = fitModel or FitModelKind.spill
    if kind is FitModelKind.engine_table:
        raise _problem(
            422,
            "Fit not estimated",
            (
                "An engine with its own fit table answers from it, as its node reports "
                "it: ask POST /v1/eligibility with `fit` and the engine's descriptor."
            ),
        )
    if kind is FitModelKind.reserved_share and gpuMemoryUtilization is None:
        raise _problem(
            422,
            "Fit not estimated",
            "A share-taking engine's fit needs its share: pass `gpuMemoryUtilization`.",
        )

    model = store.get_model(model_id)
    if model is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND, "Unknown model", f"No model with id {model_id!r}."
        )
    if model.format is ModelFormat.prepared:
        # This arithmetic is llama.cpp's; a prepared model is in its engine's
        # own format, which nothing here reads (library-sources-and-engines §6.1).
        engine = model.prepared.engine.value if model.prepared else "an engine"
        raise _problem(
            422,
            "Fit not estimated",
            (
                f"{model.name} was prepared for {engine}, in that engine's own format, "
                "which the library does not read, so no arithmetic here is its fit. An "
                "engine with its own fit table answers through POST /v1/eligibility."
            ),
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
    budget = fit_mod.budget_from_hardware(
        detected,
        vram_override=vramBytes,
        ram_override=ramBytes,
        unified_override=unifiedMemory,
        gpu_count_override=gpuCount,
        vram_total_override=vramTotalBytes,
    )
    context = contextLength or config.guidance_context_length()
    shape = _shape_for(model)
    # The disk footprint includes an optional vision projector. Merely
    # finding it beside a GGUF does not put --mmproj on the launch line.
    # Catalogue candidates already exclude it; local guidance must agree.
    projector_bytes = model_fit.projector_bytes(model)
    weights_bytes = model_fit.weights_of(model)
    expert_bytes = model_fit.expert_bytes_of(model)

    if kind is FitModelKind.reserved_share:
        assert gpuMemoryUtilization is not None
        share = fit_mod.compute_reserved_share(
            weights_bytes=weights_bytes,
            budget=budget,
            context_length=context,
            utilization=gpuMemoryUtilization,
            shape=shape,
            kv_cache_type=kvCacheType,
        )
        return ModelFit(
            modelId=model.id,
            path=model.path,
            fit=share,
            maxContextLength=fit_mod.max_context_reserved_share(
                weights_bytes=weights_bytes,
                budget=budget,
                shape=shape,
                utilization=gpuMemoryUtilization,
                kv_cache_type=kvCacheType,
            ),
            modelContextLength=model.contextLength,
        )

    result = fit_mod.compute(
        weights_bytes=weights_bytes,
        budget=budget,
        context_length=context,
        shape=shape,
        kv_cache_type=kvCacheType,
        expert_bytes=expert_bytes,
    )
    if projector_bytes:
        result.notes = [
            *(result.notes or []),
            "This estimate excludes the separate vision projector. Loading it for "
            "images needs additional memory; finding it on disk does not load it.",
        ]
    return ModelFit(
        modelId=model.id,
        path=model.path,
        fit=result,
        maxContextLength=fit_mod.max_context_that_fits(
            weights_bytes=weights_bytes,
            budget=budget,
            shape=shape,
            kv_cache_type=kvCacheType,
        ),
        modelContextLength=model.contextLength,
        maxContextExpertsInRam=fit_mod.max_context_experts_in_ram(
            weights_bytes=weights_bytes,
            expert_bytes=expert_bytes,
            budget=budget,
            shape=shape,
            kv_cache_type=kvCacheType,
        ),
    )


# The one shape reader, now beside the judge's use of it (LS6); the name
# stays for the tests that pin it.
_shape_for = model_fit.shape_of
