"""Will this model run here, and at what context?

One computation with three callers — a catalogue candidate, a model
already on disk, and a preflight of a remote file — because it is one
question asked from three places.

## The arithmetic

    required = weights + kv_cache(ctx) + overhead
    kv_cache = ctx * attention_layers * head_count_kv
                   * (key_length + value_length) * bytes_per_element

Every term but `overhead` comes from the model's own metadata. Overhead
is a flat allowance for compute buffers, the graph and the accelerator
context, and it is the term that makes this an estimate rather than a
calculation.

## The trap that would have made this a liar

`attention_layers` is not the layer count. A current 27B declares
`block_count: 65` with `full_attention_interval: 4`, so 16 layers hold a
KV cache and the rest are state-space. Measured against the naive form:

    ctx  32768   hybrid  2.00 GiB   naive   8.12 GiB   x4.1 over
    ctx 262144   hybrid 16.00 GiB   naive  65.00 GiB   x4.1 over

At the model's own trained context that is the difference between
"fits on a 5090" and "no candidate in this repo is runnable". Hybrid
architectures are mainstream now, so the attention count is read from
the metadata where it exists — and where it does not, the layer count is
assumed and `notes` says so, which errs pessimistic.

## What it will not do

Rank quality. Bits per weight is arithmetic and is reported; which of
`IQ2_S` and `Q2_K` is *better* is upstream research, model-dependent,
and would have to be invented. See `quants.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._generated.models import (
    Basis,
    Fit,
    FitVerdict,
    HostHardware,
    KvCacheType,
    MemoryBudget,
    Source,
)

GIB = 1024 * 1024 * 1024

DEFAULT_OVERHEAD_BYTES = 1 * GIB
"""Compute buffers, the CUDA/Metal context, and the graph. A flat
allowance because it varies with batch size, parallel slots and the
engine build, none of which are known here. Deliberately generous: an
estimate that is slightly pessimistic produces a `tight` where the truth
is `fits`, which costs the operator a second look. The opposite produces
a confident `fits` that OOMs, which costs them their trust."""

DEFAULT_CONTEXT_LENGTH = 8192
"""Fallback when neither the caller nor config says. Not the model's
trained context: current models declare 262144 and almost nobody can
hold that, so defaulting to it would report `no` for everything."""

KV_ELEMENT_BYTES: dict[KvCacheType, float] = {
    KvCacheType.f16: 2.0,
    KvCacheType.q8_0: 1.0625,
    KvCacheType.q4_0: 0.5625,
}
"""Bytes per KV element. `f16` is llama.cpp's default and exact; the
quantized types carry a per-block scale, so 32 four-bit weights plus one
f16 scale is 18 bytes for 32 elements, not 16."""


@dataclass(frozen=True)
class ModelShape:
    """What the KV-cache term needs from a model's metadata.

    A typed carrier rather than a dict so a renamed field fails at the
    call site instead of silently producing a zero-sized cache.
    """

    block_count: int | None = None
    attention_layers: int | None = None
    """Layers that actually hold KV. Differs from `block_count` on a
    hybrid attention/SSM model — see the module docstring."""

    head_count_kv: int | None = None
    key_length: int | None = None
    value_length: int | None = None
    embedding_length: int | None = None
    head_count: int | None = None
    context_length: int | None = None
    parameters: int | None = None

    @property
    def complete(self) -> bool:
        """Is there enough here to compute a real KV cache size?"""
        return (
            self.effective_attention_layers is not None and self._per_token_elements() is not None
        )

    @property
    def effective_attention_layers(self) -> int | None:
        return self.attention_layers or self.block_count

    def _per_token_elements(self) -> int | None:
        """K and V elements cached per token, per attention layer.

        `key_length`/`value_length` are the per-head dimensions and are
        stated outright by modern files. Where they are absent, the
        classic identity `head_dim = embedding_length / head_count`
        recovers them, which is what a GGUF from before those keys
        existed requires.
        """
        heads_kv = self.head_count_kv
        key_len = self.key_length
        value_len = self.value_length

        if (key_len is None or value_len is None) and self.embedding_length and self.head_count:
            head_dim = self.embedding_length // self.head_count
            key_len = key_len or head_dim
            value_len = value_len or head_dim

        if heads_kv is None and self.head_count is not None:
            # No grouped-query metadata: assume every head has its own
            # KV, which is the pre-GQA layout and the pessimistic read.
            heads_kv = self.head_count

        if not heads_kv or not key_len or not value_len:
            return None
        return heads_kv * (key_len + value_len)

    def kv_bytes(self, context_length: int, kv_cache_type: KvCacheType) -> int | None:
        layers = self.effective_attention_layers
        per_token = self._per_token_elements()
        if not layers or not per_token:
            return None
        element = KV_ELEMENT_BYTES[kv_cache_type]
        return int(context_length * layers * per_token * element)


def budget_from_hardware(
    hardware: HostHardware,
    *,
    vram_override: int | None = None,
    ram_override: int | None = None,
) -> MemoryBudget:
    """Collapse detected hardware into the numbers a verdict needs.

    `largestGpuFreeBytes` is separate from the VRAM sum because
    llama.cpp splits layers across cards by default: "fits on one card"
    and "fits across all of them" are different questions with different
    performance, and two replicas on two cards — M5's case — needs the
    per-card number rather than the total.

    A card that reports no free memory falls back to its total, which
    makes the verdict optimistic for that card alone; Intel is the case
    that does this, and it is named in the hardware warnings.
    """
    per_gpu_free = [
        (gpu.vramFreeBytes if gpu.vramFreeBytes is not None else gpu.vramTotalBytes)
        for gpu in hardware.gpus or []
    ]
    vram_free = sum(per_gpu_free)
    vram_total = sum(gpu.vramTotalBytes for gpu in hardware.gpus or [])
    largest = max(per_gpu_free, default=0)

    source = Source.detected
    if vram_override is not None:
        vram_free = vram_total = largest = vram_override
        source = Source.override

    ram_total = hardware.ramTotalBytes or 0
    ram_available = hardware.ramAvailableBytes or ram_total
    if ram_override is not None:
        ram_total = ram_available = ram_override
        source = Source.override

    return MemoryBudget(
        vramFreeBytes=vram_free,
        vramTotalBytes=vram_total,
        largestGpuFreeBytes=largest,
        ramAvailableBytes=ram_available,
        ramTotalBytes=ram_total,
        gpuCount=len(hardware.gpus or []),
        unifiedMemory=bool(hardware.unifiedMemory),
        source=source,
    )


def _verdict(required: int, budget: MemoryBudget) -> FitVerdict:
    vram_free = budget.vramFreeBytes or 0
    vram_total = budget.vramTotalBytes or 0
    ram_available = budget.ramAvailableBytes or 0

    if budget.unifiedMemory:
        # One pool. `tight` is what "would fit if you closed something"
        # means here, and there is no partial-offload case to describe
        # because there is nothing to offload *to*.
        if required <= vram_free or (vram_free == 0 and required <= ram_available):
            return FitVerdict.fits
        if required <= vram_total or required <= (budget.ramTotalBytes or 0):
            return FitVerdict.tight
        return FitVerdict.no

    if vram_total == 0:
        # No accelerator: everything runs on the CPU, so there is no
        # "fits in VRAM" to report and `split` would be a lie about a
        # split that is not happening.
        if required <= ram_available:
            return FitVerdict.fits
        if required <= (budget.ramTotalBytes or 0):
            return FitVerdict.tight
        return FitVerdict.no

    if required <= vram_free:
        return FitVerdict.fits
    if required <= vram_total:
        return FitVerdict.tight
    if required <= vram_free + ram_available:
        return FitVerdict.split
    return FitVerdict.no


def compute(
    *,
    weights_bytes: int,
    budget: MemoryBudget,
    context_length: int,
    shape: ModelShape | None = None,
    kv_cache_type: KvCacheType = KvCacheType.f16,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> Fit:
    """Score one candidate. Reports its own inputs, always.

    Guidance that does not state its assumptions cannot be argued with,
    and this one will sometimes be wrong — so `notes` carries the
    assumptions in words and `basis` says whether the KV term came from
    metadata or from a guess with a number on it.
    """
    notes: list[str] = []
    shape = shape or ModelShape()

    kv_bytes = shape.kv_bytes(context_length, kv_cache_type)
    if kv_bytes is None:
        # No usable shape. A flat fraction of the weights is a poor
        # estimate and an honest one; `basis: estimate` is what says not
        # to trust the breakdown.
        kv_bytes = int(weights_bytes * 0.15)
        basis = Basis.estimate
        notes.append(
            "KV cache is a rough 15% of the weights: this model's layer and attention "
            "metadata was not available. Preflight the file (or scan it, once it is on "
            "disk) for a real figure."
        )
    else:
        basis = Basis.metadata
        if shape.attention_layers is None and shape.block_count:
            notes.append(
                f"assumed all {shape.block_count} layers hold a KV cache. A hybrid "
                "attention/SSM model has far fewer, so this over-estimates rather than "
                "under-estimates."
            )

    notes.append(f"assumes full GPU offload and a {kv_cache_type.value} KV cache")
    notes.append(f"includes a flat {overhead_bytes / GIB:.1f} GiB allowance for compute buffers")
    if (budget.gpuCount or 0) > 1:
        notes.append(
            f"scored against {budget.gpuCount} GPUs summed; the largest single card has "
            f"{(budget.largestGpuFreeBytes or 0) / GIB:.1f} GiB free, which is the number "
            "that matters if you pin one card"
        )
    if budget.source == Source.override:
        notes.append("scored against a caller-supplied budget, not this host's detected memory")

    required = weights_bytes + kv_bytes + overhead_bytes
    return Fit(
        verdict=_verdict(required, budget),
        requiredBytes=required,
        weightsBytes=weights_bytes,
        kvCacheBytes=kv_bytes,
        overheadBytes=overhead_bytes,
        contextLength=context_length,
        kvCacheType=kv_cache_type,
        attentionLayers=shape.effective_attention_layers,
        basis=basis,
        budget=budget,
        notes=notes,
    )


def max_context_that_fits(
    *,
    weights_bytes: int,
    budget: MemoryBudget,
    shape: ModelShape,
    kv_cache_type: KvCacheType = KvCacheType.f16,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
    ceiling: int | None = None,
) -> int | None:
    """Largest context whose weights + KV + overhead stay in free VRAM.

    Solved directly rather than searched: the KV term is linear in
    context, so this is one division. More useful than a yes/no at one
    context — it is the number that goes in a profile's `-c`, and it
    answers the question a launch actually asks.

    None when the shape is unusable (nothing to solve) or when the
    weights alone do not fit (no context makes that true). Rounded down
    to a multiple of 256, because a context of 31,847 is not a number
    anyone should be handed.
    """
    per_token = shape.kv_bytes(1, kv_cache_type)
    if not per_token:
        return None

    vram_free = budget.vramFreeBytes or 0
    available = (vram_free if vram_free else (budget.ramAvailableBytes or 0)) - overhead_bytes
    available -= weights_bytes
    if available <= 0:
        return None

    context = int(available // per_token)
    limit = ceiling if ceiling is not None else shape.context_length
    if limit:
        context = min(context, limit)
    if context < 256:
        return context if context > 0 else None
    return (context // 256) * 256


def bits_per_weight(weights_bytes: int, parameters: int | None) -> float | None:
    """`size x 8 / parameters`, the one honest ordering number.

    Across one real repo it ordered all 25 candidates monotonically from
    1.81 to exactly **16.00** for BF16 — that last value being the check
    that the parameter count is a parameter count. It is what makes
    `IQ2_S`, `Q2_K` and `UD-Q3_K_XL` comparable when their names come
    from three different upstream naming schemes.

    None without a parameter count, which is most GGUFs on disk: the
    format carries no numeric count at all. The hub reports one for a
    GGUF *repo*, which is why this is computable there and not here.
    """
    if not parameters or parameters <= 0 or weights_bytes <= 0:
        return None
    return round(weights_bytes * 8 / parameters, 2)


def format_bytes(count: int | None) -> str:
    """`16.5 GiB`. For the prose in a recommendation's `reason`, where a
    raw byte count is unreadable and the operator is being asked to
    trust the arithmetic."""
    if not count:
        return "0 B"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = min(int(math.log(count, 1024)), len(units) - 1) if count > 0 else 0
    if index == 0:
        return f"{count} B"
    return f"{count / (1024**index):.2f} {units[index]}"
