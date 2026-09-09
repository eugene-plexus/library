"""The fit arithmetic, and the three ways it goes wrong.

Every number here is one measured on real hardware or read off a real
model file — see `docs/design/m3-discovery-download-guidance.md` in the
specs repo. A test written against invented numbers would agree with
whatever the code believes.
"""

from __future__ import annotations

from eugene_plexus_library import fit
from eugene_plexus_library._generated.models import (
    Arch,
    Basis,
    FitVerdict,
    Gpu,
    HostHardware,
    KvCacheType,
    Os,
    Source,
    Vendor,
)

GIB = 1024**3
MIB = 1024**2

# Read off unsloth/Qwen3.8-27B-GGUF's Q4_K_M over HTTP Range.
QWEN_SHAPE = fit.ModelShape(
    block_count=65,
    attention_layers=16,  # 65 // full_attention_interval 4
    head_count_kv=4,
    key_length=256,
    value_length=256,
    context_length=262144,
    parameters=27_320_697_856,
)

# The dev box: one RTX 5090, 2.9 GiB already held by the desktop.
DEV_BOX = HostHardware(
    hostname="dev",
    os=Os.windows,
    arch=Arch.x64,
    ramTotalBytes=int(93.56 * GIB),
    ramAvailableBytes=int(58.75 * GIB),
    gpus=[
        Gpu(
            index=0,
            name="NVIDIA GeForce RTX 5090",
            vendor=Vendor.nvidia,
            vramTotalBytes=32607 * MIB,
            vramFreeBytes=29582 * MIB,
        )
    ],
)


def test_hybrid_attention_is_not_the_layer_count() -> None:
    """The trap that would have made guidance a liar.

    65 blocks with a full-attention interval of 4 means 16 layers hold a
    KV cache. Assuming all 65 over-estimates by 4.1x at every context
    length — 8.12 GiB instead of 2.00 at 32k — which at this model's own
    trained context is the difference between "fits on a 5090" and "no
    candidate in this repo is runnable".
    """
    hybrid = QWEN_SHAPE.kv_bytes(32768, KvCacheType.f16)
    naive = fit.ModelShape(
        block_count=65,
        head_count_kv=4,
        key_length=256,
        value_length=256,
    ).kv_bytes(32768, KvCacheType.f16)

    assert hybrid is not None and naive is not None
    assert round(hybrid / GIB, 2) == 2.00
    assert round(naive / GIB, 2) == 8.12
    assert round(naive / hybrid, 1) == 4.1


def test_kv_cache_is_linear_in_context() -> None:
    """Measured at four context lengths on the same model."""
    expected = {4096: 0.25, 32768: 2.00, 131072: 8.00, 262144: 16.00}
    for context, gib in expected.items():
        kv = QWEN_SHAPE.kv_bytes(context, KvCacheType.f16)
        assert kv is not None
        assert round(kv / GIB, 2) == gib


def test_quantized_kv_cache_is_not_simply_half() -> None:
    """`q8_0` carries a per-block scale, so 32 8-bit weights plus an f16
    scale is 34 bytes for 32 elements — not 32."""
    f16 = QWEN_SHAPE.kv_bytes(32768, KvCacheType.f16)
    q8 = QWEN_SHAPE.kv_bytes(32768, KvCacheType.q8_0)
    assert f16 is not None and q8 is not None
    assert 0.5 < q8 / f16 < 0.55


def test_free_not_total_decides_the_verdict() -> None:
    """2.9 GiB of a 32 GiB card is held before an engine starts.

    A candidate that fits in total but not in free memory is `tight`,
    not `fits`: it would fit on an idle GPU, and closing something is
    the operator's call. Scoring against total would promise a fit that
    OOMs.
    """
    budget = fit.budget_from_hardware(DEV_BOX)
    assert budget.vramFreeBytes is not None and budget.vramTotalBytes is not None
    assert budget.vramFreeBytes < budget.vramTotalBytes

    between = budget.vramFreeBytes + GIB  # over free, under total
    result = fit.compute(
        weights_bytes=between - fit.DEFAULT_OVERHEAD_BYTES - 100 * MIB,
        budget=budget,
        context_length=1024,
        shape=QWEN_SHAPE,
    )
    assert result.verdict is FitVerdict.tight


def test_the_four_verdicts_are_reachable() -> None:
    budget = fit.budget_from_hardware(DEV_BOX)
    free = budget.vramFreeBytes or 0
    total = budget.vramTotalBytes or 0
    ram = budget.ramAvailableBytes or 0
    overhead = fit.DEFAULT_OVERHEAD_BYTES

    def verdict(weights: int) -> FitVerdict:
        return fit.compute(
            weights_bytes=weights,
            budget=budget,
            context_length=256,
            shape=QWEN_SHAPE,
            overhead_bytes=overhead,
        ).verdict

    assert verdict(2 * GIB) is FitVerdict.fits
    assert verdict(free - overhead + GIB) is FitVerdict.tight
    assert verdict(total - overhead + GIB) is FitVerdict.split
    assert verdict(free + ram) is FitVerdict.no


def test_a_shapeless_model_says_it_is_estimating() -> None:
    """No layer metadata means the KV term is a guess, and the caller has
    to be able to see that — it is the difference between arithmetic and
    a number with no arithmetic behind it."""
    result = fit.compute(
        weights_bytes=8 * GIB,
        budget=fit.budget_from_hardware(DEV_BOX),
        context_length=8192,
    )
    assert result.basis is Basis.estimate
    assert any("15%" in note for note in result.notes or [])


def test_a_shaped_model_says_it_is_not() -> None:
    result = fit.compute(
        weights_bytes=8 * GIB,
        budget=fit.budget_from_hardware(DEV_BOX),
        context_length=8192,
        shape=QWEN_SHAPE,
    )
    assert result.basis is Basis.metadata
    assert result.attentionLayers == 16


def test_assuming_every_layer_caches_is_stated_not_hidden() -> None:
    """Where the metadata does not distinguish attention layers, the
    assumption errs pessimistic and says so in `notes`."""
    result = fit.compute(
        weights_bytes=8 * GIB,
        budget=fit.budget_from_hardware(DEV_BOX),
        context_length=8192,
        shape=fit.ModelShape(block_count=65, head_count_kv=4, key_length=256, value_length=256),
    )
    assert any("all 65 layers" in note for note in result.notes or [])


def test_the_arithmetic_is_reported_with_the_verdict() -> None:
    """Differentiator #6 is "show why". A verdict with no numbers behind
    it cannot be argued with, and this one will sometimes be wrong."""
    result = fit.compute(
        weights_bytes=15 * GIB,
        budget=fit.budget_from_hardware(DEV_BOX),
        context_length=32768,
        shape=QWEN_SHAPE,
    )
    assert result.requiredBytes == (
        (result.weightsBytes or 0) + (result.kvCacheBytes or 0) + (result.overheadBytes or 0)
    )
    assert result.contextLength == 32768
    assert result.budget is not None
    assert result.notes


def test_an_override_is_labelled_as_one() -> None:
    """Scoring against another host's memory is the only honest answer
    when the GPU is in a different building, and a UI has to be able to
    say the verdict is not about the machine it measured."""
    budget = fit.budget_from_hardware(DEV_BOX, vram_override=24 * GIB)
    assert budget.source is Source.override
    assert budget.vramFreeBytes == 24 * GIB
    result = fit.compute(
        weights_bytes=2 * GIB, budget=budget, context_length=1024, shape=QWEN_SHAPE
    )
    assert any("caller-supplied" in note for note in result.notes or [])


def test_no_accelerator_scores_against_host_memory() -> None:
    """A CPU-only box has no "fits in VRAM" to report, and `split` would
    describe a split that is not happening."""
    cpu_only = HostHardware(
        hostname="cpu",
        os=Os.linux,
        arch=Arch.x64,
        ramTotalBytes=32 * GIB,
        ramAvailableBytes=24 * GIB,
        gpus=[],
    )
    budget = fit.budget_from_hardware(cpu_only)
    assert budget.gpuCount == 0
    assert (
        fit.compute(
            weights_bytes=8 * GIB, budget=budget, context_length=4096, shape=QWEN_SHAPE
        ).verdict
        is FitVerdict.fits
    )
    # 30 GiB needed against 32 GiB of total RAM is `tight`, not `no`:
    # it would fit on an idle box, which is a different problem from not
    # fitting at all.
    assert (
        fit.compute(
            weights_bytes=30 * GIB, budget=budget, context_length=4096, shape=QWEN_SHAPE
        ).verdict
        is FitVerdict.tight
    )
    assert (
        fit.compute(
            weights_bytes=64 * GIB, budget=budget, context_length=4096, shape=QWEN_SHAPE
        ).verdict
        is FitVerdict.no
    )


def test_unified_memory_has_no_split_verdict() -> None:
    """On Apple silicon there is nothing to offload *to* — one pool."""
    mac = HostHardware(
        hostname="mac",
        os=Os.macos,
        arch=Arch.arm64,
        ramTotalBytes=96 * GIB,
        ramAvailableBytes=80 * GIB,
        unifiedMemory=True,
        gpus=[Gpu(index=0, name="Apple M3 Ultra", vendor=Vendor.apple, vramTotalBytes=72 * GIB)],
    )
    budget = fit.budget_from_hardware(mac)
    assert budget.unifiedMemory is True
    verdicts = {
        fit.compute(weights_bytes=w, budget=budget, context_length=4096, shape=QWEN_SHAPE).verdict
        for w in (20 * GIB, 71 * GIB, 200 * GIB)
    }
    assert FitVerdict.split not in verdicts


def test_largest_gpu_is_reported_separately_from_the_sum() -> None:
    """ "Fits on one card" and "fits across all of them" are different
    questions with different performance — and two replicas on two cards
    needs the per-card number."""
    two_cards = HostHardware(
        hostname="two",
        os=Os.linux,
        arch=Arch.x64,
        ramTotalBytes=96 * GIB,
        ramAvailableBytes=64 * GIB,
        gpus=[
            Gpu(index=0, name="RTX 3090", vramTotalBytes=24 * GIB, vramFreeBytes=23 * GIB),
            Gpu(index=1, name="RTX 5090", vramTotalBytes=32 * GIB, vramFreeBytes=31 * GIB),
        ],
    )
    budget = fit.budget_from_hardware(two_cards)
    assert budget.vramFreeBytes == 54 * GIB
    assert budget.largestGpuFreeBytes == 31 * GIB
    assert budget.gpuCount == 2

    result = fit.compute(
        weights_bytes=40 * GIB, budget=budget, context_length=4096, shape=QWEN_SHAPE
    )
    assert any("largest single card" in note for note in result.notes or [])


def test_a_card_with_no_free_reading_falls_back_to_total() -> None:
    """Intel's tool does not report free memory in any stable form. The
    fallback is optimistic for that card alone, and the hardware
    warnings name it."""
    intel = HostHardware(
        hostname="arc",
        os=Os.linux,
        arch=Arch.x64,
        ramTotalBytes=32 * GIB,
        ramAvailableBytes=24 * GIB,
        gpus=[Gpu(index=0, name="Arc A770", vendor=Vendor.intel, vramTotalBytes=16 * GIB)],
    )
    budget = fit.budget_from_hardware(intel)
    assert budget.vramFreeBytes == 16 * GIB


def test_bits_per_weight_lands_on_16_for_bf16() -> None:
    """The check that a parameter count is a parameter count.

    Real sizes from the verified repo: BF16's two shards sum to
    54,657,735,616 bytes against 27,320,697,856 parameters.
    """
    assert fit.bits_per_weight(54_657_735_616, 27_320_697_856) == 16.0
    assert fit.bits_per_weight(16_464_440_224, 27_320_697_856) == 4.82
    assert fit.bits_per_weight(6_192_222_208, 27_320_697_856) == 1.81


def test_bits_per_weight_is_absent_without_a_parameter_count() -> None:
    """Which is most GGUFs on disk: the format carries no numeric count.
    A fabricated one would be worse than none."""
    assert fit.bits_per_weight(16 * GIB, None) is None
    assert fit.bits_per_weight(16 * GIB, 0) is None


def test_max_context_is_solved_not_searched() -> None:
    """One division, because the KV term is linear in context. The result
    is the number that goes in a profile's `-c`."""
    budget = fit.budget_from_hardware(DEV_BOX)
    largest = fit.max_context_that_fits(weights_bytes=16 * GIB, budget=budget, shape=QWEN_SHAPE)
    assert largest is not None
    assert largest % 256 == 0

    # It really is the boundary: at that context it fits, and a long way
    # past it, it does not.
    assert (
        fit.compute(
            weights_bytes=16 * GIB, budget=budget, context_length=largest, shape=QWEN_SHAPE
        ).verdict
        is FitVerdict.fits
    )
    assert (
        fit.compute(
            weights_bytes=16 * GIB,
            budget=budget,
            context_length=largest * 4,
            shape=QWEN_SHAPE,
        ).verdict
        is not FitVerdict.fits
    )


def test_max_context_is_capped_by_what_the_model_was_trained_for() -> None:
    """Offering 400k on a model trained for 262k would be arithmetic
    detached from the model."""
    budget = fit.budget_from_hardware(DEV_BOX, vram_override=200 * GIB)
    largest = fit.max_context_that_fits(weights_bytes=16 * GIB, budget=budget, shape=QWEN_SHAPE)
    assert largest == QWEN_SHAPE.context_length


def test_max_context_is_none_when_the_weights_alone_do_not_fit() -> None:
    """No context makes an oversized model fit, and 0 would read as a
    context of zero rather than as "not this way"."""
    assert (
        fit.max_context_that_fits(
            weights_bytes=500 * GIB,
            budget=fit.budget_from_hardware(DEV_BOX),
            shape=QWEN_SHAPE,
        )
        is None
    )


def test_max_context_is_none_without_a_usable_shape() -> None:
    assert (
        fit.max_context_that_fits(
            weights_bytes=8 * GIB,
            budget=fit.budget_from_hardware(DEV_BOX),
            shape=fit.ModelShape(),
        )
        is None
    )


def test_head_dim_is_recovered_from_embedding_and_head_count() -> None:
    """Older GGUFs carry no `key_length`; the classic identity recovers
    it, and without that every pre-2025 model would score as shapeless."""
    shape = fit.ModelShape(
        block_count=32,
        head_count=32,
        head_count_kv=8,
        embedding_length=4096,
    )
    # 32 layers x 8 kv heads x (128 + 128) x 2 bytes = 128 KiB/token.
    kv = shape.kv_bytes(1024, KvCacheType.f16)
    assert kv is not None
    assert kv == 1024 * 32 * 8 * 256 * 2


def test_format_bytes_reads_like_a_size() -> None:
    assert fit.format_bytes(0) == "0 B"
    assert fit.format_bytes(512) == "512 B"
    assert fit.format_bytes(16_464_440_224) == "15.33 GiB"
