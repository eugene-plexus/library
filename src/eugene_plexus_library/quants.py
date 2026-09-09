"""What the quant tiers mean.

Static reference content, served from the component so there is one copy
to revise as upstream's families churn and so a headless install can
print it. This is the direct answer to the complaint the whole milestone
exists for: *"Q3_K_S vs 2Q_K_M? No one fucking knows."*

## The line this file does not cross

**No per-model judgement, ever.** Every entry here describes a
*scheme* — how the bits are allocated and what the naming convention
means — and the shape of the quality curve, which is broadly agreed:
noticeable degradation below roughly 4 bits per weight, severe below 3.

What is deliberately absent is any claim that `IQ2_S` beats `Q2_K` on a
given model. That is upstream research, it is model-dependent, and we
would be inventing it. A fabricated quality score is worse than none —
the same rule that makes an unrecognised `general.file_type` degrade to
the raw integer rather than to a plausible neighbour.

The per-model numbers this component *does* assert are size, measured
bits per weight, and whether it fits. All three are arithmetic.

## nominalBitsPerWeight is nominal

A K-quant mixes precisions per tensor — attention and feed-forward
weights at different widths, the embedding and output layers often
larger — so the real figure for a specific file is always its own
`sizeBytes * 8 / parameters` and is routinely half a bit off the
nominal. The nominal value is here for sorting and for explaining the
family, not for arithmetic.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ._generated.models import QuantTable, QuantTier

TABLE_REVISED = datetime(2026, 9, 8, tzinfo=UTC)

_LOW = (
    "Below 3 bits per weight quality degrades severely. Reach for these only when "
    "the alternative is not running the model at all — and prefer a smaller model "
    "at a higher quant if one exists, which is almost always the better trade."
)

TIERS: list[QuantTier] = [
    QuantTier(
        tier="F32",
        family="unquantized",
        nominalBitsPerWeight=32,
        summary="Full 32-bit floats. Not a quant — the training precision, unmodified.",
        guidance=(
            "Twice the size of BF16 for no quality gain on a model that was trained in "
            "16-bit, which is all of them. Present in repos as a conversion artifact."
        ),
    ),
    QuantTier(
        tier="F16",
        family="unquantized",
        nominalBitsPerWeight=16,
        summary="Half-precision floats. The reference point every quant is measured against.",
        guidance=(
            "The quality ceiling, and roughly 2 GB per billion parameters. Worth it only "
            "if the model fits with room for the KV cache; a Q8_0 is half the size and "
            "the difference is not something people reliably notice."
        ),
    ),
    QuantTier(
        tier="BF16",
        family="unquantized",
        nominalBitsPerWeight=16,
        summary="Brain float: same size as F16, wider exponent, less precision.",
        guidance=(
            "What most models are trained in, so a BF16 GGUF is a straight conversion "
            "with no quantization step at all. Interchangeable with F16 in practice for "
            "inference."
        ),
    ),
    QuantTier(
        tier="Q8_0",
        family="legacy",
        nominalBitsPerWeight=8.5,
        summary="8-bit, one scale per 32 weights. The largest quant anyone needs.",
        guidance=(
            "Half the size of F16 and the point where quantization loss stops being a "
            "practical concern. A sensible ceiling: spend the memory you save on a "
            "longer context or a bigger model instead."
        ),
    ),
    QuantTier(
        tier="Q6_K",
        family="k_quant",
        nominalBitsPerWeight=6.6,
        summary="6-bit K-quant. Very close to Q8_0 at three quarters the size.",
        guidance=(
            "The usual answer when Q8_0 will not fit. Differences from the full-precision "
            "model are hard to demonstrate at this width."
        ),
    ),
    QuantTier(
        tier="Q5_K_M",
        family="k_quant",
        nominalBitsPerWeight=5.7,
        summary="5-bit K-quant, medium mixture. A step below Q6_K.",
        guidance=(
            "A reasonable choice when Q6_K is a little too large. `_S` variants of the "
            "same width are smaller and slightly worse; `_M` is the one to prefer if both "
            "fit."
        ),
    ),
    QuantTier(
        tier="Q4_K_M",
        family="k_quant",
        nominalBitsPerWeight=4.8,
        summary="4-bit K-quant, medium mixture. The de-facto default for local inference.",
        guidance=(
            "The most common choice in the field, and the one to start from: about a "
            "quarter the size of F16 with quality most people find hard to distinguish "
            "in ordinary use. If you do not know what to pick, pick this."
        ),
    ),
    QuantTier(
        tier="Q4_K_S",
        family="k_quant",
        nominalBitsPerWeight=4.5,
        summary="4-bit K-quant, small mixture. Q4_K_M's smaller sibling.",
        guidance=(
            "Around 5% smaller than Q4_K_M and measurably worse. Worth it only when that "
            "5% is the difference between fitting and not — which the fit verdict will "
            "tell you."
        ),
    ),
    QuantTier(
        tier="Q4_0",
        family="legacy",
        nominalBitsPerWeight=4.5,
        summary="The original 4-bit scheme: one scale per block, no per-tensor mixing.",
        guidance=(
            "Superseded by the K-quants, which spend their bits where they matter and are "
            "better at the same size. Still published for compatibility; prefer Q4_K_M "
            "when both exist."
        ),
    ),
    QuantTier(
        tier="IQ4_XS",
        family="i_quant",
        nominalBitsPerWeight=4.25,
        summary="4-bit importance-matrix quant. Smaller than Q4_K_S at similar quality.",
        guidance=(
            "The `IQ` family uses a calibration pass (an importance matrix, often shipped "
            "in the same repo) to decide which weights can afford to lose precision. It "
            "buys roughly half a bit at the same quality, and it is where the sub-4-bit "
            "tiers become usable at all."
        ),
    ),
    QuantTier(
        tier="Q3_K_M",
        family="k_quant",
        nominalBitsPerWeight=3.9,
        summary="3-bit K-quant, medium mixture. The edge of comfortable.",
        guidance=(
            "Degradation starts to be noticeable here — more repetition, weaker "
            "instruction-following, worse at long reasoning chains. Reasonable for a "
            "large model that will not fit any other way; a smaller model at Q4_K_M is "
            "often the better trade."
        ),
    ),
    QuantTier(
        tier="IQ3_S",
        family="i_quant",
        nominalBitsPerWeight=3.4,
        summary="3-bit importance-matrix quant.",
        guidance=(
            "Better than a plain 3-bit K-quant at the same size, and still below the "
            "width where quality holds up. This is the tier where a 70B on a 24 GB card "
            "becomes arithmetically possible and qualitatively a compromise."
        ),
    ),
    QuantTier(
        tier="Q2_K",
        family="k_quant",
        nominalBitsPerWeight=2.9,
        summary="2-bit K-quant. Severe compression.",
        guidance=_LOW,
    ),
    QuantTier(
        tier="IQ2_S",
        family="i_quant",
        nominalBitsPerWeight=2.4,
        summary="2-bit importance-matrix quant.",
        guidance=_LOW,
    ),
    QuantTier(
        tier="IQ1_S",
        family="i_quant",
        nominalBitsPerWeight=1.8,
        summary="Roughly 1.5-2 bits per weight. The smallest thing published.",
        guidance=(
            "An experiment more than a deployment target. Coherent output is not "
            "guaranteed. If this is the only tier that fits, the honest options are a "
            "shorter context, partial CPU offload, or a smaller model."
        ),
    ),
]

_SUFFIX_NOTES: dict[str, str] = {
    "UD": (
        "`UD-` marks Unsloth Dynamic: the publisher chose the width per tensor rather than "
        "taking the scheme's defaults, so a UD file of a given tier is usually a little "
        "larger and a little better than the plain one. It is a repacking of the same "
        "family, not a different scheme."
    ),
    "XL": (
        "`_XL` is wider than `_L`, which is wider than `_M`, which is wider than `_S`, "
        "within one tier. The letters are the mixture, not the bit width."
    ),
}


def table() -> QuantTable:
    """The full table, as served by `GET /v1/quants`."""
    return QuantTable(tiers=list(TIERS), updatedAt=TABLE_REVISED)


_BY_TIER: dict[str, QuantTier] = {t.tier: t for t in TIERS}


def describe(tier: str | None) -> QuantTier | None:
    """Look up one tier, tolerating the prefixes publishers add.

    `UD-Q4_K_XL` is a repacking of `Q4_K_M`'s family and should not read
    as unknown; an exact match wins, then the tier with the longest name
    that the label ends with. Returns None rather than a nearest guess
    when nothing matches — a tier we do not recognise is reported by its
    label and its measured bits per weight, which is the truth.
    """
    if not tier:
        return None
    label = tier.upper()
    exact = _BY_TIER.get(label)
    if exact is not None:
        return exact

    # A tier the label *ends* with: `UD-Q4_K_M` is a repacking of
    # `Q4_K_M`, and the `UD-` prefix is the publisher's, not a scheme.
    suffixed = [t for t in TIERS if label.endswith(t.tier)]
    if suffixed:
        return max(suffixed, key=lambda t: len(t.tier))

    # Otherwise the closest *family*. `Q4_K_XL` and `UD-Q8_K_XL` are
    # widths nobody upstream defines, and reporting a family we do
    # describe as unknown left a third of one real repo's candidates
    # with no reference text at all. Matched on the shared prefix rather
    # than by guessing a neighbour: three characters is `Q4_`, which is
    # a family, while `Q9_ULTRA` shares only `Q` with anything here and
    # correctly resolves to nothing.
    bare = label.removeprefix("UD-").removeprefix("UD_")
    scored: list[tuple[int, int, int, QuantTier]] = []
    for index, candidate in enumerate(TIERS):
        shared = 0
        for left, right in zip(bare, candidate.tier, strict=False):
            if left != right:
                break
            shared += 1
        if shared >= _FAMILY_PREFIX_MINIMUM:
            # Longest shared prefix wins; on a tie, the tier closest in
            # length. Without that second key `Q4_1` resolves to the
            # K-quant `Q4_K_M` rather than to `Q4_0`, which is the
            # legacy scheme it actually belongs to.
            scored.append((-shared, abs(len(bare) - len(candidate.tier)), index, candidate))
    if not scored:
        return None
    return min(scored)[3]


_FAMILY_PREFIX_MINIMUM = 3
"""How much of a tier name has to match to call it the same family.

Three characters is `Q4_`, `IQ3` or `Q6_` — a real family. Two would
make `Q4_K_XL` and `Q8_0` relatives, and one would match everything
beginning with `Q`."""


def suffix_notes(tier: str | None) -> list[str]:
    """Explanations for the naming decorations on a specific label."""
    if not tier:
        return []
    label = tier.upper()
    notes = []
    if label.startswith("UD-") or label.startswith("UD_"):
        notes.append(_SUFFIX_NOTES["UD"])
    if label.endswith(("_XL", "_L", "_M", "_S")):
        notes.append(_SUFFIX_NOTES["XL"])
    return notes
