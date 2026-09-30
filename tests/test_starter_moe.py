"""The starter set's MoE class (moe-aware-fit A3c, call B).

A MoE entry may be recommended where it runs with its experts in system
memory, and it wins only over a dense entry of a smaller size class. The
numbers are design §0's: the 30B-A3B Q4_K_M is 18.56 GB of which 16.35 GiB
are experts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library import starter
from eugene_plexus_library import starter_review as review
from eugene_plexus_library._generated.models import (
    FitOffload,
    FitVerdict,
    KvCacheType,
    MemoryBudget,
    Source,
)

GIB = 1024**3
MOE_SIZE = int(18.56e9)
MOE_EXPERTS = int(16.348 * GIB)


def budget(vram: int, *, ram: int) -> MemoryBudget:
    return MemoryBudget(
        vramFreeBytes=vram,
        vramTotalBytes=vram,
        largestGpuFreeBytes=vram,
        ramAvailableBytes=ram,
        ramTotalBytes=ram * 2,
        gpuCount=1 if vram else 0,
        unifiedMemory=False,
        source=Source.detected,
    )


def entry(size_class: str, *, size: int, params: int, experts: int | None = None) -> dict:
    recommended: dict[str, Any] = {
        "file": f"model-{size_class}.gguf",
        "label": "Q4_K_M",
        "sizeBytes": size,
    }
    if experts is not None:
        recommended["expertBytes"] = experts
    return {
        "class": size_class,
        "baseModel": f"vendor/model-{size_class}",
        "repo": f"publisher/model-{size_class}-GGUF",
        "why": "most downloaded in its class",
        "parameters": params,
        "contextLength": 262144,
        "recommended": recommended,
        "shape": {
            "blockCount": 48,
            "attentionLayers": 48,
            "headCountKv": 4,
            "keyLength": 128,
            "valueLength": 128,
        },
    }


DENSE = [
    entry("4B", size=2_740_937_888, params=4_205_751_296),
    entry("8B", size=5_335_291_936, params=7_518_069_290),
    entry("14B", size=7_121_861_440, params=11_907_350_576),
    entry("30B", size=16_464_440_224, params=27_320_697_856),
]
MOE = entry("30B MoE", size=MOE_SIZE, params=30_532_122_624, experts=MOE_EXPERTS)


def scored(tmp_path: Path, classes: list[dict], machine: MemoryBudget, context: int = 8192):
    path = tmp_path / "starter.yaml"
    path.write_text(
        yaml.safe_dump({"reviewed": "2026-09-30", "classes": classes}), encoding="utf-8"
    )
    return starter.build(
        starter.load(str(path)),
        budget=machine,
        context_length=context,
        kv_cache_type=KvCacheType.f16,
    )


def model(result, size_class: str):
    return next(m for m in result.models if m.sizeClass == size_class)


def test_an_8gb_card_is_offered_the_moe_model_with_its_experts_in_ram(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE, MOE], budget(int(7.5 * GIB), ram=24 * GIB))
    moe = model(result, "30B MoE")
    assert moe.fit.verdict is FitVerdict.split and moe.fit.offload is FitOffload.experts
    assert model(result, "8B").fit.verdict is FitVerdict.fits
    assert result.recommended.sizeClass == "30B MoE"
    reason = result.recommended.reason
    assert "experts in system memory" in reason
    assert "vendor/model-8B" in reason  # the dense pick it displaced, named
    assert f"{moe.maxContextExpertsInRam:,} tokens" in reason
    assert "tok/s" not in reason  # no speed is predicted


def test_a_card_that_holds_the_dense_30b_keeps_it(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE, MOE], budget(32 * GIB, ram=64 * GIB))
    assert model(result, "30B MoE").fit.verdict is FitVerdict.fits
    assert model(result, "30B").fit.verdict is FitVerdict.fits
    # One class, both fit, the MoE file is larger: the dense one still wins.
    assert result.recommended.sizeClass == "30B"


def test_where_ram_does_not_hold_the_experts_the_dense_pick_stands(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE, MOE], budget(int(7.5 * GIB), ram=8 * GIB))
    assert model(result, "30B MoE").fit.verdict is FitVerdict.no
    assert result.recommended.sizeClass == "8B"


def test_a_moe_entry_whose_rest_does_not_fit_the_card_is_not_offered(tmp_path: Path) -> None:
    # 1.5 GiB holds neither the non-expert part plus cache nor any dense entry:
    # a MoE split by whole layers is the 4.9 tok/s case, never a first model.
    result = scored(tmp_path, [*DENSE, MOE], budget(int(1.5 * GIB), ram=60 * GIB))
    moe = model(result, "30B MoE")
    assert moe.fit.verdict is FitVerdict.split and moe.fit.offload is FitOffload.layers
    assert result.recommended.sizeClass is None


def test_a_moe_entry_of_unknown_expert_share_is_scored_as_dense(tmp_path: Path) -> None:
    unknown = entry("30B MoE", size=MOE_SIZE, params=30_532_122_624)
    result = scored(tmp_path, [*DENSE, unknown], budget(int(7.5 * GIB), ram=24 * GIB))
    assert model(result, "30B MoE").fit.offload is None
    assert model(result, "30B MoE").maxContextExpertsInRam is None
    assert result.recommended.sizeClass == "8B"


def test_a_moe_entry_that_fits_entirely_beats_a_smaller_dense_class(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE[:3], MOE], budget(24 * GIB, ram=64 * GIB))
    assert model(result, "30B MoE").fit.verdict is FitVerdict.fits
    assert result.recommended.sizeClass == "30B MoE"
    assert "runs entirely in GPU memory" in result.recommended.reason


def test_a_machine_with_no_graphics_card_still_gets_the_smallest(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE, MOE], budget(0, ram=64 * GIB))
    assert result.recommended.sizeClass == "4B"


def test_only_a_moe_entry_carries_the_experts_in_ram_context(tmp_path: Path) -> None:
    result = scored(tmp_path, [*DENSE, MOE], budget(int(7.5 * GIB), ram=24 * GIB))
    assert model(result, "30B MoE").maxContextExpertsInRam
    assert all(model(result, c).maxContextExpertsInRam is None for c in ("4B", "8B", "30B"))


def test_size_rank_compares_dense_and_moe_by_total_parameters() -> None:
    assert starter.size_rank(None) == -1
    assert starter.size_rank(30_532_122_624) == starter.size_rank(27_320_697_856)
    assert starter.size_rank(30_532_122_624) > starter.size_rank(7_518_069_290)


# --- the review --------------------------------------------------------


def row(repo: str, *, base: str, total: int, architecture: str) -> dict[str, Any]:
    return {
        "id": repo,
        "downloads": 10,
        "gguf": {
            "architecture": architecture,
            "context_length": 262144,
            "total": total,
            "chat_template": "{{ messages }}",
        },
        "cardData": {"base_model": base, "license": "apache-2.0"},
        "gated": False,
        "tags": [],
    }


def test_a_moe_model_ranks_in_its_own_class_by_name_or_architecture() -> None:
    candidates, _ = review.rank(
        [
            # The architecture alone says MoE here; the next says it by name alone.
            row("unsloth/a", base="V/Experts-30B", total=30_532_122_624, architecture="qwen3moe"),
            row("bartowski/a", base="V/Experts-30B", total=30_532_122_624, architecture="qwen3moe"),
            row("unsloth/b", base="V/Suffix-35B-A3B", total=35_000_000_000, architecture="x"),
            row("bartowski/b", base="V/Suffix-35B-A3B", total=35_000_000_000, architecture="x"),
            row("unsloth/c", base="V/Dense-27B", total=27_320_697_856, architecture="qwen35"),
            row("bartowski/c", base="V/Dense-27B", total=27_320_697_856, architecture="qwen35"),
            row("unsloth/d", base="V/Big-235B-A22B", total=235_000_000_000, architecture="x"),
            row("bartowski/d", base="V/Big-235B-A22B", total=235_000_000_000, architecture="x"),
        ]
    )
    by_class = review.eligible(candidates)
    assert sorted(c.name for c in by_class["30B MoE"]) == ["V/Experts-30B", "V/Suffix-35B-A3B"]
    assert [c.name for c in by_class["30B"]] == ["V/Dense-27B"]
    # A MoE outside the MoE class's range is in no class, never a dense one.
    assert all("V/Big-235B-A22B" not in [c.name for c in rows] for rows in by_class.values())


@pytest.mark.parametrize("name", ["V/Model-Alpha-7B", "V/gemma-4-e4b-it", "V/llama-3.1-8b-a"])
def test_a_name_without_an_active_suffix_is_not_moe(name: str) -> None:
    c = review.Candidate(key=name.lower())
    c.names[name] = 1
    c.architectures["llama"] = 1
    assert not c.looks_moe


def _build(monkeypatch, *, size_class: str, expert_bytes: int | None, fail: bool = False):
    chosen = SimpleNamespace(
        files=[SimpleNamespace(path="m-Q4_K_M.gguf")], label="Q4_K_M", size=MOE_SIZE
    )

    async def pick(_client, _candidate):
        return "unsloth/m-GGUF", chosen

    async def header(_client, *, repo, revision, path):
        if fail:
            raise review.HubError("upstream said no")
        meta = SimpleNamespace(
            is_embedding=False,
            architecture="qwen3moe",
            context_length=262144,
            expert_bytes=expert_bytes,
        )
        return meta, 1024

    monkeypatch.setattr(review, "_pick_repo_and_quant", pick)
    monkeypatch.setattr(review.preflight_mod, "read_gguf_header", header)
    monkeypatch.setattr(
        review.preflight_mod, "shape_from_gguf", lambda _meta: fit_mod.ModelShape(block_count=48)
    )
    c = review.Candidate(key="v/m-30b-a3b")
    c.names["V/m-30B-A3B"] = 1
    c.parameters[30_532_122_624] = 1
    return asyncio.run(review.build_entry(object(), c, size_class=size_class))  # type: ignore[arg-type]


def test_a_moe_entry_carries_the_expert_bytes_its_file_has(monkeypatch) -> None:
    built = _build(monkeypatch, size_class="30B MoE", expert_bytes=MOE_EXPERTS)
    assert built is not None and built["recommended"]["expertBytes"] == MOE_EXPERTS


@pytest.mark.parametrize("experts", [0, None])
def test_a_moe_class_refuses_a_file_without_expert_tensors(monkeypatch, experts) -> None:
    assert _build(monkeypatch, size_class="30B MoE", expert_bytes=experts) is None


def test_a_moe_class_refuses_a_file_whose_header_cannot_be_read(monkeypatch) -> None:
    assert _build(monkeypatch, size_class="30B MoE", expert_bytes=None, fail=True) is None
    # A dense class keeps the entry and says the shape is missing.
    dense = _build(monkeypatch, size_class="30B", expert_bytes=None, fail=True)
    assert dense is not None and "shapeUnavailable" in dense


def test_a_dense_entry_records_zero_expert_bytes_and_unknown_as_absent(monkeypatch) -> None:
    zero = _build(monkeypatch, size_class="30B", expert_bytes=0)
    assert zero is not None and zero["recommended"]["expertBytes"] == 0
    unknown = _build(monkeypatch, size_class="30B", expert_bytes=None)
    assert unknown is not None and "expertBytes" not in unknown["recommended"]


def test_the_review_gives_the_moe_class_a_verdict_and_proposes_its_leader(
    monkeypatch, tmp_path: Path
) -> None:
    rows = [
        row("unsloth/m", base="V/m-30B-A3B", total=30_532_122_624, architecture="qwen3moe"),
        row("bartowski/m", base="V/m-30B-A3B", total=30_532_122_624, architecture="qwen3moe"),
    ]

    async def list_models(self, *, limit, pages):
        return rows

    async def aclose(self):
        return None

    async def architectures(_tag):
        return {"qwen3moe"}

    async def build(_client, candidate, *, size_class):
        return {"class": size_class, "baseModel": candidate.name}

    monkeypatch.setattr(review.HubClient, "list_models", list_models)
    monkeypatch.setattr(review.HubClient, "aclose", aclose)
    monkeypatch.setattr(review, "engine_architectures", architectures)
    monkeypatch.setattr(review, "build_entry", build)
    _text, proposed, verdicts = asyncio.run(
        review.review(
            starter_file=tmp_path / "none.yaml",
            engine_tag="b1",
            pages=1,
            limit=1,
            base_url="https://hub.invalid",
            token=None,
        )
    )
    moe = next(v for v in verdicts if v.size_class == "30B MoE")
    # Nothing is shipped in the class, so the review says REPLACE, and the
    # release gate refuses that until a person accepts an entry.
    assert moe.verdict == "REPLACE"
    assert [c["class"] for c in proposed["classes"]] == ["30B MoE"]
