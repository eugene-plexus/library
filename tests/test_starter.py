"""The starter set: the list, the scoring, and which entry a machine takes.

Every case here is a machine the recommendation has to get right, and the
two that matter most are the ones measurement produced rather than
reasoning: a per-layer KV cache (without which a current 12B reads 43x
too large and the card refuses it on a 32 GB card), and a machine with no
accelerator (where "fits" and "usable" have opposite answers).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library import starter
from eugene_plexus_library._generated.models import (
    FitVerdict,
    KvCacheType,
    MemoryBudget,
    Source,
    StarterSetSource,
)

GIB = 1024**3


def budget(vram: int, *, ram: int = 32 * GIB, unified: bool = False) -> MemoryBudget:
    return MemoryBudget(
        vramFreeBytes=vram,
        vramTotalBytes=vram,
        largestGpuFreeBytes=vram,
        ramAvailableBytes=ram,
        ramTotalBytes=ram * 2,
        gpuCount=1 if vram else 0,
        unifiedMemory=unified,
        source=Source.detected,
    )


def write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "starter.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def entry(size_class: str, *, size: int, params: int, **shape: object) -> dict:
    return {
        "class": size_class,
        "baseModel": f"vendor/model-{size_class}",
        "repo": f"publisher/model-{size_class}-GGUF",
        "why": "most downloaded in its class",
        "license": "apache-2.0",
        "parameters": params,
        "architecture": "testarch",
        "contextLength": 262144,
        "evidence": {"downloads30d": 1_000_000},
        "recommended": {
            "file": f"model-{size_class}-Q4_K_M.gguf",
            "label": "Q4_K_M",
            "sizeBytes": size,
        },
        "shape": {
            "blockCount": 32,
            "attentionLayers": 32,
            "headCountKv": 4,
            "keyLength": 128,
            "valueLength": 128,
            **shape,
        },
    }


# --- the shipped list --------------------------------------------------


def test_the_shipped_list_loads_and_every_entry_can_be_scored() -> None:
    """The wheel's own file. A starter list that does not parse is a
    first-run screen with nothing on it, so this is the check that the
    accepted review actually shipped."""
    loaded = starter.load()
    assert loaded.source is StarterSetSource.shipped
    assert loaded.notes == []
    assert loaded.entries, "the shipped list is empty"
    for row in loaded.entries:
        assert row.size_bytes > 0
        assert row.file.endswith(".gguf")
        assert row.shape.complete, f"{row.base_model} cannot be scored from its recorded shape"


def test_the_shipped_list_names_the_engine_it_was_verified_against() -> None:
    """`engine:` is what makes "we proved it loads" a checkable claim
    rather than a sentence in a design document."""
    assert (starter.load().engine or "").startswith("llama_cpp b")


# --- degrading ---------------------------------------------------------


def test_a_missing_file_recommends_nothing_and_says_so(tmp_path: Path) -> None:
    loaded = starter.load(str(tmp_path / "nope.yaml"))
    assert loaded.entries == []
    assert loaded.source is StarterSetSource.configured
    assert any("No starter list" in note for note in loaded.notes)


def test_broken_yaml_degrades_rather_than_raising(tmp_path: Path) -> None:
    """`degraded-mode-required`, applied to a data file: the Discover
    screen still has a search box, and a traceback on the way to it
    would take that away too."""
    path = tmp_path / "starter.yaml"
    path.write_text("classes: [ unclosed", encoding="utf-8")
    loaded = starter.load(str(path))
    assert loaded.entries == []
    assert loaded.notes and "could not be read" in loaded.notes[0]


def test_an_entry_missing_its_file_is_skipped_and_named(tmp_path: Path) -> None:
    broken = entry("8B", size=5 * GIB, params=8_000_000_000)
    del broken["recommended"]["file"]
    path = write(tmp_path, {"reviewed": "2026-09-16", "classes": [broken]})
    loaded = starter.load(str(path))
    assert loaded.entries == []
    assert "recommended.file" in loaded.notes[0]


def test_an_undated_list_scores_as_stale_rather_than_fresh(tmp_path: Path) -> None:
    """A missing date must not read as today's. `_epoch` is deliberately
    ancient so the staleness note fires."""
    path = write(
        tmp_path,
        {"classes": [entry("4B", size=2 * GIB, params=4_000_000_000)]},
    )
    loaded = starter.load(str(path))
    built = starter.build(
        loaded,
        budget=budget(24 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
        today=date(2026, 9, 16),
    )
    assert built.reviewedDaysAgo is not None and built.reviewedDaysAgo > 10_000
    assert any("past the" in note for note in built.notes or [])


def test_a_list_reviewed_today_carries_no_staleness_note(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        {
            "reviewed": "2026-09-16",
            "classes": [entry("4B", size=2 * GIB, params=4_000_000_000)],
        },
    )
    built = starter.build(
        starter.load(str(path)),
        budget=budget(24 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
        today=date(2026, 9, 16),
    )
    assert built.reviewedDaysAgo == 0
    assert not any("past the" in note for note in built.notes or [])


def test_an_empty_list_recommends_nothing_at_all(tmp_path: Path) -> None:
    """Empty is a valid answer and must not be dressed up as one. A
    client renders *find a model*; inventing a recommendation from an
    empty list is the one thing this endpoint must never do."""
    path = write(tmp_path, {"reviewed": "2026-09-16", "classes": []})
    built = starter.build(
        starter.load(str(path)),
        budget=budget(24 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.models == []
    assert built.recommended is None


# --- the recommendation ------------------------------------------------


@pytest.fixture
def three_classes(tmp_path: Path) -> starter.StarterList:
    path = write(
        tmp_path,
        {
            "reviewed": "2026-09-16",
            "engine": "llama_cpp b10999",
            "classes": [
                entry("4B", size=2 * GIB, params=4_000_000_000),
                entry("8B", size=5 * GIB, params=8_000_000_000),
                entry("30B", size=16 * GIB, params=27_000_000_000),
            ],
        },
    )
    return starter.load(str(path))


def test_the_largest_that_fits_entirely_wins(three_classes: starter.StarterList) -> None:
    built = starter.build(
        three_classes,
        budget=budget(29 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.recommended is not None
    assert built.recommended.sizeClass == "30B"
    assert "largest" in built.recommended.reason


def test_a_smaller_card_walks_down_the_list(three_classes: starter.StarterList) -> None:
    """8 GiB holds the 8B (6.5 GiB with overhead and cache) and not the
    30B, so the recommendation steps down one class rather than to the
    bottom of the list."""
    built = starter.build(
        three_classes,
        budget=budget(8 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.recommended is not None
    assert built.recommended.sizeClass == "8B"

    smaller = starter.build(
        three_classes,
        budget=budget(4 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert smaller.recommended is not None
    assert smaller.recommended.sizeClass == "4B"


def test_partial_offload_is_never_recommended(three_classes: starter.StarterList) -> None:
    """`catalogue.recommend`'s rule one level up: partial offload is a
    decision someone makes knowingly, and the first model a person runs
    is the worst place to inherit one."""
    built = starter.build(
        three_classes,
        budget=budget(8 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    chosen = next(m for m in built.models if m.sizeClass == built.recommended.sizeClass)
    assert chosen.fit is not None and chosen.fit.verdict is FitVerdict.fits


def test_nothing_fitting_still_answers_with_a_sentence(
    three_classes: starter.StarterList,
) -> None:
    """A silent absence reads as "we have no opinion", which is the
    opposite of what a machine too small for everything needs to hear."""
    built = starter.build(
        three_classes,
        budget=budget(1 * GIB, ram=1 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.recommended is not None
    assert built.recommended.sizeClass is None
    assert "None of these fits" in built.recommended.reason
    assert "Discover" in built.recommended.reason


def test_a_machine_with_no_gpu_gets_the_smallest_not_the_largest(
    three_classes: starter.StarterList,
) -> None:
    """The rule inverts, and it has to. Sixteen gigabytes of weights fits
    in 32 GB of host memory and generates a couple of words a second,
    which as a first impression is indistinguishable from broken."""
    built = starter.build(
        three_classes,
        budget=budget(0, ram=32 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.recommended is not None
    assert built.recommended.sizeClass == "4B"
    assert "no graphics card" in built.recommended.reason


def test_unified_memory_is_not_a_cpu(three_classes: starter.StarterList) -> None:
    """Apple silicon has one pool and a real GPU on it, so the
    no-accelerator inversion must not fire there — it would hand a
    64 GB Mac the 4B."""
    built = starter.build(
        three_classes,
        budget=budget(24 * GIB, unified=True),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.recommended is not None
    assert built.recommended.sizeClass == "30B"


def test_the_context_decides_and_the_answer_says_so(
    three_classes: starter.StarterList,
) -> None:
    """The badge names the context because the context is what moved
    the verdict; a `fits` with no number beside it is unfalsifiable."""
    small = starter.build(
        three_classes, budget=budget(18 * GIB), context_length=4096, kv_cache_type=KvCacheType.f16
    )
    large = starter.build(
        three_classes, budget=budget(18 * GIB), context_length=131072, kv_cache_type=KvCacheType.f16
    )
    assert small.recommended is not None and large.recommended is not None
    assert small.recommended.sizeClass == "30B"
    assert large.recommended.sizeClass != "30B"
    assert "4,096" in small.recommended.reason


def test_max_context_is_reported_per_entry(three_classes: starter.StarterList) -> None:
    """The number a profile takes, and what lets a client say *fits at
    75,520* instead of *fits*."""
    built = starter.build(
        three_classes, budget=budget(29 * GIB), context_length=8192, kv_cache_type=KvCacheType.f16
    )
    for model in built.models:
        if model.fit and model.fit.verdict is FitVerdict.fits:
            assert model.maxContextLength and model.maxContextLength >= 8192


# --- the per-layer KV cache, which is why the numbers above are real ---


def test_sliding_window_layers_are_not_charged_the_whole_context(tmp_path: Path) -> None:
    """The measured case, as a test.

    A real 12B declares 48 layers, `head_count_kv` as an ARRAY of 48
    (eight on five of every six, one on the sixth), and a 1024-token
    sliding window on the five. The scalar arithmetic falls back to
    `head_count` and reports 24.0 GiB of KV at 16k. The truth is
    0.56 GiB. This asserts the gap, which is what decides whether the
    starter card recommends a 7 GB model or refuses it on a 32 GB card.
    """
    runs = []
    for _ in range(8):
        runs.append([5, 8, 256, 256, 1024])
        runs.append([1, 1, 512, 512, None])
    per_layer = entry(
        "14B",
        size=7 * GIB,
        params=12_000_000_000,
        blockCount=48,
        attentionLayers=48,
        keyLength=512,
        valueLength=512,
        embeddingLength=3840,
        headCount=16,
        layerRuns=runs,
    )
    del per_layer["shape"]["headCountKv"]
    path = write(tmp_path, {"reviewed": "2026-09-16", "classes": [per_layer]})
    loaded = starter.load(str(path))
    assert loaded.entries[0].shape.layers is not None
    assert len(loaded.entries[0].shape.layers) == 48

    built = starter.build(
        loaded, budget=budget(29 * GIB), context_length=16384, kv_cache_type=KvCacheType.f16
    )
    kv = built.models[0].fit.kvCacheBytes
    assert kv < GIB, f"KV came back {kv / GIB:.2f} GiB; the per-layer form was not used"
    assert built.models[0].fit.verdict is FitVerdict.fits

    # And the scalar reading of the same file, for the contrast.
    naive = fit_mod.ModelShape(
        block_count=48, attention_layers=48, key_length=512, value_length=512, head_count=16
    ).kv_bytes(16384, KvCacheType.f16)
    assert naive is not None and naive > 20 * GIB
    assert naive / kv > 20


def test_a_malformed_layer_run_falls_back_to_the_scalars(tmp_path: Path) -> None:
    """A hand-edited file must degrade to the old arithmetic rather than
    to a zero-sized cache, which would report everything as fitting."""
    broken = entry("8B", size=5 * GIB, params=8_000_000_000, layerRuns=[[5, 8, 256]])
    path = write(tmp_path, {"reviewed": "2026-09-16", "classes": [broken]})
    loaded = starter.load(str(path))
    assert loaded.entries[0].shape.layers is None
    assert loaded.entries[0].shape.complete


def test_layer_runs_round_trip() -> None:
    layers = (
        fit_mod.LayerKV(8, 256, 256, 1024),
        fit_mod.LayerKV(8, 256, 256, 1024),
        fit_mod.LayerKV(1, 512, 512, None),
    )
    runs = fit_mod.encode_layers(layers)
    assert runs == [[2, 8, 256, 256, 1024], [1, 1, 512, 512, None]]
    assert fit_mod.decode_layers(runs) == layers


def test_max_context_solves_the_affine_cache_not_a_linear_one() -> None:
    """Sliding layers stop growing at their window, so the cache is a
    line with an intercept. Dividing by a single per-token figure --
    which is what the previous code did -- under-reports the context
    that fits by the whole sliding term."""
    shape = fit_mod.ModelShape(
        layers=(fit_mod.LayerKV(8, 256, 256, 1024),) * 40
        + (fit_mod.LayerKV(1, 512, 512, None),) * 8
    )
    fits = fit_mod.max_context_that_fits(
        weights_bytes=7 * GIB, budget=budget(29 * GIB), shape=shape
    )
    assert fits is not None
    # The context it claims must actually fit, with the overhead in.
    used = 7 * GIB + (shape.kv_bytes(fits, KvCacheType.f16) or 0) + fit_mod.DEFAULT_OVERHEAD_BYTES
    assert used <= 29 * GIB
