"""One model across several cards (2026-09-27).

llama.cpp splits a model's layers across every visible card by default,
and each card then holds its own compute buffers. Two things here were
written for one card:

* the fit routes took one `vramBytes` and one overhead allowance, so a
  console describing two 5090s could only say "one 5090", and a model
  that fits across both read as spilling into system RAM;
* an override kept THIS host's card count. A console scoring a worker's
  5090 from a library with no GPU (a container without passthrough, the
  default) passed 30 GiB and a `gpuCount` of 0, and the starter set,
  which reads `gpuCount` to decide whether there is a card at all, would
  recommend the smallest model "on a machine with no graphics card". The
  live install's library has its P4000 passed through, so it kept a count
  of one: right for one card by coincidence, wrong for a node with two.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library import hardware, starter
from eugene_plexus_library._generated.models import FitVerdict, KvCacheType

from .conftest import qwen_like_kv, write_gguf

GIB = 1024**3


def _nas() -> hardware.HostHardware:
    """A library in a container without GPU passthrough: no GPU, plenty of RAM."""
    return hardware.HostHardware.model_validate(
        {
            "hostname": "nas",
            "os": "linux",
            "arch": "x64",
            "ramTotalBytes": 70 * GIB,
            "ramAvailableBytes": 60 * GIB,
            "unifiedMemory": False,
            "gpus": [],
        }
    )


# --------------------------------------------------------------------------- #
# The override's cards are the caller's
# --------------------------------------------------------------------------- #


def test_a_card_passed_from_another_host_is_a_card() -> None:
    budget = fit_mod.budget_from_hardware(_nas(), vram_override=30 * GIB, ram_override=90 * GIB)
    assert budget.gpuCount == 1
    assert not starter.has_no_accelerator(budget)


def test_zero_passed_from_another_host_is_no_card() -> None:
    """The UI passes `vramBytes=0` for a node with no accelerator."""
    budget = fit_mod.budget_from_hardware(_nas(), vram_override=0, ram_override=32 * GIB)
    assert budget.gpuCount == 0
    assert starter.has_no_accelerator(budget)


def test_the_starter_set_does_not_tell_a_5090_it_has_no_graphics_card() -> None:
    """**The defect, where a person meets it**: a GPU-less library scoring
    the 5090 every launch goes to."""
    budget = fit_mod.budget_from_hardware(_nas(), vram_override=30 * GIB, ram_override=90 * GIB)
    built = starter.build(
        starter.load(), budget=budget, context_length=16384, kv_cache_type=KvCacheType.f16
    )
    chosen = built.recommended
    assert chosen is not None
    assert "no graphics card" not in chosen.reason
    assert "GPU memory" in chosen.reason


# --------------------------------------------------------------------------- #
# Several cards: the sum, and an allowance per card
# --------------------------------------------------------------------------- #


def test_several_cards_are_one_budget_with_an_allowance_each() -> None:
    budget = fit_mod.budget_from_hardware(
        _nas(), vram_override=60 * GIB, ram_override=90 * GIB, gpu_count_override=2
    )
    assert (budget.gpuCount, budget.vramFreeBytes) == (2, 60 * GIB)
    assert budget.largestGpuFreeBytes == 30 * GIB
    assert fit_mod.overhead_for(budget) == 2 * fit_mod.DEFAULT_OVERHEAD_BYTES


def test_a_model_too_big_for_one_card_fits_across_two() -> None:
    """A 40 GiB model on two 30 GiB cards: `split` against one card,
    which a launch refuses, and `fits` against the pair."""
    one = fit_mod.budget_from_hardware(_nas(), vram_override=30 * GIB, ram_override=90 * GIB)
    two = fit_mod.budget_from_hardware(
        _nas(), vram_override=60 * GIB, ram_override=90 * GIB, gpu_count_override=2
    )
    against_one = fit_mod.compute(weights_bytes=40 * GIB, budget=one, context_length=4096)
    against_two = fit_mod.compute(weights_bytes=40 * GIB, budget=two, context_length=4096)
    assert against_one.verdict == FitVerdict.split
    assert against_two.verdict == FitVerdict.fits
    assert against_two.overheadBytes == 2 * fit_mod.DEFAULT_OVERHEAD_BYTES
    assert any("each of the 2 cards" in note for note in against_two.notes)


def test_the_second_cards_allowance_can_be_the_difference() -> None:
    """Weights that fill the pair to within one allowance: one allowance
    said `fits`, and the second card would run out at load."""
    two = fit_mod.budget_from_hardware(
        _nas(), vram_override=60 * GIB, ram_override=90 * GIB, gpu_count_override=2
    )
    weights = 60 * GIB - int(1.5 * fit_mod.DEFAULT_OVERHEAD_BYTES) - 16 * 1024**2
    result = fit_mod.compute(weights_bytes=weights, budget=two, context_length=256)
    assert result.verdict != FitVerdict.fits


def test_the_largest_context_leaves_every_cards_allowance() -> None:
    shape = fit_mod.ModelShape(
        block_count=32,
        head_count=32,
        head_count_kv=8,
        key_length=128,
        value_length=128,
        embedding_length=4096,
        context_length=1_000_000,
    )
    one = fit_mod.budget_from_hardware(_nas(), vram_override=30 * GIB, ram_override=90 * GIB)
    two = fit_mod.budget_from_hardware(
        _nas(), vram_override=30 * GIB, ram_override=90 * GIB, gpu_count_override=2
    )
    alone = fit_mod.max_context_that_fits(weights_bytes=10 * GIB, budget=one, shape=shape)
    split = fit_mod.max_context_that_fits(weights_bytes=10 * GIB, budget=two, shape=shape)
    assert alone is not None and split is not None
    assert split < alone


def test_the_fit_route_takes_the_card_count(
    configured_client: TestClient, models_dir: Path
) -> None:
    write_gguf(models_dir / "m.gguf", qwen_like_kv(vocab=64))
    configured_client.post("/v1/scan")
    for _ in range(200):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
    model_id = configured_client.get("/v1/models").json()["models"][0]["id"]
    body = configured_client.get(
        f"/v1/models/{model_id}/fit",
        params={"vramBytes": 60 * GIB, "ramBytes": 64 * GIB, "gpuCount": 2},
    ).json()
    assert body["fit"]["budget"]["gpuCount"] == 2
    assert body["fit"]["overheadBytes"] == 2 * fit_mod.DEFAULT_OVERHEAD_BYTES
