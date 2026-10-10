"""Each engine's fit, by its own fit model (LS6, Troy's L11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_library import eligibility, engine_fit
from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library._generated.models import (
    EligibilityCandidate,
    EligibilityEngine,
    EligibilityLevel,
    FitQuestion,
    FitVerdict,
    LibraryModel,
    MemoryBudget,
    Source,
)

from .conftest import qwen_like_kv, write_gguf
from .test_eligibility import _scan

GIB = 1024**3

# The engines as their adapters declare them in the agent (LS6).
LLAMA = {
    "engine": "llama_cpp",
    "accepts": [{"format": "gguf", "preference": 10}],
    "fit": {"kind": "spill"},
}
VLLM = {
    "engine": "vllm",
    "accepts": [{"format": "safetensors", "authority": "engine", "preference": 20}],
    "fit": {"kind": "reserved_share", "gpuMemoryUtilization": 0.92},
}
MLX = {"engine": "mlx", "accepts": [{"format": "safetensors", "preference": 30}]}
IQ2 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
STRATA = {
    "engine": "strata",
    "experimental": True,
    "accepts": [
        {"format": "prepared", "preparedFor": "strata", "preference": 50},
        {
            "format": "gguf",
            "architectures": ["qwen4exp"],
            "files": [IQ2],
            "preparation": {"recipe": "strata-prepare"},
            "preference": 50,
        },
    ],
    "fit": {
        "kind": "engine_table",
        "table": [
            {
                "file": IQ2,
                "supportedModel": "IQ2_XS",
                "fit": {
                    "estimated": True,
                    "verdict": "fits",
                    "reason": "Strata keeps its 35.5 GB of experts in RAM",
                    "ramBytes": 48 * GIB,
                },
            }
        ],
    },
}


def engine(spec: dict[str, Any], *, available: bool = True) -> EligibilityEngine:
    return EligibilityEngine.model_validate({**spec, "available": available})


def card(free: int, total: int, *, cards: int = 1, ram: int = 64 * GIB) -> MemoryBudget:
    return MemoryBudget(
        vramFreeBytes=free,
        vramTotalBytes=total,
        largestGpuFreeBytes=free // cards if cards else 0,
        ramAvailableBytes=ram,
        ramTotalBytes=ram,
        gpuCount=cards,
        unifiedMemory=False,
        source=Source.override,
    )


def question(budget: MemoryBudget, context: int = 8192) -> engine_fit.Question:
    return engine_fit.Question(budget=budget, context_length=context)


# --- reserved_share: vLLM's own rule ----------------------------------------


def share(weights: int, budget: MemoryBudget, utilization: float = 0.9):
    return fit_mod.compute_reserved_share(
        weights_bytes=weights, budget=budget, context_length=8192, utilization=utilization
    )


def test_a_share_taking_engine_fits_within_its_share_of_the_total() -> None:
    # 10 GiB of weights, an estimated 1.5 GiB of cache, 1 GiB of buffers:
    # 12.5 GiB inside 90% of a 24 GiB card, which is free.
    result = share(10 * GIB, card(free=23 * GIB, total=24 * GIB))
    assert result.verdict is FitVerdict.fits
    assert result.model.value == "reserved_share"


def test_less_free_than_its_share_is_tight_never_split() -> None:
    # It fits 90% of the card, but only 20 of 24 GiB is free: vLLM refuses
    # to start until its share is free (`request_memory`).
    result = share(10 * GIB, card(free=20 * GIB, total=24 * GIB))
    assert result.verdict is FitVerdict.tight


def test_more_than_its_share_is_no_even_with_ram_to_spare() -> None:
    # llama.cpp would split this into system memory; vLLM moves nothing there.
    # 22 GiB of weights, 3.3 GiB of estimated cache and 1 GiB of buffers.
    budget = card(free=24 * GIB, total=24 * GIB, ram=256 * GIB)
    assert share(22 * GIB, budget).verdict is FitVerdict.no
    assert (
        fit_mod.compute(weights_bytes=22 * GIB, budget=budget, context_length=8192).verdict
        is FitVerdict.split
    )


def test_its_share_is_of_each_card_and_the_model_splits_evenly() -> None:
    # 30 GiB of weights and 4.5 GiB of estimated cache on two 24 GiB cards:
    # 18.25 GiB each with its buffers, inside 22.08 GiB (92%) on each.
    two = card(free=48 * GIB, total=48 * GIB, cards=2)
    assert share(30 * GIB, two, utilization=0.92).verdict is FitVerdict.fits
    one = card(free=24 * GIB, total=24 * GIB)
    assert share(30 * GIB, one, utilization=0.92).verdict is FitVerdict.no


def test_no_card_is_unknown_not_a_ram_verdict() -> None:
    result = share(1 * GIB, card(free=0, total=0, cards=0))
    assert result.verdict is FitVerdict.unknown


def test_the_longest_context_is_what_its_share_holds() -> None:
    shape = fit_mod.ModelShape(
        block_count=32, head_count_kv=8, key_length=128, value_length=128, context_length=131072
    )
    budget = card(free=24 * GIB, total=24 * GIB)
    longest = fit_mod.max_context_reserved_share(
        weights_bytes=10 * GIB, budget=budget, shape=shape, utilization=0.9
    )
    # (24 * 0.9 - 1 - 10) GiB over 128 KiB a token (f16, 32 layers, 8 heads).
    per_token = 32 * 8 * 256 * 2
    expected = int((int(24 * GIB * 0.9) - GIB - 10 * GIB) // per_token) // 256 * 256
    assert longest == expected
    at_that = fit_mod.compute_reserved_share(
        weights_bytes=10 * GIB,
        budget=budget,
        context_length=expected,
        utilization=0.9,
        shape=shape,
    )
    assert at_that.verdict is FitVerdict.fits


# --- the judge answers each engine's fit ------------------------------------

FLASH = LibraryModel.model_validate(
    {
        "id": "flash",
        "path": f"D:\\models\\{IQ2}",
        "format": "gguf",
        "name": "flash",
        "status": "present",
        "architecture": "qwen4exp",
        "sizeBytes": 68 * 10**9,
        "gguf": {"quantization": "IQ2_XS"},
    }
)
FIVE_090 = card(free=30 * GIB, total=32 * GIB, ram=20 * GIB)


def test_each_engine_answers_by_its_own_model() -> None:
    answer = eligibility.judge(
        FLASH, [engine(LLAMA), engine(STRATA), engine(VLLM)], question(FIVE_090)
    )
    by = {v.engine.value: v for v in answer.engines}
    llama = by["llama_cpp"].fit
    assert llama is not None and llama.estimated and llama.verdict is FitVerdict.no
    assert llama.model.value == "spill" and "too large here" in llama.reason
    strata = by["strata"].fit
    assert strata is not None and strata.verdict is FitVerdict.fits
    assert strata.model.value == "engine_table" and "35.5 GB" in strata.reason
    # vLLM does not load a GGUF: a `no` verdict carries no fit.
    assert by["vllm"].fit is None


def test_too_large_for_llama_cpp_but_strata_prepares_it_is_another_engine() -> None:
    answer = eligibility.judge(FLASH, [engine(LLAMA), engine(STRATA)], question(FIVE_090))
    assert answer.level is EligibilityLevel.other_engine


def test_too_large_for_every_engine_that_would_run_it_is_not_here() -> None:
    no_row = {**STRATA, "fit": {**STRATA["fit"], "table": [{**STRATA["fit"]["table"][0]}]}}
    no_row["fit"]["table"][0]["fit"] = {
        "estimated": True,
        "verdict": "no",
        "reason": "Strata needs about 48 GB of RAM",
    }
    answer = eligibility.judge(FLASH, [engine(LLAMA), engine(no_row)], question(FIVE_090))
    assert answer.level is EligibilityLevel.not_here


def test_not_estimated_never_counts_against_a_model() -> None:
    no_model = {k: v for k, v in LLAMA.items() if k != "fit"}
    answer = eligibility.judge(FLASH, [engine(no_model)], question(FIVE_090))
    llama = answer.engines[0].fit
    assert llama is not None and not llama.estimated and llama.verdict is None
    assert "no fit estimate" in llama.reason
    assert answer.level is EligibilityLevel.works_here


def test_without_a_fit_question_nothing_changes() -> None:
    answer = eligibility.judge(FLASH, [engine(LLAMA), engine(STRATA)])
    assert all(v.fit is None for v in answer.engines)
    assert answer.level is EligibilityLevel.works_here


def test_a_prepared_model_is_found_in_the_table_by_what_it_was_made_from() -> None:
    prepared = LibraryModel.model_validate(
        {
            "id": "p",
            "path": "D:\\models\\Strata-data\\iq2_xs.eugene-prepared.json",
            "format": "prepared",
            "name": "p",
            "status": "present",
            "prepared": {
                "engine": "strata",
                "entry": "strata-iq2_xs.json",
                "source": {"file": f"IQ2_XS/{IQ2.upper()}"},
            },
        }
    )
    answer = eligibility.judge(prepared, [engine(STRATA)], question(FIVE_090))
    fit = answer.engines[0].fit
    assert fit is not None and fit.estimated and fit.verdict is FitVerdict.fits


def test_a_model_off_the_table_is_not_estimated() -> None:
    other = FLASH.model_copy(update={"path": "D:\\models\\renamed.gguf"})
    fit = engine_fit.engine_fit(
        engine_fit.sizing_of_model(other), engine(STRATA), question(FIVE_090)
    )
    assert not fit.estimated and "no row" in fit.reason


def test_a_candidate_is_sized_by_its_facts_and_marked_approximate() -> None:
    version = EligibilityCandidate(id="v", format="gguf", file="small-Q4_K_M.gguf", sizeBytes=GIB)
    answer = eligibility.judge_candidate(version, [engine(LLAMA)], question(FIVE_090))
    fit = answer.engines[0].fit
    assert fit is not None and fit.verdict is FitVerdict.fits and fit.approximate
    row = EligibilityCandidate(id="r", format="gguf", approximate=True)
    unsized = eligibility.judge_candidate(row, [engine(LLAMA)], question(FIVE_090))
    fit = unsized.engines[0].fit
    assert fit is not None and not fit.estimated and "size is not known" in fit.reason


def test_the_question_is_the_callers_node_spread_over_its_cards() -> None:
    budget = engine_fit.budget_of(
        FitQuestion(vramFreeBytes=40 * GIB, vramTotalBytes=48 * GIB, gpuCount=2, ramTotalBytes=GIB),
        None,
    )
    assert budget.source is Source.override
    assert (budget.largestGpuFreeBytes, budget.vramTotalBytes) == (20 * GIB, 48 * GIB)
    assert budget.ramAvailableBytes == GIB
    lone = engine_fit.budget_of(FitQuestion(vramFreeBytes=8 * GIB), None)
    assert (lone.gpuCount, lone.vramTotalBytes) == (1, 8 * GIB)


# --- the routes -------------------------------------------------------------


def _one_model(client: TestClient, models_dir: Path) -> str:
    write_gguf(models_dir / "m-Q4_K_M.gguf", qwen_like_kv(vocab=64))
    _scan(client)
    return client.get("/v1/models").json()["models"][0]["id"]


def test_the_fit_route_takes_the_engines_fit_model(
    configured_client: TestClient, models_dir: Path
) -> None:
    model_id = _one_model(configured_client, models_dir)
    budget = {"vramBytes": 20 * GIB, "vramTotalBytes": 24 * GIB, "ramBytes": 64 * GIB}
    spill = configured_client.get(f"/v1/models/{model_id}/fit", params=budget).json()
    assert spill["fit"]["model"] == "spill"
    shared = configured_client.get(
        f"/v1/models/{model_id}/fit",
        params={**budget, "fitModel": "reserved_share", "gpuMemoryUtilization": 0.9},
    ).json()
    assert shared["fit"]["model"] == "reserved_share"
    # Less than 90% of the card is free: vLLM would not start.
    assert shared["fit"]["verdict"] == "tight"
    assert shared["fit"]["budget"]["vramTotalBytes"] == 24 * GIB


def test_the_fit_route_refuses_what_it_cannot_answer(
    configured_client: TestClient, models_dir: Path
) -> None:
    model_id = _one_model(configured_client, models_dir)
    url = f"/v1/models/{model_id}/fit"
    no_share = configured_client.get(url, params={"fitModel": "reserved_share"})
    assert no_share.status_code == 422
    assert no_share.json()["detail"]["title"] == "Fit not estimated"
    table = configured_client.get(url, params={"fitModel": "engine_table"})
    assert table.status_code == 422 and "eligibility" in table.json()["detail"]["detail"]


def test_the_judge_route_answers_fit_when_asked(
    configured_client: TestClient, models_dir: Path
) -> None:
    model_id = _one_model(configured_client, models_dir)
    body = {
        "models": [model_id],
        "engines": [{**LLAMA, "available": True}, {**MLX, "available": True}],
        "fit": {"vramFreeBytes": 0, "vramTotalBytes": 0, "gpuCount": 0, "ramAvailableBytes": 1},
    }
    answer = configured_client.post("/v1/eligibility", json=body).json()["models"][0]
    llama = next(e for e in answer["engines"] if e["engine"] == "llama_cpp")
    # One byte of RAM and no card: measured, and `no`.
    assert llama["fit"]["estimated"] is True and llama["fit"]["verdict"] == "no"
    assert answer["level"] == "not_here"


def test_a_spill_engine_under_a_callers_budget_compares_free_memory() -> None:
    """As `GET /v1/models/{id}/fit` does with `vramBytes` alone: the cards'
    total is a share-taking engine's, and must not turn the page's `split`
    into the popover's `tight`."""
    # 20 GiB of weights against 12 of 24 GiB free: llama.cpp spills.
    budget = card(free=12 * GIB, total=24 * GIB)
    model = FLASH.model_copy(update={"sizeBytes": 20 * GIB})
    answer = eligibility.judge(model, [engine(LLAMA)], question(budget))
    fit = answer.engines[0].fit
    assert fit is not None and fit.verdict is FitVerdict.split


def test_the_starter_sets_facts_carry_each_entrys_size() -> None:
    from eugene_plexus_library import starter
    from eugene_plexus_library._generated.models import KvCacheType

    built = starter.build(
        starter.load(),
        budget=card(free=24 * GIB, total=24 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    assert built.models
    for model in built.models:
        assert model.facts is not None and model.facts.sizeBytes == model.sizeBytes


def test_a_fit_is_approximate_where_the_candidates_facts_were_a_guess() -> None:
    """Strata's row is exact for a version whose file was read off the hub;
    the same row for facts that were guessed says so."""
    read = EligibilityCandidate(id="v", format="gguf", file=IQ2, sizeBytes=68 * 10**9)
    guessed = read.model_copy(update={"id": "g", "approximate": True})
    exact = eligibility.judge_candidate(read, [engine(STRATA)], question(FIVE_090))
    rough = eligibility.judge_candidate(guessed, [engine(STRATA)], question(FIVE_090))
    assert exact.engines[0].fit is not None and exact.engines[0].fit.approximate is False
    assert rough.engines[0].fit is not None and rough.engines[0].fit.approximate is True
