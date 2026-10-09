"""The one judge of model against engine (library-sources-and-engines.md, LS1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_library import eligibility
from eugene_plexus_library._generated.models import (
    EligibilityCandidate,
    EligibilityEngine,
    EligibilityLevel,
    EngineVerdictKind,
    LibraryModel,
    ModelEligibility,
)

from .conftest import qwen_like_kv, write_gguf, write_hf_model

# The engines as their adapters declare them in the agent (LS1).
LLAMA = {"engine": "llama_cpp", "accepts": [{"format": "gguf", "preference": 10}]}
VLLM = {
    "engine": "vllm",
    "accepts": [
        {
            "format": "safetensors",
            "mlxQuantization": "forbidden",
            "authority": "engine",
            "preference": 20,
            "note": "vLLM checks the architecture when it loads",
        }
    ],
}
MLX = {
    "engine": "mlx",
    "accepts": [
        {"format": "safetensors", "mlxQuantization": "required", "preference": 10},
        {"format": "safetensors", "authority": "engine", "preference": 30},
    ],
}
STRATA = {
    "engine": "strata",
    "experimental": True,
    "accepts": [
        {
            "format": "gguf",
            "architectures": ["qwen4exp"],
            "preparation": {"recipe": "strata-prepare", "note": "an expert pack and MTP helper"},
            "preference": 50,
        }
    ],
}
KEV = {"engine": "kev", "accepts": [{"format": "kev_checkpoint"}]}


def engine(spec: dict[str, Any], *, available: bool = True, installable: bool = False):
    return EligibilityEngine.model_validate(
        {**spec, "available": available, "installable": installable}
    )


def model(fmt: str, **over: Any) -> LibraryModel:
    return LibraryModel.model_validate(
        {"id": "m", "path": "/models/m", "format": fmt, "name": "m", "status": "present", **over}
    )


GGUF = model("gguf", architecture="llama", gguf={"quantization": "Q4_K_M"})
FLASH_NEXT = model("gguf", architecture="qwen4exp", gguf={"quantization": "IQ2_XS"})
PLAIN_HF = model("safetensors", architecture="LlamaForCausalLM", safetensors={})
MLX_HF = model(
    "safetensors",
    architecture="Qwen3ForCausalLM",
    safetensors={"mlxQuantization": {"bits": 4, "groupSize": 64}},
)


def verdicts(m: LibraryModel, *engines: EligibilityEngine) -> dict[str, tuple[str, str]]:
    judged = eligibility.judge(m, list(engines))
    return {str(v.engine): (str(v.verdict), v.reason) for v in judged.engines}


def test_a_gguf_runs_on_llama_cpp_and_is_refused_by_vllm_with_the_format_named() -> None:
    got = verdicts(GGUF, engine(LLAMA), engine(VLLM))
    assert got["llama_cpp"] == ("runs", "runs it as it is")
    assert got["vllm"] == ("no", "loads safetensors models, and this one is gguf")


def test_the_mlx_rule_lives_here_once() -> None:
    # Was written by hand in the console and again in the agent.
    marked = verdicts(MLX_HF, engine(VLLM), engine(MLX))
    assert marked["mlx"][0] == "runs"
    assert marked["vllm"] == ("no", "cannot load MLX-quantized weights; only MLX reads them")
    plain = verdicts(PLAIN_HF, engine(VLLM), engine(MLX))
    assert plain["vllm"] == ("may_run", "vLLM checks the architecture when it loads")
    assert plain["mlx"][0] == "may_run"


def test_strata_runs_flash_next_after_preparation_and_names_the_architecture_otherwise() -> None:
    got = verdicts(FLASH_NEXT, engine(LLAMA), engine(STRATA))
    assert got["strata"] == (
        "after_preparation",
        "runs it after preparing it (an expert pack and MTP helper)",
    )
    assert got["llama_cpp"][0] == "runs"
    other = verdicts(GGUF, engine(STRATA))
    assert other["strata"] == (
        "no",
        "loads only the qwen4exp architecture, and this one is llama",
    )


def test_an_engine_that_loads_nothing_from_the_library_says_so() -> None:
    bare = engine({"engine": "strata", "accepts": []})
    assert verdicts(GGUF, bare)["strata"] == ("no", "loads no model from the library")


def test_the_three_levels() -> None:
    def level(m: LibraryModel, *engines: EligibilityEngine) -> EligibilityLevel:
        return eligibility.judge(m, list(engines)).level

    # An available engine runs it: works here now.
    assert level(GGUF, engine(LLAMA)) == EligibilityLevel.works_here
    # Only an engine this machine could install: a different engine.
    assert level(PLAIN_HF, engine(LLAMA), engine(VLLM, available=False, installable=True)) == (
        EligibilityLevel.other_engine
    )
    # Only after preparation by an installed engine: one more step, amber.
    assert level(FLASH_NEXT, engine(STRATA)) == EligibilityLevel.other_engine
    # An engine this hardware cannot have (MLX on Windows): not here.
    assert level(MLX_HF, engine(VLLM), engine(MLX, available=False)) == EligibilityLevel.not_here


def test_best_first_available_then_verdict_then_preference() -> None:
    judged = eligibility.judge(
        PLAIN_HF, [engine(LLAMA), engine(MLX, available=False, installable=True), engine(VLLM)]
    )
    assert [str(v.engine) for v in judged.engines] == ["vllm", "llama_cpp", "mlx"]
    assert judged.engines[0].verdict == EngineVerdictKind.may_run


# --- the route ---------------------------------------------------------------


def _scan(client: TestClient) -> None:
    client.post("/v1/scan")
    for _ in range(200):
        if client.get("/v1/scan").json()["state"] != "scanning":
            return
    raise AssertionError("scan never finished")


def test_the_route_judges_scanned_models_and_leaves_out_unknown_ids(
    configured_client: TestClient, models_dir: Path
) -> None:
    write_gguf(models_dir / "a-Q4_K_M.gguf", qwen_like_kv(name="A"))
    folder = write_hf_model(models_dir / "mlx-model")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config["quantization"] = {"bits": 4, "group_size": 64}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _scan(configured_client)
    ids = {m["name"]: m["id"] for m in configured_client.get("/v1/models").json()["models"]}

    engines = [
        {**LLAMA, "available": True},
        {**VLLM, "available": True},
        {**MLX, "available": False},
    ]
    answer = configured_client.post(
        "/v1/eligibility",
        json={"models": [ids["a-Q4_K_M"], ids["mlx-model"], "no-such-id"], "engines": engines},
    )
    assert answer.status_code == 200
    by_id = {m["modelId"]: m for m in answer.json()["models"]}
    assert set(by_id) == {ids["a-Q4_K_M"], ids["mlx-model"]}
    gguf = by_id[ids["a-Q4_K_M"]]
    assert gguf["level"] == "works_here"
    assert gguf["engines"][0]["engine"] == "llama_cpp"
    mlx = by_id[ids["mlx-model"]]
    assert mlx["level"] == "not_here"
    assert {e["engine"]: e["verdict"] for e in mlx["engines"]} == {
        "llama_cpp": "no",
        "vllm": "no",
        "mlx": "runs",
    }

    every = configured_client.post("/v1/eligibility", json={"engines": engines}).json()
    assert len(every["models"]) == 2


# --- candidates not downloaded yet (LS2) ------------------------------------

# llama.cpp as an installed build declares it: every architecture it knows.
BUILD = ["llama", "qwen2", "qwen3", "gemma3", "mistral3", "phi3"]
LLAMA_BUILD = {"engine": "llama_cpp", "accepts": [{"format": "gguf", "architectures": BUILD}]}


def candidate(fmt: str, **facts: Any) -> EligibilityCandidate:
    return EligibilityCandidate.model_validate({"id": "c", "format": fmt, **facts})


def judged(c: EligibilityCandidate, *engines: EligibilityEngine) -> ModelEligibility:
    return eligibility.judge_candidate(c, list(engines))


def by_engine(answer: ModelEligibility) -> dict[str, tuple[str, str]]:
    return {str(v.engine): (str(v.verdict), v.reason) for v in answer.engines}


def test_a_candidate_is_judged_by_the_same_rules_as_a_library_model() -> None:
    answer = judged(candidate("gguf", architecture="qwen3"), engine(LLAMA_BUILD), engine(VLLM))
    assert by_engine(answer)["llama_cpp"] == ("runs", "runs it as it is")
    assert by_engine(answer)["vllm"][0] == "no"
    assert answer.level == EligibilityLevel.works_here
    assert answer.approximate is False
    assert answer.modelId == "c"


def test_a_long_architecture_list_is_counted_not_read_out() -> None:
    answer = judged(candidate("gguf", architecture="qwen4exp"), engine(LLAMA_BUILD))
    assert by_engine(answer)["llama_cpp"] == (
        "no",
        "loads 6 named architectures, and qwen4exp is not one of them",
    )


def test_an_unknown_architecture_is_assumed_and_said_never_runs() -> None:
    answer = judged(candidate("gguf"), engine(LLAMA_BUILD), engine(STRATA))
    got = by_engine(answer)
    assert got["llama_cpp"] == (
        "may_run",
        "runs it if its architecture is one of the 6 it loads, which is not known yet",
    )
    assert got["strata"] == (
        "after_preparation",
        "runs it after preparing it (an expert pack and MTP helper) if its architecture is "
        "qwen4exp, which is not known yet",
    )
    assert answer.approximate is True
    assert answer.level == EligibilityLevel.works_here


def test_on_a_library_model_an_absent_architecture_is_unreadable_not_unknown() -> None:
    unread = model("gguf", gguf={"quantization": "Q4_K_M"})
    assert verdicts(unread, engine(LLAMA_BUILD))["llama_cpp"] == (
        "no",
        "loads 6 named architectures, and this library could not read this one's",
    )


def test_flash_next_from_the_hub_is_amber_where_llama_cpp_does_not_know_it() -> None:
    answer = judged(candidate("gguf", architecture="qwen4exp"), engine(LLAMA_BUILD), engine(STRATA))
    assert by_engine(answer)["strata"][0] == "after_preparation"
    assert by_engine(answer)["llama_cpp"][0] == "no"
    assert answer.level == EligibilityLevel.other_engine


def test_the_mlx_marker_read_from_a_remote_config_decides_before_download() -> None:
    marked = judged(candidate("safetensors", mlxQuantized=True), engine(VLLM), engine(MLX))
    assert by_engine(marked)["mlx"][0] == "runs"
    assert by_engine(marked)["vllm"][0] == "no"
    plain = judged(candidate("safetensors", mlxQuantized=False), engine(VLLM), engine(MLX))
    assert by_engine(plain)["vllm"] == ("may_run", "vLLM checks the architecture when it loads")
    assert plain.approximate is False


def test_an_unread_marker_leaves_vllm_may_run_and_says_what_it_rests_on() -> None:
    answer = judged(candidate("safetensors"), engine(VLLM), engine(MLX))
    got = by_engine(answer)
    assert got["vllm"] == (
        "may_run",
        "vLLM checks the architecture when it loads; and only if it is not MLX-quantized, "
        "which is not known yet",
    )
    # MLX's own rule for a plain folder needs no guess, so it is preferred
    # over the MLX-quantized rule that would.
    assert got["mlx"] == ("may_run", "may run it; only the engine can tell, when it loads")
    assert answer.approximate is True


def test_a_search_row_is_approximate_whatever_it_assumed() -> None:
    row = candidate("gguf", architecture="qwen3", approximate=True)
    assert judged(row, engine(LLAMA_BUILD)).approximate is True


def test_the_route_judges_candidates_after_models_and_alone_judges_only_them(
    configured_client: TestClient, models_dir: Path
) -> None:
    write_gguf(models_dir / "a-Q4_K_M.gguf", qwen_like_kv(name="A"))
    _scan(configured_client)
    engines = [{**LLAMA_BUILD, "available": True}, {**STRATA, "available": True}]
    rows = [
        {"id": "row:1", "format": "gguf", "architecture": "qwen4exp", "approximate": True},
        {"id": "row:2", "format": "safetensors", "mlxQuantized": True},
    ]
    only = configured_client.post(
        "/v1/eligibility", json={"candidates": rows, "engines": engines}
    ).json()["models"]
    assert [m["modelId"] for m in only] == ["row:1", "row:2"]
    assert only[0]["level"] == "other_engine" and only[0]["approximate"] is True
    assert only[1]["level"] == "not_here"
    ids = [m["id"] for m in configured_client.get("/v1/models").json()["models"]]
    both = configured_client.post(
        "/v1/eligibility", json={"models": ids, "candidates": rows[:1], "engines": engines}
    ).json()["models"]
    assert [m["modelId"] for m in both] == [*ids, "row:1"]
