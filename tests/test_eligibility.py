"""The one judge of model against engine (library-sources-and-engines.md, LS1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_library import eligibility
from eugene_plexus_library._generated.models import (
    EligibilityEngine,
    EligibilityLevel,
    EngineVerdictKind,
    LibraryModel,
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
