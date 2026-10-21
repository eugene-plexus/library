import json
from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import LibraryModel
from eugene_plexus_library.scan_manager import ScanManager
from eugene_plexus_library.scanner import Scanner
from eugene_plexus_library.store import StateStore

from .conftest import write_gguf, write_hf_model


def scan(root: Path) -> LibraryModel:
    manager = ScanManager(StateStore(root / "state.json"), roots=lambda: [root])
    return manager._walk(Scanner(), [root], []).models[0]


def test_generation_config_supplies_only_author_sampling(tmp_path: Path) -> None:
    directory = write_hf_model(tmp_path / "model")
    (directory / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.05,
                "repetition_penalty": 1.2,
            }
        ),
        encoding="utf-8",
    )
    model = scan(tmp_path)
    assert model.safetensors.recommendedSampling.model_dump(exclude_none=True) == {
        "temperature": 0.6,
        "topP": 0.95,
        "topK": 20,
        "minP": 0.05,
        "source": "generation_config",
    }


def test_greedy_config_is_not_applied(tmp_path: Path) -> None:
    directory = write_hf_model(tmp_path / "model")
    config = directory / "generation_config.json"
    config.write_text('{"do_sample": true, "temperature": 0.6}', encoding="utf-8")
    manager = ScanManager(StateStore(tmp_path / "state.json"), roots=lambda: [tmp_path])
    first = manager._walk(Scanner(), [tmp_path], []).models[0]
    assert first.safetensors.recommendedSampling.temperature == 0.6
    for raw in ('{"do_sample": false, "temperature": 0.6}', '{"temperature": 0.6}'):
        config.write_text(raw, encoding="utf-8")
        # A cached model must not keep yesterday's author values.
        result = manager._walk(Scanner(cache_lookup=lambda *_: first), [tmp_path], [])
        assert result.models[0].safetensors.recommendedSampling is None


@pytest.mark.parametrize("raw", ["{", "[]", '{"do_sample": "true", "temperature": 1}'])
def test_bad_generation_config_does_not_break_scan(tmp_path: Path, raw: str) -> None:
    directory = write_hf_model(tmp_path / "model")
    (directory / "generation_config.json").write_text(raw, encoding="utf-8")
    assert scan(tmp_path).safetensors.recommendedSampling is None


def test_out_of_range_values_are_dropped_with_a_warning(tmp_path: Path, caplog) -> None:
    directory = write_hf_model(tmp_path / "model")
    (directory / "generation_config.json").write_text(
        '{"do_sample": true, "temperature": 5, "top_p": -1, "top_k": 1.5, "min_p": 0}',
        encoding="utf-8",
    )
    assert scan(tmp_path).safetensors.recommendedSampling.model_dump(exclude_none=True) == {
        "minP": 0,
        "source": "generation_config",
    }
    assert "The file's temperature 5 is outside 0-2, so it is not used." in caplog.text


@pytest.mark.parametrize("temperature", [True, "0.6", float("nan"), float("inf"), 10**400])
def test_invalid_sampling_never_breaks_a_scan(tmp_path: Path, temperature: object) -> None:
    directory = write_hf_model(tmp_path / "model")
    (directory / "generation_config.json").write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_k": False,
                "top_p": 0,
            }
        ),
        encoding="utf-8",
    )
    sampling = scan(tmp_path).safetensors.recommendedSampling
    assert sampling.model_dump(exclude_none=True) == {"topP": 0, "source": "generation_config"}


def test_gguf_sampling_is_validated_and_has_a_source(tmp_path: Path, caplog) -> None:
    write_gguf(
        tmp_path / "model.gguf",
        kv={
            "general.sampling.temp": 5.0,
            "general.sampling.top_k": 0,
            "general.sampling.top_p": 0.0,
            "general.sampling.min_p": 0.0,
        },
    )
    assert scan(tmp_path).gguf.recommendedSampling.model_dump(exclude_none=True) == {
        "topK": 0,
        "topP": 0,
        "minP": 0,
        "source": "gguf",
    }
    assert "outside 0-2" in caplog.text
