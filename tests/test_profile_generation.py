from pathlib import Path

import pytest
from pydantic import ValidationError

from eugene_plexus_library._generated.models import ModelProfileSpec
from eugene_plexus_library.store import StateStore


def test_generation_defaults_survive_save_replace_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)
    profile = store.create_profile(
        "model",
        ModelProfileSpec(
            name="first",
            engine="llama_cpp",
            maxTokens=111,
            temperature=0,
            topP=0,
        ),
    )
    reloaded = StateStore(path)
    reloaded.load()
    saved = reloaded.get_profile("model", profile.id)
    assert saved is not None
    assert (saved.maxTokens, saved.temperature, saved.topP) == (111, 0, 0)
    reloaded.replace_profile(
        "model",
        profile.id,
        ModelProfileSpec(
            name="first",
            engine="llama_cpp",
            maxTokens=222,
            temperature=0.4,
            topP=0.9,
        ),
    )
    again = StateStore(path)
    again.load()
    saved = again.get_profile("model", profile.id)
    assert saved is not None
    assert (saved.maxTokens, saved.temperature, saved.topP) == (222, 0.4, 0.9)
    cleared = again.replace_profile(
        "model", profile.id, ModelProfileSpec(name="first", engine="llama_cpp")
    )
    assert cleared is not None
    assert (cleared.maxTokens, cleared.temperature, cleared.topP) == (None, None, None)


@pytest.mark.parametrize(
    "values",
    [
        {"maxTokens": 0},
        {"maxTokens": 1.5},
        {"temperature": -1},
        {"temperature": 2.1},
        {"topP": -0.1},
        {"topP": 1.1},
    ],
)
def test_invalid_generation_defaults_rejected(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        ModelProfileSpec.model_validate({"name": "x", "engine": "llama_cpp", **values})


def test_a_built_by_record_survives_a_restart_and_an_edit(tmp_path: Path) -> None:
    """The record is stored with the profile, reloaded with it, and kept by
    a replace that does not mention it (PB2)."""
    path = tmp_path / "state.json"
    store = StateStore(path)
    built = {
        "buildId": "b-2",
        "node": "laptop",
        "accuracy": "max",
        "builtAt": "2026-09-30T12:00:00Z",
        "flags": {"contextSize": 40960, "cacheType": "f16"},
    }
    profile = store.create_profile(
        "model",
        ModelProfileSpec.model_validate(
            {"name": "built", "engine": "llama_cpp", "flags": built["flags"], "builtBy": built}
        ),
    )
    again = StateStore(path)
    again.load()
    edited = again.replace_profile(
        "model",
        profile.id,
        ModelProfileSpec.model_validate(
            {"name": "built", "engine": "llama_cpp", "flags": {"contextSize": 8192}}
        ),
    )
    assert edited is not None and edited.builtBy is not None
    assert edited.builtBy.buildId == "b-2"
    assert edited.builtBy.flags == {"contextSize": 40960, "cacheType": "f16"}
    third = StateStore(path)
    third.load()
    kept = third.get_profile("model", profile.id)
    assert kept is not None and kept.builtBy is not None and kept.builtBy.node == "laptop"
