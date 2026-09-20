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
