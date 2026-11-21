"""library#22: which engine flags an operations token may write into a profile.

`operations.refuse_owner_fields` decides from three explicit lists per
engine (owner-only flags, path flags, and the flags an operations token may
set), never from the spelling of a key. The lists are the agent's (agent#53),
duplicated: components share schemas, never code. The Library holds no
engine adapters, so `fixtures/engine_flags.json` records, for every flag the
agent's adapters declare, the class the agent gives it. The first test
compares class with class: a flag the Library has on no list, or in another
class than the agent's, fails.

To regenerate the fixture after the agent adds or reclassifies a flag, run
`python tests/fixtures/engine_flags.py <agent checkout>/src` with the
agent's venv, and name the agent commit in the fixture's `source`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_library import operations
from eugene_plexus_library._generated.models import EngineKind

FIXTURE = Path(__file__).parent / "fixtures" / "engine_flags.json"


def _agent_classes() -> dict[str, dict[str, str | None]]:
    engines: dict[str, dict[str, str | None]] = json.loads(FIXTURE.read_text(encoding="utf-8"))[
        "engines"
    ]
    return engines


def _pairs() -> set[tuple[str, str]]:
    return {(engine, key) for engine, flags in _agent_classes().items() for key in flags}


def test_the_fixture_covers_every_engine_kind() -> None:
    assert set(_agent_classes()) == {kind.value for kind in EngineKind}


def test_every_engine_flag_has_the_agents_class() -> None:
    differ = sorted(
        f"{engine}.{key}: agent {agent!r}, library {operations.flag_class(engine, key)!r}"
        for engine, flags in _agent_classes().items()
        for key, agent in flags.items()
        if agent is None or operations.flag_class(engine, key) != agent
    )
    assert differ == [], (
        "classify these engine flags in operations.py (OWNER_FLAGS, PATH_FLAGS or "
        f"OPERATIONS_FLAGS) as the agent does: {differ}"
    )


def test_the_lists_name_only_declared_flags_and_no_flag_twice() -> None:
    declared = _pairs()
    lists = {
        "OWNER_FLAGS": set(operations.OWNER_FLAGS),
        "PATH_FLAGS": set(operations.PATH_FLAGS),
        "OPERATIONS_FLAGS": set(operations.OPERATIONS_FLAGS),
    }
    for name, pairs in lists.items():
        assert pairs <= declared, f"{name} names flags no engine declares: {pairs - declared}"
    names = list(lists)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            both = lists[first] & lists[second]
            assert not both, f"{both} is in both {first} and {second}"


@pytest.mark.parametrize(
    ("engine", "key", "kind"),
    [
        ("mlx", "trustRemoteCode", "owner"),
        ("vllm", "trustRemoteCode", "owner"),
        ("mlx", "draftModel", "path"),
        ("mlx", "adapterPath", "path"),
        ("vllm", "tokenizer", "path"),
        ("vllm", "draftModelPath", "path"),
        ("llama_cpp", "chatTemplateFile", "path"),
        ("llama_cpp", "projectorPath", "path"),
        ("llama_cpp", "draftModelPath", "path"),
        ("kev", "fallback", "path"),
        ("mlx", "chatTemplate", "operations"),
        ("llama_cpp", "contextSize", "operations"),
    ],
)
def test_the_flags_that_reach_past_running_a_model(engine: str, key: str, kind: str) -> None:
    assert operations.flag_class(engine, key) == kind
    assert operations.flag_class(EngineKind(engine), key) == kind


def test_a_key_is_classified_per_engine() -> None:
    """MLX's `chatTemplate` is the template's text; a key of the same name on
    another engine is not classified until someone reviews it there."""
    assert operations.flag_class("mlx", "chatTemplate") == "operations"
    assert operations.flag_class("vllm", "chatTemplate") is None
    assert operations.flag_class("llama_cpp", "trustRemoteCode") is None
    assert operations.flag_class("llama_cpp", "notAFlag") is None
