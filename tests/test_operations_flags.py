"""library#22: which engine flags an operations token may write into a profile.

`operations.refuse_owner_fields` decides from three explicit lists per
engine (owner-only flags, path flags, and the flags an operations token may
set), never from the spelling of a key. The Library holds no engine
adapters, so `fixtures/engine_flags.json` records every flag the agent's
adapters declare, per engine. The first test walks it: a flag on no list
fails, so a new flag is classified here before an operations token can
write it. The lists are the agent's (agent#53), duplicated: components
share schemas, never code.

To refresh the fixture after the agent adds a flag, run this from an agent
checkout with its own venv, put the output under `engines`, and name the
agent commit in `source`:

    python -c "import json; from eugene_plexus_agent.engines import ADAPTERS; \
print(json.dumps({k.value: {f.key: f.valueType.value for f in a.flag_schema().fields} \
for k, a in ADAPTERS.items()}, indent=2))"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_library import operations
from eugene_plexus_library._generated.models import EngineKind

FIXTURE = Path(__file__).parent / "fixtures" / "engine_flags.json"
#: Declared value types that name files or folders on a machine.
PATH_VALUE_TYPES = frozenset({"file_path", "path_list"})


def _declared() -> dict[str, dict[str, str]]:
    engines: dict[str, dict[str, str]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["engines"]
    return engines


def _pairs() -> set[tuple[str, str]]:
    return {(engine, key) for engine, flags in _declared().items() for key in flags}


def test_the_fixture_covers_every_engine_kind() -> None:
    assert set(_declared()) == {kind.value for kind in EngineKind}


def test_every_engine_flag_is_classified_for_operations_tokens() -> None:
    unclassified = sorted(
        f"{engine}.{key}" for engine, key in _pairs() if operations.flag_class(engine, key) is None
    )
    assert unclassified == [], (
        "classify these engine flags in operations.py (OWNER_FLAGS, PATH_FLAGS or "
        f"OPERATIONS_FLAGS): {unclassified}"
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


def test_a_flag_declared_as_a_path_is_a_path_flag() -> None:
    for engine, flags in _declared().items():
        for key, value_type in flags.items():
            if value_type in PATH_VALUE_TYPES:
                assert operations.flag_class(engine, key) == "path", f"{engine}.{key}"


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
