"""The state store: the profile store, and the cache that makes rescans cheap."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import (
    EngineKind,
    LibraryModel,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelProfileSpec,
    ModelStatus,
)
from eugene_plexus_library.paths import model_id
from eugene_plexus_library.scanner import Scanner, cache_key
from eugene_plexus_library.store import StateStore

from .conftest import qwen_like_kv, write_gguf


def _model(path: Path, *, size: int = 100) -> LibraryModel:
    return LibraryModel(
        id=model_id(path),
        path=str(path),
        format=ModelFormat.gguf,
        name=path.stem,
        status=ModelStatus.present,
        sizeBytes=size,
        files=[ModelFile(path=str(path), role=ModelFileRole.weights, sizeBytes=size)],
        modifiedAt=datetime.now(tz=UTC),
    )


def _spec(name: str = "default", **kwargs: object) -> ModelProfileSpec:
    return ModelProfileSpec(name=name, engine=EngineKind.llama_cpp, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.json")


# --- the incremental-rescan cache ----------------------------------------


def test_a_real_stat_matches_a_stored_entry(models_dir: Path, store: StateStore) -> None:
    """The end-to-end version of the cache contract: scan, store, then
    build the key from a fresh stat of the same untouched file."""
    path = write_gguf(models_dir / "model.gguf", qwen_like_kv())
    found = Scanner().scan([models_dir]).models
    store.replace_models(found, scanned_at=datetime.now(tz=UTC))

    assert store.lookup(cache_key(path, path.stat())) is not None


def test_the_cache_misses_after_the_file_changes(models_dir: Path, store: StateStore) -> None:
    path = write_gguf(models_dir / "model.gguf", qwen_like_kv())
    store.replace_models(Scanner().scan([models_dir]).models, scanned_at=datetime.now(tz=UTC))
    write_gguf(models_dir / "model.gguf", qwen_like_kv(), payload=b"different")

    assert store.lookup(cache_key(path, path.stat())) is None


def test_the_cache_compares_the_weights_file_not_the_total(
    models_dir: Path, store: StateStore
) -> None:
    """`sizeBytes` is the sum across shards and the projector, while the
    key carries one file's stat. Comparing those two would make every
    sharded or multimodal model miss the cache on every scan."""
    first = write_gguf(
        models_dir / "m-00001-of-00002.gguf", qwen_like_kv(name="M"), payload=b"x" * 500
    )
    write_gguf(models_dir / "m-00002-of-00002.gguf", qwen_like_kv(name="M"), payload=b"x" * 900)
    found = Scanner().scan([models_dir]).models
    store.replace_models(found, scanned_at=datetime.now(tz=UTC))

    assert found[0].sizeBytes != first.stat().st_size, "fixture no longer exercises the bug"
    assert store.lookup(cache_key(first, first.stat())) is not None


def test_a_missing_entry_is_never_a_cache_hit(tmp_path: Path, store: StateStore) -> None:
    path = tmp_path / "gone.gguf"
    path.write_bytes(b"x" * 10)
    store.replace_models([_model(path, size=10)], scanned_at=datetime.now(tz=UTC))
    store.create_profile(model_id(path), _spec())
    store.replace_models([], scanned_at=datetime.now(tz=UTC))

    assert store.get_model(model_id(path)).status == ModelStatus.missing
    assert store.lookup(cache_key(path, path.stat())) is None


# --- what survives a scan --------------------------------------------------


def test_an_entry_with_profiles_survives_as_missing(tmp_path: Path, store: StateStore) -> None:
    """Dropping it would drop the profiles, which are the one thing here
    that cannot be recovered by looking at the disk again."""
    path = tmp_path / "model.gguf"
    identifier = model_id(path)
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))
    store.create_profile(identifier, _spec("tuned"))

    counts = store.replace_models([], scanned_at=datetime.now(tz=UTC))

    assert counts["missing"] == 1
    survivor = store.get_model(identifier)
    assert survivor is not None
    assert survivor.status == ModelStatus.missing
    assert [p.name for p in store.list_profiles(identifier)] == ["tuned"]


def test_an_entry_without_profiles_is_dropped(tmp_path: Path, store: StateStore) -> None:
    """Keeping a record of a file somebody deleted, with nothing attached
    to it, is clutter."""
    path = tmp_path / "model.gguf"
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))
    counts = store.replace_models([], scanned_at=datetime.now(tz=UTC))

    assert counts["missing"] == 0
    assert store.get_model(model_id(path)) is None


def test_first_seen_survives_a_rescan(tmp_path: Path, store: StateStore) -> None:
    path = tmp_path / "model.gguf"
    first = datetime(2026, 1, 1, tzinfo=UTC)
    store.replace_models([_model(path)], scanned_at=first)
    later = datetime(2026, 6, 1, tzinfo=UTC)
    store.replace_models([_model(path)], scanned_at=later)

    stored = store.get_model(model_id(path))
    assert stored.firstSeenAt == first
    assert stored.lastSeenAt == later


def test_counts_added_and_updated(tmp_path: Path, store: StateStore) -> None:
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    assert store.replace_models([_model(a)], scanned_at=datetime.now(tz=UTC))["added"] == 1

    changed = _model(a).model_copy(update={"modifiedAt": datetime(2030, 1, 1, tzinfo=UTC)})
    counts = store.replace_models([changed, _model(b)], scanned_at=datetime.now(tz=UTC))
    assert counts == {"added": 1, "updated": 1, "missing": 0}


# --- profiles -----------------------------------------------------------------


def test_the_first_profile_is_the_default(tmp_path: Path, store: StateStore) -> None:
    """A model with profiles and no default would make every launch flow
    handle an empty case that need not exist."""
    identifier = model_id(tmp_path / "m.gguf")
    profile = store.create_profile(identifier, _spec("fast"))

    assert profile.default is True


def test_setting_a_default_clears_the_previous_one(tmp_path: Path, store: StateStore) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    first = store.create_profile(identifier, _spec("fast"))
    second = store.create_profile(identifier, _spec("long-context", default=True))

    profiles = {p.id: p.default for p in store.list_profiles(identifier)}
    assert profiles[second.id] is True
    assert profiles[first.id] is False


def test_the_default_is_listed_first(tmp_path: Path, store: StateStore) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    store.create_profile(identifier, _spec("fast"))
    store.create_profile(identifier, _spec("long", default=True))

    assert store.list_profiles(identifier)[0].name == "long"


def test_duplicate_names_are_refused(tmp_path: Path, store: StateStore) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    store.create_profile(identifier, _spec("fast"))

    with pytest.raises(ValueError, match="already exists"):
        store.create_profile(identifier, _spec("fast"))


def test_replace_is_a_whole_document_swap(tmp_path: Path, store: StateStore) -> None:
    """Merge semantics give no way to remove a flag."""
    identifier = model_id(tmp_path / "m.gguf")
    created = store.create_profile(
        identifier, _spec("fast", flags={"nGpuLayers": 99, "contextSize": 8192})
    )
    updated = store.replace_profile(
        identifier, created.id, _spec("fast", flags={"contextSize": 4096})
    )

    assert updated is not None
    assert updated.flags == {"contextSize": 4096}
    assert updated.id == created.id
    assert updated.createdAt == created.createdAt


def test_replacing_a_missing_profile_returns_none(tmp_path: Path, store: StateStore) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    store.create_profile(identifier, _spec("fast"))
    assert store.replace_profile(identifier, "nope", _spec("other")) is None


def test_deleting_the_default_promotes_the_oldest_survivor(
    tmp_path: Path, store: StateStore
) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    first = store.create_profile(identifier, _spec("first"))
    store.create_profile(identifier, _spec("second"))
    store.create_profile(identifier, _spec("third"))
    store.replace_profile(identifier, first.id, _spec("first", default=True))

    assert store.delete_profile(identifier, first.id) is True
    remaining = store.list_profiles(identifier)
    assert sum(1 for p in remaining if p.default) == 1
    assert remaining[0].name == "second"


def test_profiles_carry_env_for_the_two_gpu_case(tmp_path: Path, store: StateStore) -> None:
    """The strongest argument for profiles being plural: two replicas of
    one model is the same profile twice with a different device."""
    identifier = model_id(tmp_path / "m.gguf")
    store.create_profile(identifier, _spec("gpu0", env={"CUDA_VISIBLE_DEVICES": "0"}))
    second = store.create_profile(identifier, _spec("gpu1", env={"CUDA_VISIBLE_DEVICES": "1"}))

    assert second.env == {"CUDA_VISIBLE_DEVICES": "1"}
    assert len(store.list_profiles(identifier)) == 2


def test_profile_count_appears_on_the_model(tmp_path: Path, store: StateStore) -> None:
    path = tmp_path / "m.gguf"
    identifier = model_id(path)
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))
    store.create_profile(identifier, _spec("a"))
    store.create_profile(identifier, _spec("b"))

    assert store.get_model(identifier).profileCount == 2


# --- persistence ----------------------------------------------------------------


def test_state_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    model_path = tmp_path / "m.gguf"
    identifier = model_id(model_path)

    first = StateStore(path)
    first.replace_models([_model(model_path)], scanned_at=datetime.now(tz=UTC))
    first.create_profile(identifier, _spec("tuned", notes="OOMs above 24 layers"))

    second = StateStore(path)
    second.load()

    assert second.get_model(identifier) is not None
    assert second.list_profiles(identifier)[0].notes == "OOMs above 24 layers"


def test_a_corrupt_state_file_does_not_stop_startup(tmp_path: Path) -> None:
    """Refusing to boot over a damaged *cache* would break the
    degraded-mode rule for the least important data in the system."""
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")

    store = StateStore(path)
    store.load()

    assert store.list_models() == []


def test_one_bad_record_does_not_lose_the_others(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    good = _model(tmp_path / "good.gguf").model_dump(mode="json", exclude_none=True)
    path.write_text(
        json.dumps({"version": 1, "models": [good, {"nonsense": True}], "profiles": {}}),
        encoding="utf-8",
    )

    store = StateStore(path)
    store.load()

    assert len(store.list_models()) == 1


def test_writes_are_atomic(tmp_path: Path) -> None:
    """A crash mid-write must leave the previous state, not a truncated
    file — losing a scan cache is free, losing profiles is not."""
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.create_profile("abc", _spec("one"))

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


# --- forgetting --------------------------------------------------------------------


def test_forget_drops_the_entry_and_its_profiles(tmp_path: Path, store: StateStore) -> None:
    path = tmp_path / "m.gguf"
    identifier = model_id(path)
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))
    store.create_profile(identifier, _spec("tuned"))

    assert store.forget_model(identifier) is True
    assert store.get_model(identifier) is None
    assert store.list_profiles(identifier) == []


def test_forget_never_touches_the_file(tmp_path: Path, store: StateStore) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"the operator's model")
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))

    store.forget_model(model_id(path))

    assert path.read_bytes() == b"the operator's model"


def test_find_by_path_normalizes(tmp_path: Path, store: StateStore) -> None:
    """The whole point of the reverse lookup: a caller holding a path
    should not have to reproduce the id derivation."""
    path = tmp_path / "sub" / "m.gguf"
    path.parent.mkdir()
    store.replace_models([_model(path)], scanned_at=datetime.now(tz=UTC))

    assert store.find_by_path(str(tmp_path / "sub" / ".." / "sub" / "m.gguf")) is not None
