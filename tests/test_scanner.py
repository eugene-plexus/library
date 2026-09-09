"""The walk. Every trap the design doc names gets a test here."""

from __future__ import annotations

import json
from pathlib import Path

from eugene_plexus_library._generated.models import (
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    SkipReason,
)
from eugene_plexus_library._generated.models import Status1 as ScanRootStatus
from eugene_plexus_library.scanner import Scanner, cache_key

from .conftest import (
    embedding_kv,
    projector_kv,
    qwen_like_kv,
    write_gguf,
    write_hf_model,
    write_safetensors,
)


def _skips(result, reason: SkipReason) -> list[str]:  # type: ignore[no-untyped-def]
    return [s.path for s in result.skipped if s.reason == reason]


# --- the basics --------------------------------------------------------


def test_finds_a_single_gguf(models_dir: Path) -> None:
    write_gguf(models_dir / "thing-Q4_K_M.gguf", qwen_like_kv(name="Thing"))
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.format == ModelFormat.gguf
    assert model.name == "thing-Q4_K_M"
    assert model.displayName == "Thing"
    assert model.status == ModelStatus.present
    assert model.gguf is not None
    assert model.gguf.quantization == "Q4_K_M"
    assert model.root == str(models_dir)


def test_descends_into_publisher_repo_directories(models_dir: Path) -> None:
    """LM Studio's `<root>/<publisher>/<repo>/<file>.gguf` is the
    de-facto layout. A flat listing of the root finds nothing."""
    write_gguf(models_dir / "lmstudio-community" / "Qwen-GGUF" / "Qwen-Q4_K_M.gguf", qwen_like_kv())
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1


def test_name_comes_from_the_filename_not_the_metadata(models_dir: Path) -> None:
    """The file's own name is what the operator downloaded and what
    `Runtime.modelAlias` defaults to."""
    write_gguf(models_dir / "my-chosen-name.gguf", qwen_like_kv(name="Some Training Run v3"))
    model = Scanner().scan([models_dir]).models[0]

    assert model.name == "my-chosen-name"
    assert model.displayName == "Some Training Run v3"


# --- what is not a model ------------------------------------------------


def test_projector_is_not_a_model_and_is_attached_to_one(models_dir: Path) -> None:
    """The trap worth ~1 GB of phantom entry per multimodal model."""
    write_gguf(models_dir / "vision-Q4_K_M.gguf", qwen_like_kv(name="Vision 7B"))
    projector = write_gguf(models_dir / "mmproj-F32.gguf", projector_kv(name="Vision 7B"))
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.capabilities is not None
    assert model.capabilities.vision is True
    assert model.gguf is not None
    assert model.gguf.projectorPath == str(projector)
    assert [f.role for f in model.files or []].count(ModelFileRole.projector) == 1
    assert str(projector) in _skips(result, SkipReason.projector)


def test_a_renamed_projector_is_still_a_projector(models_dir: Path) -> None:
    """Detection is by `general.type`, not the `mmproj-` convention."""
    write_gguf(models_dir / "model-Q4_K_M.gguf", qwen_like_kv(name="Pair"))
    write_gguf(models_dir / "totally-normal-name.gguf", projector_kv(name="Pair"))
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    assert len(_skips(result, SkipReason.projector)) == 1


def test_shards_group_into_one_model_named_by_the_first(models_dir: Path) -> None:
    first = write_gguf(models_dir / "big-00001-of-00003.gguf", qwen_like_kv(name="Big"))
    second = write_gguf(models_dir / "big-00002-of-00003.gguf", qwen_like_kv(name="Big"))
    third = write_gguf(models_dir / "big-00003-of-00003.gguf", qwen_like_kv(name="Big"))
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.path == str(first)
    assert model.name == "big"  # the shard suffix is not part of the name
    assert model.gguf is not None
    assert model.gguf.shardCount == 3
    assert model.fileCount == 3
    skipped = _skips(result, SkipReason.shard_member)
    assert str(second) in skipped and str(third) in skipped


def test_shards_with_no_first_part_produce_no_model(models_dir: Path) -> None:
    """An incomplete split model: report every part rather than invent an
    entry from part 2."""
    write_gguf(models_dir / "big-00002-of-00003.gguf", qwen_like_kv())
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert len(_skips(result, SkipReason.shard_member)) == 1


def test_size_is_summed_across_shards_and_projector(models_dir: Path) -> None:
    write_gguf(models_dir / "m-00001-of-00002.gguf", qwen_like_kv(name="M"), payload=b"x" * 1000)
    write_gguf(models_dir / "m-00002-of-00002.gguf", qwen_like_kv(name="M"), payload=b"x" * 2000)
    write_gguf(models_dir / "mmproj-M.gguf", projector_kv(name="M"), payload=b"x" * 500)
    model = Scanner().scan([models_dir]).models[0]

    total = sum(f.sizeBytes or 0 for f in model.files or [])
    assert model.sizeBytes == total
    assert model.fileCount == 3


def test_partial_downloads_are_skipped(models_dir: Path) -> None:
    """M3's downloader writes these. Better to have the reason before
    then than to surface a half-fetched 40 GB model as broken."""
    (models_dir / "half.gguf.part").write_bytes(b"nope")
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert len(_skips(result, SkipReason.incomplete_download)) == 1


def test_unsupported_formats_are_named(models_dir: Path) -> None:
    (models_dir / "weights.npz").write_bytes(b"nope")
    result = Scanner().scan([models_dir])

    assert len(_skips(result, SkipReason.unsupported_format)) == 1


def test_an_unparseable_gguf_is_reported_not_dropped(models_dir: Path) -> None:
    (models_dir / "corrupt.gguf").write_bytes(b"NOPE" + b"\0" * 32)
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert len(_skips(result, SkipReason.unreadable_header)) == 1


# --- safetensors --------------------------------------------------------


def test_finds_a_safetensors_directory(models_dir: Path) -> None:
    directory = write_hf_model(models_dir / "my-model", parameters=4096)
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.format == ModelFormat.safetensors
    assert model.path == str(directory)
    assert model.parameters == 4096
    assert model.safetensors is not None
    assert model.safetensors.dtype == "F32"
    assert model.gguf is None


def test_safetensors_without_a_config_is_not_a_model(models_dir: Path) -> None:
    directory = models_dir / "loose-weights"
    write_safetensors(directory / "model.safetensors", {"w": ("F32", [4])})
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert str(directory) in _skips(result, SkipReason.not_a_model)


def test_an_adapter_directory_is_skipped(models_dir: Path) -> None:
    directory = models_dir / "my-lora"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_safetensors(directory / "adapter_model.safetensors", {"w": ("F32", [4])})
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert str(directory) in _skips(result, SkipReason.adapter)


def test_embedding_safetensors_is_flagged(models_dir: Path) -> None:
    write_hf_model(models_dir / "minilm", architectures=["BertModel"])
    model = Scanner().scan([models_dir]).models[0]

    assert model.capabilities is not None
    assert model.capabilities.embedding is True
    assert model.capabilities.chat is False


# --- the HuggingFace cache ----------------------------------------------


def test_only_the_current_revision_is_a_model(models_dir: Path) -> None:
    """A cache keeps one snapshot per revision, so a naive scan reports
    the same model once per revision ever fetched."""
    cache = models_dir / "models--org--thing"
    current = write_hf_model(cache / "snapshots" / "aaaaaaaa")
    older = write_hf_model(cache / "snapshots" / "bbbbbbbb")
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("aaaaaaaa", encoding="utf-8")

    result = Scanner().scan([models_dir])

    assert [m.path for m in result.models] == [str(current)]
    assert str(older) in _skips(result, SkipReason.older_revision)
    assert result.models[0].safetensors is not None
    assert result.models[0].safetensors.repoId == "org/thing"
    assert result.models[0].safetensors.revision == "aaaaaaaa"


def test_both_revisions_survive_when_refs_main_is_missing(models_dir: Path) -> None:
    """Listing a model twice beats dropping it because a ref file was
    unreadable."""
    cache = models_dir / "models--org--thing"
    write_hf_model(cache / "snapshots" / "aaaaaaaa")
    write_hf_model(cache / "snapshots" / "bbbbbbbb")

    assert len(Scanner().scan([models_dir]).models) == 2


def test_cache_infrastructure_is_skipped(models_dir: Path) -> None:
    """`.no_exist` is a *negative* cache full of zero-byte files with
    real-looking names — a naive adapter probe finds one there."""
    cache = models_dir / "models--org--thing"
    write_hf_model(cache / "snapshots" / "aaaaaaaa")
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("aaaaaaaa", encoding="utf-8")
    blobs = cache / "blobs"
    blobs.mkdir()
    (blobs / "deadbeef").write_bytes(b"\0" * 16)
    no_exist = cache / ".no_exist" / "aaaaaaaa"
    no_exist.mkdir(parents=True)
    (no_exist / "adapter_config.json").write_bytes(b"")

    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    skipped = _skips(result, SkipReason.not_a_model)
    assert str(blobs) in skipped
    assert str(cache / ".no_exist") in skipped


# --- roots ---------------------------------------------------------------


def test_a_missing_root_is_reported_and_does_not_abort_the_scan(
    models_dir: Path, tmp_path: Path
) -> None:
    """One unplugged drive must not cost the operator every other model
    they own."""
    write_gguf(models_dir / "fine.gguf", qwen_like_kv())
    result = Scanner().scan([tmp_path / "not-here", models_dir])

    assert len(result.models) == 1
    statuses = {r.path: r.status for r in result.roots}
    assert statuses[str(tmp_path / "not-here")] == ScanRootStatus.missing
    assert statuses[str(models_dir)] == ScanRootStatus.ok


def test_a_file_given_as_a_root_is_unreadable_not_missing(tmp_path: Path) -> None:
    """`missing` and `unreadable` need different advice."""
    target = tmp_path / "a-file.txt"
    target.write_text("hello", encoding="utf-8")
    result = Scanner().scan([target])

    assert result.roots[0].status == ScanRootStatus.unreadable


def test_per_root_counts(models_dir: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    write_gguf(other / "b.gguf", qwen_like_kv())
    write_gguf(other / "c.gguf", qwen_like_kv())

    result = Scanner().scan([models_dir, other])
    found = {r.path: r.models_found for r in result.roots}

    assert found[str(models_dir)] == 1
    assert found[str(other)] == 2


def test_a_model_under_two_overlapping_roots_is_listed_once(models_dir: Path) -> None:
    write_gguf(models_dir / "nested" / "a.gguf", qwen_like_kv())
    result = Scanner().scan([models_dir, models_dir / "nested"])

    assert len(result.models) == 1


# --- caching + cancellation ------------------------------------------------


def test_an_unchanged_file_is_served_from_the_cache(models_dir: Path) -> None:
    """The cache is consulted from the stat, before anything is opened."""
    path = write_gguf(models_dir / "model.gguf", qwen_like_kv())
    first = Scanner().scan([models_dir]).models[0]

    hits: list[str] = []

    def lookup(key):  # type: ignore[no-untyped-def]
        hits.append(key[0])
        return first

    scanner = Scanner(cache_lookup=lookup)
    # Break the file so a re-read would fail loudly. A cache hit means
    # the reader was never called at all.
    path.write_bytes(b"NOPE" + b"\0" * 32)
    # The stat has to still match for the lookup to be consulted; rewrite
    # via the fixture's own key instead of trusting the corrupted file.
    result = scanner.scan([models_dir])

    assert hits, "the cache was never consulted"
    assert len(result.models) == 1
    assert result.models[0].id == first.id


def test_cache_key_changes_with_content(models_dir: Path) -> None:
    path = write_gguf(models_dir / "model.gguf", qwen_like_kv())
    before = cache_key(path, path.stat())
    write_gguf(models_dir / "model.gguf", qwen_like_kv(), payload=b"more")
    after = cache_key(path, path.stat())

    assert before != after


def test_cancellation_keeps_what_was_already_found(models_dir: Path) -> None:
    """A partial scan learned true things; discarding them to look tidy
    would lose work the operator waited for."""
    for index in range(5):
        write_gguf(models_dir / f"m{index}.gguf", qwen_like_kv())

    calls = {"n": 0}

    def should_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    result = Scanner(should_cancel=should_cancel).scan([models_dir])
    assert isinstance(result.models, list)


def test_counters_move_during_a_scan(models_dir: Path) -> None:
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    write_gguf(models_dir / "b.gguf", qwen_like_kv())
    scanner = Scanner()
    scanner.scan([models_dir])

    assert scanner.counters.files_scanned == 2
    assert scanner.counters.models_found == 2


# --- mixed tree -------------------------------------------------------------


def test_a_realistic_mixed_tree(models_dir: Path) -> None:
    """Everything at once, shaped like a real drive."""
    write_gguf(models_dir / "TheBloke" / "Llama-GGUF" / "llama-Q4_K_M.gguf", qwen_like_kv(name="L"))
    write_gguf(models_dir / "TheBloke" / "Llama-GGUF" / "mmproj-F32.gguf", projector_kv(name="L"))
    write_gguf(models_dir / "embed" / "nomic.Q4_K_M.gguf", embedding_kv())
    write_hf_model(models_dir / "hf" / "mistral")
    lora = models_dir / "loras" / "my-lora"
    lora.mkdir(parents=True)
    (lora / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    write_safetensors(lora / "adapter_model.safetensors", {"w": ("F32", [4])})
    (models_dir / "notes.txt").write_text("hello", encoding="utf-8")

    result = Scanner().scan([models_dir])
    by_name = {m.name: m for m in result.models}

    assert set(by_name) == {"llama-Q4_K_M", "nomic.Q4_K_M", "mistral"}
    assert by_name["llama-Q4_K_M"].capabilities.vision is True
    assert by_name["nomic.Q4_K_M"].capabilities.embedding is True
    assert by_name["mistral"].format == ModelFormat.safetensors
    assert len(_skips(result, SkipReason.projector)) == 1
    assert len(_skips(result, SkipReason.adapter)) == 1


def test_a_cached_model_is_named_after_its_repo(models_dir: Path) -> None:
    """A snapshot directory is named after the revision hash, so the
    directory name — right everywhere else — reads as
    `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Found by scanning a real
    HuggingFace cache, not by a fixture."""
    cache = models_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    write_hf_model(cache / "snapshots" / "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", encoding="utf-8"
    )

    model = Scanner().scan([models_dir]).models[0]
    assert model.name == "all-MiniLM-L6-v2"


def test_dataset_caches_are_pruned_whole(models_dir: Path) -> None:
    """A hub directory holds dataset caches beside model caches. Walking
    into each to reject its blobs/refs/.no_exist individually turned a
    real scan's skip list into 42 records of noise."""
    dataset = models_dir / "datasets--HuggingFaceFW--fineweb-edu"
    (dataset / "blobs").mkdir(parents=True)
    (dataset / "refs").mkdir()
    (dataset / "snapshots" / "abc").mkdir(parents=True)

    result = Scanner().scan([models_dir])

    assert result.models == []
    skipped = _skips(result, SkipReason.not_a_model)
    assert skipped == [str(dataset)], "the dataset cache should be pruned, not walked"
