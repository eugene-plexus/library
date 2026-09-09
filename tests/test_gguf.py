"""The GGUF reader, against real bytes."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from eugene_plexus_library.formats import gguf

from .conftest import embedding_kv, projector_kv, qwen_like_kv, write_gguf


def test_reads_a_models_metadata(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "model.gguf", qwen_like_kv(name="Test 7B", file_type=15))
    meta = gguf.read_metadata(path)

    assert meta.version == 3
    assert meta.architecture == "llama"
    assert meta.name == "Test 7B"
    assert meta.size_label == "7B"
    assert meta.context_length == 4096
    assert meta.file_type == 15
    assert meta.quantization == "Q4_K_M"
    assert meta.has_chat_template is True
    assert meta.is_projector is False
    assert meta.is_embedding is False


def test_reads_keys_that_sit_after_the_token_array(tmp_path: Path) -> None:
    """The quant tier is the *last* KV pair in a real file, behind a
    ~10 MB vocabulary. A reader that stops early, or that mis-steps the
    array, loses exactly this field."""
    path = write_gguf(tmp_path / "model.gguf", qwen_like_kv(vocab=500, file_type=14))
    meta = gguf.read_metadata(path)

    assert meta.quantization == "Q4_K_S"
    assert meta.vocab_size == 500


def test_does_not_materialise_the_vocabulary(tmp_path: Path) -> None:
    """Holding vocabularies costs ~10 MB per model scanned. The array is
    stepped over and only its length is kept."""
    path = write_gguf(tmp_path / "model.gguf", qwen_like_kv(vocab=2000))
    meta = gguf.read_metadata(path)

    assert meta.kv["tokenizer.ggml.tokens"] is None
    assert meta.array_lengths["tokenizer.ggml.tokens"] == 2000
    assert meta.vocab_size == 2000


def test_small_arrays_are_kept(tmp_path: Path) -> None:
    """Structural arrays carry meaning and are short. Only vocabularies
    are big enough to be worth discarding."""
    path = write_gguf(tmp_path / "model.gguf", {"llama.rope.dimension_sections": [11, 11, 10, 0]})
    meta = gguf.read_metadata(path)

    assert meta.kv["llama.rope.dimension_sections"] == [11, 11, 10, 0]


def test_architecture_prefixed_keys(tmp_path: Path) -> None:
    """There is no fixed key set: the prefix is whatever
    `general.architecture` says, so a reader with hardcoded names goes
    blind on every architecture released after it was written."""
    path = write_gguf(
        tmp_path / "model.gguf",
        qwen_like_kv(architecture="qwen35", context_length=262144),
    )
    meta = gguf.read_metadata(path)

    assert meta.architecture == "qwen35"
    assert meta.context_length == 262144
    assert meta.arch_key("block_count") == 32


def test_projector_is_identified_by_metadata_not_filename(tmp_path: Path) -> None:
    """A renamed projector still declares itself. Getting this wrong adds
    a ~1 GB phantom entry to the library."""
    path = write_gguf(tmp_path / "innocuous-name.gguf", projector_kv())
    meta = gguf.read_metadata(path)

    assert meta.is_projector is True
    assert meta.general_type == "mmproj"


def test_absent_general_type_means_model(tmp_path: Path) -> None:
    """Verified absent on a real embedding model — 23 KV pairs, no
    `general.type`. Requiring the key would reject the file."""
    path = write_gguf(tmp_path / "embed.gguf", embedding_kv())
    meta = gguf.read_metadata(path)

    assert meta.general_type is None
    assert meta.is_projector is False


def test_embedding_model_is_detected(tmp_path: Path) -> None:
    """Silent failure otherwise: launched as a chat model it starts,
    serves, and returns nonsense."""
    path = write_gguf(tmp_path / "embed.gguf", embedding_kv())
    meta = gguf.read_metadata(path)

    assert meta.is_embedding is True


def test_recommended_sampling_is_read(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "model.gguf", qwen_like_kv())
    sampling = gguf.read_metadata(path).recommended_sampling

    assert sampling.temperature == pytest.approx(1.0)
    assert sampling.top_k == 20
    assert sampling.top_p is None
    assert bool(sampling) is True


def test_no_sampling_keys_is_falsey(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "llama"})
    assert bool(gguf.read_metadata(path).recommended_sampling) is False


def test_unknown_file_type_reports_no_label(tmp_path: Path) -> None:
    """The quant families churn upstream. An unrecognised value degrades
    to the raw integer — mapping it onto a plausible neighbour would put
    a wrong quant tier in front of someone choosing a model."""
    path = write_gguf(tmp_path / "model.gguf", {"general.file_type": 999})
    meta = gguf.read_metadata(path)

    assert meta.file_type == 999
    assert meta.quantization is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Qwen3.6-27B-Q4_K_M.gguf", "Q4_K_M"),
        ("model-Q4_K_S.gguf", "Q4_K_S"),
        ("thing.IQ2_XXS.gguf", "IQ2_XXS"),
        ("Meta-Llama-3-8B-Instruct.Q8_0.gguf", "Q8_0"),
        ("mmproj-F32.gguf", "F32"),
        ("no-quant-here.gguf", None),
    ],
)
def test_quant_from_filename(filename: str, expected: str | None) -> None:
    assert gguf.quant_from_filename(filename) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("model-00001-of-00005", ("model", 1, 5)),
        ("big-thing-00003-of-00003", ("big-thing", 3, 3)),
        ("model", None),
        ("model-1-of-5", None),
    ],
)
def test_shard_position(stem: str, expected: tuple[str, int, int] | None) -> None:
    assert gguf.shard_position(Path(f"{stem}.gguf")) == expected


def test_strip_shard_suffix() -> None:
    assert gguf.strip_shard_suffix("model-00001-of-00005") == "model"
    assert gguf.strip_shard_suffix("model") == "model"


# --- malformed input --------------------------------------------------


def test_rejects_a_non_gguf_file(tmp_path: Path) -> None:
    path = tmp_path / "not-really.gguf"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 64)

    with pytest.raises(gguf.GgufError, match="not a GGUF file"):
        gguf.read_metadata(path)


def test_rejects_an_unsupported_version(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "old.gguf", {"general.architecture": "llama"}, version=1)

    with pytest.raises(gguf.GgufError, match="unsupported GGUF version"):
        gguf.read_metadata(path)


def test_rejects_a_truncated_file(tmp_path: Path) -> None:
    full = write_gguf(tmp_path / "model.gguf", qwen_like_kv()).read_bytes()
    path = tmp_path / "truncated.gguf"
    path.write_bytes(full[: len(full) // 2])

    with pytest.raises(gguf.GgufError, match="truncated"):
        gguf.read_metadata(path)


def test_rejects_an_implausible_length(tmp_path: Path) -> None:
    """A corrupt u64 length is the difference between an error and an
    attempt to allocate two exabytes."""
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 1 << 60)  # key length
    )

    with pytest.raises(gguf.GgufError, match="implausible"):
        gguf.read_metadata(path)


def test_permission_errors_are_not_disguised_as_corruption(tmp_path: Path) -> None:
    """`OSError` propagates rather than becoming a `GgufError`: a locked
    file reported as a malformed one sends the operator to the wrong
    problem entirely."""
    with pytest.raises(OSError):
        gguf.read_metadata(tmp_path / "does-not-exist.gguf")
