"""The safetensors reader and the HuggingFace-cache shapes around it."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from eugene_plexus_library.formats import safetensors

from .conftest import write_hf_model, write_safetensors


def test_reads_an_exact_parameter_count(tmp_path: Path) -> None:
    """The asymmetry that shaped the wire schema: safetensors gives an
    exact count from a ~11 KB header, GGUF gives none at all."""
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"a": ("F32", [10, 20]), "b": ("F32", [5]), "ids": ("I64", [4])},
    )
    header = safetensors.read_header(path)

    assert header.parameters == 10 * 20 + 5 + 4
    assert header.tensor_count == 3
    assert header.dtype_counts == {"F32": 205, "I64": 4}


def test_dominant_dtype_ignores_integer_bookkeeping(tmp_path: Path) -> None:
    """Verified on a real MiniLM: 512 I64 elements beside 22,713,216 F32
    ones. Counting the ids would report an F32 model as mixed."""
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"w": ("BF16", [1000]), "position_ids": ("I64", [4000])},
    )
    assert safetensors.read_header(path).dominant_dtype == "BF16"


def test_dominant_dtype_across_shards() -> None:
    """A sharded model's answer is over the sum, not any one file."""
    assert safetensors.dominant_dtype({"F16": 10, "BF16": 30}) == "BF16"
    assert safetensors.dominant_dtype({}) is None


def test_metadata_is_extracted(tmp_path: Path) -> None:
    path = write_safetensors(
        tmp_path / "model.safetensors", {"w": ("F32", [4])}, metadata={"format": "pt"}
    )
    header = safetensors.read_header(path)

    assert header.metadata == {"format": "pt"}
    # __metadata__ is not a tensor and must not be counted as one.
    assert header.tensor_count == 1


def test_reads_config_json(tmp_path: Path) -> None:
    directory = write_hf_model(tmp_path / "model", max_position_embeddings=8192)
    config = safetensors.read_config(directory / "config.json")

    assert config.architecture == "LlamaForCausalLM"
    assert config.context_length == 8192
    assert config.is_embedding is False


def test_encoder_architectures_read_as_embedding(tmp_path: Path) -> None:
    directory = write_hf_model(tmp_path / "bert", architectures=["BertModel"])
    assert safetensors.read_config(directory / "config.json").is_embedding is True


def test_falls_back_to_model_type_when_architectures_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "mistral", "n_positions": 512}), encoding="utf-8")
    config = safetensors.read_config(path)

    assert config.architecture == "mistral"
    assert config.context_length == 512


# --- what is not a model ----------------------------------------------


def test_adapter_directory_is_recognised(tmp_path: Path) -> None:
    directory = tmp_path / "lora"
    directory.mkdir()
    names = {"adapter_config.json", "adapter_model.safetensors"}
    assert safetensors.is_adapter_dir(directory, names) is True


def test_a_model_shipping_an_adapter_is_still_a_model(tmp_path: Path) -> None:
    """Requiring the *absence* of base weights is what separates the two;
    some adapter directories ship a copy of the base config, and some
    full models ship an adapter alongside."""
    directory = tmp_path / "model"
    directory.mkdir()
    names = {"config.json", "model.safetensors", "adapter_config.json"}
    assert safetensors.is_adapter_dir(directory, names) is False


def test_hf_repo_id_is_decoded(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert safetensors.hf_repo_id(snapshot) == "sentence-transformers/all-MiniLM-L6-v2"


def test_hf_repo_id_splits_on_the_first_separator(tmp_path: Path) -> None:
    """`models--a--b--c` is ambiguous; HuggingFace's own encoding splits
    on the first `--`, so an org containing a hyphen round-trips."""
    snapshot = tmp_path / "models--my-org--my--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    assert safetensors.hf_repo_id(snapshot) == "my-org/my--model"


def test_hf_revision_and_current_revision(tmp_path: Path) -> None:
    cache = tmp_path / "models--org--name"
    snapshot = cache / "snapshots" / "1110a243"
    snapshot.mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("1110a243\n", encoding="utf-8")

    assert safetensors.hf_revision(snapshot) == "1110a243"
    assert safetensors.hf_current_revision(snapshot) == "1110a243"


def test_missing_refs_main_keeps_every_revision(tmp_path: Path) -> None:
    """Dropping a model because a ref file was unreadable is worse than
    listing it twice."""
    snapshot = tmp_path / "models--org--name" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    assert safetensors.hf_current_revision(snapshot) is None


def test_a_plain_directory_has_no_revision(tmp_path: Path) -> None:
    directory = tmp_path / "just-a-folder"
    directory.mkdir()
    assert safetensors.hf_revision(directory) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model-00001-of-00003.safetensors", ("model", 1, 3)),
        ("model.safetensors", None),
    ],
)
def test_shard_position(name: str, expected: tuple[str, int, int] | None) -> None:
    assert safetensors.shard_position(Path(name)) == expected


# --- malformed input ---------------------------------------------------


def test_rejects_a_truncated_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 4096) + b"{}")

    with pytest.raises(safetensors.SafetensorsError, match="truncated"):
        safetensors.read_header(path)


def test_rejects_an_implausible_header_length(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40))

    with pytest.raises(safetensors.SafetensorsError, match="implausible"):
        safetensors.read_header(path)


def test_rejects_non_json(tmp_path: Path) -> None:
    body = b"not json at all"
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)

    with pytest.raises(safetensors.SafetensorsError, match="not valid JSON"):
        safetensors.read_header(path)


def test_rejects_a_tensor_with_no_shape(tmp_path: Path) -> None:
    body = json.dumps({"w": {"dtype": "F32"}}).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)

    with pytest.raises(safetensors.SafetensorsError, match="no shape/dtype"):
        safetensors.read_header(path)


def test_rejects_a_config_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(safetensors.SafetensorsError, match="not a JSON object"):
        safetensors.read_config(path)
