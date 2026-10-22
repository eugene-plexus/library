"""Bound drafters pair by their GGUF target shape, independent of names."""

from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import DraftKind, SkipReason
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.scanner import Scanner, cache_key

from .conftest import qwen_like_kv
from .test_speculation import _write


def target(path: Path, *, hidden: int = 5120, blocks: int = 64) -> Path:
    kv = qwen_like_kv(architecture="qwen35", vocab=128)
    kv.update({"qwen35.embedding_length": hidden, "qwen35.block_count": blocks})
    return _write(path, kv, ["blk.0.attn_norm.weight"])


def drafter(
    path: Path,
    *,
    arch: str = "dflash",
    layers: list[int] | None = None,
    markov: bool = False,
    vocab: int = 128,
) -> Path:
    kv = qwen_like_kv(architecture=arch, vocab=vocab)
    kv[f"{arch}.target_hidden_size"] = 5120
    if layers is not None:
        kv[f"{arch}.target_layers"] = layers
    return _write(path, kv, ["blk.0.attn_norm.weight", *(["markov_w1.weight"] if markov else [])])


def test_dflash_pairs_with_its_target(models_dir: Path) -> None:
    model = target(models_dir / "model.gguf")
    draft = drafter(models_dir / "renamed.gguf", layers=[1, 20, 40, 63])
    result = Scanner().scan([models_dir])
    assert [m.path for m in result.models] == [str(model)]
    speculation = result.models[0].gguf.speculation
    assert [(d.path, d.kind) for d in speculation.draftFiles] == [(str(draft), DraftKind.dflash)]
    [skip] = [s for s in result.skipped if s.path == str(draft)]
    assert skip.reason is SkipReason.draft
    assert (
        skip.detail
        == "DFlash drafter (draft-dflash) for speculative decoding with the model beside it"
    )


def test_dflash_for_another_model_is_not_paired(models_dir: Path) -> None:
    target(models_dir / "model.gguf", hidden=4096, blocks=32)
    draft = drafter(models_dir / "dflash-model.gguf", layers=[1, 20, 40, 63])
    result = Scanner().scan([models_dir])
    assert len(result.models) == 1
    assert result.models[0].gguf.speculation.draftFiles == []
    assert "it needs hidden size 5120 and at least 64 layers" in next(
        s.detail for s in result.skipped if s.path == str(draft)
    )


def test_markov_head_is_dspark(models_dir: Path) -> None:
    target(models_dir / "model.gguf")
    draft = drafter(models_dir / "dflash-model.gguf", layers=[1, 20, 40, 63], markov=True)
    assert gguf.read_metadata(draft, tensors=True).has_markov_head
    assert (
        Scanner().scan([models_dir]).models[0].gguf.speculation.draftFiles[0].kind
        is DraftKind.dspark
    )


def test_eagle3_layer_equal_to_block_count_pairs(models_dir: Path) -> None:
    target(models_dir / "model.gguf")
    drafter(models_dir / "helper.gguf", arch="eagle3", layers=[1, 20, 64])
    assert (
        Scanner().scan([models_dir]).models[0].gguf.speculation.draftFiles[0].kind
        is DraftKind.eagle3
    )


@pytest.mark.parametrize(
    "arch,layers",
    [
        ("dflash", [1, 20, 64]),
        ("eagle3", [1, 20, 65]),
        ("eagle3", [1, 20]),
        ("dflash", None),
        ("dflash", []),
        ("dflash", [-1, 20]),
        ("dflash", list(range(513))),
    ],
)
def test_invalid_target_layers_do_not_pair(
    models_dir: Path, arch: str, layers: list[int] | None
) -> None:
    target(models_dir / "model.gguf")
    drafter(models_dir / "helper.gguf", arch=arch, layers=layers)
    result = Scanner().scan([models_dir])
    assert len(result.models) == 1
    assert result.models[0].gguf.speculation.draftFiles == []
    if layers is None:
        assert "target_layers" in result.skipped[0].detail


def test_different_vocab_does_not_pair(models_dir: Path) -> None:
    target(models_dir / "model.gguf")
    drafter(models_dir / "helper.gguf", layers=[1, 20, 40, 63], vocab=256)
    result = Scanner().scan([models_dir])
    assert len(result.models) == 1
    assert result.models[0].gguf.speculation.draftFiles == []


def test_a_cached_drafter_is_reclassified(models_dir: Path) -> None:
    model_path = target(models_dir / "model.gguf")
    model = Scanner().scan([models_dir]).models[0]
    draft_path = drafter(models_dir / "helper.gguf", layers=[1, 20, 40, 63])
    old_drafter = model.model_copy(update={"path": str(draft_path), "architecture": "dflash"})

    def lookup(key):  # type: ignore[no-untyped-def]
        return old_drafter if key == cache_key(draft_path, draft_path.stat()) else model

    result = Scanner(cache_lookup=lookup).scan([models_dir])
    assert [m.path for m in result.models] == [str(model_path)]
    assert result.models[0].gguf.speculation.draftFiles[0].path == str(draft_path)


def test_missing_target_hidden_size_names_the_key(models_dir: Path) -> None:
    target(models_dir / "model.gguf")
    _write(
        models_dir / "helper.gguf",
        {"general.architecture": "dflash", "dflash.target_layers": [1, 20, 63]},
        [],
    )
    result = Scanner().scan([models_dir])
    assert len(result.models) == 1
    assert result.models[0].gguf.speculation.draftFiles == []
    assert "dflash.target_hidden_size" in result.skipped[0].detail
