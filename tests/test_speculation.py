"""SD1: what a GGUF model can draft with, read from its tensor table.

The shapes are the real ones measured on 2026-10-18: unsloth's
`Qwen3.5-9B-MTP-GGUF` carries its head in the model file (33 blocks, the
last with `nextn` tensors), and ggml-org's `mtp-Qwen3.6-27B-Q4_0.gguf` is a
separate file of 18 tensors holding only block 64 of 65. Both declare
`qwen35.nextn_predict_layers 1`, which is why the tensors decide.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from eugene_plexus_library import model_fit
from eugene_plexus_library._generated.models import DraftKind, ModelFileRole, SkipReason
from eugene_plexus_library.downloads import _role_for
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.scanner import Scanner

from .conftest import pad_gguf, qwen_like_kv, write_gguf

ARCH = "qwen35"


def _table(names: list[str]) -> bytes:
    """A tensor table of one-dimensional F32 tensors, 32 elements each."""
    out = bytearray()
    for name in names:
        encoded = name.encode()
        out += struct.pack("<Q", len(encoded)) + encoded
        out += struct.pack("<I", 1) + struct.pack("<Q", 32)
        out += struct.pack("<I", 0) + struct.pack("<Q", 0)
    return bytes(out)


def _blocks(first: int, last: int, *, head: bool) -> list[str]:
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for block in range(first, last + 1):
        names.append(f"blk.{block}.attn_norm.weight")
    if head:
        names += [
            f"blk.{last}.nextn.eh_proj.weight",
            f"blk.{last}.nextn.enorm.weight",
            f"blk.{last}.nextn.hnorm.weight",
        ]
    return names


def _write(path: Path, kv: dict[str, Any], names: list[str]) -> Path:
    return pad_gguf(write_gguf(path, kv, tensor_count=len(names), payload=_table(names)), 128)


def _kv(*, blocks: int, nextn: int | None = None, vocab: int = 128, name: str = "Q") -> dict:
    kv = qwen_like_kv(name=name, architecture=ARCH, vocab=vocab)
    kv[f"{ARCH}.block_count"] = blocks
    if nextn is not None:
        kv[f"{ARCH}.nextn_predict_layers"] = nextn
    return kv


def _model_with_head(path: Path) -> Path:
    return _write(path, _kv(blocks=33, nextn=1), _blocks(0, 32, head=True))


def _plain_model(path: Path, *, vocab: int = 128) -> Path:
    return _write(path, _kv(blocks=64, vocab=vocab), _blocks(0, 63, head=False))


def _sidecar(path: Path, *, vocab: int = 128) -> Path:
    return _write(path, _kv(blocks=65, nextn=1, vocab=vocab), _blocks(64, 64, head=True))


def _skips(result, reason: SkipReason) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {s.path: s.detail or "" for s in result.skipped if s.reason is reason}


def test_a_model_with_its_own_head_says_so(models_dir: Path) -> None:
    _model_with_head(models_dir / "Qwen3.5-9B-Q4_K_M.gguf")
    model = Scanner().scan([models_dir]).models[0]

    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.mtpHead is True
    assert model.gguf.speculation.draftFiles == []


def test_nextn_predict_layers_alone_is_not_a_head(models_dir: Path) -> None:
    """A model that declares the key with no `nextn` tensors (a conversion
    that dropped them) cannot draft with a head it does not have."""
    _write(models_dir / "m.gguf", _kv(blocks=33, nextn=1), _blocks(0, 32, head=False))
    model = Scanner().scan([models_dir]).models[0]

    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.mtpHead is False


def test_a_separate_mtp_file_belongs_to_the_model_beside_it(models_dir: Path) -> None:
    weights = _plain_model(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    sidecar = _sidecar(models_dir / "mtp-Qwen3.6-27B-Q4_0.gguf")
    result = Scanner().scan([models_dir])

    assert [m.path for m in result.models] == [str(weights)]
    model = result.models[0]
    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.mtpHead is False
    [draft] = model.gguf.speculation.draftFiles
    assert draft.path == str(sidecar) and draft.kind is DraftKind.mtp
    assert draft.sizeBytes == sidecar.stat().st_size
    assert [f.role for f in model.files or []] == [ModelFileRole.weights, ModelFileRole.draft]
    assert model.sizeBytes == weights.stat().st_size + sidecar.stat().st_size
    # A fit covers the weights a launch always loads, not the companion.
    assert model_fit.weights_of(model) == weights.stat().st_size
    assert "with the model beside it" in _skips(result, SkipReason.draft)[str(sidecar)]


def test_the_tensors_decide_not_the_name(models_dir: Path) -> None:
    _plain_model(models_dir / "model.gguf")
    renamed = _sidecar(models_dir / "helper.gguf")
    model = Scanner().scan([models_dir]).models[0]

    assert model.gguf is not None and model.gguf.speculation is not None
    assert [d.path for d in model.gguf.speculation.draftFiles] == [str(renamed)]


def test_every_quant_in_the_folder_shares_one_mtp_file(models_dir: Path) -> None:
    _plain_model(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    _plain_model(models_dir / "Qwen3.6-27B-Q8_0.gguf")
    sidecar = _sidecar(models_dir / "mtp-Qwen3.6-27B-Q4_0.gguf")
    result = Scanner().scan([models_dir])

    assert len(result.models) == 2
    for model in result.models:
        assert model.gguf is not None and model.gguf.speculation is not None
        assert [d.path for d in model.gguf.speculation.draftFiles] == [str(sidecar)]


def test_a_companion_with_no_model_is_skipped_and_says_why(models_dir: Path) -> None:
    sidecar = _sidecar(models_dir / "mtp-Qwen3.6-27B-Q4_0.gguf")
    result = Scanner().scan([models_dir])

    assert result.models == []
    assert "no model of the same architecture" in _skips(result, SkipReason.draft)[str(sidecar)]


def test_a_different_vocabulary_is_not_paired(models_dir: Path) -> None:
    _plain_model(models_dir / "model.gguf", vocab=128)
    _sidecar(models_dir / "mtp-other.gguf", vocab=256)
    model = Scanner().scan([models_dir]).models[0]

    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.draftFiles == []


def test_a_small_model_named_as_a_draft_is_a_simple_companion(models_dir: Path) -> None:
    _plain_model(models_dir / "big.gguf")
    small = _write(models_dir / "draft-small.gguf", _kv(blocks=4), _blocks(0, 3, head=False))
    result = Scanner().scan([models_dir])

    assert len(result.models) == 1
    model = result.models[0]
    assert model.gguf is not None and model.gguf.speculation is not None
    [draft] = model.gguf.speculation.draftFiles
    assert draft.path == str(small) and draft.kind is DraftKind.simple


def test_a_companion_downloaded_later_reaches_a_cached_model(models_dir: Path) -> None:
    """The model's own file, and so its cache entry, does not change when a
    companion lands beside it."""
    weights = _plain_model(models_dir / "Qwen3.6-27B-Q4_K_M.gguf")
    first = Scanner().scan([models_dir]).models[0]
    sidecar = _sidecar(models_dir / "mtp-Qwen3.6-27B-Q4_0.gguf")

    def lookup(key):  # type: ignore[no-untyped-def]
        return first if key[0] == str(weights) else None

    model = Scanner(cache_lookup=lookup).scan([models_dir]).models[0]
    assert model.gguf is not None and model.gguf.speculation is not None
    assert [d.path for d in model.gguf.speculation.draftFiles] == [str(sidecar)]
    assert model.sizeBytes == weights.stat().st_size + sidecar.stat().st_size

    # And one deleted since is gone again, its bytes with it.
    second = model
    sidecar.unlink()

    def again(key):  # type: ignore[no-untyped-def]
        return second if key[0] == str(weights) else None

    model = Scanner(cache_lookup=again).scan([models_dir]).models[0]
    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.draftFiles == []
    assert [f.role for f in model.files or []] == [ModelFileRole.weights]
    assert model.sizeBytes == weights.stat().st_size


def test_an_entry_cached_before_sd1_is_read_again(models_dir: Path) -> None:
    path = _model_with_head(models_dir / "m.gguf")
    stale = Scanner().scan([models_dir]).models[0]
    assert stale.gguf is not None
    stale = stale.model_copy(update={"gguf": stale.gguf.model_copy(update={"speculation": None})})

    model = (
        Scanner(cache_lookup=lambda key: stale if key[0] == str(path) else None)
        .scan([models_dir])
        .models[0]
    )
    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.mtpHead is True


def test_the_reader_follows_llama_cpps_rule(tmp_path: Path) -> None:
    head = gguf.read_metadata(_model_with_head(tmp_path / "a.gguf"), tensors=True)
    sidecar = gguf.read_metadata(_sidecar(tmp_path / "mtp-b.gguf"), tensors=True)
    plain = gguf.read_metadata(_plain_model(tmp_path / "c.gguf"), tensors=True)

    assert (head.mtp_head, head.draft_kind("a.gguf")) == (True, None)
    assert (sidecar.mtp_head, sidecar.draft_kind("mtp-b.gguf")) == (False, "mtp")
    assert (plain.mtp_head, plain.draft_kind("c.gguf")) == (False, None)
    # Without the tensor table nothing is claimed either way.
    unread = gguf.read_metadata(tmp_path / "mtp-b.gguf")
    assert (unread.mtp_head, unread.draft_kind("mtp-b.gguf")) == (False, None)


def test_a_downloaded_mtp_file_is_labelled_a_companion() -> None:
    assert _role_for("mtp-Qwen3.6-27B-Q4_0.gguf", index=1) is ModelFileRole.draft
    assert _role_for("Qwen3.6-27B-Q4_K_M.gguf", index=0) is ModelFileRole.weights
