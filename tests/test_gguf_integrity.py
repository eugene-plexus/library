"""Row 18: local tensor ranges, without reading any weights."""

import io
import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import SkipReason
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.scanner import Scanner, cache_key
from eugene_plexus_library.store import StateStore

from .conftest import qwen_like_kv, write_gguf


def model(path: Path, *, offset: int = 0, dtype: int = 0, alignment: int = 32) -> Path:
    kv = qwen_like_kv()
    kv["general.alignment"] = alignment
    name = b"blk.0.attn_norm.weight"
    table = struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 32, dtype, offset)
    write_gguf(path, kv, tensor_count=1, payload=table)
    header_end = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(bytes(-header_end % alignment + offset + 128))
    return path


@pytest.mark.parametrize("alignment,offset", [(32, 0), (64, 128)])
def test_tensor_data_past_eof_is_rejected(tmp_path: Path, alignment: int, offset: int) -> None:
    path = model(tmp_path / "copied.gguf", alignment=alignment, offset=offset)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(gguf.GgufError, match=r"copied.gguf:.*blk.0.attn_norm.weight.*ends.*file"):
        gguf.read_metadata(path, tensors=True)


def test_valid_tensor_range_and_header_only_probe(tmp_path: Path) -> None:
    path = model(tmp_path / "complete.gguf", alignment=64, offset=128)
    assert gguf.read_metadata(path, tensors=True).tensor_bytes == 128
    raw = path.read_bytes()
    header = gguf.read_metadata(path).header_bytes
    assert gguf.read_metadata_stream(io.BytesIO(raw[:header])).architecture == "llama"


def test_unknown_tensor_type_does_not_block_a_model(tmp_path: Path) -> None:
    path = model(tmp_path / "new-type.gguf", dtype=999)
    assert gguf.read_metadata(path, tensors=True).tensor_bytes is None
    assert len(Scanner().scan([tmp_path]).models) == 1


@pytest.mark.parametrize("split", [False, True])
def test_scan_rejects_truncated_weights_in_any_part(tmp_path: Path, split: bool) -> None:
    if split:
        model(tmp_path / "m-00001-of-00002.gguf")
        path = model(tmp_path / "m-00002-of-00002.gguf")
    else:
        path = model(tmp_path / "m.gguf")
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    result = Scanner().scan([tmp_path])
    assert result.models == []
    assert any(
        skip.path == str(path)
        and skip.reason is SkipReason.unreadable_header
        and "blk.0.attn_norm.weight" in (skip.detail or "")
        for skip in result.skipped
    )


def test_a_changed_second_shard_invalidates_the_cached_model(tmp_path: Path) -> None:
    first = model(tmp_path / "m-00001-of-00002.gguf")
    second = model(tmp_path / "m-00002-of-00002.gguf")
    store = StateStore(tmp_path / "state.json")
    store.replace_models(Scanner().scan([tmp_path]).models, scanned_at=datetime.now(UTC))
    assert store.lookup(cache_key(first, first.stat())) is not None
    with second.open("r+b") as stream:
        stream.truncate(second.stat().st_size - 1)
    assert Scanner(cache_lookup=store.lookup).scan([tmp_path]).models == []


def test_disk_cache_is_revalidated_after_restart(tmp_path: Path) -> None:
    path = model(tmp_path / "m.gguf")
    store = StateStore(tmp_path / "state.json")
    store.replace_models(Scanner().scan([tmp_path]).models, scanned_at=datetime.now(UTC))
    restarted = StateStore(tmp_path / "state.json")
    restarted.load()
    assert restarted.lookup(cache_key(path, path.stat())) is None
