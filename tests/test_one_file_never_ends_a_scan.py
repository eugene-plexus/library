"""One hostile or corrupt file must cost its own listing, never the scan.

library#21: a GGUF array whose elements are arrays recursed once per level,
12 bytes of file per level. A few thousand levels raised `RecursionError`,
which the scanner did not catch, so every later scan failed until someone
deleted the file. The other readers get the same treatment here: a JSON file
nested too deep raises the same error.
"""

from __future__ import annotations

import json
import struct
import tracemalloc
from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import ModelStatus, SkipReason
from eugene_plexus_library.formats import gguf, safetensors
from eugene_plexus_library.scanner import Scanner

from .conftest import (
    T_ARRAY,
    T_UINT32,
    qwen_like_kv,
    write_gguf,
    write_hf_model,
)

T_UINT8 = 0
DEEP = "[" * 5000 + "]" * 5000


@pytest.fixture
def deep_json_overflows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the JSON parser fail on `DEEP` as CPython does on Windows.

    Whether 5,000 levels overflows depends on the platform's C stack (it parses
    on Linux CI), so the failure is made, not hoped for.
    """
    real = json.loads

    def loads(text: str | bytes, *args: object, **kwargs: object) -> object:
        if (text.decode() if isinstance(text, bytes) else text) == DEEP:
            raise RecursionError("maximum recursion depth exceeded while decoding")
        return real(text, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(json, "loads", loads)


def _header(key: str, kv_count: int = 1) -> bytes:
    raw = key.encode()
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", kv_count)
        + struct.pack("<Q", len(raw))
        + raw
    )


def _nested_array_file(path: Path, levels: int) -> Path:
    """One key holding `levels` arrays, each the single element of the one above."""
    body = bytearray(_header("hostile.nested"))
    body += struct.pack("<I", T_ARRAY)  # the key's value type
    for _ in range(levels - 1):
        body += struct.pack("<I", T_ARRAY) + struct.pack("<Q", 1)  # element type, count
    body += struct.pack("<I", T_UINT32) + struct.pack("<Q", 0)  # the innermost, empty
    path.write_bytes(bytes(body))
    return path


def _array_of_arrays_file(path: Path, outer: int, inner: int) -> Path:
    """`outer` arrays of `inner` bytes each, the shape that multiplies."""
    body = bytearray(_header("hostile.grid"))
    body += struct.pack("<I", T_ARRAY)
    body += struct.pack("<I", T_ARRAY) + struct.pack("<Q", outer)
    row = struct.pack("<I", T_UINT8) + struct.pack("<Q", inner) + b"\0" * inner
    body += row * outer
    path.write_bytes(bytes(body))
    return path


# --- the reader's own limits ----------------------------------------------


def test_five_thousand_nested_levels_raise_the_readers_error(tmp_path: Path) -> None:
    path = _nested_array_file(tmp_path / "deep.gguf", 5000)

    with pytest.raises(gguf.GgufError, match=r"hostile\.nested.*nested"):
        gguf.read_metadata(path)


def test_a_file_nested_to_the_limit_reads(tmp_path: Path) -> None:
    path = _nested_array_file(tmp_path / "ok.gguf", gguf._MAX_ARRAY_DEPTH)

    meta = gguf.read_metadata(path)

    value = meta.kv["hostile.nested"]
    for _ in range(gguf._MAX_ARRAY_DEPTH - 1):
        assert isinstance(value, list) and len(value) == 1
        value = value[0]
    assert value == []


def test_one_level_past_the_limit_is_refused(tmp_path: Path) -> None:
    path = _nested_array_file(tmp_path / "over.gguf", gguf._MAX_ARRAY_DEPTH + 1)

    with pytest.raises(gguf.GgufError, match="nested more than"):
        gguf.read_metadata(path)


def test_a_grid_of_inline_arrays_reads_within_the_budget(tmp_path: Path) -> None:
    path = _array_of_arrays_file(tmp_path / "grid.gguf", 512, 512)

    meta = gguf.read_metadata(path)

    assert len(meta.kv["hostile.grid"]) == 512
    assert all(len(row) == 512 for row in meta.kv["hostile.grid"])


def test_a_grid_of_inline_arrays_stops_at_the_element_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count times count: each array is under the inline limit, the whole is not."""
    monkeypatch.setattr(gguf, "_MAX_ARRAY_ELEMENTS", 100_000)
    path = _array_of_arrays_file(tmp_path / "grid.gguf", 512, 512)

    with pytest.raises(gguf.GgufError, match=r"hostile\.grid.*too many array elements"):
        gguf.read_metadata(path)


def test_the_budget_is_spent_up_front_not_after_reading_the_elements(tmp_path: Path) -> None:
    """A kept array that claims 200 million elements in a tiny file is refused
    on its count, not read until the file ends."""
    body = bytearray(_header("tokenizer.ggml.tokens"))
    body += struct.pack("<I", T_ARRAY) + struct.pack("<I", T_UINT8) + struct.pack("<Q", 200_000_000)
    path = tmp_path / "claim.gguf"
    path.write_bytes(bytes(body) + b"\0" * 64)

    with pytest.raises(gguf.GgufError, match="too many array elements"):
        gguf.read_metadata_stream(
            path.open("rb"), name=path.name, keep_arrays=frozenset({"tokenizer.ggml.tokens"})
        )


def test_a_string_longer_than_the_file_is_refused_without_allocating_it(tmp_path: Path) -> None:
    """A string value claims 16 MiB in a 60-byte file. The read must not ask for 16 MiB."""
    body = _header("general.name") + struct.pack("<I", 8) + struct.pack("<Q", gguf._MAX_STRING)
    path = tmp_path / "claim.gguf"
    path.write_bytes(body)

    tracemalloc.start()
    try:
        with pytest.raises(gguf.GgufTruncated):
            gguf.read_metadata(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1 << 20


def test_a_key_longer_than_a_key_can_be_is_refused(tmp_path: Path) -> None:
    body = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", gguf._MAX_KEY + 1)
    )
    path = tmp_path / "longkey.gguf"
    path.write_bytes(body + b"k" * (gguf._MAX_KEY + 1))

    with pytest.raises(gguf.GgufError, match="implausible key length"):
        gguf.read_metadata(path)


# --- the scanner ----------------------------------------------------------


def test_a_deeply_nested_gguf_is_listed_unreadable_and_a_healthy_model_is_still_listed(
    models_dir: Path,
) -> None:
    write_gguf(models_dir / "healthy-Q4_K_M.gguf", qwen_like_kv(name="Healthy"))
    _nested_array_file(models_dir / "hostile-Q4_K_M.gguf", 5000)

    result = Scanner().scan([models_dir])

    assert [m.name for m in result.models] == ["healthy-Q4_K_M"]
    skipped = [s for s in result.skipped if s.reason == SkipReason.unreadable_header]
    assert len(skipped) == 1
    assert skipped[0].path.endswith("hostile-Q4_K_M.gguf")
    assert "nested" in (skipped[0].detail or "")


@pytest.mark.parametrize("error", [RecursionError("maximum recursion depth"), MemoryError()])
def test_a_reader_failure_the_reader_did_not_expect_is_that_files_skip(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    write_gguf(models_dir / "healthy-Q4_K_M.gguf", qwen_like_kv(name="Healthy"))
    write_gguf(models_dir / "odd-Q4_K_M.gguf", qwen_like_kv(name="Odd"))
    real = gguf.read_metadata

    def read(path: Path, **kwargs: object) -> gguf.GgufMetadata:
        if path.name.startswith("odd"):
            raise error
        return real(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gguf, "read_metadata", read)

    result = Scanner().scan([models_dir])

    assert [m.name for m in result.models] == ["healthy-Q4_K_M"]
    skipped = [s for s in result.skipped if s.reason == SkipReason.unreadable_header]
    assert len(skipped) == 1
    assert skipped[0].path.endswith("odd-Q4_K_M.gguf")
    assert type(error).__name__ in (skipped[0].detail or "")


def test_a_config_nested_past_the_json_limit_is_that_models_skip_not_the_scans(
    models_dir: Path, deep_json_overflows: None
) -> None:
    write_hf_model(models_dir / "healthy")
    hostile = write_hf_model(models_dir / "hostile")
    (hostile / "config.json").write_text(DEEP, encoding="utf-8")

    result = Scanner().scan([models_dir])

    by_name = {m.name: m for m in result.models}
    assert by_name["healthy"].status == ModelStatus.present
    assert by_name["hostile"].status == ModelStatus.unreadable
    assert "nested" in (by_name["hostile"].error or "")


def test_a_safetensors_header_nested_past_the_json_limit_is_refused(
    deep_json_overflows: None,
) -> None:
    raw = DEEP.encode()

    with pytest.raises(safetensors.SafetensorsError, match="nested"):
        safetensors.parse_header(raw, name="deep.safetensors")


def test_a_directory_the_scanner_could_not_read_for_any_reason_is_one_skip(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last net: a failure nobody foresaw in one folder ends that folder."""
    write_hf_model(models_dir / "healthy")
    write_hf_model(models_dir / "odd")
    real = safetensors.read_config

    def read(path: Path) -> safetensors.ModelConfig:
        if path.parent.name == "odd":
            raise ZeroDivisionError("the reader had a bug")
        return real(path)

    monkeypatch.setattr(safetensors, "read_config", read)

    result = Scanner().scan([models_dir])

    assert [m.name for m in result.models] == ["healthy"]
    skipped = [s for s in result.skipped if s.reason == SkipReason.unreadable_header]
    assert len(skipped) == 1
    assert skipped[0].path.endswith("odd")
    assert "ZeroDivisionError" in (skipped[0].detail or "")


def test_a_nested_generation_config_is_no_sampling_not_a_failed_scan(
    tmp_path: Path, deep_json_overflows: None
) -> None:
    from eugene_plexus_library.formats import generation_config

    (tmp_path / "generation_config.json").write_text(DEEP, encoding="utf-8")

    assert generation_config.read(tmp_path) is None


def test_a_nested_ollama_manifest_is_a_named_manifest_error(
    tmp_path: Path, deep_json_overflows: None
) -> None:
    from eugene_plexus_library.formats import ollama

    manifest = tmp_path / "manifest"
    manifest.write_text(DEEP, encoding="utf-8")

    with pytest.raises(ollama.ManifestError, match="nested too deeply"):
        ollama._load(manifest)
