"""The content fingerprint: proof that two files are the same file (row 22).

Every file here is a real GGUF or safetensors file on disk, read back by the
real scanner, so a fingerprint is checked against what a scan produces and
not against a model built by hand.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from eugene_plexus_library import fingerprint
from eugene_plexus_library._generated.models import LibraryModel
from eugene_plexus_library.scanner import Scanner

from .conftest import qwen_like_kv, write_gguf, write_hf_model, write_safetensors

# Larger than the 1.56 MiB under which a file is read whole.
BIG = 3 * 1024 * 1024
# Large enough that the head, the eight blocks and the tail are apart.
HUGE = 24 * 1024 * 1024


def _scan(root: Path) -> list[LibraryModel]:
    return Scanner().scan([root]).models


def _only(root: Path) -> LibraryModel:
    models = _scan(root)
    assert len(models) == 1, models
    return models[0]


def _flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


def _big_gguf(path: Path, size: int = BIG) -> Path:
    return write_gguf(path, qwen_like_kv(), payload=bytes(size))


def _block_offset(size: int, index: int) -> int:
    """Where sampled block `index` (1..8) starts: the contract says evenly
    spaced, so a test names the spacing rather than the code's arithmetic."""
    return (size - fingerprint.BLOCK_BYTES) * index // (fingerprint.BLOCK_COUNT + 1)


def test_a_fingerprint_names_its_version(tmp_path: Path) -> None:
    _big_gguf(tmp_path / "a" / "m.gguf")

    found = fingerprint.of(_only(tmp_path / "a"))

    assert found is not None and found.startswith("v1:")


def test_a_copy_at_another_path_has_the_same_fingerprint(tmp_path: Path) -> None:
    """It names no path and no date: that is the whole point of it."""
    first = _big_gguf(tmp_path / "a" / "m.gguf")
    other = tmp_path / "b" / "deeper" / "renamed.gguf"
    other.parent.mkdir(parents=True)
    shutil.copy2(first, other)
    (tmp_path / "b" / "deeper" / "renamed.gguf").touch()  # a new mtime

    assert fingerprint.of(_only(tmp_path / "a")) == fingerprint.of(_only(tmp_path / "b"))


@pytest.mark.parametrize(
    "where",
    ["head", *[f"block{i}" for i in range(1, 9)], "tail"],
)
def test_a_changed_byte_in_every_sampled_region_changes_it(tmp_path: Path, where: str) -> None:
    path = _big_gguf(tmp_path / "a" / "m.gguf", HUGE)
    model = _only(tmp_path / "a")
    before = fingerprint.of(model)
    size = path.stat().st_size
    offset = {
        "head": fingerprint.HEAD_BYTES - 1,
        "tail": size - 1,
    }.get(where) or _block_offset(size, int(where.removeprefix("block"))) + 7

    _flip(path, offset)

    assert fingerprint.of(model) != before


def test_a_byte_between_the_samples_is_not_seen(tmp_path: Path) -> None:
    """The limit, written down: the fingerprint is sampled, so two files agree
    when their sizes, headers and samples agree. Eugene says "same" only for
    that, and the person is the judge of the rest."""
    path = _big_gguf(tmp_path / "a" / "m.gguf", HUGE)
    model = _only(tmp_path / "a")
    before = fingerprint.of(model)
    size = path.stat().st_size
    gap = fingerprint.HEAD_BYTES + 100

    assert gap < _block_offset(size, 1)
    _flip(path, gap)

    assert fingerprint.of(model) == before


def test_a_small_file_is_read_whole(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "a" / "m.gguf", qwen_like_kv(), payload=bytes(100_000))
    model = _only(tmp_path / "a")
    before = fingerprint.of(model)

    _flip(path, path.stat().st_size // 2)

    assert fingerprint.of(model) != before


def test_a_different_size_is_a_different_fingerprint(tmp_path: Path) -> None:
    write_gguf(tmp_path / "a" / "m.gguf", qwen_like_kv(), payload=bytes(BIG))
    write_gguf(tmp_path / "b" / "m.gguf", qwen_like_kv(), payload=bytes(BIG + 1))

    assert fingerprint.of(_only(tmp_path / "a")) != fingerprint.of(_only(tmp_path / "b"))


def test_no_more_than_about_one_and_a_half_mib_is_read_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never the whole of a 40 GB model."""
    write_gguf(tmp_path / "a" / "m.gguf", qwen_like_kv(), payload=bytes(30 * 1024 * 1024))
    model = _only(tmp_path / "a")
    read: list[int] = []
    real = fingerprint._sample

    def counted(handle, offset, length):  # type: ignore[no-untyped-def]
        data = real(handle, offset, length)
        read.append(len(data))
        return data

    monkeypatch.setattr(fingerprint, "_sample", counted)

    assert fingerprint.of(model) is not None
    assert sum(read) <= fingerprint.WHOLE_FILE_BYTES


def test_a_safetensors_header_is_read_whole(tmp_path: Path) -> None:
    """A header can be longer than the first 1 MiB, and it is what says what
    the tensors are: a change in it, past the first MiB and clear of every
    block, is a different model."""
    tensors = {f"layer.{i}.weight": ("F32", [1]) for i in range(24_000)}
    tensors["big"] = ("F32", [4 * 1024 * 1024])
    path = write_safetensors(tmp_path / "a" / "model.safetensors", tensors)
    (tmp_path / "a" / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}', encoding="utf-8"
    )
    header_end = 8 + int.from_bytes(path.read_bytes()[:8], "little")
    assert header_end > fingerprint.HEAD_BYTES + 100_000
    assert header_end < _block_offset(path.stat().st_size, 1)
    model = _only(tmp_path / "a")
    before = fingerprint.of(model)

    # A digit in the header's JSON, past the first MiB: still valid JSON.
    data = bytearray(path.read_bytes())
    at = next(
        i for i in range(fingerprint.HEAD_BYTES + 50_000, header_end) if data[i : i + 1].isdigit()
    )
    data[at] = ord("7") if data[at] != ord("7") else ord("8")
    path.write_bytes(bytes(data))

    assert fingerprint.of(model) != before


def test_every_file_of_a_directory_model_counts(tmp_path: Path) -> None:
    write_hf_model(tmp_path / "a" / "m")
    shutil.copytree(tmp_path / "a" / "m", tmp_path / "b" / "m")
    assert fingerprint.of(_only(tmp_path / "a")) == fingerprint.of(_only(tmp_path / "b"))

    (tmp_path / "b" / "m" / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")

    assert fingerprint.of(_only(tmp_path / "a")) != fingerprint.of(_only(tmp_path / "b"))


def test_a_file_beside_the_model_that_is_not_part_of_it_does_not_count(tmp_path: Path) -> None:
    """A copy of the model need not carry a README or a licence."""
    write_hf_model(tmp_path / "a" / "m")
    shutil.copytree(tmp_path / "a" / "m", tmp_path / "b" / "m")
    (tmp_path / "b" / "m" / "README.md").write_text("notes", encoding="utf-8")

    assert fingerprint.of(_only(tmp_path / "a")) == fingerprint.of(_only(tmp_path / "b"))


def test_a_file_that_is_gone_says_so(tmp_path: Path) -> None:
    path = _big_gguf(tmp_path / "a" / "m.gguf")
    model = _only(tmp_path / "a")
    path.unlink()

    with pytest.raises(fingerprint.FingerprintError) as raised:
        fingerprint.of(model)

    assert raised.value.absent is True
    assert raised.value.path == str(path)


def test_a_file_that_cannot_be_read_gives_no_fingerprint_and_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _big_gguf(tmp_path / "a" / "m.gguf")
    model = _only(tmp_path / "a")
    path.unlink()

    with caplog.at_level(logging.WARNING, logger=fingerprint.log.name):
        stamped = fingerprint.stamp(model)

    assert stamped.fingerprint is None
    assert "no fingerprint" in caplog.text


def test_stamping_leaves_a_stamped_model_alone(tmp_path: Path) -> None:
    """A rescan reads nothing for an entry that has its fingerprint."""
    path = _big_gguf(tmp_path / "a" / "m.gguf")
    model = _only(tmp_path / "a").model_copy(update={"fingerprint": "v1:kept"})
    path.unlink()  # a read would now fail and drop it

    assert fingerprint.stamp_all([model])[0].fingerprint == "v1:kept"


def test_stamping_stops_between_models_when_cancelled(tmp_path: Path) -> None:
    _big_gguf(tmp_path / "a" / "one.gguf")
    _big_gguf(tmp_path / "a" / "two.gguf")
    models = _scan(tmp_path / "a")
    calls: list[int] = []

    def cancel() -> bool:
        calls.append(1)
        return len(calls) > 1

    stamped = fingerprint.stamp_all(models, should_cancel=cancel)

    assert [m.fingerprint is not None for m in stamped] == [True, False]


def test_the_first_difference_is_named(tmp_path: Path) -> None:
    write_gguf(tmp_path / "a" / "m.gguf", qwen_like_kv(), payload=bytes(BIG))
    write_gguf(tmp_path / "b" / "m.gguf", qwen_like_kv(), payload=bytes(BIG + 1000))
    write_gguf(tmp_path / "c" / "m.gguf", qwen_like_kv(context_length=8192), payload=bytes(BIG))
    write_gguf(tmp_path / "d" / "m.gguf", qwen_like_kv(), payload=bytes(BIG))
    a, b, c, d = (_only(tmp_path / n) for n in "abcd")

    sizes = fingerprint.difference(a, b) or ""
    assert "weights file" in sizes
    assert f"{b.sizeBytes:,}" in sizes and f"{a.sizeBytes:,}" in sizes
    # Same size, so the sizes cannot tell: the details can, and then nothing can.
    assert "context length" in (fingerprint.difference(a, c) or "")
    assert fingerprint.difference(a, d) is None
