"""Path normalization and the id derived from it.

Getting this wrong twice in two different ways produces two library
entries for one file, each holding half the operator's profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eugene_plexus_library.paths import ID_LENGTH, is_within, model_id, normalize, same_path


def test_the_id_is_stable_and_url_safe(tmp_path: Path) -> None:
    path = tmp_path / "some model.gguf"
    identifier = model_id(path)

    assert len(identifier) == ID_LENGTH
    assert identifier.isalnum()
    assert model_id(path) == identifier


def test_the_id_never_reads_the_file(tmp_path: Path) -> None:
    """Not content addressing. Hashing 40 GB on every scan is not on the
    table, and a content-derived identity would change when nothing the
    operator did changed."""
    path = tmp_path / "model.gguf"
    path.write_bytes(b"first contents")
    before = model_id(path)
    path.write_bytes(b"completely different contents")

    assert model_id(path) == before


def test_the_id_is_computable_for_a_path_that_does_not_exist(tmp_path: Path) -> None:
    """A `missing` entry keeps its id after its file is gone — that is
    what keeps its profiles reachable."""
    assert model_id(tmp_path / "never-existed.gguf")


def test_relative_segments_collapse(tmp_path: Path) -> None:
    direct = tmp_path / "sub" / "m.gguf"
    roundabout = tmp_path / "sub" / ".." / "sub" / "m.gguf"

    assert same_path(direct, roundabout)
    assert model_id(direct) == model_id(roundabout)


def test_a_trailing_separator_does_not_change_identity(tmp_path: Path) -> None:
    directory = tmp_path / "model-dir"
    assert same_path(directory, Path(f"{directory}{'/'}"))


def test_forward_and_back_slashes_agree_on_windows(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("separator equivalence is a Windows concern")
    assert same_path(str(tmp_path / "a" / "b"), f"{tmp_path}/a/b")


def test_case_folding_matches_the_platform(tmp_path: Path) -> None:
    """Two spellings name one file on Windows and macOS, two files on
    Linux. Folding on the wrong platform is a split-profiles bug either
    way."""
    lower = tmp_path / "model.gguf"
    upper = tmp_path / "MODEL.GGUF"

    if sys.platform in ("win32", "darwin"):
        assert same_path(lower, upper)
    else:
        assert not same_path(lower, upper)


def test_symlinks_are_not_resolved(tmp_path: Path) -> None:
    """Resolving them turns a HuggingFace snapshot into `blobs/<sha>` —
    content addressing arriving through the back door."""
    target = tmp_path / "real.gguf"
    target.write_bytes(b"x")
    link = tmp_path / "link.gguf"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/user cannot create symlinks")

    assert not same_path(link, target)
    assert model_id(link) != model_id(target)


def test_is_within_does_not_match_on_a_string_prefix(tmp_path: Path) -> None:
    """`/models2` does not live inside `/models`."""
    root = tmp_path / "models"
    sibling = tmp_path / "models2"
    root.mkdir()
    sibling.mkdir()

    assert is_within(root / "a" / "b.gguf", root)
    assert not is_within(sibling / "b.gguf", root)


def test_normalize_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert Path(normalize("relative.gguf")).is_absolute()
