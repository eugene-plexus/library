"""R1.2: a download writes inside a Library folder, or it does not write.

Roadmap `docs/design/release-roadmap.md` §2.2, finding review §6.2 #14.

**This is calling #3 — "the user's model files stay theirs" — which is
the non-negotiable one.** A rule the product does not enforce is a
slogan, and this one was unenforced on the field beside the one that
enforces it: `subdirectory` was `is_within`-checked in
`resolve_destination` and `filename` was joined onto the resolved
directory two functions later with nothing in between. The adjacent
field's own traversal test sat right there, which is what makes the gap
sharp rather than obscure.

Operator-gated, so this is post-authentication and never
network-reachable. It is in R1 for the invariant, not for an exposure.

**Refused, never sanitised.** A silently renamed file is worse than a
400: the operator asked for a name, got a different one, and finds out
when a runtime points at a path that is not there — or does not find
out, and keeps a library whose names no longer say what is in them. The
code is `PathTraversal`, the one `resolve_destination` already uses for
the sibling field, so the UI needs no new case.

**And the check is on the resolved path of every file, not on the
shape of one field.** A rule written as "reject `..`" is a rule about a
spelling; this one is about where the bytes land, and covers the
upstream repo path by the same line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eugene_plexus_library import downloads


def test_a_traversing_filename_is_refused(tmp_path: Path) -> None:
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_file(
            directory=tmp_path / "unsloth" / "Qwen3-8B-GGUF",
            root=tmp_path,
            name="../../../elsewhere.gguf",
        )
    assert raised.value.code == "PathTraversal"


def test_an_absolute_filename_is_refused(tmp_path: Path) -> None:
    """`Path("/models") / "/etc/x"` is `/etc/x`: an absolute component
    discards everything to its left, so this needs no `..` at all and
    would survive any check written against that spelling."""
    absolute = str(Path(tmp_path.anchor or "/") / "somewhere" / "else.gguf")
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_file(directory=tmp_path / "a", root=tmp_path, name=absolute)
    assert raised.value.code == "PathTraversal"


@pytest.mark.skipif(Path("C:/").anchor != "C:\\", reason="Windows drive letters")
def test_a_filename_on_another_drive_is_refused(tmp_path: Path) -> None:
    """The Windows shape of the same thing, which is the platform both
    installers ship to first."""
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_file(directory=tmp_path / "a", root=tmp_path, name="Z:/models/x.gguf")
    assert raised.value.code == "PathTraversal"


def test_a_filename_that_climbs_back_inside_is_allowed(tmp_path: Path) -> None:
    """The subject is where the bytes land, not how the name is spelled.
    A name that resolves back under the root is fine, and a check that
    refused it would be a check about `..`."""
    directory = tmp_path / "unsloth" / "Qwen3-8B-GGUF"
    assert downloads.resolve_file(
        directory=directory, root=tmp_path, name="../Qwen3-8B-GGUF/model.gguf"
    ) == (directory / "model.gguf")


def test_an_ordinary_rename_still_works(tmp_path: Path) -> None:
    """`filename` exists so one file can be renamed on the way down.
    Breaking that would be a worse bug than the one being fixed."""
    directory = tmp_path / "unsloth" / "Qwen3-8B-GGUF"
    assert downloads.resolve_file(directory=directory, root=tmp_path, name="q4.gguf") == (
        directory / "q4.gguf"
    )


def test_a_subdirectory_inside_the_name_still_works(tmp_path: Path) -> None:
    """A multi-file repo's own layout arrives this way — `mmproj/x.gguf`
    is a relative path, not a traversal."""
    directory = tmp_path / "repo"
    assert downloads.resolve_file(directory=directory, root=tmp_path, name="mmproj/x.gguf") == (
        directory / "mmproj" / "x.gguf"
    )


def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    """`directory / ""` is the directory, so this would queue a transfer
    onto the folder itself and fail deep inside the writer."""
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_file(directory=tmp_path / "a", root=tmp_path, name="")
    assert raised.value.code == "PathTraversal"


async def test_the_manager_refuses_a_spec_that_would_leave_the_root(tmp_path: Path) -> None:
    """The route's own path, not just the helper.

    `resolve_file` being right proves nothing about `start()` calling
    it -- that is the shape this project has shipped twice (a pure
    helper asserted instead of its caller), so the reproduction drives
    the manager the API actually uses.
    """
    from eugene_plexus_library._generated.models import DownloadSpec

    from .test_downloads import FakeHub, build_manager

    manager, _ = build_manager(tmp_path, FakeHub())
    with pytest.raises(downloads.DownloadError) as raised:
        await manager.start(
            # Deep enough to leave the root, not merely the repo
            # folder: the first draft of this check climbed two levels
            # from `<root>/org/repo` and landed back inside `<root>`,
            # where nothing is wrong and nothing should be refused.
            DownloadSpec(repo="org/repo", files=["model.gguf"], filename="../../../../escaped.gguf")
        )
    assert raised.value.code == "PathTraversal"
    # And nothing was queued: a refusal that still left a record would
    # leave a job pointing outside every Library folder.
    assert manager.records() == []
