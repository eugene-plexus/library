"""Opt-in: scan the real models on this machine.

Synthetic fixtures agree with whatever the reader believes. Real files
disagree when it is wrong — which is how the two defects in the first
version of the scanner were found (a model named after a HuggingFace
snapshot hash, and 42 skip records of dataset-cache noise), neither of
which any fixture would have surfaced.

Enable by pointing it at directories that hold models:

    EUGENE_PLEXUS_LIBRARY_LIVE_ROOTS="C:/Users/me/.lmstudio/models" pytest

Multiple roots separate with `os.pathsep`. Skipped when unset, so CI —
which has no models — stays green.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from eugene_plexus_library._generated.models import ModelFormat, ModelStatus
from eugene_plexus_library.scanner import Scanner

LIVE_ROOTS_ENV = "EUGENE_PLEXUS_LIBRARY_LIVE_ROOTS"


def _roots() -> list[Path]:
    raw = os.environ.get(LIVE_ROOTS_ENV, "").strip()
    if not raw:
        return []
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


pytestmark = pytest.mark.skipif(
    not _roots(), reason=f"set {LIVE_ROOTS_ENV} to scan real model directories"
)


@pytest.fixture(scope="module")
def live_scan():  # type: ignore[no-untyped-def]
    roots = [r for r in _roots() if r.exists()]
    assert roots, f"none of the {LIVE_ROOTS_ENV} paths exist"
    started = time.perf_counter()
    result = Scanner().scan(roots)
    result_elapsed = time.perf_counter() - started
    print(
        f"\nscanned {len(result.models)} models from {result.files_scanned} files "
        f"in {result_elapsed:.2f}s"
    )
    return result


def test_it_finds_something(live_scan) -> None:  # type: ignore[no-untyped-def]
    assert live_scan.models, "no models found — check the configured roots"


def test_every_entry_is_coherent(live_scan) -> None:  # type: ignore[no-untyped-def]
    for model in live_scan.models:
        assert model.id and len(model.id) == 16
        assert Path(model.path).exists(), f"{model.path} does not exist"
        assert model.name, f"{model.path} produced an empty name"
        if model.status == ModelStatus.present:
            assert model.sizeBytes and model.sizeBytes > 0


def test_no_entry_is_named_after_a_hash(live_scan) -> None:  # type: ignore[no-untyped-def]
    """A HuggingFace snapshot directory is named after the revision, so
    a model in one reads as `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
    unless the repo name is used instead. Found this way, not by a
    fixture."""
    for model in live_scan.models:
        name = model.name
        assert not (len(name) == 40 and all(c in "0123456789abcdef" for c in name)), (
            f"{model.path} is named after a revision hash"
        )


def test_no_projector_became_a_model(live_scan) -> None:  # type: ignore[no-untyped-def]
    """~1 GB of phantom entry each when this goes wrong."""
    for model in live_scan.models:
        assert model.architecture != "clip", f"{model.path} is a projector"
        assert not Path(model.path).name.startswith("mmproj-")


def test_gguf_entries_report_a_quant(live_scan) -> None:  # type: ignore[no-untyped-def]
    for model in live_scan.models:
        if model.format is not ModelFormat.gguf or model.status is not ModelStatus.present:
            continue
        assert model.gguf is not None
        # Either the metadata knew it or the filename did. Both being
        # absent means the reader learned nothing useful about the file.
        assert model.gguf.quantization or model.gguf.fileType is not None, (
            f"{model.path} produced no quant information at all"
        )


def test_safetensors_entries_report_exact_parameters(live_scan) -> None:  # type: ignore[no-untyped-def]
    for model in live_scan.models:
        if model.format is not ModelFormat.safetensors or model.status is not ModelStatus.present:
            continue
        assert model.parameters and model.parameters > 0


def test_no_duplicate_ids(live_scan) -> None:  # type: ignore[no-untyped-def]
    ids = [m.id for m in live_scan.models]
    assert len(ids) == len(set(ids))


def test_every_skip_has_a_reason_and_a_detail(live_scan) -> None:  # type: ignore[no-untyped-def]
    """The skip list is the answer to "why isn't my model showing up".
    A reason with no detail does not answer it."""
    for skipped in live_scan.skipped:
        assert skipped.reason is not None
        assert skipped.detail, f"{skipped.path} was skipped with no explanation"
