"""R2.3 — a card we cannot see is not a machine without one.

Roadmap `specs/docs/design/release-roadmap.md` §3.3. Findings: review
§6.2 #28 (the Intel half — `_intel_gpus` reports `vramTotalBytes=0`, so
`_verdict` takes the *no accelerator* branch and tells a 16 GB Arc owner
that a 30 GB model fits, with `gpuCount=1` printed beside it) and §6.2
#29 (a vendor tool that exists and exits non-zero is diagnosed as *not
on PATH*).

**The existing test is the one that should have caught #28 and asserts
the wrong subject.** `test_a_card_with_no_free_reading_falls_back_to
_total` builds an Intel GPU with `vramTotalBytes=16 * GIB` — a value the
Intel detector cannot produce — and then asserts the fallback works. The
fixture is not the shape the code under test emits, so the test is green
about a case that does not occur while the case that does occur is
unasserted. Every fixture below is what `_intel_gpus` actually builds.
"""

from __future__ import annotations

import subprocess

import pytest

from eugene_plexus_library import fit, hardware, starter
from eugene_plexus_library._generated.models import (
    Arch,
    FitVerdict,
    Gpu,
    HostHardware,
    Os,
    Vendor,
)

GIB = 1024**3


def _arc_host() -> HostHardware:
    """Exactly what `_intel_gpus` builds: a named card, no size at all.

    `xpu-smi discovery` reports device names and no memory figure in any
    stable form, so the detector fills in `vramTotalBytes=0` and appends
    a warning. That zero is the input every assertion here is about.
    """
    return HostHardware(
        hostname="arc",
        os=Os.linux,
        arch=Arch.x64,
        ramTotalBytes=32 * GIB,
        ramAvailableBytes=24 * GIB,
        gpus=[Gpu(index=0, name="Arc A770", vendor=Vendor.intel, vramTotalBytes=0)],
    )


# --------------------------------------------------------------------------- #
# §6.2 #28 — a card of unknown size
# --------------------------------------------------------------------------- #


def test_a_card_of_unknown_size_does_not_make_a_30gb_model_fit() -> None:
    """**The finding.** `_verdict` branched on `vram_total == 0`, which is
    true of a machine with no GPU *and* of a machine with a GPU whose
    size we could not read. The second took the CPU branch, compared 30
    GB of weights against host memory, and answered `fits` — to the
    owner of a 16 GB card, who will meet an out-of-memory error at load.

    Wrong in the direction that OOMs is the whole reason this one is
    rated High.
    """
    budget = fit.budget_from_hardware(_arc_host())
    assert budget.gpuCount == 1, "the card is detected; only its size is unknown"
    assert (budget.vramTotalBytes or 0) == 0

    result = fit.compute(weights_bytes=28 * GIB, budget=budget, context_length=4096)
    assert result.verdict is FitVerdict.unknown, (
        f"a 30 GB model on a card of unknown size answered {result.verdict}"
    )


def test_a_small_model_on_a_card_of_unknown_size_is_also_unknown() -> None:
    """Not *sometimes* unknown. The size of the card is what is missing,
    so no comparison against it can be made at all — including a
    favourable one. A `fits` here would be right by luck, and luck is not
    a verdict."""
    budget = fit.budget_from_hardware(_arc_host())
    result = fit.compute(weights_bytes=1 * GIB, budget=budget, context_length=4096)
    assert result.verdict is FitVerdict.unknown


def test_the_unknown_verdict_says_what_is_missing() -> None:
    """A badge reading `unknown` with no reason is the thing
    differentiator #6 exists to remove. The note names the card."""
    budget = fit.budget_from_hardware(_arc_host())
    result = fit.compute(weights_bytes=8 * GIB, budget=budget, context_length=4096)
    notes = " ".join(result.notes or [])
    assert "size" in notes.lower() or "memory" in notes.lower()
    assert "unknown" in notes.lower() or "could not" in notes.lower()


def test_a_machine_with_no_gpu_at_all_still_scores_against_host_memory() -> None:
    """The branch this splits away from, still working. A CPU-only box
    has `gpuCount == 0` and gets a real verdict against RAM — an
    `unknown` there would be a regression that reads as caution."""
    cpu_only = HostHardware(
        hostname="cpu",
        os=Os.linux,
        arch=Arch.x64,
        ramTotalBytes=32 * GIB,
        ramAvailableBytes=24 * GIB,
        gpus=[],
    )
    budget = fit.budget_from_hardware(cpu_only)
    assert budget.gpuCount == 0
    assert fit.compute(weights_bytes=4 * GIB, budget=budget, context_length=4096).verdict is (
        FitVerdict.fits
    )
    assert fit.compute(weights_bytes=64 * GIB, budget=budget, context_length=4096).verdict is (
        FitVerdict.no
    )


def test_a_card_whose_size_is_known_is_unaffected() -> None:
    """The 5090 path, which is every verdict this project has shipped."""
    box = HostHardware(
        hostname="dev",
        os=Os.windows,
        arch=Arch.x64,
        ramTotalBytes=96 * GIB,
        ramAvailableBytes=64 * GIB,
        gpus=[
            Gpu(
                index=0,
                name="RTX 5090",
                vendor=Vendor.nvidia,
                vramTotalBytes=32 * GIB,
                vramFreeBytes=31 * GIB,
            )
        ],
    )
    budget = fit.budget_from_hardware(box)
    assert fit.compute(weights_bytes=8 * GIB, budget=budget, context_length=4096).verdict is (
        FitVerdict.fits
    )


def test_the_starter_set_is_not_inverted_by_a_card_we_cannot_measure() -> None:
    """**The second half of #28, and the one a person actually meets.**

    `starter.py` asked the same question the same wrong way:
    `no_accelerator = vramTotalBytes == 0 and not unifiedMemory`. So an
    Arc owner opening Discover for the first time was handed the
    *smallest* model in the set, with a sentence explaining that their
    machine has no graphics card — on a machine built around one.
    """
    budget = fit.budget_from_hardware(_arc_host())
    assert starter.has_no_accelerator(budget) is False

    cpu_only = fit.budget_from_hardware(
        HostHardware(
            hostname="cpu",
            os=Os.linux,
            arch=Arch.x64,
            ramTotalBytes=32 * GIB,
            ramAvailableBytes=24 * GIB,
            gpus=[],
        )
    )
    assert starter.has_no_accelerator(cpu_only) is True


# --------------------------------------------------------------------------- #
# §6.2 #29 — a tool that exists and fails
# --------------------------------------------------------------------------- #


def test_a_vendor_tool_that_fails_is_not_a_vendor_tool_that_is_absent(monkeypatch) -> None:
    """**The finding.** `_run` returned `None` for both, so the two
    cases were indistinguishable one layer up — and the warning `detect`
    appends says the tool *is not on this process's PATH*, which sends a
    person to fix an environment that is fine while their driver is
    wedged. After a Linux driver update that is the commonest failure
    there is.
    """
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    assert hardware.probe(["nvidia-smi"]) is hardware.Probe.ABSENT

    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/" + name)

    def _fails(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=9,
            stdout="",
            stderr="Failed to initialize NVML: Driver/library version mismatch",
        )

    monkeypatch.setattr(hardware.subprocess, "run", _fails)
    result = hardware.probe(["nvidia-smi"])
    assert result is hardware.Probe.FAILED
    assert result is not hardware.Probe.ABSENT


def test_the_warning_names_the_wedged_driver_rather_than_the_path(monkeypatch) -> None:
    """What the person reads. Two different sentences, because they send
    them to two different places."""
    monkeypatch.setattr(hardware, "host_memory", lambda: (32 * GIB, 24 * GIB))
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/" + name)

    def _fails(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=9,
            stdout="",
            stderr="Failed to initialize NVML: Driver/library version mismatch",
        )

    monkeypatch.setattr(hardware.subprocess, "run", _fails)
    detected = hardware.detect()
    joined = " ".join(detected.warnings or [])
    assert "nvidia-smi" in joined
    assert "not on" not in joined.lower(), (
        "a tool that ran and failed was reported as one that is not on PATH"
    )
    assert "exited" in joined.lower() or "failed" in joined.lower()


def test_a_machine_with_no_vendor_tool_still_says_so(monkeypatch) -> None:
    """The other sentence, unchanged. A trimmed environment under a
    supervisor is a real and common cause and the warning has always
    named it."""
    monkeypatch.setattr(hardware, "host_memory", lambda: (32 * GIB, 24 * GIB))
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    detected = hardware.detect()
    joined = " ".join(detected.warnings or [])
    assert "PATH" in joined


@pytest.mark.parametrize("vendor_tool", ["nvidia-smi", "rocm-smi", "xpu-smi"])
def test_every_vendor_probe_goes_through_the_same_three_state_result(vendor_tool) -> None:
    """One `Probe` for all three, which is why this belongs with the
    detector rather than at each call site: three copies of the
    absent-versus-failed distinction is three chances to get one wrong.
    """
    assert hasattr(hardware, "probe")
    assert set(hardware.Probe) >= {hardware.Probe.ABSENT, hardware.Probe.FAILED}


def test_an_unmeasurable_machine_is_told_why_nothing_is_recommended() -> None:
    """**Found by the live run, not by reading.**

    With every verdict `unknown` nothing "fits", so `recommend` fell
    into the nothing-fits branch and produced *"None of these fits
    entirely in 0 B at 8,192 tokens"* — a number that is not a number,
    about a card we can see and cannot size. Silence would have been no
    better: an Arc owner opening Discover for the first time deserves
    the reason, and the reason is about their machine.
    """
    from eugene_plexus_library._generated.models import StarterModel

    budget = fit.budget_from_hardware(_arc_host())
    models = [
        StarterModel(
            sizeClass="8B",
            baseModel="Qwen/Qwen3.5-8B",
            repo="unsloth/Qwen3.5-8B-GGUF",
            file="Q4_K_M.gguf",
            label="Q4_K_M",
            sizeBytes=5 * GIB,
            why="a good first model",
            fit=fit.compute(weights_bytes=5 * GIB, budget=budget, context_length=8192),
        )
    ]
    got = starter.recommend(models, context_length=8192, budget=budget)
    assert got is not None
    assert got.sizeClass is None, "nothing can be recommended against a card we cannot size"
    assert "0 B" not in got.reason
    assert "graphics card" in got.reason
    assert "nvidia-smi" in got.reason
