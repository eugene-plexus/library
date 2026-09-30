"""A4: the library's Apple budget is Metal's working set, not 75% of RAM.

Measured 2026-09-30 on GitHub's macOS runners (docs/acceptance/
a4-macos-runner-run.md in specs): Metal's `recommendedMaxWorkingSetSize`
is two thirds of RAM there (5,010,800,640 of 7,516,192,768 bytes), exactly
MLX's own figure, while the library scored every fit against
`0.75 * hw.memsize` and so called a model that fits one Metal will not
hold. The same fix is in the agent; `gpu_probe.py` is the same file in both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eugene_plexus_library import gpu_probe, hardware
from eugene_plexus_library._generated.models import Vendor

RAM = 7_516_192_768
WORKING_SET = 5_010_800_640


def test_the_budget_is_metals_working_set() -> None:
    warnings: list[str] = []
    metal = gpu_probe.MetalDevice(
        name="Apple M2 Pro", working_set_bytes=WORKING_SET, unified_memory=True
    )
    [gpu] = hardware._apple_gpu(RAM, warnings, metal=lambda: metal)
    assert gpu.vramTotalBytes == WORKING_SET
    assert gpu.vramTotalBytes != int(RAM * 0.75)
    assert gpu.vendor is Vendor.apple
    assert gpu.name == "Apple M2 Pro"
    assert gpu.vramFreeBytes is None
    assert warnings == []


def test_with_no_metal_it_falls_back_to_two_thirds_and_says_so(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(hardware, "_apple_chip", lambda: "Apple M1 (Virtual)")
    warnings: list[str] = []
    [gpu] = hardware._apple_gpu(RAM, warnings, metal=lambda: None)
    assert gpu.vramTotalBytes == int(RAM * 2 / 3)
    assert gpu.name == "Apple M1 (Virtual)"
    assert any("Metal could not be asked" in w for w in warnings), warnings


def test_metal_is_not_asked_off_a_mac() -> None:
    if sys.platform == "darwin":
        pytest.skip("this is the off-Mac half")
    assert gpu_probe.metal_device() is None


def test_gpu_probe_is_the_agents_file_byte_for_byte() -> None:
    agent = (
        Path(__file__).resolve().parents[2] / "agent/src/eugene_plexus_agent/engines/gpu_probe.py"
    )
    if not agent.is_file():
        pytest.skip("no agent checkout beside this one")
    ours = Path(gpu_probe.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert ours == agent.read_bytes().replace(b"\r\n", b"\n")
