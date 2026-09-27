"""An Intel Arc mini PC read "no GPU" on alpha.3 (2026-09-26).

The library's hardware reading asked only `nvidia-smi`, `rocm-smi` and
`xpu-smi`, none of which a Windows machine with an Arc or a Radeon has
unless its owner installed an SDK. So Discover, the starter set and every
fit said "no accelerator was detected" about a machine whose Vulkan build
was using its GPU, and scored against system RAM.

`gpu_probe` asks the operating system, and is the agent's module copied:
the agent's copy carries the tests of the probe itself (DXCore memory,
sysfs, which adapters a build uses). These are about what the library
does with the answer, and about `unifiedMemory` on the fit routes, which
is how a console scores another host's integrated GPU as the one pool it
is rather than as a card beside RAM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library import gpu_probe, hardware
from eugene_plexus_library._generated.models import FitVerdict, Vendor

from .conftest import qwen_like_kv, write_gguf

GIB = 1024**3
MIB = 1024**2

ARC_IGPU = gpu_probe.Adapter(
    name="Intel(R) Arc(TM) Graphics",
    vendor=gpu_probe.INTEL,
    integrated=True,
    dedicated_bytes=128 * MIB,
    shared_bytes=16 * GIB,
    dedicated_used_bytes=16 * MIB,
    shared_used_bytes=1 * GIB,
)
RTX_5090 = gpu_probe.Adapter(
    name="NVIDIA GeForce RTX 5090",
    vendor=gpu_probe.NVIDIA,
    integrated=False,
    dedicated_bytes=31 * GIB,
    shared_bytes=74 * GIB,
    dedicated_used_bytes=2 * GIB,
    shared_used_bytes=0,
)
RADEON_IGPU = gpu_probe.Adapter(
    name="AMD Radeon(TM) Graphics",
    vendor=gpu_probe.AMD,
    integrated=True,
    dedicated_bytes=2 * GIB,
    shared_bytes=74 * GIB,
    dedicated_used_bytes=0,
    shared_used_bytes=0,
)


@pytest.fixture
def windows_with(monkeypatch: pytest.MonkeyPatch):
    """A Windows x64 box with no vendor tool, 32 GiB of RAM and 26 free."""
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware, "host_memory", lambda: (32 * GIB, 26 * GIB))
    monkeypatch.setattr(hardware, "_rocm_installed", lambda os_kind: False)
    monkeypatch.setattr(gpu_probe, "vulkan_loader_present", lambda os_name: True)

    def _set(adapters: list[gpu_probe.Adapter]) -> None:
        monkeypatch.setattr(gpu_probe, "adapters", lambda os_name=None: list(adapters))

    return _set


def test_an_arc_mini_pc_has_a_gpu_and_it_shares_host_memory(windows_with) -> None:
    """**The report.** "no GPU" about a machine running the Vulkan build."""
    windows_with([ARC_IGPU])
    detected = hardware.detect()
    [arc] = detected.gpus or []
    assert (arc.name, arc.vendor) == ("Intel(R) Arc(TM) Graphics", Vendor.intel)
    assert arc.vramTotalBytes == 128 * MIB + 16 * GIB
    assert arc.vramFreeBytes == (128 - 16) * MIB + 15 * GIB
    assert detected.unifiedMemory is True
    assert not any("no accelerator" in w for w in detected.warnings or [])


def test_the_card_is_the_budget_not_the_integrated_gpu_beside_it(windows_with) -> None:
    """The development box with `nvidia-smi` out of the way: a 5090 and a
    Radeon whose 74 GiB shared allowance would outrank it."""
    windows_with([RADEON_IGPU, RTX_5090])
    detected = hardware.detect()
    assert [g.name for g in detected.gpus or []] == ["NVIDIA GeForce RTX 5090"]
    assert detected.unifiedMemory is False


def test_an_os_that_could_not_be_asked_is_said(windows_with, monkeypatch) -> None:
    def _broken(os_name=None):
        raise gpu_probe.GpuProbeError("DXCore is not on this machine")

    monkeypatch.setattr(gpu_probe, "adapters", _broken)
    detected = hardware.detect()
    assert detected.gpus == []
    joined = " ".join(detected.warnings or [])
    assert "DXCore is not on this machine" in joined and "no accelerator" in joined


def test_a_card_of_unreadable_size_is_unknown_not_a_cpu(windows_with) -> None:
    """R2.3's rule, reached by the new path: a GPU whose size is not known
    is not a machine without one."""
    intel_linux = gpu_probe.Adapter("Intel GPU (0000:03:00.0)", gpu_probe.INTEL, False, None, None)
    windows_with([intel_linux])
    detected = hardware.detect()
    [gpu] = detected.gpus or []
    assert gpu.vramTotalBytes == 0
    budget = fit_mod.budget_from_hardware(detected)
    assert fit_mod.card_of_unknown_size(budget)


# --------------------------------------------------------------------------- #
# `unifiedMemory` on the fit routes
# --------------------------------------------------------------------------- #


def _hardware(**overrides: object) -> hardware.HostHardware:
    body: dict[str, object] = {
        "hostname": "h",
        "os": "linux",
        "arch": "x64",
        "ramTotalBytes": 32 * GIB,
        "ramAvailableBytes": 26 * GIB,
        "unifiedMemory": False,
        "gpus": [],
    }
    body.update(overrides)
    return hardware.HostHardware.model_validate(body)


def test_a_caller_can_say_the_budget_it_passes_is_one_shared_pool() -> None:
    budget = fit_mod.budget_from_hardware(
        _hardware(), vram_override=15 * GIB, ram_override=26 * GIB, unified_override=True
    )
    assert budget.unifiedMemory is True
    assert budget.source == fit_mod.Source.override


def test_without_it_the_default_is_the_detected_one() -> None:
    assert fit_mod.budget_from_hardware(_hardware(unifiedMemory=True)).unifiedMemory is True
    assert fit_mod.budget_from_hardware(_hardware()).unifiedMemory is False


def _scan_one(client: TestClient, models_dir: Path) -> str:
    write_gguf(models_dir / "m.gguf", qwen_like_kv(vocab=64))
    client.post("/v1/scan")
    for _ in range(200):
        if client.get("/v1/scan").json()["state"] != "scanning":
            break
    return str(client.get("/v1/models").json()["models"][0]["id"])


def test_a_shared_pool_is_not_scored_as_spilling_into_itself(
    configured_client: TestClient, models_dir: Path
) -> None:
    """**The case the parameter exists for.** A model larger than an
    integrated GPU's pool, scored against that pool passed as `vramBytes`:
    read as a card, the same RAM is counted again as spill and the verdict
    is `split`, which a launch reads as "fits with partial offload"."""
    model_id = _scan_one(configured_client, models_dir)
    url = f"/v1/models/{model_id}/fit"
    base = configured_client.get(url, params={"contextLength": 4096}).json()
    required = base["fit"]["requiredBytes"]
    params = {"contextLength": 4096, "vramBytes": required // 2, "ramBytes": 64 * GIB}

    as_a_card = configured_client.get(url, params=params).json()
    assert as_a_card["fit"]["verdict"] == FitVerdict.split.value

    shared = configured_client.get(url, params={**params, "unifiedMemory": "true"}).json()
    assert shared["fit"]["budget"]["unifiedMemory"] is True
    assert shared["fit"]["verdict"] != FitVerdict.split.value


RX_7900 = gpu_probe.Adapter(
    name="AMD Radeon RX 7900 XTX",
    vendor=gpu_probe.AMD,
    integrated=False,
    dedicated_bytes=24 * GIB,
    shared_bytes=32 * GIB,
    dedicated_used_bytes=1 * GIB,
    shared_used_bytes=0,
)


def test_a_discrete_card_beside_nvidia_is_counted_too(windows_with, monkeypatch) -> None:
    """The agent's build for this machine uses both cards, so the library
    scores against both (2026-09-27). The integrated GPU stays out."""
    windows_with([RADEON_IGPU, RTX_5090, RX_7900])
    monkeypatch.setattr(
        hardware,
        "_nvidia_gpus",
        lambda warnings: [
            hardware.Gpu(
                index=0,
                name="NVIDIA GeForce RTX 5090",
                vendor=Vendor.nvidia,
                vramTotalBytes=32 * GIB,
                vramFreeBytes=29 * GIB,
            )
        ],
    )
    detected = hardware.detect()
    assert [(g.index, g.name) for g in detected.gpus or []] == [
        (0, "NVIDIA GeForce RTX 5090"),
        (1, "AMD Radeon RX 7900 XTX"),
    ]
