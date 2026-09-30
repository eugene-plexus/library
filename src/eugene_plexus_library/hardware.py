"""What this host has to spend on a model.

The input to every fit verdict, and the reason the guidance surface
lives in this component: the agent's `HostAccelerator` answers "which
engine build do I fetch" and its own description says the
VRAM-and-quant-fit surface belongs here.

## Free, not total

Measured on the dev box with nothing unusual running:

    RTX 5090   32,607 MiB total   29,582 MiB free   2,606 MiB already held
    RAM        93.56 GiB total     58.75 GiB avail   34.81 GiB held (37% load)

2.9 GiB of VRAM is gone to the desktop before an engine starts, and on a
24 GB card that is 11% of the budget. So both numbers are reported and
scoring uses **free** — a verdict computed from total promises a fit
that OOMs, and hiding total conceals what quitting a browser buys back.

## Stdlib plus vendor CLIs, no new dependency

`ctypes` on Windows, `/proc/meminfo` on Linux, `sysctl` on macOS, and
`nvidia-smi` / `rocm-smi` / `xpu-smi` for accelerators. Deliberately not
`psutil`: the library is `pip install -e`'d into the agent's venv so
the supervisor can spawn it, and every dependency added here has to be
added there too — a lesson this project has now learned four times.

## What is unverified, and says so

Only NVIDIA-on-Windows detection has been run against real hardware,
and Apple silicon against a virtual Mac: GitHub's macOS runners, where
the budget is Metal's own working-set figure (A4, 2026-09-30). AMD and
Intel are written from their tools' documented output and are
**untested**; each failure appends to
`warnings` rather than defaulting silently, because a fit verdict
computed from a wrong budget is worse than no verdict.

Apple is the one that matters most: the VRAM/RAM split does not exist
there, and the naive reading of "VRAM" on a 96 GB Mac is zero — which
would tell one of the better local-inference boxes on the market that it
has no GPU.
"""

from __future__ import annotations

import ctypes
import enum
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime

from . import gpu_probe
from ._generated.models import Arch, Gpu, HostHardware, Os, Vendor

log = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0
"""Vendor CLIs are quick, but `nvidia-smi` on a machine with a wedged
driver can hang indefinitely. This surface is called from a request
handler, so it is bounded."""

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

APPLE_WIRED_LIMIT_FRACTION = 2 / 3
"""macOS reserves part of unified memory for the CPU side; what the GPU
may hold is Metal's `recommendedMaxWorkingSetSize`, which
`gpu_probe.metal_device()` reads from Metal itself and which is reported
as the GPU's "VRAM" on Apple silicon, because that is the number that
decides whether a model loads. This fraction is only the fallback when
Metal cannot be asked. It was 0.75, from documentation; Metal reports two
thirds on every GitHub macOS runner measured (A4, 2026-09-30), and the
agent carries the same figure."""


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


class Probe(enum.Enum):
    """Why a vendor CLI produced nothing — review §6.2 #29.

    `_run` used to answer `None` for both *not installed* and *ran and
    failed*, so one layer up the two were indistinguishable and
    `detect()` reported the second as the first: *"its vendor tool is
    not on this process's PATH"*. That sends a person to fix an
    environment that is fine while their driver is wedged — and a
    driver/library version mismatch after an update is the commonest
    Linux failure there is.

    Three states, in one place, because `nvidia-smi`, `rocm-smi` and
    `xpu-smi` all need the same distinction and three copies of it is
    three chances to get one wrong.
    """

    ABSENT = "absent"
    """Not on PATH. The common case: most machines have exactly one of
    these tools, and the absence of the other two says nothing."""

    FAILED = "failed"
    """It is installed and it did not work. Interesting, and the thing
    a person has to be told."""


def probe(argv: list[str]) -> Probe | str:
    """Run a vendor CLI: its stdout, or which kind of nothing."""
    exe = shutil.which(argv[0])
    if exe is None:
        return Probe.ABSENT
    try:
        completed = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("%s could not be run (%s)", argv[0], exc)
        return Probe.FAILED
    if completed.returncode != 0:
        log.warning(
            "%s exited %d: %s",
            argv[0],
            completed.returncode,
            (completed.stderr or "").strip()[:200],
        )
        return Probe.FAILED
    return completed.stdout


def _run(argv: list[str], warnings: list[str] | None = None) -> str | None:
    """`probe`, with the FAILED case turned into a warning in passing.

    **Recorded where it happens, not re-established afterwards.** The
    first version of this fix scanned all three vendor tools again at
    the end of `detect()` to find out which had failed -- which meant up
    to three extra subprocesses on every hardware read, on a request
    path, for a fact the detectors had already learned and thrown away.
    In WSL2 (where `nvidia-smi` is real and takes about a second) the
    acceptance run caught it as a library process still alive after its
    own shutdown. Same family as review §6.1 #5.
    """
    result = probe(argv)
    if result is Probe.FAILED and warnings is not None:
        name = argv[0]
        warnings.append(
            f"{name} is installed here and exited non-zero, so any GPU it manages could "
            "not be read. On Linux that is usually a driver/library version mismatch "
            f"after an update -- `{name}` itself will say so. This is not a PATH problem."
        )
    return None if isinstance(result, Probe) else result


def detect_os() -> Os:
    system = platform.system()
    if system == "Windows":
        return Os.windows
    if system == "Darwin":
        return Os.macos
    return Os.linux


def detect_arch() -> Arch:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return Arch.arm64
    return Arch.x64


def _windows_memory() -> tuple[int | None, int | None]:
    # `sys.platform`, not `platform.system()`, and checked inside the
    # function rather than only at the call site: type checkers narrow on
    # this exact comparison and nothing else, so it is what makes
    # `ctypes.windll` -- which exists only on Windows -- check cleanly on
    # a Linux CI runner *and* on a Windows dev box. A `type: ignore`
    # cannot do both: it is required on Linux and flagged as unused on
    # Windows, and this repo treats both as errors.
    if sys.platform != "win32":  # pragma: no cover - unreachable on Windows
        return None, None

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError) as exc:
        log.warning("GlobalMemoryStatusEx failed (%s)", exc)
        return None, None
    if not ok:
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _linux_memory() -> tuple[int | None, int | None]:
    """`MemAvailable`, not `MemFree`.

    `MemFree` on Linux is almost always small because the kernel uses
    everything spare as page cache, and reporting it would tell a box
    with 64 GB of cache that it has 400 MB. `MemAvailable` is the
    kernel's own estimate of what a new allocation can have without
    swapping, which is exactly the question being asked.
    """
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024  # kB
    except OSError as exc:
        log.warning("/proc/meminfo unreadable (%s)", exc)
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if available is None and total is not None:
        free = values.get("MemFree", 0)
        cached = values.get("Cached", 0)
        buffers = values.get("Buffers", 0)
        available = free + cached + buffers
    return total, available


def _macos_memory() -> tuple[int | None, int | None]:
    total: int | None = None
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out and out.strip().isdigit():
        total = int(out.strip())

    # `vm_stat` reports pages. Free plus inactive plus speculative is
    # the closest analogue to Linux's MemAvailable; wired and active are
    # genuinely in use.
    available: int | None = None
    stats = _run(["vm_stat"])
    if stats:
        page_size = 4096
        header = re.search(r"page size of (\d+) bytes", stats)
        if header:
            page_size = int(header.group(1))
        pages = dict(re.findall(r"^(.*?):\s+(\d+)\.?$", stats, flags=re.MULTILINE))
        wanted = ("Pages free", "Pages inactive", "Pages speculative")
        counted = [int(pages[k]) for k in wanted if k in pages]
        if counted:
            available = sum(counted) * page_size
    return total, available


def host_memory() -> tuple[int | None, int | None]:
    """`(total, available)` in bytes, either possibly None."""
    if sys.platform == "win32":
        return _windows_memory()
    if sys.platform == "darwin":
        return _macos_memory()
    return _linux_memory()


def _nvidia_gpus(warnings: list[str]) -> list[Gpu]:
    """Verified against a real RTX 5090 on Windows.

    `nounits` keeps the CSV numeric; the memory columns are MiB, which
    is `nvidia-smi`'s own unit and not negotiable.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        warnings,
    )
    if out is None:
        return []

    gpus: list[Gpu] = []
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            index = int(fields[0])
            total = int(float(fields[2])) * MIB
            free = int(float(fields[3])) * MIB
        except ValueError:
            warnings.append(f"could not parse an nvidia-smi row: {line.strip()!r}")
            continue
        gpus.append(
            Gpu(
                index=index,
                name=fields[1],
                vendor=Vendor.nvidia,
                vramTotalBytes=total,
                vramFreeBytes=free,
                computeCapability=fields[4] if len(fields) > 4 and fields[4] else None,
                driverVersion=fields[5] if len(fields) > 5 and fields[5] else None,
            )
        )
    return gpus


def _amd_gpus(warnings: list[str]) -> list[Gpu]:
    """UNVERIFIED — no AMD hardware or `rocm-smi` on the dev box.

    Parses the CSV form of `rocm-smi --showmeminfo vram`, whose column
    names have changed between ROCm releases. A parse failure appends a
    warning and reports no GPU rather than guessing a size.
    """
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"], warnings)
    if out is None:
        return []

    lines = [line for line in out.splitlines() if line.strip()]
    if len(lines) < 2:
        warnings.append("rocm-smi returned no VRAM rows; AMD VRAM is unknown")
        return []

    header = [h.strip().lower() for h in lines[0].split(",")]
    total_col = next((i for i, h in enumerate(header) if "total" in h and "vram" in h), None)
    used_col = next((i for i, h in enumerate(header) if "used" in h and "vram" in h), None)
    if total_col is None:
        warnings.append(
            "rocm-smi output has no recognisable VRAM total column "
            f"(saw {header}); AMD VRAM is unknown"
        )
        return []

    gpus: list[Gpu] = []
    for index, line in enumerate(lines[1:]):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) <= total_col:
            continue
        try:
            total = int(fields[total_col])
            used = int(fields[used_col]) if used_col is not None and len(fields) > used_col else 0
        except ValueError:
            warnings.append(f"could not parse a rocm-smi row: {line.strip()!r}")
            continue
        gpus.append(
            Gpu(
                index=index,
                name=fields[0] or f"AMD GPU {index}",
                vendor=Vendor.amd,
                vramTotalBytes=total,
                vramFreeBytes=max(total - used, 0),
            )
        )
    return gpus


def _intel_gpus(warnings: list[str]) -> list[Gpu]:
    """UNVERIFIED — no Intel Arc hardware or `xpu-smi` on the dev box.

    `xpu-smi discovery` reports device names but not free memory in any
    stable form, so this reports the device with no `vramFreeBytes` and
    says why. Fit then falls back to total for that card, which the
    verdict labels `tight` rather than `fits`.
    """
    out = _run(["xpu-smi", "discovery", "--dump", "1,2"], warnings)
    if out is None:
        return []
    gpus: list[Gpu] = []
    for index, line in enumerate(out.splitlines()[1:]):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        gpus.append(
            Gpu(
                index=int(fields[0]),
                name=fields[1] or f"Intel GPU {index}",
                vendor=Vendor.intel,
                vramTotalBytes=0,
            )
        )
    if gpus:
        warnings.append(
            "xpu-smi does not report VRAM size in a stable form, so this card's memory is "
            "unknown and every fit verdict on this host is `unknown` rather than a guess. "
            "Detection here is untested against real hardware."
        )
    return gpus


def _apple_chip() -> str | None:
    """`Apple M2 Pro`, where `platform.processor()` says only `arm`."""
    return (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or None


def _apple_gpu(
    ram_total: int | None,
    warnings: list[str],
    metal: Callable[[], gpu_probe.MetalDevice | None] | None = None,
) -> list[Gpu]:
    """On Apple silicon there is no separate VRAM pool: the GPU addresses
    host RAM, up to the working set Metal allows. Reporting
    `vramTotalBytes: 0` here would tell a 96 GB M-series Mac it has no
    GPU, which is the single worst answer this component could give.

    The budget is Metal's own `recommendedMaxWorkingSetSize`, read on this
    host (A4, 2026-09-30: it matched MLX's figure on GitHub's macOS
    runners). Only when Metal cannot be asked is it a fraction of RAM, and
    then a warning says so.
    """
    device = (metal or gpu_probe.metal_device)()
    if device is not None:
        budget = device.working_set_bytes
        name = device.name or _apple_chip() or "Apple silicon"
    elif ram_total is None:
        warnings.append("could not read total memory, so the unified-memory budget is unknown")
        return []
    else:
        budget = int(ram_total * APPLE_WIRED_LIMIT_FRACTION)
        name = _apple_chip() or "Apple silicon"
        warnings.append(
            f"unified memory: Metal could not be asked for its working-set limit, so "
            f"{APPLE_WIRED_LIMIT_FRACTION:.0%} of RAM is reported as the GPU budget"
        )
    return [
        Gpu(
            index=0,
            name=name,
            vendor=Vendor.apple,
            vramTotalBytes=budget,
            vramFreeBytes=None,
        )
    ]


_VENDOR = {
    gpu_probe.NVIDIA: Vendor.nvidia,
    gpu_probe.AMD: Vendor.amd,
    gpu_probe.INTEL: Vendor.intel,
}

_HIP_SDK_PATHS = (r"C:\Program Files\AMD\ROCm", r"C:\Program Files\AMD\HIP SDK")


def _rocm_installed(os_kind: Os) -> bool:
    """AMD's SDK on disk: the agent's `rocm_installed`, restated here
    because components share schemas and not code."""
    if os_kind == Os.windows:
        env = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
        if env and os.path.isdir(env):
            return True
        return any(os.path.isdir(p) for p in _HIP_SDK_PATHS)
    return os.path.isdir("/opt/rocm")


def _os_gpus(os_kind: Os, ram_available: int | None, warnings: list[str]) -> tuple[list[Gpu], bool]:
    """The GPUs no vendor tool answered for, from the operating system.

    **The Arc report (2026-09-27).** An Intel Arc or an AMD Radeon on
    Windows has none of `nvidia-smi`, `rocm-smi` and `xpu-smi` unless its
    owner installed an SDK, so this surface said "no accelerator was
    detected" about a machine whose Vulkan build was using its GPU, and
    every fit was scored against RAM. `gpu_probe` is the same module the
    agent uses, copied, and `gpu_probe.family` the same decision, so the
    library scores against the card the agent's build computes on. One
    difference, named: this side passes `sycl=False`, because it runs no
    `sycl-ls`; that changes the answer only for a machine with an AMD and
    an Intel GPU and oneAPI installed.

    Returns the GPUs and whether they share host memory.
    """
    try:
        found = gpu_probe.adapters(os_kind.value)
    except gpu_probe.GpuProbeError as exc:
        warnings.append(f"could not list this machine's GPUs from the operating system: {exc}")
        return [], False
    if not found:
        return [], False
    chosen = gpu_probe.family(
        os_kind.value,
        detect_arch().value,
        found,
        vulkan_loader=gpu_probe.vulkan_loader_present(os_kind.value),
        rocm=_rocm_installed(os_kind),
    )
    warnings.extend(chosen.notes)
    gpus: list[Gpu] = []
    for index, adapter in enumerate(chosen.adapters):
        total, free = adapter.budget(ram_available)
        if total is None:
            warnings.append(
                f"{adapter.name} does not report its memory on this platform, so every fit "
                "against it is unknown rather than a guess."
            )
        elif free is None:
            warnings.append(
                f"how much of {adapter.name}'s memory is in use could not be read, so its "
                "free memory is unknown and scoring uses its total."
            )
        gpus.append(
            Gpu(
                index=index,
                name=adapter.name,
                vendor=_VENDOR.get(adapter.vendor, Vendor.unknown),
                vramTotalBytes=total or 0,
                vramFreeBytes=free,
            )
        )
    shared = bool(chosen.adapters) and all(a.integrated for a in chosen.adapters)
    return gpus, shared


def _beside_nvidia(os_kind: Os, first_index: int, ram_available: int | None) -> list[Gpu]:
    """A discrete AMD or Intel card beside the NVIDIA ones (2026-09-27).

    On Windows the build for such a machine is the CUDA one with the
    Vulkan backend added, and it uses both cards; `gpu_probe.beside_nvidia`
    is the agent's rule, copied, so the two components count the same
    cards. On Linux the builds cannot be combined and this is empty.
    """
    try:
        found = gpu_probe.adapters(os_kind.value)
    except gpu_probe.GpuProbeError:
        return []
    extra = gpu_probe.beside_nvidia(
        os_kind.value,
        detect_arch().value,
        found,
        vulkan_loader=gpu_probe.vulkan_loader_present(os_kind.value),
    )
    gpus: list[Gpu] = []
    for offset, adapter in enumerate(extra):
        total, free = adapter.budget(ram_available)
        gpus.append(
            Gpu(
                index=first_index + offset,
                name=adapter.name,
                vendor=_VENDOR.get(adapter.vendor, Vendor.unknown),
                vramTotalBytes=total or 0,
                vramFreeBytes=free,
            )
        )
    return gpus


def detect() -> HostHardware:
    """Read this host's memory and accelerators.

    Never raises: every probe failure becomes a `warnings` entry, so a
    machine whose vendor tool is broken still reports its RAM and gets a
    verdict that says what it could not see.
    """
    warnings: list[str] = []
    os_kind = detect_os()
    ram_total, ram_available = host_memory()
    if ram_total is None:
        warnings.append("could not read total host memory on this platform")

    unified = os_kind == Os.macos and detect_arch() == Arch.arm64

    gpus: list[Gpu] = []
    if unified:
        gpus = _apple_gpu(ram_total, warnings)
    else:
        gpus = _nvidia_gpus(warnings)
        if gpus:
            gpus += _beside_nvidia(os_kind, len(gpus), ram_available)
        if not gpus:
            gpus = _amd_gpus(warnings)
        if not gpus:
            gpus = _intel_gpus(warnings)
        if not gpus:
            gpus, unified = _os_gpus(os_kind, ram_available, warnings)

    # **Which kind of nothing** (review §6.2 #29). A tool that is not
    # installed and a tool that is installed and broken are two different
    # problems with two different fixes, and reporting the second as the
    # first sends a person to audit their PATH while their driver is the
    # thing that is wrong. Each detector above records its own failure as
    # it happens, so the distinction costs no extra process -- and it is
    # recorded whether or not ANOTHER vendor's card was found, because on
    # a machine with a working Intel card and a wedged NVIDIA driver the
    # NVIDIA card is the one the person cares about.
    wedged = any("exited non-zero" in w for w in warnings)

    if not gpus and not wedged:
        warnings.append(
            "no accelerator was detected, by a vendor tool or by the operating system, so "
            "fit is scored against host memory alone. If you have a GPU, its vendor tool "
            "(nvidia-smi, rocm-smi, xpu-smi) is not on this process's PATH -- which is a "
            "common outcome when a supervisor spawns a child with a trimmed environment."
        )
    elif not gpus:
        warnings.append("fit is scored against host memory alone, because no GPU was read.")

    return HostHardware(
        hostname=socket.gethostname(),
        os=os_kind,
        arch=detect_arch(),
        cpuCount=os.cpu_count() or 1,
        ramTotalBytes=ram_total,
        ramAvailableBytes=ram_available,
        unifiedMemory=unified,
        gpus=gpus,
        detectedAt=datetime.now(tz=UTC),
        warnings=warnings,
    )
