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

Only NVIDIA-on-Windows detection has been run against real hardware.
AMD, Intel, and Apple unified memory are written from their tools'
documented output and are **untested**; each failure appends to
`warnings` rather than defaulting silently, because a fit verdict
computed from a wrong budget is worse than no verdict.

Apple is the one that matters most: the VRAM/RAM split does not exist
there, and the naive reading of "VRAM" on a 96 GB Mac is zero — which
would tell one of the better local-inference boxes on the market that it
has no GPU.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime

from ._generated.models import Arch, Gpu, HostHardware, Os, Vendor

log = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0
"""Vendor CLIs are quick, but `nvidia-smi` on a machine with a wedged
driver can hang indefinitely. This surface is called from a request
handler, so it is bounded."""

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

APPLE_WIRED_LIMIT_FRACTION = 0.75
"""macOS reserves part of unified memory for the CPU side. The default
wired limit is roughly 75% of RAM (`iogpu.wired_limit_pct`), which is
what an engine can actually claim for weights. Reported as the GPU's
"VRAM" on Apple silicon because that is the number that decides whether
a model loads."""


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


def _run(argv: list[str]) -> str | None:
    """Run a vendor CLI and return stdout, or None if it is not usable.

    Absence is the common case and is not an error: most machines have
    exactly one of these tools. A tool that exists and fails *is*
    interesting, so it is logged.
    """
    exe = shutil.which(argv[0])
    if exe is None:
        return None
    try:
        completed = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("%s could not be run (%s)", argv[0], exc)
        return None
    if completed.returncode != 0:
        log.warning(
            "%s exited %d: %s",
            argv[0],
            completed.returncode,
            (completed.stderr or "").strip()[:200],
        )
        return None
    return completed.stdout


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
        ]
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
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
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
    out = _run(["xpu-smi", "discovery", "--dump", "1,2"])
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
            "xpu-smi does not report VRAM size in a stable form, so Intel GPU memory is "
            "unknown and fit verdicts fall back to host memory. Detection here is "
            "untested against real hardware."
        )
    return gpus


def _apple_gpu(ram_total: int | None, warnings: list[str]) -> list[Gpu]:
    """UNVERIFIED — written from Apple's documented behaviour.

    On Apple silicon there is no separate VRAM pool: the GPU addresses
    host RAM, capped by the wired limit. Reporting `vramTotalBytes: 0`
    here would tell a 96 GB M-series Mac it has no GPU, which is the
    single worst answer this component could give.
    """
    if ram_total is None:
        warnings.append("could not read total memory, so the unified-memory budget is unknown")
        return []
    budget = int(ram_total * APPLE_WIRED_LIMIT_FRACTION)
    name = platform.processor() or "Apple silicon"
    warnings.append(
        f"unified memory: reporting {APPLE_WIRED_LIMIT_FRACTION:.0%} of RAM as the GPU budget, "
        "which is the default iogpu.wired_limit_pct. Raise that limit and more is available. "
        "This path is untested on real hardware."
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
        if not gpus:
            gpus = _amd_gpus(warnings)
        if not gpus:
            gpus = _intel_gpus(warnings)

    if not gpus:
        warnings.append(
            "no accelerator was detected, so fit is scored against host memory alone. "
            "If you have a GPU, its vendor tool (nvidia-smi, rocm-smi, xpu-smi) is not on "
            "this process's PATH -- which is a common outcome when a supervisor spawns a "
            "child with a trimmed environment."
        )

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
