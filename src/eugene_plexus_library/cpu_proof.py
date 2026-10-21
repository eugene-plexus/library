"""Measure a starter file with the pinned Linux CPU build of llama.cpp."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

from ._generated.models import CpuProof
from ._http import egress_client

CPU_BUILD_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-x64.zip"
)
MAX_MODEL_BYTES = 8 * 1024**3


class CpuProofError(Exception):
    """A download, load or measurement failed, with its observed cause."""


def processor(cpuinfo: Path = Path("/proc/cpuinfo")) -> tuple[int, str]:
    """Read physical cores and the processor name on the measurement host."""
    try:
        records = [
            dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            for block in cpuinfo.read_text(encoding="utf-8").strip().split("\n\n")
        ]
        records = [{k.strip(): v.strip() for k, v in row.items()} for row in records]
        cores = {
            (row["physical id"], row["core id"])
            for row in records
            if "physical id" in row and "core id" in row
        }
        name = next(row["model name"] for row in records if row.get("model name"))
        if not cores:
            raise ValueError("physical core IDs are absent")
    except (OSError, ValueError, StopIteration) as exc:
        raise CpuProofError(f"Could not read the test processor from {cpuinfo}: {exc}") from exc
    return len(cores), f"{name} ({len(cores)} cores)"


def last_lines(*outputs: str) -> str:
    return "\n".join("\n".join(outputs).strip().splitlines()[-12:])[-4000:]


def parse_json(
    output: str,
    *,
    engine_tag: str,
    threads: int,
    machine: str,
    statistics: dict[str, float] | None = None,
) -> CpuProof:
    """The separate prompt and generation rows carry their own mean rates."""
    try:
        rows = json.loads(output)
        if not isinstance(rows, list):
            raise ValueError("expected an array of benchmark rows")
        prompt = next(r for r in rows if r.get("n_prompt") == 512 and r.get("n_gen") == 0)
        decode = next(r for r in rows if r.get("n_gen") == 128 and r.get("n_prompt") == 0)
        for row in (prompt, decode):
            if not math.isfinite(row["avg_ts"]) or row["avg_ts"] < 0:
                raise ValueError("avg_ts must be a finite, non-negative speed")
            if row.get("n_threads") != threads:
                raise ValueError("benchmark thread count differs from the physical core count")
        if statistics is not None:
            for name, row in (("prompt", prompt), ("decode", decode)):
                spread = row.get("stddev_ts")
                if isinstance(spread, int | float) and math.isfinite(spread) and spread >= 0:
                    statistics[name] = float(spread)
        return CpuProof(
            decodeTokensPerSecond=decode["avg_ts"],
            promptTokensPerSecond=prompt["avg_ts"],
            promptTokens=512,
            decodeTokens=128,
            threads=threads,
            machine=machine,
            engine=f"llama_cpp {engine_tag}",
            measuredOn=date.today(),
        )
    except (ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise CpuProofError(
            f"llama-bench's JSON could not be read ({exc or 'missing prompt or decode row'}). "
            f"{last_lines(output)}"
        ) from exc


def measure(
    bench: Path,
    model: Path,
    engine_tag: str,
    *,
    statistics: dict[str, float] | None = None,
) -> CpuProof:
    """Run the CPU executable; stderr survives both load and JSON failures."""
    threads, machine = processor()
    command = [
        str(bench),
        "-m",
        str(model),
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "3",
        "-o",
        "json",
        "-t",
        str(threads),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(bench.parent), env.get("LD_LIBRARY_PATH", "")])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            env=env,
            cwd=bench.parent,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:

        def decoded(value: Any) -> str:
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        raise CpuProofError(
            f"llama-bench timed out after 900 seconds. "
            f"{last_lines(decoded(exc.stdout), decoded(exc.stderr))}"
        ) from exc
    except OSError as exc:
        raise CpuProofError(f"llama-bench could not start: {exc}") from exc
    if result.returncode:
        raise CpuProofError(
            f"llama-bench exited {result.returncode}: {last_lines(result.stdout, result.stderr)}"
        )
    try:
        return parse_json(
            result.stdout,
            engine_tag=engine_tag,
            threads=threads,
            machine=machine,
            statistics=statistics,
        )
    except CpuProofError as exc:
        raise CpuProofError(f"{exc}\n{last_lines(result.stderr)}") from exc


def _unpack(archive: Path, directory: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(directory)
    bench = next(directory.rglob("llama-bench"), None)
    if bench is None:
        raise CpuProofError("The CPU build archive contains no llama-bench executable.")
    bench.chmod(bench.stat().st_mode | 0o111)
    archive.unlink()
    return bench.resolve()


async def download_engine(tag: str, directory: Path) -> Path:
    """Fetch one pinned CPU build, including the libraries beside the executable."""
    archive = directory / "engine.zip"
    try:
        async with (
            egress_client(timeout=120, follow_redirects=True) as client,
            client.stream("GET", CPU_BUILD_URL.format(tag=tag)) as response,
        ):
            response.raise_for_status()
            handle = await asyncio.to_thread(archive.open, "wb")
            try:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    await asyncio.to_thread(handle.write, chunk)
            finally:
                await asyncio.to_thread(handle.close)
        return await asyncio.to_thread(_unpack, archive, directory)
    except Exception as exc:
        raise CpuProofError(str(exc)) from exc
