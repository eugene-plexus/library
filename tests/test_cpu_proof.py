from __future__ import annotations

import asyncio
import io
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import httpx
import pytest


def test_bench_json_parsed(monkeypatch, tmp_path: Path) -> None:
    from eugene_plexus_library import cpu_proof

    recorded = (Path(__file__).parent / "fixtures/llama-bench-cpu.json").read_text()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, recorded, "backend: CPU")

    monkeypatch.setattr(cpu_proof.subprocess, "run", run)
    monkeypatch.setattr(cpu_proof, "processor", lambda: (16, "Recorded processor (16 cores)"))
    statistics = {}
    measured = cpu_proof.measure(
        tmp_path / "llama-bench", tmp_path / "model.gguf", "b10948", statistics=statistics
    )
    assert statistics == {"decode": 0.27, "prompt": 3.5}
    assert measured.decodeTokensPerSecond == 13.85
    assert measured.promptTokensPerSecond == 175.4
    assert (measured.promptTokens, measured.decodeTokens, measured.threads) == (512, 128, 16)
    assert measured.engine == "llama_cpp b10948"
    assert measured.machine == "Recorded processor (16 cores)"
    assert measured.measuredOn == date.today()
    assert calls[0][0] == [
        str(tmp_path / "llama-bench"),
        "-m",
        str(tmp_path / "model.gguf"),
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "3",
        "-o",
        "json",
        "-t",
        "16",
    ]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"


@pytest.mark.parametrize("returncode, output", [(1, ""), (0, "not JSON"), (0, "[]")])
def test_bench_failure_names_cause_and_last_lines(monkeypatch, tmp_path, returncode, output):
    from eugene_plexus_library import cpu_proof

    monkeypatch.setattr(cpu_proof, "processor", lambda: (16, "Test CPU (16 cores)"))
    monkeypatch.setattr(
        cpu_proof.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a[0], returncode, output, "loading weights\nunsupported model architecture: qwen35moe"
        ),
    )
    with pytest.raises(cpu_proof.CpuProofError, match="unsupported model architecture"):
        cpu_proof.measure(tmp_path / "llama-bench", tmp_path / "model.gguf", "b10948")


def test_processor_counts_physical_cores_not_threads(tmp_path):
    from eugene_plexus_library import cpu_proof

    info = tmp_path / "cpuinfo"
    info.write_text(
        "\n\n".join(
            f"processor : {i}\nmodel name : Recorded CPU\nphysical id : {i // 4}\ncore id : {i % 2}"
            for i in range(8)
        )
    )
    threads, machine = cpu_proof.processor(info)
    assert threads == 4
    assert "Recorded CPU" in machine and "4 cores" in machine


@pytest.mark.parametrize("bad_speed", [-1, float("nan"), float("inf")])
def test_invalid_bench_rates_are_rejected(bad_speed):
    from eugene_plexus_library import cpu_proof

    rows = json.loads((Path(__file__).parent / "fixtures/llama-bench-cpu.json").read_text())
    rows[1]["avg_ts"] = bad_speed
    with pytest.raises(cpu_proof.CpuProofError):
        cpu_proof.parse_json(json.dumps(rows), engine_tag="b10948", threads=16, machine="Test CPU")


def test_cpu_build_download_keeps_shared_libraries(monkeypatch, tmp_path):
    from eugene_plexus_library import cpu_proof

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("build/bin/llama-bench", b"recorded executable")
        bundle.writestr("build/bin/libggml-cpu.so", b"recorded library")
    urls = []

    def response(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=archive.getvalue())

    monkeypatch.setattr(
        cpu_proof,
        "egress_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(response), **kwargs),
    )
    bench = asyncio.run(cpu_proof.download_engine("b10948", tmp_path))
    assert bench.read_bytes() == b"recorded executable"
    assert (bench.parent / "libggml-cpu.so").read_bytes() == b"recorded library"
    assert urls == [
        "https://github.com/ggml-org/llama.cpp/releases/download/b10948/llama-b10948-bin-ubuntu-x64.zip"
    ]
    assert not (tmp_path / "engine.zip").exists()


def test_bench_timeout_keeps_its_last_lines(monkeypatch, tmp_path):
    from eugene_plexus_library import cpu_proof

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 900, stderr=b"loading model stalled")

    monkeypatch.setattr(cpu_proof, "processor", lambda: (16, "Test CPU (16 cores)"))
    monkeypatch.setattr(cpu_proof.subprocess, "run", timeout)
    with pytest.raises(cpu_proof.CpuProofError, match="loading model stalled"):
        cpu_proof.measure(tmp_path / "llama-bench", tmp_path / "model.gguf", "b10948")
