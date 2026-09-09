"""The ranged metadata read, against real GGUF bytes over a mock transport.

The fixture builds an actual GGUF and serves it in slices, so the growth
logic meets the same "the KV block is bigger than my window" situation
the live hub produces — the one the whole endpoint exists to survive.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from eugene_plexus_library import fit, hub, preflight
from eugene_plexus_library._generated.models import (
    Arch,
    Basis,
    Gpu,
    HostHardware,
    ModelFormat,
    Os,
    Vendor,
)
from eugene_plexus_library.formats import gguf

from .conftest import write_gguf

GIB = 1024**3


def hybrid_kv(vocab: int) -> dict:
    """A hybrid attention/SSM model's KV block, shaped like the real one.

    65 blocks with a full-attention interval of 4 — the layout that
    breaks the naive KV-cache formula by 4.1x.
    """
    return {
        "general.architecture": "qwen35",
        "general.type": "model",
        "general.name": "Hybrid 27B",
        "general.file_type": 15,
        "general.size_label": "27B",
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        "general.sampling.top_p": 0.95,
        "qwen35.block_count": 65,
        "qwen35.full_attention_interval": 4,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "tokenizer.chat_template": "{{ x }}",
        "tokenizer.ggml.tokens": [f"tok{n}" for n in range(vocab)],
    }


@pytest.fixture
def budget():
    return fit.budget_from_hardware(
        HostHardware(
            hostname="dev",
            os=Os.windows,
            arch=Arch.x64,
            ramTotalBytes=96 * GIB,
            ramAvailableBytes=58 * GIB,
            gpus=[
                Gpu(
                    index=0,
                    name="RTX 5090",
                    vendor=Vendor.nvidia,
                    vramTotalBytes=32 * GIB,
                    vramFreeBytes=29 * GIB,
                )
            ],
        )
    )


class SlicedFile:
    """Serves one local file over Range, counting the requests."""

    def __init__(self, path: Path) -> None:
        self.data = path.read_bytes()
        self.ranges: list[tuple[int, int]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://cdn.example/blob",
                    "X-Linked-Size": str(len(self.data)),
                    "X-Linked-ETag": '"' + "a" * 64 + '"',
                    "X-Repo-Commit": "commit1",
                    "Accept-Ranges": "bytes",
                },
            )
        header = request.headers["Range"]
        first, _, last = header.removeprefix("bytes=").partition("-")
        start = int(first)
        stop = min(int(last) + 1, len(self.data)) if last else len(self.data)
        self.ranges.append((start, stop))
        return httpx.Response(206, content=self.data[start:stop])


def client_for(served: SlicedFile) -> hub.HubClient:
    inner = httpx.AsyncClient(transport=httpx.MockTransport(served.handler), follow_redirects=False)
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    return client


async def test_a_small_header_costs_one_request(tmp_path: Path, budget) -> None:
    """A projector's KV block is 1,743 bytes. It should not cost the same
    as a 248k-vocab model's."""
    path = tmp_path / "small.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="small.gguf",
        budget=budget,
        context_length=8192,
    )
    assert len(served.ranges) == 1
    assert result.format is ModelFormat.gguf
    assert result.bytesRead <= preflight.INITIAL_WINDOW


async def test_a_header_past_the_window_grows_and_appends(tmp_path: Path, budget) -> None:
    """The growth path. Each grow appends rather than re-reading from
    zero — re-reading cost 1+2+4+8+16 MB of transfer to reach an 11 MB
    block on the live hub."""
    path = tmp_path / "big.gguf"
    # A vocabulary large enough to overrun the initial 1 MB window.
    write_gguf(path, hybrid_kv(vocab=140_000))
    served = SlicedFile(path)
    assert path.stat().st_size > preflight.INITIAL_WINDOW

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="big.gguf",
        budget=budget,
        context_length=8192,
    )
    assert len(served.ranges) >= 2
    # Contiguous and non-overlapping: nothing was fetched twice.
    for (_, previous_stop), (next_start, _) in zip(served.ranges, served.ranges[1:], strict=False):
        assert next_start == previous_stop
    assert result.vocabSize == 140_000


async def test_the_quant_comes_from_metadata_not_the_filename(tmp_path: Path, budget) -> None:
    """The whole point: the same standard a local scan holds a model to."""
    path = tmp_path / "model-Q4_K_M.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="model-Q4_K_M.gguf",
        budget=budget,
        context_length=8192,
    )
    assert result.fileType == 15
    assert result.quantization == "Q4_K_M"
    assert result.agreesWithFilename is True


async def test_a_filename_that_lies_is_reported_as_a_disagreement(tmp_path: Path, budget) -> None:
    """A requantized file keeps its old name more often than a metadata
    field is wrong — but not always, so both are reported and neither is
    silently preferred."""
    path = tmp_path / "model-Q8_0.gguf"
    write_gguf(path, hybrid_kv(vocab=8))  # file_type 15 = Q4_K_M
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="model-Q8_0.gguf",
        budget=budget,
        context_length=8192,
    )
    assert result.agreesWithFilename is False


async def test_hybrid_attention_is_read_and_the_fit_becomes_arithmetic(
    tmp_path: Path, budget
) -> None:
    """16 of 65 layers hold a KV cache, and the fit says `metadata`
    rather than `estimate` because there is now something to compute."""
    path = tmp_path / "hybrid.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="hybrid.gguf",
        budget=budget,
        context_length=32768,
        candidate_size=16 * GIB,
    )
    assert result.blockCount == 65
    assert result.attentionLayers == 16
    assert result.fit is not None
    assert result.fit.basis is Basis.metadata
    assert result.fit.attentionLayers == 16
    # 32768 x 16 layers x 4 kv heads x 512 x 2 bytes = 2 GiB exactly.
    assert result.fit.kvCacheBytes == 2 * GIB


async def test_the_candidate_size_wins_over_the_probed_file(tmp_path: Path, budget) -> None:
    """A split model's shards sum to more than the file being probed, and
    scoring the probe would understate the candidate by every other
    shard."""
    path = tmp_path / "shard1.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="shard1.gguf",
        budget=budget,
        context_length=4096,
        candidate_size=50 * GIB,
    )
    assert result.fit is not None
    assert result.fit.weightsBytes == 50 * GIB


async def test_the_authors_sampling_is_reported_and_applied_by_nothing(
    tmp_path: Path, budget
) -> None:
    """The gateway owns every parameter that affects output. The file
    having an opinion is still information the operator would otherwise
    have to go read the model card for."""
    path = tmp_path / "m.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="m.gguf",
        budget=budget,
        context_length=4096,
    )
    assert result.recommendedSampling is not None
    assert result.recommendedSampling.temperature == 1.0
    assert result.recommendedSampling.topK == 20


async def test_a_truncated_upload_is_not_mistaken_for_a_small_window(
    tmp_path: Path, budget
) -> None:
    """A file shorter than the window with an incomplete header is
    broken, not under-fetched — and growing the window forever would
    never say so."""
    path = tmp_path / "truncated.gguf"
    write_gguf(path, hybrid_kv(vocab=8))
    path.write_bytes(path.read_bytes()[:40])  # cut mid-header
    served = SlicedFile(path)

    with pytest.raises(hub.HubError) as raised:
        await preflight.preflight(
            client_for(served),
            repo="org/repo",
            revision="main",
            path="truncated.gguf",
            budget=budget,
            context_length=4096,
        )
    assert "truncated" in str(raised.value).lower()


async def test_a_file_that_is_not_a_model_is_refused_by_name(tmp_path: Path, budget) -> None:
    path = tmp_path / "README.md"
    path.write_text("not a model")
    served = SlicedFile(path)

    with pytest.raises(hub.HubError) as raised:
        await preflight.preflight(
            client_for(served),
            repo="org/repo",
            revision="main",
            path="README.md",
            budget=budget,
            context_length=4096,
        )
    assert raised.value.status == 400


async def test_a_safetensors_header_costs_two_small_reads(tmp_path: Path, budget) -> None:
    """Eight bytes for the length, then the header. Two orders of
    magnitude cheaper than a GGUF's, and the parameter count is exact."""
    import json
    import struct

    header = {
        "__metadata__": {"format": "pt"},
        "model.embed.weight": {"dtype": "BF16", "shape": [1000, 64], "data_offsets": [0, 128000]},
        "model.layers.0.q.weight": {
            "dtype": "BF16",
            "shape": [64, 64],
            "data_offsets": [128000, 136192],
        },
    }
    blob = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 100)
    served = SlicedFile(path)

    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="model.safetensors",
        budget=budget,
        context_length=4096,
    )
    assert result.format is ModelFormat.safetensors
    assert result.parameters == 1000 * 64 + 64 * 64
    assert result.dtype == "BF16"
    assert len(served.ranges) == 2
    assert served.ranges[0] == (0, 8)
    assert result.bytesRead == 8 + len(blob)
    # No layer metadata in a safetensors header — the shape lives in
    # config.json beside it, so the fit stays honest about estimating.
    assert result.fit is not None
    assert result.fit.basis is Basis.estimate


def test_attention_layers_falls_back_to_the_block_count() -> None:
    """A model using a hybrid convention we have not seen errs
    pessimistic rather than under-counting."""
    meta = gguf.GgufMetadata(
        version=3,
        tensor_count=0,
        kv_count=0,
        header_bytes=0,
        kv={"general.architecture": "llama", "llama.block_count": 32},
    )
    assert preflight.attention_layers(meta) == 32


def test_attention_layers_reads_an_explicit_index_list() -> None:
    meta = gguf.GgufMetadata(
        version=3,
        tensor_count=0,
        kv_count=0,
        header_bytes=0,
        kv={
            "general.architecture": "jamba",
            "jamba.block_count": 32,
            "jamba.attention.layer_indices": [3, 7, 11, 15],
        },
    )
    assert preflight.attention_layers(meta) == 4


def test_attention_layers_is_none_without_a_block_count() -> None:
    meta = gguf.GgufMetadata(
        version=3, tensor_count=0, kv_count=0, header_bytes=0, kv={"general.architecture": "x"}
    )
    assert preflight.attention_layers(meta) is None
