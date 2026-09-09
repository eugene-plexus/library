"""Fixtures, and the builders that write real model files.

The format readers are the part of this component most likely to be
wrong, so the suite writes **actual GGUF and safetensors bytes** into
`tmp_path` and reads them back rather than mocking the readers out. A
mocked reader agrees with whatever the code believes; a real file
disagrees when the code is wrong, which is the entire point.

The builders below encode the layouts independently of the readers —
`struct.pack` against the documented format, not a call back into
`formats.gguf` — so a bug in the reader cannot be papered over by the
same bug in the fixture.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library.app import create_app
from eugene_plexus_library.settings import Settings

# GGUF value type tags, restated here rather than imported: a fixture
# that shares constants with the code under test can agree with it about
# something wrong.
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(value: Any) -> bytes:
    """Encode one KV value, tag included."""
    if isinstance(value, str):
        return struct.pack("<I", T_STRING) + _gguf_string(value)
    if isinstance(value, bool):
        return struct.pack("<I", T_BOOL) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", T_UINT32) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", T_FLOAT32) + struct.pack("<f", value)
    if isinstance(value, list):
        if not value:
            return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_INT32) + struct.pack("<Q", 0)
        if isinstance(value[0], str):
            body = b"".join(_gguf_string(v) for v in value)
            element = T_STRING
        else:
            body = b"".join(struct.pack("<i", v) for v in value)
            element = T_INT32
        head = (
            struct.pack("<I", T_ARRAY) + struct.pack("<I", element) + struct.pack("<Q", len(value))
        )
        return head + body
    raise TypeError(f"no GGUF encoding for {type(value).__name__}")


def write_gguf(
    path: Path,
    kv: dict[str, Any],
    *,
    version: int = 3,
    tensor_count: int = 0,
    payload: bytes = b"",
    magic: bytes = b"GGUF",
) -> Path:
    """Write a GGUF file with the given KV block.

    `magic`, `version` and a truncated `payload` are parameters so the
    malformed-input tests can build genuinely malformed files instead of
    asserting against a mock that raises on request.
    """
    body = bytearray()
    body += magic
    body += struct.pack("<I", version)
    body += struct.pack("<Q", tensor_count)
    body += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        body += _gguf_string(key)
        body += _gguf_value(value)
    body += payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return path


def qwen_like_kv(
    *,
    name: str = "Test 7B",
    architecture: str = "llama",
    file_type: int = 15,
    context_length: int = 4096,
    vocab: int = 128,
) -> dict[str, Any]:
    """A KV block shaped like a real model's.

    Ordered so `general.file_type` lands **after** the token array, which
    is where a real 27B puts it — the reason the reader cannot stop
    early. A fixture with the quant helpfully near the front would let a
    short-circuiting reader pass.
    """
    return {
        "general.architecture": architecture,
        "general.type": "model",
        "general.name": name,
        "general.size_label": "7B",
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        f"{architecture}.block_count": 32,
        f"{architecture}.context_length": context_length,
        f"{architecture}.embedding_length": 4096,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(vocab)],
        "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "general.quantization_version": 2,
        "general.file_type": file_type,
    }


def projector_kv(*, name: str = "Test 7B") -> dict[str, Any]:
    """A vision projector: `general.type: mmproj`, `clip` architecture,
    and the same `general.name` as the model it belongs to — which is
    how a real pair looks."""
    return {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "general.name": name,
        "general.file_type": 32,
        "clip.has_vision_encoder": True,
    }


def embedding_kv(*, name: str = "Test Embed") -> dict[str, Any]:
    """An embedding model: a pooling type, non-causal attention, and
    **no `general.type`** — verified absent on a real one."""
    return {
        "general.architecture": "nomic-bert",
        "general.name": name,
        "nomic-bert.block_count": 12,
        "nomic-bert.context_length": 2048,
        "nomic-bert.pooling_type": 1,
        "nomic-bert.attention.causal": False,
        "general.file_type": 15,
    }


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a safetensors file whose header declares `tensors`.

    `{name: (dtype, shape)}`. Data offsets are computed so the header is
    self-consistent; the tensor bytes themselves are zeros, because
    nothing here ever reads past the header.
    """
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U8": 1}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        length = count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    if metadata is not None:
        header["__metadata__"] = metadata

    raw = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def write_hf_model(
    directory: Path,
    *,
    architectures: list[str] | None = None,
    max_position_embeddings: int = 2048,
    parameters: int = 1024,
) -> Path:
    """A minimal HuggingFace model directory: config, weights, tokenizer."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": architectures or ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": max_position_embeddings,
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(directory / "model.safetensors", {"weight": ("F32", [parameters])})
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


# --------------------------------------------------------------------- #
# App fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "library-state.json",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def configured_client(settings: Settings, models_dir: Path) -> Iterator[TestClient]:
    """A client whose config already points at `models_dir`.

    Written to the config file before startup rather than PATCHed
    afterwards, so the startup-scan path is the one under test.
    """
    settings.config_file.write_text(
        f"modelRoots:\n  - {models_dir.as_posix()}\nscanOnStartup: true\n",
        encoding="utf-8",
    )
    with TestClient(create_app(settings=settings)) as c:
        yield c
