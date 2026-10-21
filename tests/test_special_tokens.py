"""DI4: the spellings a model's engine reads as special tokens in text.

The vocabulary is longer than the reader ever keeps inline (512), as every
real one is, so the test exercises the whole-array pass and not the
small-array shortcut.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_library.formats import gguf

from .conftest import write_gguf
from .test_store import _model

NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
#: DeepSeek's spelling, with its full-width bars.
DEEPSEEK_USER = "<" + chr(0xFF5C) + "User" + chr(0xFF5C) + ">"


def _vocab_gguf(path: Path) -> Path:
    special = [
        ("<|im_start|>", CONTROL),
        ("<|im_end|>", CONTROL),
        ("<|endoftext|>", CONTROL),
        ("<tool_call>", USER_DEFINED),
        ("[INST]", CONTROL),
        (DEEPSEEK_USER, USER_DEFINED),
        ("<unk>", UNKNOWN),
        ("    ", USER_DEFINED),  # whitespace: left alone
        ("hello", USER_DEFINED),  # a word: left alone
        ("<>", CONTROL),  # too short to be markup worth breaking
        ("<0x0A>", BYTE),
        ("<|unused_7|>", UNUSED),
        ("<|not_special|>", NORMAL),
    ]
    tokens = [f"tok{i}" for i in range(600)] + [t for t, _ in special]
    types = [NORMAL] * 600 + [k for _, k in special]
    return write_gguf(
        path,
        {
            "general.architecture": "qwen2",
            "general.name": "Vocab",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "general.file_type": 15,
        },
    )


def test_the_special_spellings_are_read_from_the_whole_vocabulary(tmp_path: Path) -> None:
    found = gguf.special_tokens(_vocab_gguf(tmp_path / "m.gguf"))
    assert set(found) == {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<tool_call>",
        "[INST]",
        DEEPSEEK_USER,
        "<unk>",
    }
    assert [len(t) for t in found] == sorted((len(t) for t in found), reverse=True)


def test_ordinary_metadata_reads_still_skip_the_vocabulary(tmp_path: Path) -> None:
    meta = gguf.read_metadata(_vocab_gguf(tmp_path / "m.gguf"))
    assert meta.kv["tokenizer.ggml.tokens"] is None
    assert meta.array_lengths["tokenizer.ggml.tokens"] == 613


def test_the_route_answers_a_models_spellings(client: TestClient, tmp_path: Path) -> None:
    path = _vocab_gguf(tmp_path / "models" / "m.gguf")
    model = _model(path, size=path.stat().st_size)
    client.app.state.state_store.replace_models([model], scanned_at=datetime.now(UTC))  # type: ignore[attr-defined]
    response = client.get(f"/v1/models/{model.id}/special-tokens")
    assert response.status_code == 200, response.text
    assert "<|im_end|>" in response.json()["tokens"]
    assert "hello" not in response.json()["tokens"]
    assert client.get("/v1/models/nope/special-tokens").status_code == 404


def test_a_file_that_is_not_a_readable_gguf_has_none(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "models" / "broken.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a gguf")
    model = _model(path, size=10)
    client.app.state.state_store.replace_models([model], scanned_at=datetime.now(UTC))  # type: ignore[attr-defined]
    response = client.get(f"/v1/models/{model.id}/special-tokens")
    assert response.status_code == 200, response.text
    assert response.json() == {"tokens": []}
