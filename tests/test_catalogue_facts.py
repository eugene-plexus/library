"""What the catalogue tells the judge before anything is downloaded (LS2).

Search rows, a repo's versions and the starter set each carry an
`EligibilityCandidate`; a safetensors folder's comes from one small read of
its remote `config.json`, so MLX is told from vLLM before 20 GB moves.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import catalogue, hardware, hub, starter
from eugene_plexus_library._generated.models import (
    KvCacheType,
    LibraryModel,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
)
from eugene_plexus_library.formats import safetensors
from eugene_plexus_library.hub import FileMetadata, RepoInfo
from eugene_plexus_library.store import StateStore

from .conftest import mock_hubs
from .test_catalogue_routes import HOST
from .test_starter import budget as starter_budget
from .test_starter import entry, write

GIB = 1024**3
MLX_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "quantization": {"bits": 4, "group_size": 64},
}
PLAIN_CONFIG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}


def client_for(handler) -> hub.HubClient:
    inner = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    return client


# --- the hub ------------------------------------------------------------------


async def test_a_search_names_the_gguf_block_and_every_field_a_row_shows() -> None:
    """`expand[]` replaces `full=true` upstream, so every field is named."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=[])

    await client_for(handler).search(query="q")
    expanded = seen[0].params.get_list("expand[]")
    assert "gguf" in expanded
    assert {"downloads", "tags", "library_name", "gated", "lastModified"} <= set(expanded)
    assert "full" not in seen[0].params


async def test_a_small_read_takes_a_range_and_accepts_either_answer() -> None:
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ranges.append(request.headers["Range"])
        status = 206 if request.url.path.endswith("a/config.json") else 200
        return httpx.Response(status, content=b'{"model_type": "llama"}')

    client = client_for(handler)
    assert await client.read_small("org/repo", revision="main", path="a/config.json") == (
        b'{"model_type": "llama"}'
    )
    # A whole small file is a fine answer here, unlike a model's Range read.
    assert await client.read_small("org/repo", revision="main", path="b/config.json") == (
        b'{"model_type": "llama"}'
    )
    assert ranges[0] == f"bytes=0-{hub.CONFIG_READ_LIMIT - 1}"


async def test_a_small_read_never_takes_more_than_its_limit_and_is_cached() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"x" * 5000)

    client = client_for(handler)
    data = await client.read_small("org/repo", revision="main", path="config.json", limit=100)
    assert data == b"x" * 100
    await client.read_small("org/repo", revision="main", path="config.json", limit=100)
    assert calls == 1


async def test_a_small_read_that_fails_is_a_hub_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Entry not found"})

    with pytest.raises(hub.HubError):
        await client_for(handler).read_small("org/repo", revision="main", path="config.json")


def test_a_remote_config_parses_by_the_scanners_rules() -> None:
    config = safetensors.parse_config(json.dumps(MLX_CONFIG).encode())
    assert config.architecture == "Qwen3ForCausalLM"
    assert config.mlx_quantization == (4, 64)
    with pytest.raises(safetensors.SafetensorsError):
        safetensors.parse_config(b'{"architectures": ["Qwe')


# --- the catalogue --------------------------------------------------------------


def _file(path: str, size: int) -> FileMetadata:
    return FileMetadata(path=path, size=size, sha256=None, git_blob_sha1=None, lfs=False)


FOLDER = [
    _file("config.json", 900),
    _file("model-00001-of-00002.safetensors", 5 * GIB),
    _file("model-00002-of-00002.safetensors", 3 * GIB),
    _file("tokenizer.model", 500_000),
    _file("tokenizer_config.json", 2_000),
]


def _build(info: RepoInfo, files: list[FileMetadata], store: StateStore, configs=None):
    return catalogue.build_model(
        info=info,
        files=files,
        revision="main",
        store=store,
        budget=starter_budget(32 * GIB),
        context_length=8192,
        configs=configs,
    )


def test_a_gguf_version_carries_the_hubs_architecture_and_its_quant(tmp_path: Path) -> None:
    info = RepoInfo(
        repo="org/repo-GGUF",
        raw={"gguf": {"architecture": "qwen3", "total": 8 * 10**9}, "config": {"model_type": "x"}},
    )
    files = [_file("Repo-UD-Q4_K_XL.gguf", 5 * GIB), _file("Repo-IQ2_XS.gguf", 3 * GIB)]
    model = _build(info, files, StateStore(tmp_path / "s.json"))
    facts = {c.label: c.facts for c in model.candidates}
    assert facts["UD-Q4_K_XL"].architecture == "qwen3"
    # Not a name a file's own metadata uses: not known until it is read.
    assert facts["UD-Q4_K_XL"].quantization is None
    assert facts["IQ2_XS"].quantization == "IQ2_XS"
    assert facts["IQ2_XS"].format == ModelFormat.gguf
    assert facts["IQ2_XS"].id == "catalogue:org/repo-GGUF:IQ2_XS"
    assert not facts["IQ2_XS"].approximate
    # The file's own name (LS4), which an engine that runs only named files reads.
    assert facts["IQ2_XS"].file == "Repo-IQ2_XS.gguf"
    # Its size (LS6), which an engine's fit is computed from.
    assert facts["IQ2_XS"].sizeBytes == 3 * GIB


def test_a_folder_carries_what_its_remote_config_says(tmp_path: Path) -> None:
    info = RepoInfo(repo="org/Model-4bit", raw={})
    assert catalogue.config_paths(FOLDER, repo=info.repo) == ["config.json"]
    store = StateStore(tmp_path / "s.json")
    read = {"config.json": safetensors.parse_config(json.dumps(MLX_CONFIG).encode())}
    facts = _build(info, FOLDER, store, read).candidates[0].facts
    assert facts.mlxQuantized is True
    assert facts.architecture == "Qwen3ForCausalLM"
    plain = {"config.json": safetensors.parse_config(json.dumps(PLAIN_CONFIG).encode())}
    assert _build(info, FOLDER, store, plain).candidates[0].facts.mlxQuantized is False
    # Unread: not known, never "not MLX".
    unread = _build(info, FOLDER, store, {"config.json": None}).candidates[0].facts
    assert unread.mlxQuantized is None and unread.architecture is None


def test_a_folder_download_brings_its_sentencepiece_tokenizer(tmp_path: Path) -> None:
    """Design doc section 7: `tokenizer.model` was left behind."""
    model = _build(RepoInfo(repo="org/m", raw={}), FOLDER, StateStore(tmp_path / "s.json"))
    assert "tokenizer.model" in [f.path for f in model.candidates[0].files]


def test_a_folder_already_on_disk_says_so(tmp_path: Path) -> None:
    """Design doc section 7: the folder's path never matched a file's name."""
    folder = tmp_path / "Model-4bit"
    on_disk = [
        ModelFile(path=str(folder / f.path), role=ModelFileRole.other, sizeBytes=f.size)
        for f in FOLDER
    ]
    store = StateStore(tmp_path / "s.json")
    store.replace_models(
        [
            LibraryModel(
                id="m1",
                path=str(folder),
                name="Model-4bit",
                format=ModelFormat.safetensors,
                status=ModelStatus.present,
                sizeBytes=sum(f.size for f in FOLDER),
                files=on_disk,
            )
        ],
        scanned_at=datetime.now(UTC),
    )
    owned = _build(RepoInfo(repo="org/Model-4bit", raw={}), FOLDER, store).candidates[0]
    assert owned.alreadyOwned is not None and owned.alreadyOwned.modelId == "m1"
    # A different-sized set under the same names is not the same download.
    bigger = [*FOLDER[:1], _file("model-00001-of-00002.safetensors", 6 * GIB), *FOLDER[2:]]
    assert _build(RepoInfo(repo="org/x", raw={}), bigger, store).candidates[0].alreadyOwned is None


def test_a_search_row_guesses_from_tags_and_says_it_guessed() -> None:
    row = catalogue.build_search_result(
        {
            "id": "mlx-community/Qwen3-4B-4bit",
            "tags": ["mlx", "safetensors"],
            "library_name": "mlx",
        }
    )
    assert [f.mlxQuantized for f in row.facts or []] == [True]
    assert all(f.approximate for f in row.facts or [])
    gguf = catalogue.build_search_result(
        {"id": "org/R-GGUF", "tags": ["gguf"], "gguf": {"architecture": "qwen4exp"}}
    )
    assert [(f.id, f.architecture) for f in gguf.facts or []] == [
        ("search:org/R-GGUF:gguf", "qwen4exp")
    ]
    plain = catalogue.build_search_result({"id": "org/P", "tags": ["safetensors"]})
    assert [f.mlxQuantized for f in plain.facts or []] == [False]


def test_a_starter_entry_carries_what_the_review_recorded(tmp_path: Path) -> None:
    listed = starter.load(
        str(
            write(
                tmp_path,
                {
                    "reviewed": "2026-10-01",
                    "engine": "llama_cpp b10999",
                    "classes": [entry("8B", size=5 * GIB, params=8 * 10**9)],
                },
            )
        )
    )
    built = starter.build(
        listed,
        budget=starter_budget(32 * GIB),
        context_length=8192,
        kv_cache_type=KvCacheType.f16,
    )
    facts = built.models[0].facts
    assert facts is not None
    assert facts.format == ModelFormat.gguf
    assert facts.architecture == built.models[0].architecture
    assert facts.id.startswith("starter:")
    assert facts.file and facts.file.endswith(".gguf") and "/" not in facts.file


# --- the detail route -----------------------------------------------------------

MLX_TREE = [
    {"type": "file", "path": "config.json", "size": 900, "oid": "1" * 40},
    {
        "type": "file",
        "path": "model.safetensors",
        "size": 2 * GIB,
        "oid": "2" * 40,
        "lfs": {"oid": "a" * 64, "size": 2 * GIB},
    },
    {"type": "file", "path": "tokenizer.json", "size": 9_000, "oid": "3" * 40},
]


def test_the_detail_reads_a_folders_config_before_download(
    configured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/tree/" in path:
            return httpx.Response(200, json=MLX_TREE)
        if path.startswith("/api/models/"):
            return httpx.Response(200, json={"sha": "c1", "tags": ["mlx"]})
        if path.endswith("/resolve/main/config.json"):
            reads.append(request.headers.get("Range", ""))
            return httpx.Response(206, content=json.dumps(MLX_CONFIG).encode())
        return httpx.Response(404)

    inner = httpx.AsyncClient(transport=httpx.MockTransport(upstream), follow_redirects=False)
    mock_hubs(configured_client.app, inner)
    monkeypatch.setattr(hardware, "detect", lambda: HOST)
    body = configured_client.get(
        "/v1/catalogue/model", params={"repo": "mlx-community/M-4bit"}
    ).json()
    facts = body["candidates"][0]["facts"]
    assert facts["mlxQuantized"] is True
    assert facts["architecture"] == "Qwen3ForCausalLM"
    assert reads and reads[0].startswith("bytes=0-")


def test_a_gguf_versions_architecture_is_never_the_configs_model_type(tmp_path: Path) -> None:
    """`general.architecture` and a config's `model_type` are two vocabularies;
    with no GGUF block from the hub the fact is not known, not borrowed."""
    info = RepoInfo(repo="org/r-GGUF", raw={"config": {"model_type": "llama"}})
    model = _build(info, [_file("r-Q4_K_M.gguf", GIB)], StateStore(tmp_path / "s.json"))
    assert model.candidates[0].facts is not None
    assert model.candidates[0].facts.architecture is None
