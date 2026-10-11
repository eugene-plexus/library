"""Which models are search models, and what a profile says about it (MR3).

The agent starts an embedding or reranking model the way llama-server needs
(`--embeddings`, `--reranking`) and decides from the file's own header, by the same
two keys this reads. The console predicts with these, so each of them is checked
against real GGUF bytes shaped like the real files (read off this box on
2026-10-11): an embedder is a pooling type, a reranker is rank pooling (4) or
classifier labels, and a chat model is neither.

`embedding` is false for a reranker, though its header carries a pooling type: it
has its own flag, and a console that listed it as "embedding" would offer to start
it as the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import fit, preflight
from eugene_plexus_library._generated.models import (
    Arch,
    Gpu,
    HostHardware,
    ModelProfileSpec,
    Os,
    ProfileMode,
    Vendor,
)
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.paths import model_id
from eugene_plexus_library.scanner import Scanner
from eugene_plexus_library.store import StateStore

from .conftest import (
    cross_encoder_kv,
    decoder_embedding_kv,
    embedding_kv,
    qwen_like_kv,
    reranker_kv,
    write_gguf,
)
from .test_preflight import SlicedFile, client_for
from .test_routes import _one_model


def _meta(tmp_path: Path, kv: dict) -> gguf.GgufMetadata:
    return gguf.read_metadata(write_gguf(tmp_path / "m.gguf", kv))


# --- the header ----------------------------------------------------------------------


def test_rank_pooling_is_a_reranker(tmp_path: Path) -> None:
    meta = _meta(tmp_path, reranker_kv())
    assert meta.is_rerank is True
    assert meta.is_search_embedding is False
    assert meta.is_embedding is True, "rank pooling is a pooling type"


def test_pooling_type_4_alone_is_a_reranker(tmp_path: Path) -> None:
    kv = reranker_kv()
    del kv["qwen3.classifier.output_labels"]
    assert _meta(tmp_path, kv).is_rerank is True


def test_classifier_labels_alone_are_a_reranker(tmp_path: Path) -> None:
    kv = reranker_kv()
    del kv["qwen3.pooling_type"]
    meta = _meta(tmp_path, kv)
    assert meta.is_rerank is True
    assert meta.is_embedding is False, "no pooling type, and attention is not stated"


def test_empty_classifier_labels_are_not_a_reranker(tmp_path: Path) -> None:
    kv = qwen_like_kv(architecture="qwen3")
    kv["qwen3.classifier.output_labels"] = []
    assert _meta(tmp_path, kv).is_rerank is False


def test_a_long_list_of_labels_is_still_labels(tmp_path: Path) -> None:
    """The reader steps over an array longer than 512 rather than keeping it; the
    length it recorded is the answer."""
    kv = qwen_like_kv(architecture="qwen3")
    kv["qwen3.classifier.output_labels"] = [f"class{n}" for n in range(700)]
    meta = _meta(tmp_path, kv)
    assert meta.arch_key("classifier.output_labels") is None
    assert meta.is_rerank is True


def test_an_embedder_is_not_a_reranker(tmp_path: Path) -> None:
    """Qwen3-Embedding: last pooling (3), a decoder, and a chat template it kept."""
    meta = _meta(tmp_path, decoder_embedding_kv())
    assert (meta.is_embedding, meta.is_rerank, meta.is_search_embedding) == (True, False, True)


def test_an_encoder_with_mean_pooling_is_an_embedder(tmp_path: Path) -> None:
    meta = _meta(tmp_path, embedding_kv())
    assert (meta.is_embedding, meta.is_rerank, meta.is_search_embedding) == (True, False, True)


def test_a_chat_model_is_neither(tmp_path: Path) -> None:
    meta = _meta(tmp_path, qwen_like_kv())
    assert (meta.is_embedding, meta.is_rerank, meta.is_search_embedding) == (False, False, False)


def test_the_cross_encoder_the_header_cannot_name_reads_as_an_embedder(tmp_path: Path) -> None:
    """The gpustack bge-reranker-v2-m3: non-causal, with its classifier head in the tensors
    and neither key in the metadata. The honest answer from the header is the wrong one, and
    the contract says so; `ModelProfileSpec.mode` is how a person says otherwise."""
    meta = _meta(tmp_path, cross_encoder_kv())
    assert (meta.is_embedding, meta.is_rerank) == (True, False)


def test_rerank_is_read_from_the_architectures_own_key(tmp_path: Path) -> None:
    """The pooling type belongs to `general.architecture`'s prefix; another prefix's is not it."""
    kv = qwen_like_kv(architecture="llama")
    kv["clip.pooling_type"] = 4
    assert _meta(tmp_path, kv).is_rerank is False


# --- the scanner ---------------------------------------------------------------------


def test_the_scanner_says_what_each_file_is(tmp_path: Path) -> None:
    write_gguf(tmp_path / "chat" / "chat.gguf", qwen_like_kv(name="chat"))
    write_gguf(tmp_path / "embed" / "nomic.gguf", embedding_kv(name="nomic"))
    write_gguf(tmp_path / "embed" / "qwen3-embed.gguf", decoder_embedding_kv(name="qwen3-embed"))
    write_gguf(tmp_path / "rerank" / "qwen3-rerank.gguf", reranker_kv(name="qwen3-rerank"))
    write_gguf(tmp_path / "rerank" / "bge.gguf", cross_encoder_kv(name="bge"))

    by_name = {m.name: m.capabilities for m in Scanner().scan([tmp_path]).models}

    def kinds(name: str) -> tuple[bool, bool, bool]:
        caps = by_name[name]
        assert caps is not None
        return (bool(caps.chat), bool(caps.embedding), bool(caps.rerank))

    assert kinds("chat") == (True, False, False)
    assert kinds("nomic") == (False, True, False)
    assert kinds("qwen3-embed") == (False, True, False)
    assert kinds("qwen3-rerank") == (False, False, True), (
        "a reranker is not a chat or an embedding model"
    )
    assert kinds("bge") == (False, True, False), "the header's honest answer"


def test_a_reranker_is_not_listed_as_chat_even_with_a_chat_template(tmp_path: Path) -> None:
    """Its file keeps the template of the model it was made from, so a template is no evidence."""
    write_gguf(tmp_path / "r.gguf", reranker_kv(name="r"))
    [model] = Scanner().scan([tmp_path]).models
    assert model.capabilities is not None
    assert model.capabilities.chatTemplate is True and model.capabilities.chat is False


def _labels_only_reranker_kv() -> dict:
    """A reranker the header names by its classifier labels alone: no pooling type."""
    kv = reranker_kv(name="labels")
    del kv["qwen3.pooling_type"]
    return kv


def test_a_reranker_the_header_names_only_by_its_labels_is_not_listed_as_chat(
    tmp_path: Path,
) -> None:
    """With no pooling type it is not an embedding model either, so the labels are all
    there is to say what it is, and `chat` has to follow them and not only the pooling
    type (a break that listed it as chat was the one sabotage did not catch)."""
    write_gguf(tmp_path / "labels.gguf", _labels_only_reranker_kv())
    [model] = Scanner().scan([tmp_path]).models
    caps = model.capabilities
    assert caps is not None
    assert (bool(caps.chat), bool(caps.embedding), bool(caps.rerank)) == (False, False, True)


# --- before a download -----------------------------------------------------------------


@pytest.fixture
def budget():
    gib = 1024**3
    return fit.budget_from_hardware(
        HostHardware(
            hostname="dev",
            os=Os.windows,
            arch=Arch.x64,
            ramTotalBytes=96 * gib,
            ramAvailableBytes=58 * gib,
            gpus=[
                Gpu(
                    index=0,
                    name="RTX 5090",
                    vendor=Vendor.nvidia,
                    vramTotalBytes=32 * gib,
                    vramFreeBytes=29 * gib,
                )
            ],
        )
    )


@pytest.mark.parametrize(
    ("kv", "chat", "embedding", "rerank"),
    [
        (qwen_like_kv(), True, False, False),
        (embedding_kv(), False, True, False),
        (decoder_embedding_kv(), False, True, False),
        (reranker_kv(), False, False, True),
        (_labels_only_reranker_kv(), False, False, True),
    ],
)
async def test_preflight_says_it_before_the_file_is_downloaded(
    tmp_path: Path,
    budget: object,
    kv: dict,
    chat: bool,
    embedding: bool,
    rerank: bool,
) -> None:
    path = write_gguf(tmp_path / "m.gguf", kv)
    served = SlicedFile(path)
    result = await preflight.preflight(
        client_for(served),
        repo="org/repo",
        revision="main",
        path="m.gguf",
        budget=budget,
        context_length=8192,
    )
    caps = result.capabilities
    assert caps is not None
    assert (bool(caps.chat), bool(caps.embedding), bool(caps.rerank)) == (chat, embedding, rerank)


# --- a profile keeps what a person said ------------------------------------------------


def _spec(**fields: object) -> ModelProfileSpec:
    return ModelProfileSpec.model_validate({"name": "default", "engine": "llama_cpp", **fields})


@pytest.fixture
def store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.json")


def test_a_profile_keeps_its_mode(tmp_path: Path, store: StateStore) -> None:
    identifier = model_id(tmp_path / "m.gguf")
    created = store.create_profile(identifier, _spec(mode="reranking"))
    assert created.mode == ProfileMode.reranking
    assert store.get_profile(identifier, created.id).mode == ProfileMode.reranking  # type: ignore[union-attr]


def test_replacing_a_profile_replaces_its_mode_too(tmp_path: Path, store: StateStore) -> None:
    """A whole-document swap: the mode a person set last is the mode it has, and
    saying nothing clears it back to the agent's own reading."""
    identifier = model_id(tmp_path / "m.gguf")
    created = store.create_profile(identifier, _spec(mode="chat"))
    swapped = store.replace_profile(identifier, created.id, _spec(mode="embedding"))
    assert swapped is not None and swapped.mode == ProfileMode.embedding
    cleared = store.replace_profile(identifier, created.id, _spec())
    assert cleared is not None and cleared.mode is None


def test_a_profile_that_says_no_mode_has_none(tmp_path: Path, store: StateStore) -> None:
    """Absent means auto, so the stored file is not filled with `mode: auto`."""
    identifier = model_id(tmp_path / "m.gguf")
    profile = store.create_profile(identifier, _spec())
    assert profile.mode is None


def test_a_mode_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    identifier = model_id(tmp_path / "m.gguf")
    StateStore(path).create_profile(identifier, _spec(mode="chat"))

    second = StateStore(path)
    second.load()
    [profile] = second.list_profiles(identifier)
    assert profile.mode == ProfileMode.chat


def test_the_stored_file_holds_a_mode_only_when_one_was_said(tmp_path: Path) -> None:
    """Absent means auto: the file is not filled with `mode: auto` for every profile."""
    from eugene_plexus_library import profile_storage

    identifier = model_id(tmp_path / "m.gguf")
    store = StateStore(tmp_path / "state.json")
    store.create_profile(identifier, _spec(name="plain"))
    store.create_profile(identifier, _spec(name="said", mode="reranking"))

    stored, _identities = profile_storage.read(store._profiles_path)
    entries = {entry["name"]: entry for entry in stored[identifier]}
    assert "mode" not in entries["plain"]
    assert entries["said"]["mode"] == "reranking"


def test_the_http_surface_takes_and_returns_a_mode(
    configured_client: TestClient, models_dir: Path
) -> None:
    model = _one_model(configured_client, models_dir)
    created = configured_client.post(
        f"/v1/models/{model}/profiles",
        json={"name": "as chat", "engine": "llama_cpp", "mode": "chat"},
    )
    assert created.status_code == 201, created.text
    profile = created.json()
    assert profile["mode"] == "chat"

    replaced = configured_client.put(
        f"/v1/models/{model}/profiles/{profile['id']}",
        json={"name": "as chat", "engine": "llama_cpp", "mode": "embedding"},
    )
    assert replaced.json()["mode"] == "embedding"
    listed = configured_client.get(f"/v1/models/{model}/profiles").json()["profiles"]
    assert [p["mode"] for p in listed] == ["embedding"]


def test_the_http_surface_refuses_a_mode_the_contract_does_not_know(
    configured_client: TestClient, models_dir: Path
) -> None:
    model = _one_model(configured_client, models_dir)
    refused = configured_client.post(
        f"/v1/models/{model}/profiles",
        json={"name": "x", "engine": "llama_cpp", "mode": "summarising"},
    )
    assert refused.status_code == 422


def test_the_model_list_carries_rerank(configured_client: TestClient, models_dir: Path) -> None:
    write_gguf(models_dir / "r.gguf", reranker_kv(name="r"))
    configured_client.post("/v1/scan")
    from .test_routes import _wait_for_scan

    _wait_for_scan(configured_client)
    [model] = configured_client.get("/v1/models").json()["models"]
    assert model["capabilities"]["rerank"] is True
    assert model["capabilities"]["embedding"] is False
    assert model["capabilities"]["chat"] is False
