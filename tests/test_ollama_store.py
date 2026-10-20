"""CX9: models already in Ollama are listed in place, read-only.

A fake store is written the way Ollama writes one (hash-named blobs, a
manifest per tag, the legacy and the v2 layout, lists with `llamacpp`, `ggml`
and `mlx` children) and read back through the scanner and the routes. The
manifest-list layout was read from Ollama's `manifest/manifest.go`
(2026-10-20): a child is inline or a reference to a blob holding its manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_library import catalogue, deletion, ollama_detect
from eugene_plexus_library._generated.models import ModelFormat
from eugene_plexus_library.app import create_app
from eugene_plexus_library.downloads import DownloadError, resolve_destination
from eugene_plexus_library.folders import coerce_folders, validate_folders
from eugene_plexus_library.formats import ollama as ollama_store
from eugene_plexus_library.routes import run_operations as run_routes
from eugene_plexus_library.run_operations import PreparedRequest
from eugene_plexus_library.scanner import Scanner
from eugene_plexus_library.settings import Settings

from .conftest import projector_kv, qwen_like_kv, wait_for_scan, write_gguf
from .test_run_preparation import _to_preparing

MODEL = "application/vnd.ollama.image.model"
PROJECTOR = "application/vnd.ollama.image.projector"
TEMPLATE = "application/vnd.ollama.image.template"
TENSOR = "application/vnd.ollama.image.tensor"
LIST = "application/vnd.ollama.manifest.list.v2+json"
MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"


def digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def layer(media: str, seed: str) -> dict[str, Any]:
    return {"mediaType": media, "digest": f"sha256:{digest(seed)}", "size": 1}


def manifest(*layers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "mediaType": MANIFEST,
        "config": layer("application/vnd.docker.container.image.v1+json", "config"),
        "layers": list(layers),
        **extra,
    }


def blob(store: Path, seed: str, kv: dict[str, Any] | None = None) -> Path:
    path = store / "blobs" / f"sha256-{digest(seed)}"
    if kv is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a gguf")
    else:
        write_gguf(path, kv)
    return path


def named(
    store: Path,
    directory: str,
    name: str,
    body: dict[str, Any],
    *,
    host: str = "registry.ollama.ai",
    namespace: str = "library",
) -> Path:
    model, _, tag = name.partition(":")
    path = store / directory / host / namespace / model / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def blob_manifest(store: Path, seed: str, body: dict[str, Any]) -> dict[str, Any]:
    """A list child that is a reference: the child's JSON lives in a blob."""
    path = store / "blobs" / f"sha256-{digest(seed)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return {"mediaType": MANIFEST, "digest": f"sha256:{digest(seed)}"}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """What Ollama v0.40 leaves behind, one of each case the design names."""
    root = tmp_path / "ollama" / "models"
    # qwen3: a legacy manifest and a v2 one pointing at the same blob.
    blob(root, "qwen3", qwen_like_kv(name="Qwen3 8B", architecture="qwen3"))
    named(root, "manifests", "qwen3:8b", manifest(layer(MODEL, "qwen3"), layer(TEMPLATE, "t")))
    named(root, "manifests-v2", "qwen3:latest", manifest(layer(MODEL, "qwen3")), host="ollama.com")
    # The same name in both layouts: the v2 one wins.
    blob(root, "llama-v2", qwen_like_kv(name="Llama V2"))
    named(root, "manifests", "llama3:latest", manifest(layer(MODEL, "llama-gone")))
    named(
        root, "manifests-v2", "llama3:latest", manifest(layer(MODEL, "llama-v2")), host="ollama.com"
    )
    # Migrated gemma3: a list whose llamacpp child is a reference to a blob.
    blob(root, "gemma-new", qwen_like_kv(name="Gemma", architecture="gemma3"))
    blob(root, "gemma-old", qwen_like_kv(name="Gemma", architecture="gemma3"))
    blob(root, "gemma-proj", projector_kv(name="Gemma"))
    new_child = blob_manifest(
        root, "gemma-child", manifest(layer(MODEL, "gemma-new"), layer(PROJECTOR, "gemma-proj"))
    )
    named(
        root,
        "manifests-v2",
        "gemma3:4b",
        {
            "schemaVersion": 2,
            "mediaType": LIST,
            "manifests": [
                {"runner": "mlx", "format": "safetensors", **manifest(layer(TENSOR, "mlx"))},
                {"runner": "ggml", "format": "gguf", **manifest(layer(MODEL, "gemma-old"))},
                {**new_child, "runner": "llamacpp", "format": "gguf"},
            ],
        },
        host="ollama.com",
    )
    # Ollama's internal name for the migrated one is never listed.
    named(
        root,
        "manifests-v2",
        f"llamacpp:{digest('gemma-new')}",
        manifest(layer(MODEL, "gemma-new")),
        host="ollama.com",
    )
    # An older gemma3 that Ollama has not migrated.
    named(root, "manifests", "gemma3:old", manifest(layer(MODEL, "gemma-old")))
    # An MLX/safetensors model llama.cpp cannot load.
    named(
        root,
        "manifests-v2",
        "qwen3-mlx:latest",
        manifest(layer(TENSOR, "mlx-t"), runner="mlx", format="safetensors"),
        host="ollama.com",
    )
    # A list with only an MLX child.
    named(
        root,
        "manifests-v2",
        "mlx-only:latest",
        {
            "schemaVersion": 2,
            "mediaType": LIST,
            "manifests": [
                {"runner": "mlx", "format": "safetensors", **manifest(layer(TENSOR, "m"))}
            ],
        },
        host="ollama.com",
    )
    # A model by someone else, and one from another registry.
    blob(root, "someone", qwen_like_kv(name="Someone"))
    named(
        root,
        "manifests-v2",
        "cool:q4",
        manifest(layer(MODEL, "someone")),
        host="ollama.com",
        namespace="someone",
    )
    named(
        root,
        "manifests-v2",
        "m:t",
        manifest(layer(MODEL, "someone")),
        host="example.org",
        namespace="team",
    )
    # A download in progress: never a model.
    (root / "blobs" / f"sha256-{digest('big')}-partial").write_bytes(b"half")
    (root / "blobs" / f"sha256-{digest('big')}-partial-0").write_bytes(b"half")
    return root


def scan(store: Path) -> Any:
    return Scanner().scan([store], ollama_roots=[store])


def by_name(result: Any) -> dict[str, Any]:
    return {m.name: m for m in result.models}


# --- finding the store ----------------------------------------------------------


def make_store(path: Path) -> Path:
    (path / "blobs").mkdir(parents=True)
    (path / "manifests").mkdir()
    return path


def test_the_variable_names_the_store_and_nothing_else_is_tried(tmp_path: Path) -> None:
    home = tmp_path / "home"
    make_store(home / ".ollama" / "models")
    mine = make_store(tmp_path / "mine")
    assert ollama_detect.find_store(environ={"OLLAMA_MODELS": str(mine)}, home=home) == mine
    # Ollama itself would read the variable's folder, so a variable that names no store finds none.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ollama_detect.find_store(environ={"OLLAMA_MODELS": str(empty)}, home=home) is None


def test_the_home_folder_is_the_default_place(tmp_path: Path) -> None:
    home = tmp_path / "home"
    found = make_store(home / ".ollama" / "models")
    assert ollama_detect.find_store(environ={}, home=home, platform="win32") == found


def test_the_linux_service_store_is_a_last_place_on_linux_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_store(tmp_path / "usr" / "share" / "ollama" / ".ollama" / "models")
    monkeypatch.setattr(ollama_detect, "LINUX_SERVICE_STORE", str(service))
    home = tmp_path / "home"
    assert ollama_detect.find_store(environ={}, home=home, platform="linux") == service
    assert ollama_detect.find_store(environ={}, home=home, platform="darwin") is None


def test_a_store_needs_blobs_and_a_manifest_folder(tmp_path: Path) -> None:
    (tmp_path / "a" / "blobs").mkdir(parents=True)
    (tmp_path / "b" / "manifests-v2").mkdir(parents=True)
    (tmp_path / "c" / "blobs").mkdir(parents=True)
    (tmp_path / "c" / "manifests-v2").mkdir()
    assert not ollama_store.is_store(tmp_path / "a")
    assert not ollama_store.is_store(tmp_path / "b")
    assert ollama_store.is_store(tmp_path / "c")
    assert not ollama_store.is_store(tmp_path / "missing")


# --- reading the store ------------------------------------------------------------


def test_one_model_per_blob_named_by_its_tags(store: Path) -> None:
    found = by_name(scan(store))
    qwen = found["qwen3:8b"]
    assert qwen.ollama.tags == ["qwen3:8b", "qwen3:latest"]
    assert qwen.path == str(store / "blobs" / f"sha256-{digest('qwen3')}")
    assert qwen.root == str(store)
    assert qwen.gguf.quantization == "Q4_K_M"
    assert qwen.architecture == "qwen3"
    assert "qwen3:latest" not in found


def test_the_v2_manifest_wins_over_the_legacy_one_of_the_same_name(store: Path) -> None:
    found = by_name(scan(store))
    assert found["llama3:latest"].path.endswith(f"sha256-{digest('llama-v2')}")


def test_a_list_is_read_through_its_llamacpp_child_resolved_by_digest(store: Path) -> None:
    found = by_name(scan(store))
    gemma = found["gemma3:4b"]
    assert gemma.path.endswith(f"sha256-{digest('gemma-new')}")
    assert gemma.ollama.needsMigration is None
    # The ggml child's blob is the older gemma3, listed only under its own tag.
    old = found["gemma3:old"]
    assert old.path.endswith(f"sha256-{digest('gemma-old')}")
    assert old.ollama.tags == ["gemma3:old"]


def test_the_ggml_child_is_used_when_there_is_no_llamacpp_one(store: Path) -> None:
    named(
        store,
        "manifests-v2",
        "ggml-only:latest",
        {
            "schemaVersion": 2,
            "mediaType": LIST,
            "manifests": [
                {"runner": "mlx", "format": "safetensors", **manifest(layer(TENSOR, "m"))},
                {"runner": "ggml", "format": "gguf", **manifest(layer(MODEL, "qwen3"))},
            ],
        },
        host="ollama.com",
    )
    found = by_name(scan(store))
    assert found["ggml-only:latest"].ollama.tags == ["ggml-only:latest", "qwen3:8b", "qwen3:latest"]


def test_the_internal_llamacpp_name_is_not_a_tag(store: Path) -> None:
    for model in scan(store).models:
        assert not any(tag.startswith("llamacpp") for tag in model.ollama.tags)


def test_a_projector_layer_is_the_models_projector(store: Path) -> None:
    gemma = by_name(scan(store))["gemma3:4b"]
    assert gemma.gguf.projectorPath == str(store / "blobs" / f"sha256-{digest('gemma-proj')}")
    assert gemma.capabilities.vision is True
    assert [f.role.value for f in gemma.files] == ["weights", "projector"]
    assert by_name(scan(store))["qwen3:8b"].capabilities.vision is False


def test_a_model_the_migration_has_not_reached_is_marked(store: Path) -> None:
    found = by_name(scan(store))
    assert found["gemma3:old"].ollama.needsMigration is True
    assert found["qwen3:8b"].ollama.needsMigration is None


def test_models_llama_cpp_cannot_load_are_skipped_with_the_reason(store: Path) -> None:
    result = scan(store)
    skipped = {Path(s.path).parent.name + ":" + Path(s.path).name: s for s in result.skipped}
    assert "not for llama.cpp" in skipped["qwen3-mlx:latest"].detail
    assert "not for llama.cpp" in skipped["mlx-only:latest"].detail
    names = {m.name for m in result.models}
    assert not names & {"qwen3-mlx:latest", "mlx-only:latest"}


def test_other_namespaces_and_registries_keep_their_names(store: Path) -> None:
    found = by_name(scan(store))
    # Two names for one blob: one model, named by the first of its sorted tags.
    assert found["example.org/team/m:t"].ollama.tags == ["example.org/team/m:t", "someone/cool:q4"]


def test_partial_blobs_are_never_models(store: Path) -> None:
    result = scan(store)
    assert not any("-partial" in Path(m.path).name for m in result.models)


def test_an_ordinary_walk_finds_nothing_in_a_store(store: Path) -> None:
    """The reason a store is its own kind: `blobs` is pruned and nothing ends `.gguf`."""
    assert Scanner().scan([store]).models == []


def test_a_blob_missing_from_the_store_is_skipped_not_listed(store: Path) -> None:
    named(
        store, "manifests-v2", "gone:latest", manifest(layer(MODEL, "nowhere")), host="ollama.com"
    )
    result = scan(store)
    assert "gone:latest" not in {m.name for m in result.models}
    assert any("not in blobs" in s.detail for s in result.skipped)


def test_a_folder_marked_ollama_that_is_not_a_store_says_so(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    result = Scanner().scan([tmp_path / "plain"], ollama_roots=[tmp_path / "plain"])
    assert result.models == []
    assert any("not an Ollama model store" in s.detail for s in result.skipped)


def test_the_cache_is_used_and_tags_still_follow_the_manifests(store: Path) -> None:
    first = scan(store)
    cached = {m.path: m for m in first.models}
    from eugene_plexus_library.scanner import CacheKey

    def lookup(key: CacheKey) -> Any:
        return cached.get(key[0]) or next(
            (m for p, m in cached.items() if p.casefold() == key[0].casefold()), None
        )

    named(store, "manifests-v2", "qwen3:new", manifest(layer(MODEL, "qwen3")), host="ollama.com")
    again = Scanner(cache_lookup=lookup).scan([store], ollama_roots=[store])
    assert by_name(again)["qwen3:8b"].ollama.tags == ["qwen3:8b", "qwen3:latest", "qwen3:new"]


# --- folders in the config ------------------------------------------------------


def test_a_folder_kind_is_kept_validated_and_defaults_to_a_folder(tmp_path: Path) -> None:
    assert coerce_folders([{"path": "/a", "kind": "ollama"}]) == [
        {"path": "/a", "mounts": [], "kind": "ollama"}
    ]
    assert coerce_folders([{"path": "/a", "kind": "folder"}, {"path": "/b", "kind": "nope"}]) == [
        {"path": "/a", "mounts": []},
        {"path": "/b", "mounts": []},
    ]
    assert validate_folders([{"path": "/a", "kind": "ollama"}]) is None
    assert "`kind` must be one of" in (validate_folders([{"path": "/a", "kind": "nope"}]) or "")
    assert "`kind` must be one of" in (validate_folders([{"path": "/a", "kind": ["x"]}]) or "")


@pytest.fixture
def ollama_client(
    tmp_path: Path, models_dir: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A library over `models_dir` on a machine whose Ollama store is `store`."""
    monkeypatch.setattr(ollama_detect, "find_store", lambda **_: store)
    settings = Settings(config_file=tmp_path / "config.yaml", state_file=tmp_path / "state.json")
    settings.config_file.write_text(
        f"modelRoots:\n  - path: {models_dir.as_posix()}\n    mounts: []\n", encoding="utf-8"
    )
    with TestClient(create_app(settings=settings)) as client:
        wait_for_scan(client)
        client.post("/v1/scan")
        wait_for_scan(client)
        yield client


def test_the_found_store_is_a_folder_of_kind_ollama_and_not_in_the_config(
    ollama_client: TestClient, models_dir: Path, store: Path, tmp_path: Path
) -> None:
    folders = ollama_client.get("/v1/folders").json()["folders"]
    assert folders == [
        {"path": folders[0]["path"], "mounts": [], "kind": "folder"},
        {"path": str(store), "mounts": [], "kind": "ollama", "found": True},
    ]
    config = ollama_client.get("/v1/config").json()
    assert [Path(f["path"]) for f in config["modelRoots"]] == [models_dir]
    assert Path(folders[0]["path"]) == models_dir
    assert config["findOllama"] is True
    names = {m["name"] for m in ollama_client.get("/v1/models").json()["models"]}
    assert {"qwen3:8b", "gemma3:4b"} <= names


def test_turning_find_ollama_off_removes_the_store_and_its_models(
    ollama_client: TestClient, models_dir: Path
) -> None:
    assert ollama_client.patch("/v1/config", json={"findOllama": False}).status_code == 200
    assert [Path(f["path"]) for f in ollama_client.get("/v1/folders").json()["folders"]] == [
        models_dir
    ]
    ollama_client.post("/v1/scan")
    wait_for_scan(ollama_client)
    models = ollama_client.get("/v1/models").json()["models"]
    assert not [m for m in models if m["status"] == "present" and m.get("ollama")]


def test_a_store_added_by_hand_is_listed_once_and_suppresses_the_find(
    ollama_client: TestClient, models_dir: Path, store: Path
) -> None:
    answer = ollama_client.patch(
        "/v1/config",
        json={
            "modelRoots": [
                {"path": str(models_dir), "mounts": []},
                {"path": str(store), "mounts": [], "kind": "ollama"},
            ]
        },
    )
    assert answer.json()["rejected"] == []
    folders = ollama_client.get("/v1/folders").json()["folders"]
    assert [(f["path"], f["kind"], f.get("found")) for f in folders] == [
        (str(models_dir), "folder", None),
        (str(store), "ollama", None),
    ]
    ollama_client.post("/v1/scan")
    wait_for_scan(ollama_client)
    models = ollama_client.get("/v1/models").json()["models"]
    assert sum(1 for m in models if m["name"] == "qwen3:8b") == 1


# --- read-only, enforced in code -------------------------------------------------


def test_a_download_may_not_name_an_ollama_folder(
    ollama_client: TestClient, models_dir: Path, store: Path
) -> None:
    before = sorted(p.name for p in store.rglob("*"))
    answer = ollama_client.post(
        "/v1/downloads", json={"repo": "a/b", "files": ["m.gguf"], "root": str(store)}
    )
    assert answer.status_code == 409, answer.text
    assert "Ollama" in json.dumps(answer.json())
    assert sorted(p.name for p in store.rglob("*")) == before


def test_resolve_destination_refuses_a_read_only_root_and_never_defaults_to_one(
    tmp_path: Path,
) -> None:
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    with pytest.raises(DownloadError) as refused:
        resolve_destination(
            repo="a/b",
            roots=[ours],
            root=str(theirs),
            subdirectory=None,
            layout="flat",
            read_only=[theirs],
        )
    assert refused.value.status == 409 and refused.value.code == "ReadOnlyFolder"
    # With nothing writable, the default is no folder rather than the store.
    with pytest.raises(DownloadError) as none:
        resolve_destination(
            repo="a/b", roots=[], root=None, subdirectory=None, layout="flat", read_only=[theirs]
        )
    assert none.value.code == "NoRootsConfigured"


def test_only_a_found_store_leaves_a_download_nowhere_to_go(
    tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama_detect, "find_store", lambda **_: store)
    settings = Settings(config_file=tmp_path / "c.yaml", state_file=tmp_path / "s.json")
    with TestClient(create_app(settings=settings)) as client:
        answer = client.post("/v1/downloads", json={"repo": "a/b", "files": ["m.gguf"]})
        assert answer.status_code == 409
        assert answer.json()["detail"]["detail"].startswith("No model directories")
        assert not (store / "a").exists()


def test_provenance_never_lands_in_a_store(
    ollama_client: TestClient, models_dir: Path, store: Path
) -> None:
    before = sorted(str(p) for p in store.rglob("*"))
    entry = store / "blobs" / f"sha256-{digest('qwen3')}"
    named_root = ollama_client.post(
        "/v1/models/prepared",
        json={
            "name": "x",
            "root": str(store),
            "provenance": {"engine": "strata", "entry": "x.json"},
        },
    )
    assert named_root.status_code == 409
    assert named_root.json()["detail"]["title"] == "Ollama manages this folder"
    beside = ollama_client.post(
        "/v1/models/prepared",
        json={"name": "x", "provenance": {"engine": "strata", "entry": str(entry)}},
    )
    assert beside.status_code == 400, beside.text
    assert sorted(str(p) for p in store.rglob("*")) == before
    assert not list(models_dir.glob("*.eugene-prepared.json"))


def _qwen(client: TestClient) -> dict[str, Any]:
    models = client.get("/v1/models").json()["models"]
    return next(m for m in models if m["name"] == "qwen3:8b")


def test_an_ollama_model_cannot_be_deleted_or_planned_for_deletion(
    ollama_client: TestClient, store: Path
) -> None:
    model = _qwen(ollama_client)
    blob_file = Path(model["path"])
    plan = ollama_client.get(f"/v1/models/{model['id']}/deletion")
    assert plan.status_code == 409
    assert plan.json()["detail"]["title"] == "Ollama manages this model"
    assert "Ollama's file" in plan.json()["detail"]["detail"]
    gone = ollama_client.post(f"/v1/models/{model['id']}/delete", json={"token": "x"})
    assert gone.status_code == 409
    assert gone.json()["detail"]["title"] == "Ollama manages this model"
    forget = ollama_client.delete(f"/v1/models/{model['id']}")
    assert forget.status_code == 409
    assert forget.json()["detail"]["title"] == "Ollama manages this model"
    assert blob_file.is_file()


def test_the_removal_itself_refuses_a_file_in_a_store(store: Path) -> None:
    held = store / "blobs" / f"sha256-{digest('qwen3')}"
    mine = store.parent / "mine.gguf"
    mine.write_bytes(b"x")
    with pytest.raises(deletion.DeleteRefused):
        deletion.remove([str(mine), str(held)], [store.parent], protected=[store])
    assert held.is_file() and mine.is_file()


def test_discover_never_offers_an_ollama_file_as_already_on_disk(
    ollama_client: TestClient,
) -> None:
    """A candidate with the blob's own name and size is still not "already owned":
    that would point a new file at Ollama's store."""
    path = Path(_qwen(ollama_client)["path"])
    group = SimpleNamespace(
        files=[SimpleNamespace(path=path.name, size=path.stat().st_size)],
        format=ModelFormat.gguf,
        size=path.stat().st_size,
    )
    assert catalogue._owned(group, ollama_client.app.state.state_store) is None


# --- what is prepared from an Ollama model ----------------------------------------


def test_a_preparation_from_an_ollama_model_is_received_and_adopted_in_the_first_writable_folder(
    ollama_client: TestClient, models_dir: Path, store: Path
) -> None:
    model = _qwen(ollama_client)
    intent = {"modelId": model["id"], "preparation": {"engine": "strata"}}
    assert ollama_client.put("/v1/run-operations/p", json=intent).status_code == 202
    lease = _to_preparing(ollama_client)
    # The node spells the folder as the model's root: Ollama's store.
    sent = ollama_client.put(
        "/v1/run-operations/p/files",
        params={"path": "Strata-data/strata.json", "offset": 0, "lease": lease},
        content=b"{}",
    )
    assert sent.status_code == 200, sent.text
    assert (models_dir / "Strata-data" / "strata.json.eugene-upload").is_file()
    assert not (store / "Strata-data").exists()


def _prepared_body(entry: Path) -> PreparedRequest:
    return PreparedRequest.model_validate(
        {
            "lease": "l",
            "name": "from-ollama",
            "provenance": {"engine": "strata", "entry": str(entry)},
        }
    )


def test_adopting_what_was_prepared_from_a_store_model_lands_in_the_writable_folder(
    tmp_path: Path, store: Path
) -> None:
    mine = tmp_path / "mine"
    (mine / "Strata-data").mkdir(parents=True)
    (mine / "Strata-data" / "strata.json").write_text("{}", encoding="utf-8")
    target, root = run_routes._adopt(
        [mine], _prepared_body(store / "Strata-data" / "strata.json"), [store]
    )
    assert root == mine and target.parent == mine / "Strata-data"
    assert not (store / "Strata-data").exists()
    assert not [p for p in store.rglob("*.eugene-prepared.json")]


def test_adopting_with_no_writable_folder_refuses_plainly(tmp_path: Path, store: Path) -> None:
    with pytest.raises(HTTPException) as refused:
        run_routes._adopt([], _prepared_body(store / "Strata-data" / "x.json"), [store])
    assert refused.value.status_code == 409
    assert "Ollama store" in str(refused.value.detail)
    assert not (store / "Strata-data").exists()


# --- review fixes: startup, nesting, hostile manifests -----------------------------


def test_detection_never_raises_when_there_is_no_home_folder(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    with caplog.at_level("WARNING"):
        assert ollama_detect.find_store(environ={}) is None
    assert sum("could not look for Ollama" in r.message for r in caplog.records) == 1

    def no_expansion(self: Path) -> Path:
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "expanduser", no_expansion)
    assert ollama_detect.find_store(environ={"OLLAMA_MODELS": "~/m"}) is None


def test_the_library_starts_and_answers_with_no_home_folder(
    tmp_path: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)

    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    settings = Settings(config_file=tmp_path / "c.yaml", state_file=tmp_path / "s.json")
    settings.config_file.write_text(
        f"modelRoots:\n  - path: {models_dir.as_posix()}\n    mounts: []\n", encoding="utf-8"
    )
    with TestClient(create_app(settings=settings)) as client:
        wait_for_scan(client)
        assert client.post("/v1/scan").status_code in (200, 202)
        wait_for_scan(client)
        assert [f["kind"] for f in client.get("/v1/folders").json()["folders"]] == ["folder"]


def _patch(values: dict[str, Any]) -> Any:
    from eugene_plexus_library._generated.models import ConfigUpdateRequest

    return ConfigUpdateRequest.model_validate(values)


def test_detection_is_kept_for_a_few_seconds_and_the_setting_still_decides(
    tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_library import config as config_mod
    from eugene_plexus_library.config import ConfigStore

    asked: list[int] = []

    def find(**_: Any) -> Path:
        asked.append(1)
        return store

    monkeypatch.setattr(ollama_detect, "find_store", find)
    configs = ConfigStore(tmp_path / "c.yaml")
    configs.load()
    for _ in range(3):
        assert configs.found_ollama() == store
    assert len(asked) == 1
    assert configs.apply_patch(_patch({"findOllama": False})).applied == ["findOllama"]
    assert configs.found_ollama() is None
    monkeypatch.setattr(config_mod, "DETECTION_SECONDS", 0.0)
    configs.apply_patch(_patch({"findOllama": True}))
    assert configs.found_ollama() == store and len(asked) == 2


def _build(root: Path) -> Path:
    blob(root, "qwen3", qwen_like_kv(name="Qwen3 8B", architecture="qwen3"))
    named(root, "manifests", "qwen3:8b", manifest(layer(MODEL, "qwen3")))
    return root


@pytest.fixture
def nested(
    tmp_path: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Path]]:
    """`OLLAMA_MODELS` inside a configured, writable folder."""
    import shutil

    inner = models_dir / "ollama"
    shutil.copytree(_build(tmp_path / "scratch"), inner)
    monkeypatch.setattr(ollama_detect, "find_store", lambda **_: inner)
    settings = Settings(config_file=tmp_path / "c.yaml", state_file=tmp_path / "s.json")
    settings.config_file.write_text(
        f"modelRoots:\n  - path: {models_dir.as_posix()}\n    mounts: []\n", encoding="utf-8"
    )
    with TestClient(create_app(settings=settings)) as client:
        wait_for_scan(client)
        client.post("/v1/scan")
        wait_for_scan(client)
        yield client, inner


def test_the_destination_is_checked_where_it_comes_to_not_at_the_root(tmp_path: Path) -> None:
    from eugene_plexus_library.downloads import resolve_file

    outer = tmp_path / "models"
    inner = outer / "ollama"
    for subdirectory in ("ollama/blobs", "ollama"):
        with pytest.raises(DownloadError) as refused:
            resolve_destination(
                repo="a/b",
                roots=[outer],
                root=str(outer),
                subdirectory=subdirectory,
                layout="flat",
                read_only=[inner],
            )
        assert refused.value.code == "ReadOnlyFolder"
    assert resolve_destination(
        repo="a/b", roots=[outer], root=None, subdirectory="x", layout="flat", read_only=[inner]
    ) == (outer, outer / "x")
    with pytest.raises(DownloadError):
        resolve_file(directory=outer, root=outer, name="ollama/blobs/x.gguf", read_only=[inner])


def test_a_download_into_a_store_inside_a_writable_folder_is_refused(
    nested: tuple[TestClient, Path],
) -> None:
    client, inner = nested
    before = sorted(str(p) for p in inner.rglob("*"))
    answer = client.post(
        "/v1/downloads",
        json={"repo": "a/b", "files": ["m.gguf"], "subdirectory": "ollama/blobs"},
    )
    assert answer.status_code == 409, answer.text
    assert sorted(str(p) for p in inner.rglob("*")) == before


def test_provenance_beside_an_entry_in_a_nested_store_is_refused(
    nested: tuple[TestClient, Path],
) -> None:
    client, inner = nested
    before = sorted(str(p) for p in inner.rglob("*"))
    entry = inner / "blobs" / f"sha256-{digest('qwen3')}"
    answer = client.post(
        "/v1/models/prepared",
        json={"name": "x", "provenance": {"engine": "strata", "entry": str(entry)}},
    )
    assert answer.status_code == 409, answer.text
    assert answer.json()["detail"]["title"] == "Ollama manages this folder"
    assert sorted(str(p) for p in inner.rglob("*")) == before


def test_a_nested_store_is_listed_only_as_the_store_and_cannot_be_deleted(
    nested: tuple[TestClient, Path],
) -> None:
    from datetime import UTC, datetime

    from eugene_plexus_library._generated.models import LibraryModel

    client, inner = nested
    models = client.get("/v1/models").json()["models"]
    inside = [m for m in models if Path(m["path"]).is_relative_to(inner)]
    assert [m["name"] for m in inside] == ["qwen3:8b"] and inside[0]["ollama"]
    # The same file listed as an ordinary model, however it got there, is still Ollama's.
    plain = LibraryModel.model_validate({**inside[0], "id": "plain-copy", "ollama": None})
    client.app.state.state_store.add_model(plain, seen_at=datetime.now(tz=UTC))
    for call in (
        lambda: client.get("/v1/models/plain-copy/deletion"),
        lambda: client.post("/v1/models/plain-copy/delete", json={"token": "x"}),
        lambda: client.delete("/v1/models/plain-copy"),
    ):
        answer = call()
        assert answer.status_code == 409
        assert answer.json()["detail"]["title"] == "Ollama manages this model"
    assert Path(inside[0]["path"]).is_file()


def test_the_walker_does_not_list_a_store_it_passes_through(tmp_path: Path, store: Path) -> None:
    outer = store.parent
    write_gguf(store / "loose.gguf", qwen_like_kv(name="Stray"))
    assert Scanner().scan([outer]).models != []  # without the rule it is found as a file
    walked = Scanner().scan([outer], ollama_roots=[store])
    assert walked.models == []
    assert any("Ollama model store" in s.detail for s in walked.skipped)


def test_a_writable_folder_inside_a_store_is_not_writable(tmp_path: Path, store: Path) -> None:
    import time

    from eugene_plexus_library.config import ConfigStore

    configs = ConfigStore(tmp_path / "c.yaml")
    configs.load()
    configs.apply_patch(
        _patch({"modelRoots": [{"path": str(store / "elsewhere")}, {"path": str(tmp_path / "ok")}]})
    )
    configs._detected = (time.perf_counter(), store)
    assert configs.writable_roots() == [tmp_path / "ok"]


def test_a_manifest_that_is_not_one_is_skipped_and_the_rest_are_listed(store: Path) -> None:
    host = "ollama.com"
    named(store, "manifests-v2", "weird:a", {"layers": 5}, host=host)
    named(store, "manifests-v2", "weird:b", {"mediaType": LIST, "manifests": "x"}, host=host)
    named(store, "manifests-v2", "weird:c", manifest({"mediaType": MODEL, "digest": 7}), host=host)
    named(store, "manifests-v2", "weird:d", {"layers": [3, None]}, host=host)
    (store / "manifests-v2" / host / "library" / "weird" / "e").write_text(
        "[1, 2]", encoding="utf-8"
    )
    result = scan(store)
    assert "qwen3:8b" in {m.name for m in result.models}
    assert {Path(s.path).name for s in result.skipped} >= {"a", "b", "c", "d", "e"}


def test_a_manifest_is_never_a_megabyte_and_a_digest_naming_the_model_is_not_read(
    store: Path,
) -> None:
    big = store / "manifests-v2" / "ollama.com" / "library" / "huge" / "latest"
    big.parent.mkdir(parents=True)
    big.write_text(" " * (ollama_store.MAX_MANIFEST_BYTES + 1), encoding="utf-8")
    model_blob = store / "blobs" / f"sha256-{digest('qwen3')}"
    model_blob.write_bytes(model_blob.read_bytes() + b"\0" * (ollama_store.MAX_MANIFEST_BYTES + 1))
    decoy = {"runner": "llamacpp", "format": "gguf", "digest": f"sha256:{digest('qwen3')}"}
    named(
        store,
        "manifests-v2",
        "decoy:latest",
        {"schemaVersion": 2, "mediaType": LIST, "manifests": [decoy]},
        host="ollama.com",
    )
    result = scan(store)
    skips = {Path(s.path).parent.name: s.detail for s in result.skipped}
    assert "too large" in skips["huge"]
    assert "decoy" in skips
    assert "decoy:latest" not in {t for m in result.models for t in m.ollama.tags}


def test_a_llamacpp_child_with_no_model_falls_back_to_the_ggml_one(store: Path) -> None:
    named(
        store,
        "manifests-v2",
        "fallback:latest",
        {
            "schemaVersion": 2,
            "mediaType": LIST,
            "manifests": [
                {"runner": "llamacpp", "format": "gguf", **manifest(layer(TEMPLATE, "t"))},
                {"runner": "ggml", "format": "gguf", **manifest(layer(MODEL, "llama-v2"))},
            ],
        },
        host="ollama.com",
    )
    found = by_name(scan(store))
    assert found["fallback:latest"].ollama.tags == ["fallback:latest", "llama3:latest"]
    assert found["fallback:latest"].path.endswith(f"sha256-{digest('llama-v2')}")
