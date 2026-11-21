"""library#25: what a download started with an operations token may write.

A person with the Models power downloads models. With their token a
download never replaces a file that is already there, never writes at or
inside the folder of a folder-format model the Library lists (or an
engine's own prepared folder), and never writes a file the Library would
list as a prepared model. The same download, resumed, finishes its own
files. The owner is not restricted.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import downloads, hub, prepared, tokens
from eugene_plexus_library._generated.models import DownloadSpec, DownloadState
from eugene_plexus_library.app import create_app
from eugene_plexus_library.settings import Settings
from eugene_plexus_library.store import StateStore

from .conftest import (
    FakeInstall,
    mock_hubs,
    qwen_like_kv,
    wait_for_scan,
    write_gguf,
    write_hf_model,
)
from .test_operations_gates import _bearer, _operations

MODELS = tokens.POWER_OPERATE_MODELS
EVIL = b"import os  # bytes the owner never reviewed\n"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads, "BACKOFF_SECONDS", (0, 0, 0, 0))


class Hub:
    """Serves any file named in `files`, with its git blob hash as digest."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.failing: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/tree/" in path:
            tree = [
                {"type": "file", "path": name, "size": len(body), "oid": hub.git_blob_sha1(body)}
                for name, body in self.files.items()
            ]
            return httpx.Response(200, json=tree)
        if path.startswith("/api/models/"):
            return httpx.Response(200, json={"sha": "commit1", "author": "someone"})
        name = path.rsplit("/", 1)[-1]
        if name in self.failing:
            return httpx.Response(503)
        body = self.files[name]
        if request.method == "HEAD":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://cdn.example/blob",
                    "X-Linked-Size": str(len(body)),
                    "X-Linked-ETag": f'"{hub.git_blob_sha1(body)}"',
                    "X-Repo-Commit": "commit1",
                },
            )
        return httpx.Response(200, content=body)


@pytest.fixture
def library(tmp_path: Path, install: FakeInstall) -> Iterator[tuple[TestClient, Path, Hub]]:
    """A Library folder with a safetensors model the owner trusts
    (`org/trusted`) and a plain folder of GGUF files (`plain`)."""
    models = tmp_path / "models"
    write_hf_model(models / "org" / "trusted")
    (models / "org" / "trusted" / "modeling_x.py").write_text("# reviewed\n", encoding="utf-8")
    write_gguf(models / "plain" / "a.gguf", qwen_like_kv())
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
    )
    settings.config_file.write_text(
        f"modelRoots:\n  - path: {models.as_posix()}\n    mounts: []\nscanOnStartup: true\n",
        encoding="utf-8",
    )
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    fake = Hub({"modeling_x.py": EVIL, "a.gguf": EVIL, "b.gguf": b"GGUF more"})
    with TestClient(app) as client:
        client.headers.update(_bearer(install.session()))
        wait_for_scan(client)
        formats = {m["format"] for m in client.get("/v1/models").json()["models"]}
        assert formats == {"safetensors", "gguf"}, formats
        client.headers.pop("Authorization")
        mock_hubs(app, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
        yield client, models, fake


def _finished(client: TestClient, download_id: str, headers: dict[str, str]) -> dict[str, Any]:
    deadline = time.perf_counter() + 20
    record: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        record = client.get(f"/v1/downloads/{download_id}", headers=headers).json()
        if record["state"] in ("done", "failed", "cancelled"):
            return record
        time.sleep(0.02)
    raise AssertionError(f"download did not finish within 20 seconds: {record}")


def test_models_cannot_download_into_a_model_folder(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    """The repro: a new file in the owner's trusted model folder."""
    client, models, fake = library
    fake.files["extra.py"] = EVIL
    dana = _bearer(_operations(install, MODELS))
    spec = {"repo": "evil/repo", "files": ["extra.py"], "subdirectory": "org/trusted"}
    refused = client.post("/v1/downloads", json=spec, headers=dana)
    assert refused.status_code == 403, refused.text
    detail = refused.json()["detail"]["detail"]
    assert str(models / "org" / "trusted") in detail and "owner" in detail, detail
    assert not (models / "org" / "trusted" / "extra.py").exists()
    # A folder below it is inside it too.
    deeper = {"repo": "evil/repo", "files": ["extra.py"], "subdirectory": "org/trusted/sub"}
    assert client.post("/v1/downloads", json=deeper, headers=dana).status_code == 403


def test_models_cannot_replace_a_file(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    client, models, _ = library
    before = (models / "plain" / "a.gguf").read_bytes()
    dana = _bearer(_operations(install, MODELS))
    spec = {"repo": "evil/repo", "files": ["a.gguf"], "subdirectory": "plain"}
    refused = client.post("/v1/downloads", json=spec, headers=dana)
    assert refused.status_code == 403, refused.text
    detail = refused.json()["detail"]["detail"]
    assert str(models / "plain" / "a.gguf") in detail and "already there" in detail, detail
    assert (models / "plain" / "a.gguf").read_bytes() == before


@pytest.mark.parametrize("name", ["x.eugene-prepared.json", "X.EUGENE-PREPARED.JSON"])
def test_models_cannot_download_a_prepared_model(
    library: tuple[TestClient, Path, Hub], install: FakeInstall, name: str
) -> None:
    """A provenance file is listed as a prepared model wherever it lands, and
    the owner would run what it names."""
    client, _, fake = library
    fake.files[name] = b'{"formatVersion": 1}'
    dana = _bearer(_operations(install, MODELS))
    refused = client.post("/v1/downloads", json={"repo": "evil/p", "files": [name]}, headers=dana)
    assert refused.status_code == 403, refused.text
    assert name in refused.json()["detail"]["detail"]


def test_models_cannot_download_into_an_engines_own_folder(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    client, models, fake = library
    engine = models / "prep"
    engine.mkdir()
    (engine / prepared.ENGINE_FILES_MARKER).write_text("", encoding="utf-8")
    fake.files["w.gguf"] = b"GGUF w"
    dana = _bearer(_operations(install, MODELS))
    spec = {"repo": "evil/repo", "files": ["w.gguf"], "subdirectory": "prep/inner"}
    refused = client.post("/v1/downloads", json=spec, headers=dana)
    assert refused.status_code == 403, refused.text
    assert str(engine) in refused.json()["detail"]["detail"]
    marker = {"repo": "evil/repo", "files": ["w.gguf"], "filename": prepared.ENGINE_FILES_MARKER}
    assert client.post("/v1/downloads", json=marker, headers=dana).status_code == 403


def test_models_downloads_one_more_gguf_and_a_catalogue_download(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    client, models, _ = library
    dana = _bearer(_operations(install, MODELS))
    # The companion beside a version on the disk (`subdirectory`).
    beside = {"repo": "org/more", "files": ["b.gguf"], "subdirectory": "plain"}
    started = client.post("/v1/downloads", json=beside, headers=dana)
    assert started.status_code == 202, started.text
    assert _finished(client, started.json()["id"], dana)["state"] == "done"
    assert (models / "plain" / "b.gguf").read_bytes() == b"GGUF more"
    # The catalogue's own download: repo, revision and files, nothing else.
    catalogue = {"repo": "org/fresh", "revision": "main", "files": ["b.gguf"]}
    started = client.post("/v1/downloads", json=catalogue, headers=dana)
    assert started.status_code == 202, started.text
    assert _finished(client, started.json()["id"], dana)["state"] == "done"


def test_models_cannot_bring_back_a_missing_model(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    """Second review: the owner moved a trusted folder away; the Library lists
    it missing, with its profiles. Files at that path would come back as the
    same model, with the owner's profiles on them."""
    client, models, fake = library
    owner = _bearer(install.session())
    listed = client.get("/v1/models", headers=owner).json()["models"]
    trusted = next(m for m in listed if m["path"].endswith("trusted"))
    profile = {"name": "o", "engine": "vllm", "flags": {"trustRemoteCode": True}}
    made = client.post(f"/v1/models/{trusted['id']}/profiles", json=profile, headers=owner)
    assert made.status_code == 201, made.text
    (models / "org" / "trusted").rename(models.parent / "moved-away")
    client.headers.update(owner)
    client.post("/v1/scan", json={})
    wait_for_scan(client)
    client.headers.pop("Authorization")
    listed = client.get("/v1/models", headers=owner).json()["models"]
    missing = [m for m in listed if m["path"].endswith("trusted")]
    assert [m["status"] for m in missing] == ["missing"], listed
    fake.files["config.json"] = b"{}"
    dana = _bearer(_operations(install, MODELS))
    spec = {"repo": "evil/repo", "files": ["config.json"], "subdirectory": "org/trusted"}
    refused = client.post("/v1/downloads", json=spec, headers=dana)
    assert refused.status_code == 403, refused.text
    assert str(models / "org" / "trusted") in refused.json()["detail"]["detail"]
    assert not (models / "org" / "trusted").exists()


STREAM = "::" + chr(36) + "DATA"


@pytest.mark.parametrize(
    ("spec", "part"),
    [
        ({"subdirectory": "p1", "filename": "x.eugene-prepared.json."}, "x.eugene-prepared.json."),
        ({"subdirectory": "p2", "filename": "y.eugene-prepared.json "}, "y.eugene-prepared.json "),
        ({"subdirectory": "p3", "filename": ".eugene-engine-files."}, ".eugene-engine-files."),
        ({"subdirectory": "p4", "filename": "z.eugene-prepared.json" + STREAM}, "z.eugene"),
        ({"subdirectory": "org/new "}, "new "),
        ({"subdirectory": "org/new./x"}, "new."),
    ],
    ids=["dot", "space", "marker-dot", "stream", "folder-space", "folder-dot"],
)
def test_models_cannot_use_a_name_windows_writes_as_another(
    library: tuple[TestClient, Path, Hub], install: FakeInstall, spec: dict[str, Any], part: str
) -> None:
    client, _, fake = library
    fake.files["p.json"] = b"{}"
    dana = _bearer(_operations(install, MODELS))
    refused = client.post(
        "/v1/downloads", json={"repo": "evil/p", "files": ["p.json"], **spec}, headers=dana
    )
    assert refused.status_code == 403, refused.text
    detail = refused.json()["detail"]["detail"]
    assert part in detail, detail
    assert "Windows" in detail, detail
    assert client.get("/v1/downloads", headers=dana).json()["downloads"] == []


@pytest.mark.skipif(os.name != "nt", reason="Windows drops a trailing space; Linux keeps it")
def test_a_folder_with_a_trailing_space_gets_its_cause_at_the_start(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    client, _, _ = library
    owner = _bearer(install.session())
    spec = {"repo": "org/x", "files": ["b.gguf"], "subdirectory": "org/new "}
    refused = client.post("/v1/downloads", json=spec, headers=owner)
    assert refused.status_code == 400, refused.text
    assert "dot or a space" in refused.json()["detail"]["detail"]


def test_the_owner_is_not_restricted(
    library: tuple[TestClient, Path, Hub], install: FakeInstall
) -> None:
    client, models, _ = library
    owner = _bearer(install.session())
    spec = {"repo": "evil/repo", "files": ["modeling_x.py"], "subdirectory": "org/trusted"}
    started = client.post("/v1/downloads", json=spec, headers=owner)
    assert started.status_code == 202, started.text
    assert _finished(client, started.json()["id"], owner)["state"] == "done"
    assert (models / "org" / "trusted" / "modeling_x.py").read_bytes() == EVIL


# ------------------------------------------------------- the manager itself


def _manager(tmp_path: Path, fake: Hub) -> tuple[downloads.DownloadManager, StateStore, Path]:
    root = tmp_path / "models"
    root.mkdir(exist_ok=True)
    inner = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    store = StateStore(tmp_path / "state.json")
    store.load()  # a second call is a restart: what the state file kept
    manager = downloads.DownloadManager(
        client=client,
        store=store,
        scan_manager=None,
        roots=lambda: [root],
        layout=lambda: "publisher_repo",
        concurrency=lambda: 1,
    )
    return manager, store, root


async def _drain(manager: downloads.DownloadManager, record_id: str) -> Any:
    deadline = time.perf_counter() + 20
    while time.perf_counter() < deadline:
        record = manager.get(record_id)
        assert record is not None
        if record.state in (DownloadState.done, DownloadState.failed):
            return record
        await asyncio.sleep(0.01)
    raise AssertionError("download did not finish within 20 seconds")


@pytest.mark.asyncio
async def test_a_resumed_download_finishes_its_own_files(tmp_path: Path) -> None:
    """Its first file landed, and the scan then listed its folder as a
    model: resuming it is not a new download into a model folder."""
    fake = Hub({"config.json": b"{}", "model.safetensors": b"weights"})
    fake.failing.add("model.safetensors")
    manager, store, root = _manager(tmp_path, fake)
    try:
        spec = DownloadSpec(repo="org/new", files=["config.json", "model.safetensors"])
        record = await manager.start(spec, restricted=True)
        assert (await _drain(manager, record.id)).state is DownloadState.failed
        folder = root / "org" / "new"
        assert (folder / "config.json").exists()
        _list_a_folder_model(store, folder)
        fake.failing.clear()
        await manager.resume(record.id, restricted=True)
        finished = await _drain(manager, record.id)
        assert finished.state is DownloadState.done, finished.error
        assert (folder / "model.safetensors").read_bytes() == b"weights"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_a_file_that_appears_meanwhile_is_not_replaced(tmp_path: Path) -> None:
    fake = Hub({"model.gguf": b"GGUF from the download"})
    fake.failing.add("model.gguf")
    manager, _, root = _manager(tmp_path, fake)
    try:
        record = await manager.start(
            DownloadSpec(repo="org/m", files=["model.gguf"]), restricted=True
        )
        assert (await _drain(manager, record.id)).state is DownloadState.failed
        target = root / "org" / "m" / "model.gguf"
        target.write_bytes(b"GGUF the owner put here")
        fake.failing.clear()
        await manager.resume(record.id, restricted=True)
        finished = await _drain(manager, record.id)
        assert finished.state is DownloadState.failed
        assert "already there" in (finished.error or ""), finished.error
        assert target.read_bytes() == b"GGUF the owner put here"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_after_a_restart_the_owners_resume_does_not_replace_a_file(tmp_path: Path) -> None:
    """Second review: who started a download is kept with its record. The
    owner resuming a Models person's download never replaces a file."""
    fake = Hub({"modeling_x.py": EVIL})
    fake.failing.add("modeling_x.py")
    manager, _, root = _manager(tmp_path, fake)
    spec = DownloadSpec(repo="evil/repo", files=["modeling_x.py"], subdirectory="org/trusted")
    record = await manager.start(spec, restricted=True)
    assert (await _drain(manager, record.id)).state is DownloadState.failed
    await manager.shutdown()
    target = root / "org" / "trusted" / "modeling_x.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# reviewed by the owner", encoding="utf-8")
    fake.failing.clear()
    again, _, _ = _manager(tmp_path, fake)
    try:
        await again.resume(record.id)  # the owner
        ended = await _drain(again, record.id)
        assert ended.state is DownloadState.failed
        assert "Models power started this download" in (ended.error or ""), ended.error
        assert "owner can start the download again" in (ended.error or ""), ended.error
        assert target.read_text(encoding="utf-8") == "# reviewed by the owner"
    finally:
        await again.shutdown()


@pytest.mark.asyncio
async def test_the_owners_download_is_the_owners_after_a_models_resume(tmp_path: Path) -> None:
    fake = Hub({"big.gguf": b"GGUF the newer upstream file"})
    fake.failing.add("big.gguf")
    manager, _, root = _manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/own", files=["big.gguf"]))
    assert (await _drain(manager, record.id)).state is DownloadState.failed
    target = root / "org" / "own" / "big.gguf"
    target.write_bytes(b"GGUF old")
    fake.failing.clear()
    await manager.shutdown()
    manager, _, _ = _manager(tmp_path, fake)  # after a restart, still the owner's
    try:
        # A Models person resumes it: for that run, no file is replaced.
        await manager.resume(record.id, restricted=True)
        ended = await _drain(manager, record.id)
        assert ended.state is DownloadState.failed
        assert "Only the owner can download over" in (ended.error or ""), ended.error
        assert target.read_bytes() == b"GGUF old"
        # Then the owner resumes it: not restricted.
        await manager.resume(record.id)
        assert (await _drain(manager, record.id)).state is DownloadState.done
        assert target.read_bytes() == b"GGUF the newer upstream file"
    finally:
        await manager.shutdown()


def _list_a_folder_model(store: StateStore, folder: Path) -> None:
    from eugene_plexus_library._generated.models import LibraryModel, ModelFormat, ModelStatus

    store.add_model(
        LibraryModel(
            id="m-new",
            path=str(folder),
            root=str(folder.parent.parent),
            format=ModelFormat.safetensors,
            name="new",
            status=ModelStatus.present,
        ),
        seen_at=datetime.now(tz=UTC),
    )
