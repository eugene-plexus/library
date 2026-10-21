"""Row 65: real stores, verified Keep files and source-bound worker uploads."""

import errno
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import ollama_detect
from eugene_plexus_library.app import create_app

from .conftest import qwen_like_kv, wait_for_scan, write_gguf
from .test_ollama_store import kept_setup as kept_setup


def finish(client, job):
    deadline = time.perf_counter() + 10
    while job["state"] == "running" and time.perf_counter() < deadline:
        time.sleep(0.01)
        response = client.get(f"/v1/keep-jobs/{job['id']}")
        assert response.status_code == 200, response.text
        job = response.json()
    assert job["state"] != "running"
    return job


def start(client, model):
    response = client.post(f"/v1/models/{model['id']}/keep", json={})
    assert response.status_code == 202, response.text
    return finish(client, response.json())


def test_write_through_link_into_store_refused(kept_setup, directory_link):
    client, _, _, root, store, _ = kept_setup
    link = root / "shared"
    directory_link(link, store)
    response = client.post(
        "/v1/downloads",
        json={
            "repo": "a/b",
            "files": ["m.gguf"],
            "root": str(root),
            "subdirectory": "shared",
        },
    )
    assert response.status_code == 409, response.text
    assert "Ollama" in response.text
    assert not (store / "m.gguf").exists()


def test_service_account_finds_profile_store(tmp_path):
    profile = tmp_path / "person"
    store = profile / ".ollama" / "models"
    (store / "blobs").mkdir(parents=True)
    (store / "manifests").mkdir()
    found = ollama_detect.candidates(
        environ={},
        home=Path("C:/Windows/System32/config/systemprofile"),
        platform="win32",
        profiles=[("S-1-test", profile, None)],
        machine_models=None,
    )
    assert store in found


def test_keep_links_same_volume(kept_setup):
    client, model, source, _, _, _ = kept_setup
    job = start(client, model)
    assert job["state"] == "done", job
    assert job["method"] == "link"
    target = Path(job["target"])
    assert os.path.samefile(source, target)
    source.unlink()
    assert target.is_file()


def test_keep_copies_and_checks_digest(kept_setup, monkeypatch):
    client, model, source, _, _, _ = kept_setup

    def exdev(*args, **kwargs):
        raise OSError(errno.EXDEV, "different drives")

    monkeypatch.setattr(os, "link", exdev)
    job = start(client, model)
    assert job["state"] == "done", job
    assert job["method"] == "copy"
    assert Path(job["target"]).read_bytes() == source.read_bytes()
    assert "different drives" in job["note"]


def test_keep_bad_digest_keeps_nothing(kept_setup):
    client, model, source, root, _, _ = kept_setup
    source.write_bytes(b"changed after scan")
    job = start(client, model)
    assert job["state"] == "failed", job
    assert "SHA-256" in job["error"]
    assert not list(root.rglob("*.gguf"))
    assert not list(root.rglob("*.partial"))


def test_keep_moves_profiles_and_keeps_name(kept_setup):
    client, model, _, _, _, _ = kept_setup
    profile = client.post(
        f"/v1/models/{model['id']}/profiles", json={"name": "tuned", "engine": "llama_cpp"}
    )
    assert profile.status_code == 201
    job = start(client, model)
    assert job["state"] == "done", job
    assert job["model"]["name"] == "llama3:latest"
    target = job["model"]["id"]
    assert client.get(f"/v1/models/{target}/profiles").json()["profiles"][0]["name"] == "tuned"
    client.post("/v1/scan")
    wait_for_scan(client)
    kept = client.get(f"/v1/models/{target}").json()
    assert kept["name"] == "llama3:latest" and kept["keptFrom"]["method"] == "link"
    assert client.get(f"/v1/models/{model['id']}/profiles").json()["profiles"] == []


def report(digest="sha256:" + "a" * 64):
    return {
        "stores": [
            {
                "path": "C:/Ollama/models",
                "models": [
                    {"tags": ["llama3:8b"], "digest": digest, "sizeBytes": 3, "runner": "llamacpp"}
                ],
                "skipped": [],
            }
        ],
        "readAt": "2026-10-21T12:00:00Z",
    }


def test_node_reports_only_its_own_store(settings, install):
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as client:
        response = client.put(
            "/v1/nodes/gw/ollama",
            json=report(),
            headers={"Authorization": "Bearer " + install.foreign_service()},
        )
        assert response.status_code == 403, response.text
        assert "A machine reports only its own Ollama store." in response.text


def test_unbound_development_mode_cannot_report_another_machine(client):
    response = client.put("/v1/nodes/far/ollama", json=report())
    assert response.status_code == 403


def test_report_lists_models_for_node(settings, install):
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as client:
        response = client.put(
            "/v1/nodes/far/ollama",
            json=report(),
            headers={"Authorization": "Bearer " + install.foreign_service()},
        )
        assert response.status_code == 204, response.text
        listed = client.get(
            "/v1/nodes/ollama", headers={"Authorization": "Bearer " + install.session()}
        )
        node = listed.json()["nodes"][0]
        assert node["node"] == "far"
        assert node["stores"][0]["models"][0]["tags"] == ["llama3:8b"]
        assert node["readAt"].startswith("2026-10-21T12:00:00")


@pytest.fixture
def remote_setup(settings, install, models_dir, tmp_path, monkeypatch):
    from eugene_plexus_library.routes import ollama

    async def no_send(*args):
        pass

    monkeypatch.setattr(ollama, "send_from_node", no_send)
    settings.config_file.write_text(
        json.dumps({"modelRoots": [str(models_dir)], "scanOnStartup": False})
    )
    data = write_gguf(tmp_path / "source.gguf", qwen_like_kv()).read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + install.foreign_service()}
        session = {"Authorization": "Bearer " + install.session()}
        assert (
            client.put("/v1/nodes/far/ollama", json=report(digest), headers=headers).status_code
            == 204
        )
        # The accepted report defines the size, not the uploader.
        body = report(digest)
        body["stores"][0]["models"][0]["sizeBytes"] = len(data)
        assert client.put("/v1/nodes/far/ollama", json=body, headers=headers).status_code == 204
        response = client.post(
            "/v1/nodes/far/ollama/keep", json={"digest": digest}, headers=session
        )
        assert response.status_code == 202, response.text
        yield client, response.json(), data, headers, session, models_dir


def test_worker_upload_uses_only_reserved_role_and_own_token(remote_setup, install, tmp_path):
    client, job, data, headers, session, root = remote_setup
    base = f"/v1/keep-jobs/{job['id']}/files"
    wrong = {"Authorization": "Bearer " + install.service("agent")}
    for credential in (wrong, session):
        assert (
            client.post(base + "/state", json={"role": "weights"}, headers=credential).status_code
            == 403
        )
        assert (
            client.put(
                base, params={"role": "weights", "offset": 0}, content=data, headers=credential
            ).status_code
            == 403
        )
        assert (
            client.post(
                base + "/complete", json={"role": "weights"}, headers=credential
            ).status_code
            == 403
        )
    outside = tmp_path / "outside.gguf"
    assert (
        client.post(
            base + "/state", json={"role": "weights", "path": str(outside)}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.put(
            base, params={"role": "projector", "offset": 0}, content=data, headers=headers
        ).status_code
        == 409
    )
    assert (
        client.put(
            base, params={"role": "weights", "offset": 1}, content=data, headers=headers
        ).status_code
        == 409
    )
    sent = client.put(
        base,
        params={"role": "weights", "offset": 0, "path": str(outside)},
        content=data,
        headers=headers,
    )
    assert sent.status_code == 200, sent.text
    completed = client.post(base + "/complete", json={"role": "weights"}, headers=headers)
    assert completed.status_code == 200, completed.text
    assert (
        client.post(base + "/complete", json={"role": "weights"}, headers=headers).status_code
        == 200
    )
    done = client.get(f"/v1/keep-jobs/{job['id']}", headers=session).json()
    assert done["state"] == "done", done
    assert done["model"]["keptFrom"]["node"] == "far"
    assert done["model"]["name"] == "llama3:8b"
    assert Path(done["target"]).is_relative_to(root)
    assert not outside.exists()
    listed = client.get("/v1/nodes/ollama", headers=session).json()
    assert listed["nodes"][0]["stores"][0]["models"][0]["libraryModel"] == done["model"]["id"]


def test_worker_upload_digest_mismatch_fails_without_publishing(remote_setup):
    client, job, data, headers, session, root = remote_setup
    base = f"/v1/keep-jobs/{job['id']}/files"
    assert (
        client.put(
            base, params={"role": "weights", "offset": 0}, content=bytes(len(data)), headers=headers
        ).status_code
        == 200
    )
    response = client.post(base + "/complete", json={"role": "weights"}, headers=headers)
    assert response.status_code == 409
    assert "SHA-256" in response.text
    assert client.get(f"/v1/keep-jobs/{job['id']}", headers=session).json()["state"] == "failed"
    assert not list(root.rglob("*.gguf")) and not list(root.rglob("*.partial"))


def test_report_limits_are_enforced_before_replacing(remote_setup):
    client, _, _, headers, session, _ = remote_setup
    huge = report()
    huge["stores"][0]["models"] *= 2001
    assert client.put("/v1/nodes/far/ollama", json=huge, headers=headers).status_code == 413
    assert (
        client.put(
            "/v1/nodes/far/ollama", content=b" " * (1024 * 1024 + 1), headers=headers
        ).status_code
        == 413
    )
    assert (
        len(
            client.get("/v1/nodes/ollama", headers=session).json()["nodes"][0]["stores"][0][
                "models"
            ]
        )
        == 1
    )


def test_hidden_store_configured_as_plain_folder_stays_read_only(kept_setup):
    client, _, _, root, store, _ = kept_setup
    response = client.patch(
        "/v1/config",
        json={"findOllama": False, "modelRoots": [{"path": str(root)}, {"path": str(store)}]},
    )
    assert response.status_code == 200
    assert store not in client.app.state.config_store.writable_roots()
    response = client.post(
        "/v1/downloads", json={"repo": "a/b", "files": ["m.gguf"], "root": str(store)}
    )
    assert response.status_code == 409


def test_upload_refuses_new_junction_into_store(remote_setup, tmp_path, directory_link):
    client, job, data, headers, session, root = remote_setup
    store = tmp_path / "protected"
    (store / "blobs").mkdir(parents=True)
    (store / "manifests").mkdir()
    patched = client.patch(
        "/v1/config",
        json={"modelRoots": [{"path": str(root)}, {"path": str(store), "kind": "ollama"}]},
        headers=session,
    )
    assert patched.status_code == 200
    directory_link(root / "ollama", store)
    response = client.put(
        f"/v1/keep-jobs/{job['id']}/files",
        params={"role": "weights", "offset": 0},
        content=data,
        headers=headers,
    )
    assert response.status_code == 409
    assert "Ollama" in response.text
    assert not list(store.rglob("*.partial")) and not list(store.rglob("*.gguf"))


def test_cancelled_remote_keep_removes_only_partial(remote_setup):
    client, job, data, headers, session, root = remote_setup
    base = f"/v1/keep-jobs/{job['id']}"
    assert (
        client.put(
            base + "/files",
            params={"role": "weights", "offset": 0},
            content=data[:4],
            headers=headers,
        ).status_code
        == 200
    )
    assert list(root.rglob("*.partial"))
    assert client.delete(base, headers=session).json()["state"] == "cancelled"
    assert not list(root.rglob("*.partial"))
    assert (
        client.put(
            base + "/files", params={"role": "weights", "offset": 0}, content=data, headers=headers
        ).status_code
        == 409
    )


def test_upload_never_opens_a_shared_inode_for_writing(remote_setup, tmp_path):
    client, job, data, headers, _, _ = remote_setup
    target = Path(job["target"])
    target.parent.mkdir(parents=True)
    source = tmp_path / "already-kept.gguf"
    source.write_bytes(data)
    os.link(source, target.with_name(target.name + ".partial"))
    response = client.put(
        f"/v1/keep-jobs/{job['id']}/files",
        params={"role": "weights", "offset": 0},
        content=bytes(len(data)),
        headers=headers,
    )
    assert response.status_code == 409
    assert source.read_bytes() == data
