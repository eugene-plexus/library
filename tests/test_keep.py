"""Row 65: real stores, verified Keep files and source-bound worker uploads."""

import errno
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    assert not list(target.parent.glob("*.partial.keep"))
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
    assert not list(Path(job["target"]).parent.glob("*.partial.keep"))
    assert "different drives" in job["note"]


def test_keep_bad_digest_keeps_nothing(kept_setup):
    client, model, source, root, _, _ = kept_setup
    source.write_bytes(b"changed after scan")
    job = start(client, model)
    assert job["state"] == "failed", job
    assert "SHA-256" in job["error"]
    assert not list(root.rglob("*.gguf"))
    assert not list(root.rglob("*.partial"))
    assert not list(root.rglob("*.partial.keep"))


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
    assert not list(root.rglob("*.partial.keep"))
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
    assert not list(root.rglob("*.partial.keep"))


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
    if (root / "ollama").exists():
        (root / "ollama").rename(root / "old-ollama")
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
    assert not list(root.rglob("*.partial.keep"))
    assert (
        client.put(
            base + "/files", params={"role": "weights", "offset": 0}, content=data, headers=headers
        ).status_code
        == 409
    )


def test_upload_never_opens_a_shared_inode_for_writing(remote_setup, tmp_path):
    client, job, data, headers, _, _ = remote_setup
    target = Path(job["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
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


def test_reservations_are_atomic(kept_setup):
    from eugene_plexus_library.keep import KeepError

    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    inserting, release, second_done = threading.Event(), threading.Event(), threading.Event()

    class PausedInsert(dict):
        def __setitem__(self, key, value):
            if not inserting.is_set():
                inserting.set()
                assert release.wait(5)
            super().__setitem__(key, value)

    keeps.jobs = PausedInsert()

    def reserve(second=False):
        try:
            return keeps.local(keeps.store.get_model(model["id"]), None)
        except KeepError as exc:
            return exc
        finally:
            if second:
                second_done.set()

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(reserve)
        assert inserting.wait(5)
        second = pool.submit(reserve, True)
        try:
            assert not second_done.wait(0.2), "A second reservation passed the pending insert"
        finally:
            release.set()
        assert first.result().public.state == "running"
        assert isinstance(second.result(), KeepError)
    assert len(keeps.jobs) == 1


@pytest.mark.parametrize("action", ["cancel", "fail"])
def test_cleanup_blocks_retry_until_finished(kept_setup, monkeypatch, action):
    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    source = keeps.store.get_model(model["id"])
    old = keeps.local(source, None)
    cleaning, release, attempted = threading.Event(), threading.Event(), threading.Event()
    acquired_without_wait = []
    original_lock, cleanup = keeps.lock, keeps.cleanup

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name.startswith("keep-retry"):
                acquired = original_lock.acquire(blocking=False)
                acquired_without_wait.append(acquired)
                attempted.set()
                if acquired:
                    return self
            original_lock.acquire()
            return self

        def __exit__(self, *args):
            original_lock.release()

    def paused_cleanup(job):
        cleaning.set()
        assert release.wait(5)
        cleanup(job)

    def terminate():
        if action == "cancel":
            keeps.cancel(old)
        else:
            keeps.fail(old, "transfer failed")

    def retry():
        job = keeps.local(source, None)
        keeps.write(job, job.files["weights"], 0, b"new")
        return job

    monkeypatch.setattr(keeps, "lock", ObservedLock())
    monkeypatch.setattr(keeps, "cleanup", paused_cleanup)
    with (
        ThreadPoolExecutor(1) as terminator,
        ThreadPoolExecutor(1, thread_name_prefix="keep-retry") as retries,
    ):
        ending = terminator.submit(terminate)
        try:
            assert cleaning.wait(5)
            retrying = retries.submit(retry)
            assert attempted.wait(5)
            if acquired_without_wait[0]:
                # Without the reservation lock, write the retry before old cleanup resumes.
                retrying.result(timeout=5)
        finally:
            release.set()
        ending.result(timeout=5)
        new = retrying.result(timeout=5)
    assert new.files["weights"].partial.read_bytes() == b"new"
    assert acquired_without_wait == [False]


def test_reservations_exclude_aliased_folders(kept_setup, tmp_path, directory_link, monkeypatch):
    from eugene_plexus_library.keep import KeepError

    client, model, _, root, _, _ = kept_setup
    keeps = client.app.state.keeps
    alias = tmp_path / "alias"
    directory_link(alias, root)
    monkeypatch.setattr(keeps.config, "writable_roots", lambda: [root, alias])
    source = keeps.store.get_model(model["id"])
    first = keeps.local(source, str(root))
    assert not first.files["weights"].partial.exists()
    with pytest.raises(KeepError, match="already being kept"):
        keeps.local(source, str(alias))


def test_failure_after_cancel_does_not_remove_retry_partial(kept_setup):
    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    source = keeps.store.get_model(model["id"])
    old = keeps.local(source, None)
    keeps.cancel(old)
    new = keeps.local(source, None)
    keeps.write(new, new.files["weights"], 0, b"new")
    keeps.fail(old, "late transfer error")
    assert new.files["weights"].partial.read_bytes() == b"new"
    assert old.public.state == "cancelled"


def test_reservation_refuses_marker_link(kept_setup, tmp_path, directory_link):
    from eugene_plexus_library.keep import KeepError

    client, model, _, root, _, _ = kept_setup
    keeps = client.app.state.keeps
    source = keeps.store.get_model(model["id"]).model_copy(update={"name": "foo"})
    marker = root / "ollama" / "foo" / "foo.gguf.partial.keep"
    marker.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    directory_link(marker, outside)
    with pytest.raises(KeepError):
        keeps.local(source, None)
    assert not list(outside.iterdir())


def test_startup_preserves_unmarked_partial_with_keep_name(kept_setup):
    from eugene_plexus_library.keep import Keeps

    client, _, _, root, _, _ = kept_setup
    partial = root / "ollama" / "foo" / "foo.gguf.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"a person's own file")
    keeps = client.app.state.keeps
    Keeps(keeps.config, keeps.store)
    assert partial.read_bytes() == b"a person's own file"


def test_startup_removes_marked_partial(kept_setup):
    from eugene_plexus_library.keep import Keeps

    client, _, _, root, _, _ = kept_setup
    partial = root / "ollama" / "foo" / "foo.gguf.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"interrupted Keep")
    marker = partial.with_name(partial.name + ".keep")
    marker.write_text("interrupted-job", encoding="utf-8")
    keeps = client.app.state.keeps
    Keeps(keeps.config, keeps.store)
    assert not partial.exists()
    assert not marker.exists()


def test_startup_removes_partial_for_truncated_name_ending_in_dash(kept_setup):
    from eugene_plexus_library.keep import Keeps

    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    source = keeps.store.get_model(model["id"]).model_copy(update={"name": "a" * 119 + "-tail"})
    job = keeps.local(source, None)
    file = job.files["weights"]
    assert file.path.parent.name.endswith("-")
    file.path.parent.mkdir(parents=True, exist_ok=True)
    file.partial.write_bytes(b"interrupted Keep")
    # Model a previous process's ownership record, including before marker creation is fixed.
    marker = file.partial.with_name(file.partial.name + ".keep")
    marker.write_text(job.public.id, encoding="utf-8")
    Keeps(keeps.config, keeps.store)
    assert not file.partial.exists()
    assert not marker.exists()


def test_reservation_marks_every_partial_before_transfer(kept_setup):
    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    source = keeps.store.get_model(model["id"])
    job = keeps.reserve(
        name=source.name,
        tags=source.ollama.tags,
        digest="sha256:" + "a" * 64,
        size=3,
        store=source.root,
        folder=None,
        projector_digest="sha256:" + "b" * 64,
        projector_size=2,
    )
    for file in job.files.values():
        assert not file.partial.exists()
        assert file.partial.with_name(file.partial.name + ".keep").read_text() == job.public.id
    keeps.cancel(job)
    assert not list(job.root.rglob("*.partial.keep"))


def test_startup_removes_only_orphan_keep_partials(kept_setup, tmp_path, directory_link):
    from eugene_plexus_library.keep import Keeps

    client, model, _, root, _, _ = kept_setup
    keeps = client.app.state.keeps
    job = keeps.local(keeps.store.get_model(model["id"]), None)
    file = job.files["weights"]
    file.path.parent.mkdir(parents=True, exist_ok=True)
    file.partial.write_bytes(b"interrupted transfer")
    projector = file.path.with_name("mmproj-" + file.path.name + ".partial")
    projector.write_bytes(b"interrupted projector")
    projector.with_name(projector.name + ".keep").write_text(job.public.id)
    unrelated = file.path.with_name("download.gguf.partial")
    unrelated.write_bytes(b"another module")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "escaped.gguf.partial"
    external.write_bytes(b"keep this")
    external.with_name(external.name + ".keep").write_text(job.public.id)
    directory_link(root / "ollama" / "escaped", outside)
    final = file.path
    final.write_bytes(b"already kept")
    keeps.remove_orphan_partials()
    assert file.partial.exists(), "A live job owns this partial"
    Keeps(keeps.config, keeps.store)
    assert not file.partial.exists()
    assert not projector.exists()
    assert unrelated.read_bytes() == b"another module"
    assert external.read_bytes() == b"keep this"
    assert final.read_bytes() == b"already kept"
    # The old job cannot be restored, but its abandoned transfer no longer blocks a retry.
    final.unlink()
    restarted = Keeps(keeps.config, keeps.store)
    assert restarted.local(keeps.store.get_model(model["id"]), None).public.state == "running"
    # The Windows device-name prefix can add six characters after truncation.
    reserved_name = keeps.store.get_model(model["id"]).model_copy(
        update={"name": "CON." + "a" * 116}
    )
    prefixed = restarted.local(reserved_name, None).files["weights"].partial
    prefixed.parent.mkdir(parents=True, exist_ok=True)
    prefixed.write_bytes(b"interrupted")
    Keeps(keeps.config, keeps.store)
    assert not prefixed.exists()


def test_reservation_names_existing_partial(kept_setup):
    from eugene_plexus_library.keep import KeepError

    client, model, _, _, _, _ = kept_setup
    keeps = client.app.state.keeps
    job = keeps.local(keeps.store.get_model(model["id"]), None)
    keeps.cancel(job)
    partial = job.files["weights"].partial
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"partial")
    with pytest.raises(KeepError, match=r"\.gguf\.partial already exists"):
        keeps.local(keeps.store.get_model(model["id"]), None)


@pytest.mark.parametrize("cause", [errno.ENOSPC, errno.EACCES])
def test_upload_write_error_fails_job_and_returns_cause(remote_setup, monkeypatch, cause):
    client, job, data, headers, session, root = remote_setup
    target = Path(job["target"])
    partial = target.with_name(target.name + ".partial")
    original = Path.open

    def broken_open(path, *args, **kwargs):
        if path == partial and args and args[0] in ("wb", "r+b"):
            # Leave a real partial, as a buffered write can fail on close.
            with original(path, "wb") as stream:
                stream.write(b"partial")
            raise OSError(cause, os.strerror(cause))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken_open)
    response = client.put(
        f"/v1/keep-jobs/{job['id']}/files",
        params={"role": "weights", "offset": 0},
        content=data,
        headers=headers,
    )
    assert response.status_code == (507 if cause == errno.ENOSPC else 409), response.text
    failed = client.get(f"/v1/keep-jobs/{job['id']}", headers=session).json()
    assert failed["state"] == "failed"
    expected = f"The disk holding {root} is full" if cause == errno.ENOSPC else os.strerror(cause)
    assert expected in response.json()["detail"]["detail"] and expected in failed["error"]
    assert not partial.exists() and not target.exists()


@pytest.mark.asyncio
async def test_remote_keep_uses_library_machine_credential(kept_setup, install, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import httpx

    from eugene_plexus_library.routes import ollama

    client, model, _, _, _, _ = kept_setup
    app = client.app
    app.state.auth_state = install.auth_state()
    app.state.auth_state = replace(app.state.auth_state, service_token=install.service("library"))
    app.state.settings.agent_url = "http://local-agent"
    job = app.state.keeps.local(app.state.state_store.get_model(model["id"]), None)
    job.public.node = "far"
    observed = []

    async def send(request):
        observed.append(request.headers["authorization"])
        job.public.state = type(job.public.state).done
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as transport:
        monkeypatch.setattr(ollama, "shared_internal_client", lambda *args, **kwargs: transport)
        operator = install.session(aud=[install.recipient])
        request = SimpleNamespace(app=app, headers={"authorization": "Bearer " + operator})
        await ollama.send_from_node(request, job)
    assert observed == ["Bearer " + app.state.auth_state.service_token]
