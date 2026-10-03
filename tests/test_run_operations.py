"""Crash recovery, idempotency, cancellation and node-scoped execution grants."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from eugene_plexus_library._generated.models import Download, DownloadSpec
from eugene_plexus_library.routes import run_operations
from eugene_plexus_library.run_operations import Checkpoint, Intent, Journal

from .test_store import _model


def test_reopen_keeps_intent_and_fences_the_old_worker(tmp_path):
    path = tmp_path / "runs.sqlite3"
    now = [1000.0]
    jobs = Journal(path, clock=lambda: now[0])
    jobs.create("a", Intent(node="worker", modelId="m"), {"id": "m"})
    lease = jobs.claim("a", "worker")["lease"]
    recovered = Journal(path, clock=lambda: now[0])
    with pytest.raises(HTTPException, match="409"):
        recovered.claim("a", "worker")
    now[0] += 121
    new_lease = recovered.claim("a", "worker")["lease"]
    assert new_lease != lease
    with pytest.raises(HTTPException, match="409"):
        jobs.checkpoint("a", "worker", Checkpoint(lease=lease, step="ready"))
    recovered.checkpoint(
        "a", "worker", Checkpoint(lease=new_lease, step="settings", engine="llama_cpp")
    )
    assert Journal(path).get("a")["step"] == "settings"


def test_concurrent_claims_have_one_winner(tmp_path):
    jobs = Journal(tmp_path / "runs.sqlite3")
    jobs.create(
        "a",
        Intent(modelId="m"),
        {"id": "m", "path": "/models/m.gguf", "name": "m", "format": "gguf", "status": "present"},
    )

    def claim(_):
        try:
            return jobs.claim("a", None)["lease"]
        except HTTPException as exc:
            assert exc.status_code == 409
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len([v for v in pool.map(claim, range(8)) if v]) == 1


def test_idempotent_create_and_conflicting_reuse(tmp_path):
    jobs = Journal(tmp_path / "runs.sqlite3")
    first = jobs.create(
        "a",
        Intent(modelId="m"),
        {"id": "m", "path": "/models/m.gguf", "name": "m", "format": "gguf", "status": "present"},
    )
    assert jobs.create("a", Intent(modelId="m"), None) == first
    assert jobs.create("b", Intent(modelId="m"), None) == first
    with pytest.raises(HTTPException, match="409"):
        jobs.create("a", Intent(modelId="different"), None)
    assert len(jobs.list()) == 1
    lease = jobs.claim("a", None)["lease"]
    jobs.checkpoint("a", None, Checkpoint(lease=lease, step="ready"))
    assert jobs.create("b", Intent(modelId="m"), None)["step"] == "ready"


def test_future_journal_is_untouched(tmp_path):
    path = tmp_path / "runs.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=9")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="version 9"):
        Journal(path)
    assert path.read_bytes() == before


def test_unversioned_nonempty_journal_is_untouched(tmp_path):
    path = tmp_path / "runs.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE owned_data(value TEXT)")
        db.execute("INSERT INTO owned_data VALUES ('keep me')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unversioned, nonempty"):
        Journal(path)
    assert path.read_bytes() == before


def test_retry_returns_completed_receipt_after_source_model_is_removed(client, tmp_path):
    model = _model(tmp_path / "model.gguf")
    store = client.app.state.state_store
    store.replace_models([model], scanned_at=datetime.now(UTC))
    intent = {"modelId": model.id}
    assert client.put("/v1/run-operations/a", json=intent).status_code == 202
    jobs = client.app.state.run_operations
    lease = jobs.claim("a", None)["lease"]
    jobs.checkpoint("a", None, Checkpoint(lease=lease, step="ready"))
    store.replace_models([], scanned_at=datetime.now(UTC))
    assert store.get_model(model.id) is None
    response = client.put("/v1/run-operations/a", json=intent)
    assert response.status_code == 202
    assert response.json()["step"] == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_pause", [False, True])
async def test_restart_resumes_only_transfers_not_explicitly_paused(
    tmp_path, monkeypatch, explicit_pause
):
    path = tmp_path / "runs.sqlite3"
    jobs = Journal(path)
    jobs.create("a", Intent(download=DownloadSpec(repo="org/model", files=["m.gguf"])), None)
    # An iteration may have listed this record before the pause request arrives.
    stale_listing = jobs.list()
    app = SimpleNamespace(
        state=SimpleNamespace(run_operations=jobs, run_download_lock=asyncio.Lock())
    )
    if explicit_pause:
        await run_operations.set_download_pause(
            Request({"type": "http", "app": app}), "a", paused=True
        )
    # Both worker and journal are new; no in-memory pause state survives.
    app.state.run_operations = Journal(path)
    monkeypatch.setattr(app.state.run_operations, "list", lambda **_: stale_listing)
    paused = Download(id="a", repo="org/model", state="paused", files=[])
    resume = AsyncMock(return_value=paused.model_copy(update={"state": paused.state.downloading}))
    app.state.download_manager = SimpleNamespace(get=lambda _: paused, resume=resume)

    async def end_iteration(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(run_operations.asyncio, "sleep", end_iteration)
    with pytest.raises(asyncio.CancelledError):
        await run_operations.advance_downloads(app)
    assert resume.await_count == (0 if explicit_pause else 1)
    assert Journal(path).get("a")["step"] == "downloading"


def test_cancel_fences_a_claim_before_the_next_stage(client):
    jobs = client.app.state.run_operations
    jobs.create(
        "a",
        Intent(modelId="m"),
        {"id": "m", "path": "/models/m.gguf", "name": "m", "format": "gguf", "status": "present"},
    )
    lease = client.post("/v1/run-operations/a/claim").json()["lease"]
    assert client.post("/v1/run-operations/a/cancel").json()["step"] == "cancelled"
    response = client.post(
        "/v1/run-operations/a/checkpoint", json={"lease": lease, "step": "launching"}
    )
    assert response.status_code == 409
    assert client.delete("/v1/run-operations/a").status_code == 204
    assert client.get("/v1/run-operations").json() == {"operations": []}
    assert jobs.get("a")["step"] == "cancelled"


def test_agent_grants_cannot_submit_edit_other_nodes_or_rewrite_profiles(app, install, tmp_path):
    app.state.auth_state = install.auth_state()
    with TestClient(app) as client:
        model = _model(tmp_path / "model.gguf")
        app.state.state_store.replace_models([model], scanned_at=datetime.now(UTC))
        operator = {"Authorization": "Bearer " + install.session()}
        worker = {"Authorization": "Bearer " + install.foreign_service("agent")}
        local = {"Authorization": "Bearer " + install.service("agent")}
        gateway = {"Authorization": "Bearer " + install.service("gateway")}
        intent = {"node": "far", "modelId": model.id}
        assert client.put("/v1/run-operations/a", json=intent, headers=worker).status_code == 401
        assert client.put("/v1/run-operations/a", json=intent, headers=operator).status_code == 202
        assert client.get("/v1/run-operations/assigned", headers=local).json() == {"operations": []}
        assert client.post("/v1/run-operations/a/claim", headers=local).status_code == 403
        assert client.post("/v1/run-operations/a/claim", headers=gateway).status_code == 403
        claimed = client.post("/v1/run-operations/a/claim", headers=worker).json()
        lease = claimed["lease"]
        assert client.get("/v1/run-operations", headers=worker).status_code == 401
        body = {"lease": lease, "engine": "llama_cpp", "contextSize": 4096}
        assert (
            client.post("/v1/run-operations/a/profile", json=body, headers=worker).status_code
            == 409
        )
        response = client.post(
            "/v1/run-operations/a/checkpoint",
            json={"lease": lease, "step": "settings", "engine": "llama_cpp"},
            headers=worker,
        )
        assert response.status_code == 200
        lease = client.post("/v1/run-operations/a/claim", headers=worker).json()["lease"]
        body["lease"] = lease
        first = client.post("/v1/run-operations/a/profile", json=body, headers=worker)
        assert first.status_code == 200, first.text
        # Crash between profile side effect and checkpoint: replay adopts it.
        again = client.post("/v1/run-operations/a/profile", json=body, headers=worker)
        assert again.json()["profile"] == first.json()["profile"]
        assert len(app.state.state_store.list_profiles(model.id)) == 1
        assert (
            client.post(
                f"/v1/models/{model.id}/profiles",
                json={"name": "pwn", "engine": "llama_cpp"},
                headers=worker,
            ).status_code
            == 401
        )
        assert client.post("/v1/run-operations/a/cancel", headers=worker).status_code == 401
        # Tokens and grants are absent from persisted/public operation state.
        assert (
            "lease"
            not in client.get("/v1/run-operations", headers=operator).json()["operations"][0]
        )


def test_install_answer_survives_restart_and_is_idempotent(client):
    jobs = client.app.state.run_operations
    jobs.create(
        "a",
        Intent(modelId="m"),
        {"id": "m", "path": "/models/m.gguf", "name": "m", "format": "gguf", "status": "present"},
    )
    lease = jobs.claim("a", None)["lease"]
    jobs.checkpoint("a", None, Checkpoint(lease=lease, step="awaiting-install", engine="llama_cpp"))
    assert client.post("/v1/run-operations/a/answer", json={"answer": "skip"}).status_code == 200
    assert Journal(jobs.path).get("a")["answer"] == "skip"
    lease = jobs.claim("a", None)["lease"]
    jobs.checkpoint("a", None, Checkpoint(lease=lease, step="settings"))
    assert client.post("/v1/run-operations/a/answer", json={"answer": "skip"}).status_code == 200
    assert client.post("/v1/run-operations/a/answer", json={"answer": "install"}).status_code == 409
