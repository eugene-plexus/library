"""LS5: a run operation that prepares its model (library-sources-and-engines.md §6.6).

The library's half: the intent, the `preparing` step, `POST /{id}/prepared`
writing the provenance file beside the prepared entry (B55), the guards that
keep the run on the prepared model, and the scan skipping an engine's own
folder (B44).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_library import prepared
from eugene_plexus_library.run_operations import Checkpoint, Intent, Journal, Operation, public
from eugene_plexus_library.scanner import Scanner

from .conftest import write_gguf

FIRST = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"


@pytest.fixture
def prepared_client(configured_client: TestClient, models_dir: Path) -> TestClient:
    """A library over `models_dir` holding a GGUF, and Strata-data beside it
    as a preparation leaves it: the configuration and the marker."""
    gguf = models_dir / "ISTA-DASLab" / "repo" / FIRST
    gguf.parent.mkdir(parents=True)
    write_gguf(gguf, {"general.architecture": "qwen4exp"})
    data = models_dir / "Strata-data"
    data.mkdir()
    (data / prepared.ENGINE_FILES_MARKER).write_text("engine files", encoding="utf-8")
    (data / "strata-iq2_xs.json").write_text("{}", encoding="utf-8")
    # A walk in flight from startup must not replace what this one finds.
    for _ in range(400):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
        time.sleep(0.01)
    configured_client.post("/v1/scan")
    for _ in range(400):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
        time.sleep(0.01)
    models = configured_client.app.state.state_store.list_models()
    assert [Path(m.path).name for m in models] == [FIRST]
    return configured_client


def _submit(client: TestClient, **extra: object) -> dict:
    model = client.app.state.state_store.list_models()[0]
    intent = {"modelId": model.id, "preparation": {"engine": "strata"}, **extra}
    response = client.put("/v1/run-operations/p", json=intent)
    assert response.status_code == 202, response.text
    return response.json()


def _to_preparing(client: TestClient) -> str:
    jobs = client.app.state.run_operations
    lease = jobs.claim("p", None)["lease"]
    jobs.checkpoint("p", None, Checkpoint(lease=lease, step="preparing", engine="strata"))
    return jobs.claim("p", None)["lease"]


def _body(lease: str, models_dir: Path, **provenance: object) -> dict:
    return {
        "lease": lease,
        "name": "qwen3.8-flash-next-iq2_xs",
        "provenance": {
            "engine": "strata",
            "entry": str(models_dir / "Strata-data" / "strata-iq2_xs.json"),
            "recipe": "strata-prepare",
            "recipeVersion": "v0.1.39",
            "source": {"repoId": "ISTA-DASLab/repo", "file": f"IQ2_XS/{FIRST}"},
            **provenance,
        },
    }


def test_the_intent_carries_the_engine_and_context():
    intent = Intent(modelId="m", preparation={"engine": "strata", "contextSize": 32768})
    assert intent.preparation is not None
    assert intent.preparation.engine.value == "strata"
    with pytest.raises(ValueError):
        Intent(modelId="m", preparation={"engine": "not-an-engine"})
    with pytest.raises(ValueError):
        Intent(modelId="m", preparation={"engine": "strata", "contextSize": 0})


def test_prepared_lists_the_model_and_the_run_goes_on_with_it(prepared_client, models_dir):
    record = _submit(prepared_client)
    original = record["model"]
    lease = _to_preparing(prepared_client)
    response = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["preparedFrom"]["id"] == original["id"]
    assert record["model"]["format"] == "prepared"
    target = models_dir / "Strata-data" / "qwen3.8-flash-next-iq2_xs.eugene-prepared.json"
    assert record["model"]["path"] == str(target)
    written = json.loads(target.read_text(encoding="utf-8"))
    # Beside its entry, so the entry is relative and the folder travels.
    assert written["entry"] == "strata-iq2_xs.json"
    assert written["recipe"] == "strata-prepare"
    assert written["formatVersion"] == 1
    listed = prepared_client.get("/v1/models").json()
    assert str(target) in [m["path"] for m in listed["models"]]
    # Then the run goes on, with the prepared model.
    checkpoint = {"lease": lease, "step": "settings", "engine": "strata"}
    after = prepared_client.post("/v1/run-operations/p/checkpoint", json=checkpoint)
    assert after.status_code == 200, after.text
    assert after.json()["model"]["format"] == "prepared"


def test_a_preparation_run_cannot_skip_to_settings_unlisted(prepared_client):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    checkpoint = {"lease": lease, "step": "settings", "engine": "strata"}
    response = prepared_client.post("/v1/run-operations/p/checkpoint", json=checkpoint)
    assert response.status_code == 409
    assert "not listed" in response.json()["detail"]


def test_preparing_again_replaces_its_own_file_and_no_other(prepared_client, models_dir):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    first = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert first.status_code == 200, first.text
    target = models_dir / "Strata-data" / "qwen3.8-flash-next-iq2_xs.eugene-prepared.json"
    # The same recipe for the same entry: replaced (a lost checkpoint, or a
    # second preparation of the same choice).
    again = prepared_client.post(
        "/v1/run-operations/p/prepared",
        json=_body(lease, models_dir, recipeVersion="v0.1.40"),
    )
    assert again.status_code == 200, again.text
    assert json.loads(target.read_text(encoding="utf-8"))["recipeVersion"] == "v0.1.40"
    # A file of that name made by hand (no recipe) is someone else's.
    target.write_text(
        json.dumps({"engine": "strata", "entry": "strata-iq2_xs.json"}), encoding="utf-8"
    )
    refused = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert refused.status_code == 409
    assert "recipe" not in json.loads(target.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"engine": "llama_cpp"}, 409),  # not the engine the run prepares for
        ({"entry": "OUTSIDE"}, 400),  # there, but outside every Library folder
        ({"entry": "MISSING"}, 400),  # not there
    ],
)
def test_prepared_refusals_write_nothing(prepared_client, models_dir, tmp_path, change, status):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    outside = tmp_path / "elsewhere"
    if change.get("entry") == "MISSING":
        change = {"entry": str(models_dir / "Strata-data" / "strata-missing.json")}
    elif change.get("entry") == "OUTSIDE":
        outside.mkdir()
        (outside / "strata-iq2_xs.json").write_text("{}", encoding="utf-8")
        change = {"entry": str(outside / "strata-iq2_xs.json")}
    response = prepared_client.post(
        "/v1/run-operations/p/prepared", json=_body(lease, models_dir, **change)
    )
    assert response.status_code == status, response.text
    if status == 400 and outside.exists():
        assert "not inside a Library folder" in response.json()["detail"]
    assert not list((models_dir / "Strata-data").glob("*.eugene-prepared.json"))
    if outside.exists():
        assert not list(outside.glob("*.eugene-prepared.json"))
    assert prepared_client.get("/v1/run-operations/p").json()["preparedFrom"] is None


def test_prepared_needs_the_lease_and_the_step(prepared_client, models_dir):
    _submit(prepared_client)
    jobs = prepared_client.app.state.run_operations
    lease = jobs.claim("p", None)["lease"]
    # Still checking: nothing to list yet.
    early = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert early.status_code == 409
    jobs.checkpoint("p", None, Checkpoint(lease=lease, step="preparing", engine="strata"))
    # The lease was spent by the checkpoint.
    stale = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert stale.status_code == 409
    assert not list((models_dir / "Strata-data").glob("*.eugene-prepared.json"))


def test_a_run_without_preparation_cannot_use_prepared(prepared_client, models_dir):
    model = prepared_client.app.state.state_store.list_models()[0]
    assert (
        prepared_client.put("/v1/run-operations/r", json={"modelId": model.id}).status_code == 202
    )
    jobs = prepared_client.app.state.run_operations
    lease = jobs.claim("r", None)["lease"]
    response = prepared_client.post("/v1/run-operations/r/prepared", json=_body(lease, models_dir))
    assert response.status_code == 409


def test_skip_is_not_an_answer_to_a_preparation(prepared_client):
    _submit(prepared_client)
    jobs = prepared_client.app.state.run_operations
    lease = jobs.claim("p", None)["lease"]
    jobs.checkpoint("p", None, Checkpoint(lease=lease, step="awaiting-install", engine="strata"))
    skip = prepared_client.post("/v1/run-operations/p/answer", json={"answer": "skip"})
    assert skip.status_code == 409
    assert "install it" in skip.json()["detail"]
    install = prepared_client.post("/v1/run-operations/p/answer", json={"answer": "install"})
    assert install.json()["step"] == "installing"


MODEL = {"id": "m", "path": "/models/m.gguf", "name": "m", "format": "gguf", "status": "present"}


def test_progress_is_kept_on_the_operation(tmp_path):
    jobs = Journal(tmp_path / "runs.sqlite3")
    jobs.create("p", Intent(modelId="m", preparation={"engine": "strata"}), MODEL)
    lease = jobs.claim("p", None)["lease"]
    status = {"state": "running", "step": "Step 6 of 7", "bytesWritten": 5, "bytesNeeded": 9}
    record = jobs.checkpoint(
        "p", None, Checkpoint(lease=lease, step="preparing", preparation=status)
    )
    assert record["preparation"]["step"] == "Step 6 of 7"
    seen = Operation.model_validate(public(record))
    assert seen.preparation is not None and seen.preparation.warnings == []
    with pytest.raises(ValueError):
        Checkpoint(lease=lease, step="preparing", preparation={"state": "sleeping"})


def test_failed_preparation_is_a_failed_run(tmp_path):
    jobs = Journal(tmp_path / "runs.sqlite3")
    jobs.create("p", Intent(modelId="m", preparation={"engine": "strata"}), {"id": "m"})
    lease = jobs.claim("p", None)["lease"]
    jobs.checkpoint("p", None, Checkpoint(lease=lease, step="preparing"))
    lease = jobs.claim("p", None)["lease"]
    record = jobs.checkpoint(
        "p",
        None,
        Checkpoint(lease=lease, step="failed", failedStep="prepare", error="setup stopped"),
    )
    assert record["step"] == "failed" and record["finishedAt"] is not None
    with pytest.raises(HTTPException):
        jobs.claim("p", None)


def test_the_scan_lists_only_provenance_in_an_engines_folder(tmp_path):
    """B44: Strata's MTP helper is a GGUF; under the marker it is not a model."""
    root = tmp_path / "models"
    data = root / "Strata-data"
    (data / "mtp").mkdir(parents=True)
    write_gguf(data / "mtp" / "mtp-q2_0.gguf", {"general.architecture": "qwen4exp"})
    (data / "packs" / "iq2_xs").mkdir(parents=True)
    write_gguf(data / "packs" / "iq2_xs" / "stray.gguf", {"general.architecture": "llama"})
    (data / "strata-iq2_xs.json").write_text("{}", encoding="utf-8")
    (data / "qwen.eugene-prepared.json").write_text(
        json.dumps({"engine": "strata", "entry": "strata-iq2_xs.json"}), encoding="utf-8"
    )
    write_gguf(data / "loose.gguf", {"general.architecture": "llama"})
    before = Scanner().scan([root])
    assert {Path(m.path).name for m in before.models} >= {"mtp-q2_0.gguf", "loose.gguf"}
    (data / prepared.ENGINE_FILES_MARKER).write_text("engine files", encoding="utf-8")
    after = Scanner().scan([root])
    assert [Path(m.path).name for m in after.models] == ["qwen.eugene-prepared.json"]
    assert any(s.path == str(data) for s in after.skipped)
