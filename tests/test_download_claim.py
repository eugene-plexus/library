"""The intent a download carries, and the claim that takes it.

The library never acts on `runWhenReady` — it records it so the intent
outlives the browser tab that expressed it, and hands it to exactly one
console. Both halves of that sentence are what these assert.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import hub

TREE = [
    {
        "type": "file",
        "path": "model-Q4_K_M.gguf",
        "size": 4_000_000,
        "oid": "0" * 40,
        "lfs": {"oid": "a" * 64, "size": 4_000_000},
    },
]

INFO = {"author": "org", "sha": "commit1", "gated": False, "tags": ["gguf"]}


def upstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if "/tree/" in path:
        return httpx.Response(200, json=TREE)
    if path.startswith("/api/models/"):
        return httpx.Response(200, json=INFO)
    # Never actually transfer: every test here is about the record.
    return httpx.Response(500)


@pytest.fixture
def downloads_client(configured_client: TestClient) -> Iterator[TestClient]:
    """A client whose DOWNLOAD MANAGER speaks to the mock.

    Not `app.state.hub_client`, which is the catalogue's: the manager
    holds its own client, captured when the app was built. Patching the
    other one leaves the manager talking to the real hub, which is how
    the first version of this file got a 401 from huggingface.co.
    """
    inner = httpx.AsyncClient(transport=httpx.MockTransport(upstream), follow_redirects=False)
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    configured_client.app.state.hub_client = client
    configured_client.app.state.download_manager._client = client
    yield configured_client


def start(client: TestClient, **extra: object) -> dict:
    response = client.post(
        "/v1/downloads",
        json={"repo": "org/repo", "files": ["model-Q4_K_M.gguf"], **extra},
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_a_download_records_the_intent_that_started_it(downloads_client: TestClient) -> None:
    record = start(downloads_client, runWhenReady=True)
    assert record["runWhenReady"] is True
    # And it survives a read, because the whole point is outliving the
    # tab that asked for it.
    again = downloads_client.get(f"/v1/downloads/{record['id']}").json()
    assert again["runWhenReady"] is True


def test_an_ordinary_download_carries_no_intent(downloads_client: TestClient) -> None:
    """Fetching a model is not asking to run it. The default has to be
    off, or every download anyone ever started would launch something."""
    record = start(downloads_client)
    assert not record.get("runWhenReady")


def test_exactly_one_caller_claims_it(downloads_client: TestClient) -> None:
    """Two browsers are two browsers. Every console on the install can
    see the same finished download; without this, every one of them
    would create a profile and launch a runtime for the same model."""
    record = start(downloads_client, runWhenReady=True)
    first = downloads_client.post(f"/v1/downloads/{record['id']}/claim").json()
    second = downloads_client.post(f"/v1/downloads/{record['id']}/claim").json()
    assert first["claimed"] is True
    assert second["claimed"] is False


def test_a_claim_clears_the_flag_on_the_record(downloads_client: TestClient) -> None:
    """So a console opening a week after the operator stopped that
    runtime on purpose does not see the old intent and start it again."""
    record = start(downloads_client, runWhenReady=True)
    downloads_client.post(f"/v1/downloads/{record['id']}/claim")
    assert downloads_client.get(f"/v1/downloads/{record['id']}").json()["runWhenReady"] is False


def test_claiming_a_download_with_no_intent_is_not_an_error(
    downloads_client: TestClient,
) -> None:
    """A client that retried after a timeout has not done anything
    wrong, and every console is expected to try."""
    record = start(downloads_client)
    response = downloads_client.post(f"/v1/downloads/{record['id']}/claim")
    assert response.status_code == 200
    assert response.json()["claimed"] is False


def test_claiming_an_unknown_download_is_a_404(downloads_client: TestClient) -> None:
    response = downloads_client.post("/v1/downloads/nosuchthing/claim")
    assert response.status_code == 404
    assert "nosuchthing" in response.json()["detail"]["detail"]


def test_the_claim_does_not_touch_the_transfer(downloads_client: TestClient) -> None:
    """It is an operator intent, not a control on the download. A
    claim must not pause, cancel or restart anything."""
    record = start(downloads_client, runWhenReady=True)
    before = downloads_client.get(f"/v1/downloads/{record['id']}").json()
    downloads_client.post(f"/v1/downloads/{record['id']}/claim")
    after = downloads_client.get(f"/v1/downloads/{record['id']}").json()
    for field in ("state", "files", "bytesTotal", "destinationDirectory", "resolvedCommit"):
        assert after[field] == before[field], field


# The operator-only guard is asserted in `test_auth.py`, where the
# fixtures actually configure auth: `configured_client` has no master key,
# so every route on it answers 200 and a 401 assertion here could not
# fail.
