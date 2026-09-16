"""`GET /v1/catalogue/starter`, and a pasted URL on the search endpoint.

The property worth a route test rather than a module test: the starter
endpoint must answer with the hub **unreachable**, because the screen
where a person picks their first model should not be the screen that
fails first.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import hardware, hub
from eugene_plexus_library._generated.models import Arch, Gpu, HostHardware, Os, Vendor

GIB = 1024**3

# A fixed machine, for `test_catalogue_routes`'s reason: `_budget` calls
# `hardware.detect()` per request, so an unpinned host makes the
# recommendation depend on what else is holding the GPU right now.
HOST = HostHardware(
    hostname="fixture",
    os=Os.linux,
    arch=Arch.x64,
    cpuCount=16,
    ramTotalBytes=96 * GIB,
    ramAvailableBytes=64 * GIB,
    unifiedMemory=False,
    gpus=[
        Gpu(
            index=0,
            name="Fixture GPU",
            vendor=Vendor.nvidia,
            vramTotalBytes=32 * GIB,
            vramFreeBytes=32 * GIB,
        )
    ],
)

REPO_INFO = {
    "author": "unsloth",
    "sha": "commit1",
    "downloads": 4242,
    "likes": 7,
    "gated": False,
    "tags": ["gguf", "license:apache-2.0"],
    "gguf": {"total": 8_000_000_000, "architecture": "qwen35", "context_length": 262144},
}

SEARCH = [{"id": "org/from-search", "author": "org", "downloads": 1, "tags": ["gguf"]}]


def upstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/models":
        return httpx.Response(200, json=SEARCH)
    if path == "/api/models/unsloth/Qwen3.8-27B-GGUF":
        return httpx.Response(200, json=REPO_INFO)
    return httpx.Response(404, json={"error": "Repository not found"})


@pytest.fixture
def catalogue_client(
    configured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    inner = httpx.AsyncClient(transport=httpx.MockTransport(upstream), follow_redirects=False)
    configured_client.app.state.hub_client = hub.HubClient(client=inner)
    monkeypatch.setattr(hardware, "detect", lambda: HOST)
    yield configured_client


@pytest.fixture
def offline_client(
    configured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client whose hub refuses every connection."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    inner = httpx.AsyncClient(transport=httpx.MockTransport(dead), follow_redirects=False)
    configured_client.app.state.hub_client = hub.HubClient(client=inner)
    monkeypatch.setattr(hardware, "detect", lambda: HOST)
    yield configured_client


# --- the starter set ---------------------------------------------------


def test_the_starter_set_is_served_with_a_recommendation(
    catalogue_client: TestClient,
) -> None:
    body = catalogue_client.get("/v1/catalogue/starter").json()
    assert body["source"] == "shipped"
    assert body["models"], "the shipped list came back empty"
    assert body["recommended"]["reason"]
    assert body["reviewed"]
    assert body["reviewedDaysAgo"] >= 0
    for model in body["models"]:
        assert model["fit"]["verdict"] in {"fits", "tight", "split", "no"}


def test_it_answers_with_the_hub_unreachable(offline_client: TestClient) -> None:
    """The point of carrying every measured number in the list. A first
    model can be chosen with no internet; only fetching it needs one.
    The search endpoint beside it fails, as it must."""
    assert offline_client.get("/v1/catalogue/starter").status_code == 200
    assert offline_client.get("/v1/catalogue/search", params={"q": "x"}).status_code == 503


def test_the_context_is_the_one_the_caller_asked_for(catalogue_client: TestClient) -> None:
    body = catalogue_client.get("/v1/catalogue/starter", params={"contextLength": 32768}).json()
    assert all(m["fit"]["contextLength"] == 32768 for m in body["models"])
    assert "32,768" in body["recommended"]["reason"]


def test_a_vram_override_scores_against_another_machine(
    catalogue_client: TestClient,
) -> None:
    """The multi-host case: the machine a launch runs on is not the one
    the library runs on, and the recommendation must be about the
    former."""
    big = catalogue_client.get(
        "/v1/catalogue/starter", params={"vramBytes": 80 * GIB, "contextLength": 16384}
    ).json()
    small = catalogue_client.get(
        "/v1/catalogue/starter", params={"vramBytes": 6 * GIB, "contextLength": 16384}
    ).json()
    assert big["recommended"]["sizeClass"] != small["recommended"]["sizeClass"]


def test_an_operators_own_list_replaces_the_shipped_one(
    catalogue_client: TestClient, tmp_path
) -> None:
    """`starterModelsFile`, and the `configured` source that says the
    operator did it. An empty list is a valid answer meaning "no
    recommendation" rather than a bug."""
    path = tmp_path / "mine.yaml"
    path.write_text("reviewed: 2026-09-16\nclasses: []\n", encoding="utf-8")
    response = catalogue_client.patch("/v1/config", json={"starterModelsFile": str(path)})
    assert response.status_code == 200, response.text
    body = catalogue_client.get("/v1/catalogue/starter").json()
    assert body["source"] == "configured"
    assert body["models"] == []
    assert body.get("recommended") is None


# --- a pasted reference on the search endpoint -------------------------


def test_a_pasted_url_resolves_to_one_repo(catalogue_client: TestClient) -> None:
    body = catalogue_client.get(
        "/v1/catalogue/search",
        params={"q": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main"},
    ).json()
    assert body["interpretedAs"] == "repo"
    assert body["interpretedFrom"] == "unsloth/Qwen3.8-27B-GGUF"
    assert [r["repo"] for r in body["results"]] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert body["results"][0]["downloads"] == 4242


def test_a_bare_owner_slash_name_resolves_too(catalogue_client: TestClient) -> None:
    body = catalogue_client.get(
        "/v1/catalogue/search", params={"q": "unsloth/Qwen3.8-27B-GGUF"}
    ).json()
    assert body["interpretedAs"] == "repo"


def test_a_url_that_does_not_resolve_is_a_404_naming_the_repo(
    catalogue_client: TestClient,
) -> None:
    """A URL names one repo and nothing else, so an empty result list
    would be the silent failure -- the exact shape of the dead end this
    replaces."""
    response = catalogue_client.get(
        "/v1/catalogue/search", params={"q": "https://huggingface.co/nobody/nothing"}
    )
    assert response.status_code == 404
    assert "nobody/nothing" in response.json()["detail"]["detail"]


def test_a_bare_name_that_does_not_resolve_falls_through_to_a_search(
    catalogue_client: TestClient,
) -> None:
    """It may simply be what they meant to type. A guess that misses is
    a search, not an error."""
    body = catalogue_client.get("/v1/catalogue/search", params={"q": "nobody/nothing"}).json()
    assert body["interpretedAs"] == "search"
    assert [r["repo"] for r in body["results"]] == ["org/from-search"]


def test_an_ordinary_query_is_still_an_ordinary_search(
    catalogue_client: TestClient,
) -> None:
    body = catalogue_client.get("/v1/catalogue/search", params={"q": "qwen"}).json()
    assert body["interpretedAs"] == "search"
    assert [r["repo"] for r in body["results"]] == ["org/from-search"]
