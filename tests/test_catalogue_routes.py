"""The catalogue and download endpoints, with upstream mocked out.

Covers the seam the module tests cannot: the auth level each router
sits behind, and whether an upstream failure arrives as something an
operator can act on.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import hardware, hub
from eugene_plexus_library._generated.models import Arch, Gpu, HostHardware, Os, Vendor

TREE = [
    {
        "type": "file",
        "path": "model-Q4_K_M.gguf",
        "size": 4_000_000_000,
        "oid": "0" * 40,
        "lfs": {"oid": "a" * 64, "size": 4_000_000_000},
    },
    {
        "type": "file",
        "path": "model-Q8_0.gguf",
        "size": 8_000_000_000,
        "oid": "1" * 40,
        "lfs": {"oid": "b" * 64, "size": 8_000_000_000},
    },
    {
        "type": "file",
        "path": "mmproj-F16.gguf",
        "size": 900_000_000,
        "oid": "2" * 40,
        "lfs": {"oid": "c" * 64, "size": 900_000_000},
    },
]

INFO = {
    "author": "org",
    "sha": "commit1",
    "downloads": 1234,
    "likes": 7,
    "gated": False,
    "tags": ["gguf", "license:apache-2.0"],
    "gguf": {"total": 8_000_000_000, "architecture": "llama", "context_length": 32768},
}

SEARCH = [
    {
        "id": "org/repo",
        "author": "org",
        "downloads": 1234,
        "likes": 7,
        "tags": ["gguf"],
        "library_name": "gguf",
    },
    {
        "id": "org/torch-only",
        "author": "org",
        "downloads": 99,
        "tags": ["safetensors"],
        "library_name": "transformers",
    },
]


GIB = 1024**3
MIB = 1024**2

# **A fixed machine, because the detail endpoint scores against the one
# the tests run on.** `_budget` calls `hardware.detect()` per request, so
# with nothing pinned the recommendation is decided by whatever else is
# holding the GPU: this file passed with 30 GiB free and failed with
# 509 MiB, as `TypeError: 'NoneType' object is not subscriptable`,
# because `recommend()` returns None when nothing fits entirely. CI never
# saw it -- CI has no GPU, so the detected budget is host RAM and stable.
#
# The `vramBytes` query override is not enough on its own. It replaces
# the three VRAM figures and `ramBytes` the two RAM ones, but `gpuCount`
# and `unifiedMemory` come from detection either way, and
# `unifiedMemory` picks a different branch in `_verdict` -- so an
# overridden test would still answer differently on an Apple Silicon
# machine, which is a dev platform this project now supports. Pinning
# the whole `HostHardware` closes that, and stops every request shelling out
# to nvidia-smi as a side effect.
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


def upstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if "/tree/" in path:
        return httpx.Response(200, json=TREE)
    if path == "/api/models":
        return httpx.Response(200, json=SEARCH)
    if path.startswith("/api/models/"):
        return httpx.Response(200, json=INFO)
    if path.endswith("README.md"):
        return httpx.Response(200, text="# A model\n\nIt does things.")
    return httpx.Response(404)


@pytest.fixture
def catalogue_client(
    configured_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client whose hub is a mock transport rather than the internet,
    and whose hardware is `HOST` rather than this machine."""
    inner = httpx.AsyncClient(transport=httpx.MockTransport(upstream), follow_redirects=False)
    configured_client.app.state.hub_client = hub.HubClient(client=inner)
    monkeypatch.setattr(hardware, "detect", lambda: HOST)
    yield configured_client


def test_search_returns_rows_and_a_cursor_field(catalogue_client: TestClient) -> None:
    body = catalogue_client.get("/v1/catalogue/search", params={"q": "qwen"}).json()
    assert [r["repo"] for r in body["results"]] == ["org/repo", "org/torch-only"]
    assert "nextCursor" in body


def test_a_safetensors_filter_is_applied_locally(catalogue_client: TestClient) -> None:
    """Upstream has no reliable safetensors filter, so this is a
    post-filter on tags and approximate by nature — a repo can hold both
    formats."""
    body = catalogue_client.get(
        "/v1/catalogue/search", params={"q": "x", "format": "safetensors"}
    ).json()
    assert [r["repo"] for r in body["results"]] == ["org/torch-only"]


def test_detail_groups_candidates_and_attaches_the_fit(catalogue_client: TestClient) -> None:
    body = catalogue_client.get(
        "/v1/catalogue/model", params={"repo": "org/repo", "contextLength": 4096}
    ).json()
    labels = {c["label"] for c in body["candidates"]}
    assert labels == {"Q4_K_M", "Q8_0"}
    assert len(body["projectors"]) == 1
    assert all(c["fit"]["contextLength"] == 4096 for c in body["candidates"])
    assert body["resolvedCommit"] == "commit1"
    # Nameable now that the machine is fixed rather than detected: both
    # fit in 32 GiB and `recommend` takes the largest that fits
    # entirely, so "one of them" can become "this one".
    assert body["recommended"]["label"] == "Q8_0"


def test_detail_recommends_nothing_when_nothing_fits_and_says_why(
    catalogue_client: TestClient,
) -> None:
    """`recommended` is null rather than a best-effort pick, because
    partial offload is materially slower and is the operator's call to
    make knowingly. Asserted here because that null is what broke the
    test above whenever a real GPU was busy: the branch existed, ran on
    developer machines, and was covered by nothing.

    `vramBytes` is the documented operator override, so this also checks
    that squeezing the budget by hand reaches the verdict."""
    body = catalogue_client.get(
        "/v1/catalogue/model",
        params={"repo": "org/repo", "contextLength": 4096, "vramBytes": 509 * MIB},
    ).json()
    assert body["recommended"] is None
    assert {c["label"] for c in body["candidates"]} == {"Q4_K_M", "Q8_0"}
    assert any("509.00 MiB of free GPU memory" in w for w in body["warnings"])


def test_detail_defaults_the_context_from_config(catalogue_client: TestClient) -> None:
    """`guidanceContextLength` is the single number that decides which
    quant gets recommended, which is why it is config and not a
    constant."""
    catalogue_client.patch("/v1/config", json={"guidanceContextLength": 16384})
    body = catalogue_client.get("/v1/catalogue/model", params={"repo": "org/repo"}).json()
    assert all(c["fit"]["contextLength"] == 16384 for c in body["candidates"])


def test_the_card_is_returned_as_markdown(catalogue_client: TestClient) -> None:
    body = catalogue_client.get("/v1/catalogue/model/card", params={"repo": "org/repo"}).json()
    assert body["markdown"].startswith("# A model")
    assert body["fetchedAt"]


def test_an_upstream_404_is_a_404_with_an_explanation(catalogue_client: TestClient) -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"X-Error-Code": "RepoNotFound"})

    inner = httpx.AsyncClient(transport=httpx.MockTransport(missing), follow_redirects=False)
    catalogue_client.app.state.hub_client = hub.HubClient(client=inner)

    response = catalogue_client.get("/v1/catalogue/model", params={"repo": "org/nope"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["component"] == "library"
    assert "private repo" in detail["detail"]


def test_a_gated_download_says_what_to_do(catalogue_client: TestClient) -> None:
    def gated(request: httpx.Request) -> httpx.Response:
        if "/tree/" in request.url.path:
            return httpx.Response(200, json=TREE)
        return httpx.Response(
            401,
            headers={
                "X-Error-Code": "GatedRepo",
                "X-Error-Message": "Access to model org/gated is restricted.",
            },
        )

    inner = httpx.AsyncClient(transport=httpx.MockTransport(gated), follow_redirects=False)
    catalogue_client.app.state.hub_client = hub.HubClient(client=inner)

    response = catalogue_client.post(
        "/v1/downloads", json={"repo": "org/gated", "files": ["model-Q4_K_M.gguf"]}
    )
    assert response.status_code == 403
    assert "hfToken" in response.json()["detail"]["detail"]


def test_the_catalogue_switch_refuses_every_endpoint(catalogue_client: TestClient) -> None:
    """An air-gapped install gets one legible answer, not four different
    timeouts."""
    catalogue_client.patch("/v1/config", json={"catalogueEnabled": False})
    for path, params in (
        ("/v1/catalogue/search", {"q": "x"}),
        ("/v1/catalogue/model", {"repo": "org/repo"}),
        ("/v1/catalogue/model/card", {"repo": "org/repo"}),
        ("/v1/catalogue/model/preflight", {"repo": "org/repo", "file": "model-Q4_K_M.gguf"}),
    ):
        response = catalogue_client.get(path, params=params)
        assert response.status_code == 409, path
        assert "catalogueEnabled" in response.json()["detail"]["detail"]


def test_downloads_list_is_empty_before_anything_is_asked_for(
    catalogue_client: TestClient,
) -> None:
    assert catalogue_client.get("/v1/downloads").json() == {"downloads": []}


def test_an_unknown_download_is_a_404(catalogue_client: TestClient) -> None:
    assert catalogue_client.get("/v1/downloads/nope").status_code == 404


def test_a_download_with_no_roots_configured_is_refused(catalogue_client: TestClient) -> None:
    """Nowhere to put it, and inventing a directory is the one thing this
    component will not do."""
    catalogue_client.patch("/v1/config", json={"modelRoots": []})
    response = catalogue_client.post(
        "/v1/downloads", json={"repo": "org/repo", "files": ["model-Q4_K_M.gguf"]}
    )
    assert response.status_code == 409
    assert "modelRoots" in response.json()["detail"]["detail"]
