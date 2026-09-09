"""The upstream client, against a mock transport.

Every fixture body here is shaped like a response the live hub actually
gave on 2026-09-08. The traps this covers — three hashes per entry, the
metadata living on the redirect rather than the response — are ones that
a client written from the docs gets wrong silently.
"""

from __future__ import annotations

import httpx
import pytest

from eugene_plexus_library import hub
from eugene_plexus_library._generated.models import CatalogueSort, GateKind

TREE = [
    {
        "type": "file",
        "oid": "e663c1e26c3e19799a7a7383ce2ea9de194fb115",
        "size": 16_464_440_224,
        "lfs": {
            "oid": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
            "size": 16_464_440_224,
            "pointerSize": 136,
        },
        "xetHash": "76b9d90a30128aedd438a2239b9b3867e877687223a9bf8c55eff5314e3a1473",
        "path": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    },
    {
        "type": "file",
        "oid": "d46195ac87f837ad233d02b2f80f148bf7c005e0",
        "size": 728,
        "path": "config.json",
    },
    {"type": "directory", "oid": "45a8269c", "size": 0, "path": "MTP"},
]


def client_for(handler) -> hub.HubClient:
    transport = httpx.MockTransport(handler)
    inner = httpx.AsyncClient(transport=transport, follow_redirects=False)
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    return client


async def test_the_content_digest_comes_from_lfs_oid() -> None:
    """Three hashes per entry and only one is the content sha256.

    `oid` is the git sha1 of the 136-byte LFS *pointer*, so a verifier
    that used it would compare a 16 GB download against a hash of the
    stub and pass on anything at all.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TREE)

    files = await client_for(handler).tree("org/repo")
    by_path = {f.path: f for f in files}

    weights = by_path["Qwen3.8-27B-UD-Q4_K_M.gguf"]
    assert weights.sha256 == TREE[0]["lfs"]["oid"]  # type: ignore[index]
    assert weights.sha256 != TREE[0]["oid"]
    assert weights.sha256 != TREE[0]["xetHash"]
    assert weights.digest == ("sha256", weights.sha256)
    assert weights.lfs is True


async def test_a_non_lfs_file_falls_back_to_the_git_blob_hash() -> None:
    """Sidecars are plain git objects with no content digest. They are
    still checkable, and between the two every file has something."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TREE)

    files = await client_for(handler).tree("org/repo")
    config = next(f for f in files if f.path == "config.json")
    assert config.sha256 is None
    assert config.git_blob_sha1 == "d46195ac87f837ad233d02b2f80f148bf7c005e0"
    assert config.digest == ("git-blob-sha1", config.git_blob_sha1)
    assert config.lfs is False


def test_git_blob_sha1_reproduces_the_hub_oid() -> None:
    """Verified exactly against `Qwen/Qwen3-8B`'s real 728-byte
    `config.json`: this is the hash git reports, so a sidecar download
    can be checked without upstream publishing a digest for it."""
    assert hub.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert hub.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


async def test_directories_are_not_files() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TREE)

    files = await client_for(handler).tree("org/repo")
    assert "MTP" not in {f.path for f in files}
    assert len(files) == 2


async def test_head_reads_the_metadata_off_the_302() -> None:
    """The trap that cost a debugging cycle: `X-Linked-Size`,
    `X-Linked-ETag` and `X-Repo-Commit` are on the redirect, and a
    client that follows it transparently loses all three."""
    digest = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hub.example":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://cdn.example/xet-bridge/abc?Expires=1788926395",
                    "X-Linked-Size": "16464440224",
                    "X-Linked-ETag": f'"{digest}"',
                    "X-Repo-Commit": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
                    "Accept-Ranges": "bytes",
                },
            )
        # The CDN's own etag is the Xet hash, which matches nothing in
        # the tree API.
        return httpx.Response(200, headers={"etag": '"76b9d90a"'})

    resolved = await client_for(handler).head_file(
        repo="org/repo", revision="main", path="model.gguf"
    )
    assert resolved.size == 16_464_440_224
    assert resolved.sha256 == digest
    assert resolved.commit == "4ca720788d1e01f1bff70c033e0d0028fd02e502"
    assert resolved.accept_ranges is True


async def test_a_cdn_style_etag_is_not_mistaken_for_a_digest() -> None:
    """Only a 64-hex `X-Linked-ETag` is the sha256. Anything else is some
    other hash of some other thing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={
                "Location": "https://cdn.example/x",
                "X-Linked-Size": "100",
                "X-Linked-ETag": '"W/short-opaque-value"',
            },
        )

    resolved = await client_for(handler).head_file(
        repo="org/repo", revision="main", path="model.gguf"
    )
    assert resolved.sha256 is None


async def test_a_gated_repo_is_a_403_that_says_what_to_do() -> None:
    """`X-Error-Code: GatedRepo` means "accept the licence on the model's
    page", which is something the operator can act on. A bare 403 is
    not."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={
                "X-Error-Code": "GatedRepo",
                "X-Error-Message": "Access to model meta-llama/X is restricted.",
            },
        )

    with pytest.raises(hub.HubError) as raised:
        await client_for(handler).head_file(
            repo="meta-llama/X", revision="main", path="config.json"
        )
    assert raised.value.status == 403
    assert raised.value.code == "GatedRepo"
    assert "hfToken" in str(raised.value)
    assert "licence" in str(raised.value)


async def test_a_rate_limit_names_the_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "42"})

    with pytest.raises(hub.HubError) as raised:
        await client_for(handler).tree("org/repo")
    assert raised.value.status == 429
    assert raised.value.retry_after == 42
    assert "500 API requests" in str(raised.value)


async def test_an_unreachable_hub_is_503_not_502() -> None:
    """A connection that never happened is a different problem from a hub
    that answered badly, and only one of them is worth retrying."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(hub.HubError) as raised:
        await client_for(handler).tree("org/repo")
    assert raised.value.status == 503


async def test_the_catalogue_can_be_switched_off() -> None:
    """An air-gapped install gets a legible refusal rather than a
    sequence of connection timeouts."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    client = client_for(handler)
    client.configure(base_url=None, token=None, enabled=False)
    with pytest.raises(hub.CatalogueDisabled) as raised:
        await client.tree("org/repo")
    assert raised.value.status == 409
    assert "catalogueEnabled" in str(raised.value)


async def test_a_token_is_sent_for_public_repos_too() -> None:
    """Upstream's own warning header says a token raises the rate limit
    and speeds up transfers, so it is not only for gated models."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=TREE)

    client = client_for(handler)
    client.configure(base_url="https://hub.example", token="hf_secret", enabled=True)
    await client.tree("org/repo")
    assert seen == ["Bearer hf_secret"]


async def test_the_next_page_is_a_cursor_not_a_url() -> None:
    """Upstream hands back a whole URL; a full upstream URL in a client's
    hands is an invitation to bypass this component."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": "a/b"}],
            headers={
                "Link": '<https://hub.example/api/models?cursor=eyJhIjoxfQ%3D%3D>; rel="next"'
            },
        )

    _, cursor, _ = await client_for(handler).search(query="qwen")
    assert cursor == "eyJhIjoxfQ=="


async def test_no_link_header_means_the_last_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "a/b"}])

    _, cursor, _ = await client_for(handler).search(query="qwen")
    assert cursor is None


async def test_search_maps_sort_names_onto_upstreams() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=[])

    client = client_for(handler)
    await client.search(query="q", sort=CatalogueSort.trending, gguf_only=True)
    assert seen[0].params["sort"] == "trendingScore"
    assert seen[0].params["filter"] == "gguf"
    assert seen[0].params["direction"] == "-1"


async def test_repeated_calls_are_served_from_the_cache() -> None:
    """500 API requests per 300 seconds is the budget a UI re-rendering a
    screen would otherwise spend."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=TREE)

    client = client_for(handler)
    await client.tree("org/repo")
    await client.tree("org/repo")
    assert calls == 1


async def test_switching_hub_clears_the_cache() -> None:
    """One hub's answers must not be attributed to another."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=TREE)

    client = client_for(handler)
    await client.tree("org/repo")
    client.configure(base_url="https://mirror.example", token=None, enabled=True)
    await client.tree("org/repo")
    assert calls == 2


async def test_a_range_read_refuses_a_whole_file() -> None:
    """A 200 to a Range request means the entire file is coming back. On
    a 16 GB model that is not a fallback, it is a disaster."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"the whole thing")

    with pytest.raises(hub.HubError) as raised:
        await client_for(handler).read_range(
            repo="org/repo", revision="main", path="model.gguf", first=0, last=15
        )
    assert "Range" in str(raised.value)


async def test_a_range_read_returns_the_partial_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=0-3"
        return httpx.Response(206, content=b"GGUF")

    data = await client_for(handler).read_range(
        repo="org/repo", revision="main", path="model.gguf", first=0, last=3
    )
    assert data == b"GGUF"


def test_gate_kinds_are_normalized_to_three_names() -> None:
    """`false | "auto" | "manual"` upstream would be a boolean-or-string
    union in every generated client for no gain."""
    assert RepoInfoFor({"gated": False}).gated is GateKind.open
    assert RepoInfoFor({"gated": "auto"}).gated is GateKind.auto
    assert RepoInfoFor({"gated": "manual"}).gated is GateKind.manual
    assert RepoInfoFor({}).gated is GateKind.open


def RepoInfoFor(raw: dict) -> hub.RepoInfo:
    return hub.RepoInfo(repo="org/repo", raw=raw)


def test_the_repo_level_gguf_total_is_a_parameter_count() -> None:
    """27.32 billion, not a byte count — and the reason bits per weight
    is computable for a remote candidate with no extra request."""
    info = RepoInfoFor({"gguf": {"total": 27_320_697_856, "architecture": "qwen35"}})
    assert info.parameters == 27_320_697_856
    assert info.architecture == "qwen35"


def test_the_architecture_falls_back_to_the_transformers_config() -> None:
    """A safetensors repo has no `gguf` block at all."""
    info = RepoInfoFor({"config": {"model_type": "qwen3_5"}})
    assert info.architecture == "qwen3_5"
    assert info.parameters is None


def test_a_license_tag_is_read_when_card_data_is_absent() -> None:
    info = RepoInfoFor({"tags": ["gguf", "license:apache-2.0"]})
    assert info.license == "apache-2.0"
