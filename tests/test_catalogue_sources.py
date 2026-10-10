"""Sources are a list (library-sources-and-engines.md §4.4, LS4).

- An old config file's hub and token become the first source, and the file
  keeps both old keys, so a library older than LS4 still finds them.
- `POST /v1/catalogue/search` searches every enabled source together: each
  hub with its own address and token, each engine's list from what the
  caller sent; every result names its source; one hub down is not an
  empty search.
- The calls about one repo, and a download, ask the hub named by `source`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_library import catalogue_sources as sources_mod
from eugene_plexus_library import hardware, security
from eugene_plexus_library._generated.models import ConfigUpdateRequest
from eugene_plexus_library.config import ConfigStore

from .conftest import mock_hubs
from .test_catalogue_routes import HOST

KEY = b"k" * 32
MIRROR = "https://mirror.example"
PUBLIC = "https://huggingface.co"

ROW = {"id": "org/Small-GGUF", "author": "org", "downloads": 9, "tags": ["gguf"]}
MIRROR_ROW = {"id": "corp/Inside-GGUF", "author": "corp", "downloads": 3, "tags": ["gguf"]}

IQ2_XS = {
    "id": "IQ2_XS",
    "title": "Qwen3.8-Flash-Next IQ2_XS",
    "about": "2-bit i-quant, a little better quality, close in speed",
    "publisher": "Qwen; GSQ-RCO quants by ISTA-DASLab",
    "format": "gguf",
    "architecture": "qwen4exp",
    "quantization": "IQ2_XS",
    "source": {
        "repoId": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "file": "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    },
    "sizeBytes": 68_026_093_024,
    "preparation": {"recipe": "strata-prepare"},
    "recommended": True,
}
CODER = {
    **IQ2_XS,
    "id": "coder-IQ1_M",
    "title": "Qwen3.8-Flash-Next Coder IQ1_M",
    "about": "half the experts (code, tools, images kept)",
    "quantization": "IQ1_M",
    "recommended": False,
}
LISTS = [{"engine": "strata", "models": [IQ2_XS, CODER]}]


def _store(tmp_path: Path, content: dict[str, Any] | None = None) -> ConfigStore:
    path = tmp_path / "library.yaml"
    if content is not None:
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
    store = ConfigStore(path, master_key=KEY)
    store.load()
    return store


def _on_disk(store: ConfigStore) -> dict[str, Any]:
    return yaml.safe_load(store._path.read_text(encoding="utf-8"))


# -- config ---------------------------------------------------------------------


def test_an_old_files_hub_and_sealed_token_become_the_first_source(tmp_path: Path) -> None:
    sealed = security.seal("hf_old", KEY).to_dict()
    store = _store(tmp_path, {"catalogueBaseUrl": MIRROR + "/", "hfToken": sealed})
    engines, hub = store.catalogue_sources()
    assert (hub.id, hub.kind.value, hub.address, hub.token) == (
        "huggingface",
        "hf_hub",
        MIRROR,
        "hf_old",
    )
    assert hub.label == "Hub at mirror.example", "a mirror is not called Hugging Face"
    assert (engines.kind.value, engines.engine) == ("engine_list", None)
    shown = store.as_document().model_dump()["catalogueSources"][1]
    assert shown["token"] is None and shown["hasToken"] is True
    assert "hfToken" not in store.as_document().model_dump()


def test_the_file_keeps_the_old_keys_for_an_older_library(tmp_path: Path) -> None:
    store = _store(tmp_path, {"catalogueBaseUrl": MIRROR, "hfToken": "hf_plain"})
    store.apply_patch(ConfigUpdateRequest.model_validate({"guidanceContextLength": 4096}))
    disk = _on_disk(store)
    assert disk["catalogueBaseUrl"] == MIRROR
    assert security.is_envelope(disk["hfToken"]), "the token is sealed where it is mirrored"
    assert security.open_envelope(security.Envelope.from_dict(disk["hfToken"]), KEY) == "hf_plain"
    entry = disk["catalogueSources"][1]
    assert security.is_envelope(entry["token"]) and "hasToken" not in entry
    # And it reads back the same, from the list now rather than the old keys.
    again = _store(tmp_path)
    assert again.catalogue_sources()[1].token == "hf_plain"


def test_a_new_install_has_the_public_hub_and_every_engines_list(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # The engines' lists first (LS7): a search answers in the list's order.
    assert [(s.id, s.kind.value) for s in store.catalogue_sources()] == [
        ("engines", "engine_list"),
        ("huggingface", "hf_hub"),
    ]
    field = next(f for f in store.schema().fields if f.key == "catalogueSources")
    assert field.valueType.value == "catalogue_sources"
    assert {f.key for f in store.schema().fields}.isdisjoint({"hfToken", "catalogueBaseUrl"})


@pytest.mark.parametrize(
    ("value", "said"),
    [
        ("nope", "expected a list"),
        ([{"id": "Bad Id", "kind": "hf_hub"}], "`id` must be"),
        ([{"id": "a", "kind": "hf_hub"}, {"id": "a", "kind": "engine_list"}], "used twice"),
        ([{"id": "a", "kind": "ftp"}], "`kind` must be"),
        ([{"id": "a", "kind": "hf_hub", "address": "file:///x"}], "http:// or https://"),
        ([{"id": "a", "kind": "hf_hub", "engine": "strata"}], "a hub has no `engine`"),
        ([{"id": "a", "kind": "engine_list", "token": "t"}], "no address or token"),
        ([{"id": "a", "kind": "engine_list", "engine": "nope"}], "`engine` must be one of"),
        ([{"id": "a", "kind": "hf_hub", "colour": "red"}], "unknown field(s) colour"),
    ],
)
def test_a_bad_list_is_refused_naming_the_entry(tmp_path: Path, value: Any, said: str) -> None:
    store = _store(tmp_path)
    result = store.apply_patch(ConfigUpdateRequest.model_validate({"catalogueSources": value}))
    assert result.applied == [] and said in result.rejected[0].message


def test_the_old_keys_need_a_hub_to_set(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.apply_patch(
        ConfigUpdateRequest.model_validate(
            {"catalogueSources": [{"id": "engines", "kind": "engine_list"}]}
        )
    )
    result = store.apply_patch(ConfigUpdateRequest.model_validate({"hfToken": "t"}))
    assert result.rejected and "no hub there" in result.rejected[0].message


def test_which_hub_a_call_means(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.apply_patch(
        ConfigUpdateRequest.model_validate(
            {
                "catalogueSources": [
                    {"id": "off", "kind": "hf_hub", "address": MIRROR, "enabled": False},
                    {"id": "huggingface", "kind": "hf_hub"},
                    {"id": "engines", "kind": "engine_list"},
                ]
            }
        )
    )
    listed = store.catalogue_sources()
    assert sources_mod.hub_for(listed, None).id == "huggingface", "the first ENABLED hub"
    for ident, status in (("gone", 404), ("off", 409), ("engines", 400)):
        with pytest.raises(sources_mod.SourceProblem) as raised:
            sources_mod.hub_for(listed, ident)
        assert raised.value.status == status
    assert sources_mod.hub_for_host(listed, "mirror.example").id == "huggingface", (
        "a link to a switched-off hub is looked up on the default"
    )


# -- the routes -----------------------------------------------------------------


class Hubs:
    """Two hubs behind one mock transport, told apart by host."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, str | None]] = []
        self.down: set[str] = set()
        self.searches: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.seen.append((host, request.url.path, request.headers.get("Authorization")))
        if host in self.down:
            raise httpx.ConnectError("no route to host", request=request)
        rows = [MIRROR_ROW] if host == "mirror.example" else [ROW]
        path = request.url.path
        if "/resolve/" in path:
            # Never actually transfer: these tests are about which hub is asked.
            return httpx.Response(500)
        if path == "/api/models":
            self.searches.append(dict(request.url.params))
            if (
                request.url.params.get("sort") == "downloads"
                and request.url.params.get("direction") != "-1"
            ):
                # As Hugging Face answers since 2026-10 (seen on the live install).
                return httpx.Response(
                    400,
                    json={
                        "error": "Invalid sort direction, only descending sort is supported "
                        "for downloads"
                    },
                )
            headers = {}
            if "cursor" not in request.url.params:
                headers["Link"] = f'<https://{host}/api/models?cursor=next-{host}>; rel="next"'
            return httpx.Response(200, json=rows, headers=headers)
        for row in rows:
            if path == f"/api/models/{row['id']}":
                return httpx.Response(200, json={**row, "sha": "c0ffee"})
            if path.startswith(f"/api/models/{row['id']}/tree/"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "type": "file",
                            "path": "m-Q4_K_M.gguf",
                            "size": 4_000,
                            "oid": "0" * 40,
                            "lfs": {"oid": "a" * 64, "size": 4_000},
                        }
                    ],
                )
        return httpx.Response(404, headers={"X-Error-Code": "RepoNotFound"})

    def hosts(self) -> list[str]:
        return [host for host, _, _ in self.seen]


@pytest.fixture
def two_hubs(configured_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Hubs:
    saved = configured_client.patch(
        "/v1/config",
        json={
            "catalogueSources": [
                {"id": "huggingface", "kind": "hf_hub", "label": "Hugging Face"},
                {
                    "id": "corp",
                    "kind": "hf_hub",
                    "label": "Corp hub",
                    "address": MIRROR,
                    "token": "corp_token",
                },
                {"id": "engines", "kind": "engine_list", "label": "Engines' own lists"},
            ]
        },
    )
    assert saved.json()["applied"] == ["catalogueSources"], saved.text
    hubs = Hubs()
    inner = httpx.AsyncClient(transport=httpx.MockTransport(hubs), follow_redirects=False)
    mock_hubs(configured_client.app, inner)
    monkeypatch.setattr(hardware, "detect", lambda: HOST)
    return hubs


def _search(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/v1/catalogue/search", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_every_source_is_searched_together_and_each_result_says_where_from(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    page = _search(configured_client, engines=LISTS)
    got = [(r["source"], r["repo"], r.get("hubSource")) for r in page["results"]]
    # In the list's order whatever a source's kind (LS7, Troy: the order is
    # the person's): this list has the engines' lists last.
    assert got == [
        ("huggingface", "org/Small-GGUF", "huggingface"),
        ("corp", "corp/Inside-GGUF", "corp"),
        ("engines", IQ2_XS["source"]["repoId"], "huggingface"),
        ("engines", CODER["source"]["repoId"], "huggingface"),
    ]
    listed = page["results"][2]
    assert listed["engine"] == "strata" and listed["name"] == "Qwen3.8-Flash-Next IQ2_XS"
    assert listed["supported"]["source"]["file"].endswith("IQ2_XS-00001-of-00002.gguf")
    assert listed["facts"] == [
        {
            "id": "list:engines:strata:IQ2_XS",
            "format": "gguf",
            "architecture": "qwen4exp",
            "quantization": "IQ2_XS",
            "file": "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
            # The list's own size, for an engine's fit (LS6).
            "sizeBytes": 68026093024,
            "mlxQuantized": None,
            "approximate": False,
        }
    ]
    # One repo on two hubs is two rows the judge tells apart.
    assert page["results"][0]["facts"][0]["id"] == "search:huggingface:org/Small-GGUF:gguf"
    assert [(s["id"], s["searched"], s.get("results")) for s in page["sources"]] == [
        ("huggingface", True, 1),
        ("corp", True, 1),
        ("engines", True, 2),
    ]


def test_a_search_asks_for_the_most_downloaded_first(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    """The live install, 2026-10-09: the request's default direction is the
    string "desc", which the route compared with the enum member by identity,
    so every search asked for ascending order, and Hugging Face refused it."""
    page = _search(configured_client, sort="downloads")
    hubs = [s for s in page["sources"] if s["kind"] == "hf_hub"]
    assert [s.get("problem") for s in hubs] == [None, None]
    assert {s.get("direction") for s in two_hubs.searches} == {"-1"}
    two_hubs.searches.clear()
    # Its own words, so the 60 s search cache does not answer it.
    _search(configured_client, sort="downloads", direction="desc", q="small")
    assert {s.get("direction") for s in two_hubs.searches} == {"-1"}


def test_each_hub_gets_its_own_token_and_no_other(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    _search(configured_client, engines=LISTS)
    by_host = {host: auth for host, _, auth in two_hubs.seen}
    assert by_host == {"huggingface.co": None, "mirror.example": "Bearer corp_token"}


def test_one_hub_down_is_named_and_the_others_still_answer(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    two_hubs.down.add("mirror.example")
    page = _search(configured_client, engines=LISTS)
    assert {r["source"] for r in page["results"]} == {"engines", "huggingface"}
    corp = next(s for s in page["sources"] if s["id"] == "corp")
    assert corp["results"] == 0 and "Could not reach https://mirror.example" in corp["problem"]


def test_a_source_switched_off_is_not_asked_and_says_so(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    sources = configured_client.get("/v1/config").json()["catalogueSources"]
    sources[1]["enabled"] = False
    configured_client.patch("/v1/config", json={"catalogueSources": sources})
    page = _search(configured_client, engines=LISTS)
    assert "mirror.example" not in two_hubs.hosts()
    corp = next(s for s in page["sources"] if s["id"] == "corp")
    assert corp["searched"] is False and "switched off" in corp["problem"]
    # And the token stayed through that round trip, which never saw it.
    assert configured_client.get("/v1/config").json()["catalogueSources"][1]["hasToken"] is True


def test_the_engines_lists_filter_like_a_hub_and_need_to_be_sent(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    coder = _search(configured_client, q="coder strata", sources=["engines"], engines=LISTS)
    assert [r["supported"]["id"] for r in coder["results"]] == ["coder-IQ1_M"]
    assert two_hubs.seen == [], "naming one source asks no other"
    none = _search(configured_client, sources=["engines"], format="safetensors", engines=LISTS)
    assert none["results"] == []
    unsent = _search(configured_client, sources=["engines"])
    assert "no engine's list came with the search" in unsent["sources"][2]["problem"]
    # An older agent's node sends lists, none of them its engines' own.
    older = _search(configured_client, sources=["engines"], engines=[])
    assert older["sources"][2]["problem"].startswith(
        "none of the node's engines publishes a list of its own"
    )


def test_an_engine_list_source_can_name_one_engine(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    sources = configured_client.get("/v1/config").json()["catalogueSources"]
    sources[2]["engine"] = "llama_cpp"
    configured_client.patch("/v1/config", json={"catalogueSources": sources})
    page = _search(configured_client, sources=["engines"], engines=LISTS)
    assert page["results"] == [], "Strata's list is not llama.cpp's"
    assert page["sources"][2]["problem"].startswith("llama_cpp on the node publishes")
    # Two engines' lists sent: a source naming one lists that one's alone.
    other = {**CODER, "id": "llama-pick", "title": "A llama.cpp pick"}
    both = [*LISTS, {"engine": "llama_cpp", "models": [other]}]
    sources[2]["engine"] = "strata"
    configured_client.patch("/v1/config", json={"catalogueSources": sources})
    named = _search(configured_client, sources=["engines"], engines=both)["results"]
    assert [r["supported"]["id"] for r in named] == ["IQ2_XS", "coder-IQ1_M"]
    sources[2].pop("engine")
    configured_client.patch("/v1/config", json={"catalogueSources": sources})
    every = _search(configured_client, sources=["engines"], engines=both)["results"]
    assert [r["engine"] for r in every] == ["strata", "strata", "llama_cpp"]


def test_a_later_page_asks_only_the_hubs_that_had_more(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    first = _search(configured_client, engines=LISTS)
    assert first["nextCursor"]
    two_hubs.seen.clear()
    second = _search(configured_client, cursor=first["nextCursor"], engines=LISTS)
    assert {r["source"] for r in second["results"]} == {"huggingface", "corp"}, (
        "the engines' lists are whole on the first page"
    )
    assert sorted(two_hubs.hosts()) == ["huggingface.co", "mirror.example"]
    assert second["nextCursor"] is None
    bad = configured_client.post("/v1/catalogue/search", json={"cursor": "upstream-raw"})
    assert bad.status_code == 400 and bad.json()["detail"]["title"].startswith("Not a cursor")


def test_an_unknown_source_is_refused(configured_client: TestClient, two_hubs: Hubs) -> None:
    response = configured_client.post("/v1/catalogue/search", json={"sources": ["gone"]})
    assert response.status_code == 400 and "'gone'" in response.json()["detail"]["detail"]


def test_a_pasted_link_is_looked_up_on_the_hub_at_its_host(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    page = _search(configured_client, q=f"{MIRROR}/corp/Inside-GGUF/tree/main")
    assert page["interpretedAs"] == "repo" and page["interpretedFrom"] == "corp/Inside-GGUF"
    assert page["results"][0]["source"] == "corp" and page["results"][0]["hubSource"] == "corp"
    assert two_hubs.hosts() == ["mirror.example"]


def test_the_air_gap_switch_stops_every_hub_and_leaves_the_lists(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    configured_client.patch("/v1/config", json={"catalogueEnabled": False})
    page = _search(configured_client, engines=LISTS)
    assert two_hubs.seen == []
    assert [r["source"] for r in page["results"]] == ["engines", "engines"]
    assert all("switched off" in s["problem"] for s in page["sources"] if s["kind"] == "hf_hub")


def test_the_detail_asks_the_hub_it_names(configured_client: TestClient, two_hubs: Hubs) -> None:
    answer = configured_client.get(
        "/v1/catalogue/model", params={"repo": "corp/Inside-GGUF", "source": "corp"}
    )
    assert answer.status_code == 200, answer.text
    assert set(two_hubs.hosts()) == {"mirror.example"}
    off = configured_client.get(
        "/v1/catalogue/model", params={"repo": "corp/Inside-GGUF", "source": "engines"}
    )
    assert off.status_code == 400 and off.json()["detail"]["title"] == "Not a hub"


def test_a_download_records_its_hub_and_fetches_from_it(
    configured_client: TestClient, two_hubs: Hubs
) -> None:
    started = configured_client.post(
        "/v1/downloads",
        json={"repo": "corp/Inside-GGUF", "source": "corp", "files": ["m-Q4_K_M.gguf"]},
    )
    assert started.status_code == 202, started.text
    assert started.json()["source"] == "corp"
    # The transfer itself asks that hub too (it answers 500 here), not the default.
    ident = started.json()["id"]
    for _ in range(200):
        if configured_client.get(f"/v1/downloads/{ident}").json()["state"] == "failed":
            break
        time.sleep(0.02)
    assert any("/resolve/" in path for _, path, _ in two_hubs.seen)
    assert set(two_hubs.hosts()) == {"mirror.example"}
    plain = configured_client.post(
        "/v1/downloads", json={"repo": "org/Small-GGUF", "files": ["m-Q4_K_M.gguf"]}
    )
    assert plain.status_code == 202, plain.text
    assert plain.json()["source"] == "huggingface", "no source: the default hub, recorded"


def test_a_cursor_round_trips() -> None:
    from eugene_plexus_library.routes import catalogue as routes

    hubs = {"huggingface": "abc", "corp": "x/y+z=="}
    assert routes._decode_cursor(routes._encode_cursor(hubs)) == hubs
