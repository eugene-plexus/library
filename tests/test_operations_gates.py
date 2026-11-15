"""Row 39: operations rights for people who are not the owner, on the library.

`specs/docs/design/rows/row-39.md`. An operations token addressed to this
machine opens exactly the routes the table in `operations.py` names, with
the power it holds; every other route keeps the owner's gate. Each change
is reported to this machine's agent, the person read from the token.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from eugene_plexus_library import _http, operations, tokens
from eugene_plexus_library.app import create_app
from eugene_plexus_library.settings import Settings

from .conftest import FakeInstall

MODELS = tokens.POWER_OPERATE_MODELS
MACHINES = tokens.POWER_OPERATE_MACHINES

#: The design's table (row-39.md §2), written out again so that a route
#: added to `operations.py` by mistake fails here.
EXPECTED_READS = {
    ("GET", "/v1/folders"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{model_id}"),
    ("GET", "/v1/models/{model_id}/profiles"),
    ("GET", "/v1/models/{model_id}/profiles/{profile_id}"),
    ("GET", "/v1/downloads"),
    ("GET", "/v1/downloads/{download_id}"),
    ("GET", "/v1/scan"),
}
EXPECTED_ROUTES = {
    ("POST", "/v1/downloads"): MODELS,
    ("DELETE", "/v1/downloads/{download_id}"): MODELS,
    ("POST", "/v1/downloads/{download_id}/pause"): MODELS,
    ("POST", "/v1/downloads/{download_id}/resume"): MODELS,
    # R39-DELETE: Models includes deleting model files.
    ("GET", "/v1/models/{model_id}/deletion"): MODELS,
    ("POST", "/v1/models/{model_id}/delete"): MODELS,
    ("DELETE", "/v1/models/{model_id}"): MODELS,
    ("POST", "/v1/models/{model_id}/profiles"): MODELS,
    ("PUT", "/v1/models/{model_id}/profiles/{profile_id}"): MODELS,
    ("DELETE", "/v1/models/{model_id}/profiles/{profile_id}"): MODELS,
    ("POST", "/v1/scan"): MODELS,
    ("DELETE", "/v1/scan"): MODELS,
    # Finding a model to download.
    ("POST", "/v1/catalogue/search"): MODELS,
    ("GET", "/v1/catalogue/model"): MODELS,
    ("GET", "/v1/catalogue/starter"): MODELS,
    ("GET", "/v1/catalogue/model/card"): MODELS,
    ("GET", "/v1/catalogue/model/preflight"): MODELS,
}


def _operations(install: FakeInstall, *powers: str, machine: str | None = None) -> str:
    token, _ = install.root.mint(
        typ=tokens.TYP_OPERATIONS,
        sub="p-dana",
        aud=[tokens.node_recipient(machine or install.name)],
        ttl_seconds=tokens.OPERATIONS_TTL_SECONDS,
        extra={
            "ops": list(powers),
            "name": "Dana",
            "sid": "admin-1",
            "administration_version": 2,
        },
    )
    return token


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reports() -> Iterator[list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/node/operations-log"
        seen.append({"auth": request.headers.get("authorization"), **json.loads(request.content)})
        return httpx.Response(204)

    _http.set_shared_client(
        operations.CLIENT_KEY, httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    yield seen
    _http.set_shared_client(operations.CLIENT_KEY, None)


@pytest.fixture
def app(tmp_path: Path, install: FakeInstall) -> FastAPI:
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
        agent_url="http://127.0.0.1:9/",
    )
    made: FastAPI = create_app(settings=settings)
    made.state.auth_state = install.auth_state()
    return made


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _every_route(app: FastAPI) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()

    def walk(routes: list[Any], prefix: str) -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                walk(included.routes, prefix + (route.include_context.prefix or ""))
                continue
            if isinstance(route, APIRoute):
                for method in route.methods or ():
                    found.add((method, prefix + route.path))

    walk(app.router.routes, "")
    return sorted(found)


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x-none", path)


def test_the_table_is_the_designs() -> None:
    assert operations.OPERATIONS_READS == EXPECTED_READS
    assert {k: v[0] for k, v in operations.OPERATIONS_ROUTES.items()} == EXPECTED_ROUTES


def test_every_mutating_route_is_in_the_table(
    app: FastAPI, client: TestClient, install: FakeInstall, reports: list[dict[str, Any]]
) -> None:
    everything = _bearer(_operations(install, MODELS, MACHINES))
    listed = set(EXPECTED_ROUTES) | EXPECTED_READS
    tried: list[tuple[str, str]] = []
    opened: list[tuple[str, str, int]] = []
    for method, path in _every_route(app):
        if (method, path) in listed or method == "HEAD":
            continue
        body = {} if method in ("POST", "PUT", "PATCH") else None
        if client.request(method, _concrete(path), json=body).status_code not in (401, 403):
            continue
        tried.append((method, path))
        answer = client.request(method, _concrete(path), headers=everything, json=body)
        if answer.status_code not in (401, 403):
            opened.append((method, path, answer.status_code))
    assert ("PATCH", "/v1/config") in tried
    assert len(tried) > 15, tried
    assert opened == []
    assert reports == []


def test_config_patch_stays_owner_only(client: TestClient, install: FakeInstall) -> None:
    answer = client.patch(
        "/v1/config", json={}, headers=_bearer(_operations(install, MODELS, MACHINES))
    )
    assert answer.status_code in (401, 403)


def test_download_with_operate_models(
    client: TestClient, install: FakeInstall, reports: list[dict[str, Any]]
) -> None:
    dana = _bearer(_operations(install, MODELS))
    assert client.get("/v1/models", headers=dana).status_code == 200
    assert client.get("/v1/downloads", headers=dana).status_code == 200
    # Past the gate: the body names nothing to fetch.
    started = client.post("/v1/downloads", json={}, headers=dana)
    assert started.status_code == 422, started.text
    machines = client.post(
        "/v1/downloads", json={}, headers=_bearer(_operations(install, MACHINES))
    )
    assert machines.status_code == 403
    assert "Models power" in machines.json()["detail"]["detail"]


def test_a_change_is_reported_to_the_agent_with_the_person_from_the_token(
    app: FastAPI, client: TestClient, install: FakeInstall, reports: list[dict[str, Any]]
) -> None:
    dana = _operations(install, MODELS)
    started = client.post(
        "/v1/scan",
        json={},
        headers={**_bearer(dana), "X-Eugene-Person": "Mallory"},
    )
    assert started.status_code == 202, started.text
    assert len(reports) == 1
    report = reports[0]
    assert report["auth"] == f"Bearer {app.state.auth_state.service_token}"
    assert (report["personId"], report["name"], report["owner"]) == ("p-dana", "Dana", False)
    assert (report["power"], report["action"], report["status"]) == (MODELS, "re-scan models", 202)
    # The owner's own change is recorded too.
    client.delete("/v1/scan", headers=_bearer(install.session()))
    assert reports[-1]["owner"] is True and reports[-1]["action"] == "stop scan"


def test_a_token_for_another_machine_or_a_revoked_one_is_refused(
    app: FastAPI, client: TestClient, install: FakeInstall
) -> None:
    far = client.get("/v1/models", headers=_bearer(_operations(install, MODELS, machine="far")))
    assert far.status_code == 401
    token = _operations(install, MODELS)
    assert client.get("/v1/models", headers=_bearer(token)).status_code == 200
    install.publish(revoked=(("admin-1", 4_000_000_000),))
    app.state.auth_state = install.auth_state()
    answer = client.get("/v1/models", headers=_bearer(token))
    assert answer.status_code == 401
    assert "signed out" in answer.json()["detail"]["detail"]
