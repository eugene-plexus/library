"""Row 39: operations rights for people who are not the owner, on the library.

`specs/docs/design/rows/row-39.md`. An operations token addressed to this
machine opens exactly the routes the table in `operations.py` names, with
the power it holds; every other route keeps the owner's gate. Each change
is reported to this machine's agent, the person read from the token.
"""

from __future__ import annotations

import json
import os
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
    assert answer.status_code == 403
    assert "Only the owner" in answer.json()["detail"]["detail"]
    far = _bearer(_operations(install, MODELS, machine="far"))
    assert client.patch("/v1/config", json={}, headers=far).status_code == 401


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


# ------------------------------------------------------------ review findings


@pytest.fixture
def stocked(
    tmp_path: Path, install: FakeInstall, reports: list[dict[str, Any]]
) -> Iterator[tuple[TestClient, str, Path]]:
    """A library with one folder and one model in it, and auth on."""
    from .conftest import qwen_like_kv, wait_for_scan, write_gguf

    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "a.gguf", qwen_like_kv())
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
        agent_url="http://127.0.0.1:9/",
    )
    settings.config_file.write_text(
        f"modelRoots:\n  - path: {models.as_posix()}\n    mounts: []\nscanOnStartup: true\n",
        encoding="utf-8",
    )
    made = create_app(settings=settings)
    made.state.auth_state = install.auth_state()
    owner = _bearer(install.session())
    with TestClient(made) as c:
        c.headers.update(owner)
        wait_for_scan(c)
        model_id = c.get("/v1/models").json()["models"][0]["id"]
        c.headers.pop("Authorization")
        reports.clear()
        yield c, model_id, models


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({"env": {"LD_PRELOAD": "/tmp/x.so"}}, "env"),
        ({"extraArgs": ["--api-key", "x"]}, "extraArgs"),
        ({"flags": {"chatTemplateFile": "/home/me/.eugene/node.yaml"}}, "flags.chatTemplateFile"),
        ({"flags": {"projectorPath": "/elsewhere/mmproj.gguf"}}, "flags.projectorPath"),
    ],
)
def test_models_cannot_write_a_profile_that_reaches_past_running_a_model(
    stocked: tuple[TestClient, str, Path], install: FakeInstall, extra: Any, field: str
) -> None:
    """Review B1: a profile is what a runtime is later declared from."""
    client, model_id, _ = stocked
    dana = _bearer(_operations(install, MODELS))
    body = {"name": "dana", "engine": "llama_cpp", **extra}
    refused = client.post(f"/v1/models/{model_id}/profiles", json=body, headers=dana)
    assert refused.status_code == 403, refused.text
    assert f"({field})" in refused.json()["detail"]["detail"]


def test_models_writes_a_plain_profile_and_reads_no_environment_values(
    stocked: tuple[TestClient, str, Path], install: FakeInstall
) -> None:
    client, model_id, models = stocked
    dana = _bearer(_operations(install, MODELS))
    owner = _bearer(install.session())
    inside = {"chatTemplateFile": str(models / "template.jinja"), "contextSize": 4096}
    made = client.post(
        f"/v1/models/{model_id}/profiles",
        json={"name": "dana", "engine": "llama_cpp", "flags": inside},
        headers=dana,
    )
    assert made.status_code == 201, made.text
    secret = {"name": "owner", "engine": "llama_cpp", "env": {"HF_TOKEN": "hf-secret"}}
    owned = client.post(f"/v1/models/{model_id}/profiles", json=secret, headers=owner)
    assert owned.status_code == 201
    listed = client.get(f"/v1/models/{model_id}/profiles", headers=dana).json()
    one = client.get(f"/v1/models/{model_id}/profiles/{owned.json()['id']}", headers=dana)
    assert "hf-secret" not in str(listed) and "hf-secret" not in one.text
    # Dana changes the owner's profile and keeps its environment as it is.
    kept = {**secret, "flags": {"contextSize": 8192}}
    path = f"/v1/models/{model_id}/profiles/{owned.json()['id']}"
    assert client.put(path, json=kept, headers=dana).status_code == 200
    assert client.get(path, headers=owner).json()["env"] == {"HF_TOKEN": "hf-secret"}


def test_a_search_is_not_a_change_and_a_download_names_its_repository(
    client: TestClient, install: FakeInstall, reports: list[dict[str, Any]]
) -> None:
    """Review S3: a question leaves no row; a download row says what."""
    dana = _bearer(_operations(install, MODELS))
    client.post("/v1/catalogue/search", json={}, headers=dana)
    assert reports == []
    client.post(
        "/v1/downloads",
        json={"repo": "org/model", "files": ["a.gguf"], "root": "/not-a-library-folder"},
        headers=dana,
    )
    assert [(r["action"], r["target"]) for r in reports] == [("download model", "org/model")]


@pytest.mark.parametrize("sep", ["/", "\\"], ids=["posix", "windows"])
def test_a_profile_path_that_climbs_out_of_the_library_is_refused(
    stocked: tuple[TestClient, str, Path], install: FakeInstall, sep: str
) -> None:
    """Review B1, second pass: no `..`, under either separator."""
    client, model_id, models = stocked
    climbing = sep.join([str(models), "sub", "..", "..", "outside", "secret.txt"])
    body = {"name": "dana", "engine": "llama_cpp", "flags": {"chatTemplateFile": climbing}}
    dana = _bearer(_operations(install, MODELS))
    refused = client.post(f"/v1/models/{model_id}/profiles", json=body, headers=dana)
    assert refused.status_code == 403, refused.text
    assert "(flags.chatTemplateFile)" in refused.json()["detail"]["detail"]


def test_a_profile_path_through_a_link_that_points_out_is_refused(
    stocked: tuple[TestClient, str, Path], install: FakeInstall, tmp_path: Path
) -> None:
    client, model_id, models = stocked
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "node.yaml").write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, models / "out", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.skip(f"this machine cannot make a symlink here: {exc}")
        # Windows without the symlink privilege: a junction, which needs none.
        import _winapi

        _winapi.CreateJunction(str(outside), str(models / "out"))
    dana = _bearer(_operations(install, MODELS))
    linked = {"chatTemplateFile": str(models / "out" / "node.yaml")}
    body = {"name": "dana", "engine": "llama_cpp", "flags": linked}
    refused = client.post(f"/v1/models/{model_id}/profiles", json=body, headers=dana)
    assert refused.status_code == 403, refused.text
    assert "(flags.chatTemplateFile)" in refused.json()["detail"]["detail"]
