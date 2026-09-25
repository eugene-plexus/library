"""`GET /v1/directories`: the picker `path_list` promised since M2 (M11).

The pure function is exercised in the agent's suite as well -- the two
components carry the same module by design (schemas shared, code
duplicated) -- so this file is about the library's mounting of it: the
route, its flags, its errors, and that it is operator-only where every
other read here takes a service token.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library.app import create_app
from eugene_plexus_library.directory_listing import ListingError, list_directory
from eugene_plexus_library.settings import Settings

from .conftest import FakeInstall


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    (root / "zeta").mkdir(parents=True)
    (root / "Alpha").mkdir()
    (root / ".hidden").mkdir()
    (root / "a.gguf").write_bytes(b"x")
    return root


# --- the function, as the library carries it -------------------------------


def test_lists_directories_first_and_files_on_request(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert [e.name for e in list_directory(str(root), host="nas").entries] == ["Alpha", "zeta"]
    everything = list_directory(str(root), include_files=True, show_hidden=True, host="nas")
    assert [e.name for e in everything.entries] == [".hidden", "Alpha", "zeta", "a.gguf"]
    assert everything.host == "nas"
    assert everything.parent == str(tmp_path)


def test_the_errors_carry_their_status(tmp_path: Path) -> None:
    with pytest.raises(ListingError) as missing:
        list_directory(str(tmp_path / "nope"))
    assert missing.value.status == 404
    with pytest.raises(ListingError) as not_dir:
        list_directory(str(_tree(tmp_path) / "a.gguf"))
    assert not_dir.value.status == 400


def test_the_starting_points_name_the_roots_and_home() -> None:
    names = [e.name for e in list_directory(None).entries]
    assert names[-1] == "Home"
    if sys.platform == "win32":
        assert any(n.endswith(":\\") for n in names)
    else:
        assert names[0] == "/"


# --- the route --------------------------------------------------------------


def test_the_route_lists_a_directory(client: TestClient, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    response = client.get("/v1/directories", params={"path": str(root)})
    assert response.status_code == 200
    body = response.json()
    assert [e["name"] for e in body["entries"]] == ["Alpha", "zeta"]
    assert body["entries"][0]["path"] == str(root / "Alpha")
    assert body["parent"] == str(tmp_path)
    assert "hidden" not in body["entries"][0]


def test_the_route_honours_the_flags(client: TestClient, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    body = client.get(
        "/v1/directories",
        params={"path": str(root), "includeFiles": "true", "showHidden": "true"},
    ).json()
    assert [e["name"] for e in body["entries"]] == [".hidden", "Alpha", "zeta", "a.gguf"]
    assert body["entries"][0]["hidden"] is True


def test_the_route_answers_a_problem_document(client: TestClient, tmp_path: Path) -> None:
    response = client.get("/v1/directories", params={"path": str(tmp_path / "nope")})
    assert response.status_code == 404
    problem = response.json()["detail"]
    assert problem["component"] == "library"
    assert problem["title"] == "No such directory"
    assert "does not exist on" in problem["detail"]


def test_no_path_is_the_starting_points(client: TestClient) -> None:
    body = client.get("/v1/directories").json()
    assert body["entries"][-1]["name"] == "Home"
    assert "path" not in body


# --- operator-only ----------------------------------------------------------


@pytest.fixture
def authed_client(tmp_path: Path, install: FakeInstall) -> Iterator[TestClient]:
    settings = Settings(config_file=tmp_path / "config.yaml", state_file=tmp_path / "state.json")
    app: FastAPI = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as authed:
        yield authed


def test_a_service_token_reads_models_but_does_not_walk_the_disk(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """Every other read here accepts a service token; this one does not.
    A component doing its job has no business listing the operator's
    directories."""
    service = {"Authorization": f"Bearer {install.service('gateway')}"}
    assert authed_client.get("/v1/models", headers=service).status_code == 200
    assert authed_client.get("/v1/directories", headers=service).status_code == 401
    assert authed_client.get("/v1/directories").status_code == 401
    operator = {"Authorization": f"Bearer {install.session()}"}
    assert authed_client.get("/v1/directories", headers=operator).status_code == 200
