"""Model pages continue by a stable key, including across completed rescans."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library._generated.models import LibraryModel
from eugene_plexus_library.store import StateStore

from .conftest import qwen_like_kv, wait_for_scan, write_gguf

STAMP = datetime(2026, 10, 22, tzinfo=UTC)


def model(model_id: str, name: str) -> LibraryModel:
    return LibraryModel(
        id=model_id, name=name, path=f"/models/{model_id}.gguf", format="gguf", status="present"
    )


@pytest.fixture
def populated(client: TestClient, app: FastAPI) -> TestClient:
    store: StateStore = app.state.state_store
    store.replace_models(
        [model("z", "Zulu"), model("b", "same"), model("a", "Same"), model("c", "Alpha")],
        scanned_at=STAMP,
    )
    return client


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 500])
def test_pages_match_the_full_list_and_omit_the_last_cursor(
    populated: TestClient, limit: int
) -> None:
    full = populated.get("/v1/models").json()
    assert set(full) == {"models", "lastScanAt"}
    assert [m["id"] for m in full["models"]] == ["c", "a", "b", "z"]
    assert full["lastScanAt"] == "2026-10-22T00:00:00Z"
    collected = []
    params = {"limit": str(limit)}
    for _ in range(5):
        response = populated.get("/v1/models", params=params)
        assert response.status_code == 200
        page = response.json()
        assert len(page["models"]) <= limit
        assert page["lastScanAt"] == full["lastScanAt"]
        collected.extend(page["models"])
        if "nextCursor" not in page:
            break
        assert page["models"]
        params["cursor"] = page["nextCursor"]
    else:
        pytest.fail("The model pages did not end.")
    assert collected == full["models"]


def test_cursor_without_limit_returns_every_remaining_model(populated: TestClient) -> None:
    first = populated.get("/v1/models", params={"limit": 2}).json()
    remaining = populated.get("/v1/models", params={"cursor": first["nextCursor"]}).json()
    assert [m["id"] for m in first["models"] + remaining["models"]] == ["c", "a", "b", "z"]
    assert "nextCursor" not in remaining


def test_a_cursor_not_issued_by_this_library_is_unknown(populated: TestClient) -> None:
    first = populated.get("/v1/models", params={"limit": 1}).json()
    token = first["nextCursor"]
    unsigned = token.rsplit(".", 1)[0]
    response = populated.get("/v1/models", params={"cursor": unsigned + "." + "0" * 64})
    assert response.status_code == 400
    assert "unknown" in response.json()["detail"]["detail"].lower()


def test_rescan_reorders_ties_and_adds_and_removes_entries_without_shifting_the_cursor(
    populated: TestClient, app: FastAPI
) -> None:
    first = populated.get("/v1/models", params={"limit": 2}).json()
    assert [m["id"] for m in first["models"]] == ["c", "a"]
    store: StateStore = app.state.state_store
    # Remove the cursor's own model and the earlier Alpha; insert on both
    # sides of the cursor, and reverse the scan's order for equal names.
    store.replace_models(
        [model("d", "SAME"), model("b", "same"), model("0", "Aardvark"), model("z", "Zulu")],
        scanned_at=STAMP,
    )
    second = populated.get("/v1/models", params={"limit": 2, "cursor": first["nextCursor"]}).json()
    assert [m["id"] for m in second["models"]] == ["b", "d"]
    final = populated.get("/v1/models", params={"cursor": second["nextCursor"]}).json()
    assert [m["id"] for m in final["models"]] == ["z"]
    assert "nextCursor" not in final


def test_cursor_survives_an_actual_disk_rescan(
    configured_client: TestClient, models_dir: Path
) -> None:
    for name in ("bravo", "delta", "foxtrot"):
        write_gguf(models_dir / f"{name}.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    wait_for_scan(configured_client)
    first = configured_client.get("/v1/models", params={"limit": 1}).json()
    assert [m["name"] for m in first["models"]] == ["bravo"]
    (models_dir / "bravo.gguf").unlink()
    (models_dir / "delta.gguf").unlink()
    for name in ("alpha", "echo"):
        write_gguf(models_dir / f"{name}.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    wait_for_scan(configured_client)
    rest = configured_client.get("/v1/models", params={"cursor": first["nextCursor"]}).json()
    assert [m["name"] for m in rest["models"]] == ["echo", "foxtrot"]


@pytest.mark.parametrize(
    "cursor", ["", "not-a-cursor", "v99.e30.c2ln", "v1.@@@.c2ln", "v1.e30.c2ln"]
)
def test_bad_cursors_return_a_problem_with_the_observed_reason(
    client: TestClient, cursor: str
) -> None:
    response = client.get("/v1/models", params={"cursor": cursor, "limit": 2})
    assert response.status_code == 400
    problem = response.json()["detail"]
    assert problem["component"] == "library"
    assert problem["status"] == 400
    assert "cursor" in problem["detail"].lower()
    assert any(word in problem["detail"].lower() for word in ("malformed", "version", "unknown"))
    assert problem["detail"].endswith(".")


@pytest.mark.parametrize("limit", ["", "0", "-1", "501", "1.5", "abc"])
def test_bad_limits_return_400_with_a_reason(client: TestClient, limit: str) -> None:
    response = client.get("/v1/models", params={"limit": limit})
    assert response.status_code == 400
    assert "integer between 1 and 500" in response.json()["detail"]["detail"]


def test_an_oversized_integer_limit_is_still_a_400(client: TestClient) -> None:
    response = client.get("/v1/models", params={"limit": "9" * 5000})
    assert response.status_code == 400
    assert "integer between 1 and 500" in response.json()["detail"]["detail"]


@pytest.mark.parametrize("path", ["/models/a.gguf", "/models/absent.gguf"])
@pytest.mark.parametrize("limit", ["0", "501", "abc"])
def test_path_lookup_ignores_even_invalid_paging(
    populated: TestClient, path: str, limit: str
) -> None:
    expected = populated.get("/v1/models", params={"path": path}).json()
    response = populated.get(
        "/v1/models", params={"path": path, "limit": limit, "cursor": "broken"}
    )
    assert response.status_code == 200
    assert response.json() == expected
    assert [m["id"] for m in expected["models"]] == (["a"] if path.endswith("/a.gguf") else [])
    assert "nextCursor" not in expected


def test_empty_library_has_no_continuation(client: TestClient) -> None:
    assert client.get("/v1/models", params={"limit": 200}).json() == {
        "models": [],
        "lastScanAt": None,
    }


def test_unpaginated_list_has_no_implicit_limit(client: TestClient, app: FastAPI) -> None:
    store: StateStore = app.state.state_store
    store.replace_models([model(str(i), f"Model {i:04}") for i in range(501)], scanned_at=STAMP)
    body = client.get("/v1/models").json()
    assert len(body["models"]) == 501
    assert "nextCursor" not in body
