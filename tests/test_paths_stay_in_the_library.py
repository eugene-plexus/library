"""A model's files stay inside its Library folder (library#20).

A prepared model's file list is text in its provenance file. Where that text
becomes paths (the scan, the two routes that write a provenance file), a path
that leaves the Library folder makes the model unreadable or the write is
refused, naming the path. Where files are removed (`deletion.remove`), every
path must lie in a writable Library folder, as written and after links
resolve, or nothing is moved: the second lock, for any future caller.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import deletion, prepared
from eugene_plexus_library._generated.models import (
    LibraryModel,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
)
from eugene_plexus_library.paths import is_within, model_id
from eugene_plexus_library.scanner import Scanner

from .test_run_preparation import _body, _submit, _to_preparing, prepared_client  # noqa: F401


def _link_folder(link: Path, target: Path) -> None:
    """A folder link: a symlink, or on Windows without the symlink right a
    junction, which needs no right. Skips where neither can be made."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as exc:
        if sys.platform != "win32":
            pytest.skip(f"this platform cannot make a folder link here: {exc}")
    import _winapi  # type: ignore[import-not-found]

    _winapi.CreateJunction(str(target), str(link))


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "x.bin").write_bytes(b"x" * 7)
    (folder / "e.json").write_text("{}", encoding="utf-8")
    return folder


def _provenance(models_dir: Path, **body: Any) -> Path:
    data = models_dir / "Strata-data"
    data.mkdir(exist_ok=True)
    (data / prepared.ENGINE_FILES_MARKER).write_text("engine files", encoding="utf-8")
    (data / "e.json").write_text("{}", encoding="utf-8")
    path = data / f"bad{prepared.SUFFIX}"
    path.write_text(json.dumps({"engine": "strata", "entry": "e.json", **body}), encoding="utf-8")
    return path


def _scanned(models_dir: Path) -> LibraryModel:
    found = {m.name: m for m in Scanner().scan([models_dir]).models}
    return found["bad"]


def _mentions(model: LibraryModel, text: str) -> bool:
    """Whether any file the model lists (its files, or its prepared files)
    names `text`."""
    listed = [f.path for f in model.files or []]
    if model.prepared is not None:
        listed += [f.path for f in model.prepared.files or []]
    return any(text in p or os.path.normcase(text) in os.path.normcase(p) for p in listed)


# --- where the list is read ----------------------------------------------------


@pytest.mark.parametrize(
    "listed",
    [
        "../../outside/x.bin",
        "..\\..\\outside\\x.bin" if sys.platform == "win32" else "../../outside/./x.bin",
        "ABSOLUTE",
        "\\\\server\\share\\x.bin",
        "C:x.bin",
        ".",
        "..",
    ],
)
def test_a_file_list_that_leaves_the_folder_makes_the_model_unreadable(
    models_dir: Path, outside: Path, listed: str
) -> None:
    if listed == "ABSOLUTE":
        listed = str(outside / "x.bin")
    _provenance(models_dir, files=[{"path": "e.json"}, {"path": listed}])
    model = _scanned(models_dir)
    assert model.status is ModelStatus.unreadable, model
    assert listed in (model.error or "") or str(outside) in (model.error or "")
    assert not _mentions(model, str(outside))
    assert not _mentions(model, listed)
    assert all(is_within(f.path, models_dir) for f in model.files or [])


def test_a_relative_entry_that_leaves_the_folder_makes_the_model_unreadable(
    models_dir: Path, outside: Path
) -> None:
    _provenance(models_dir, entry="../../outside/e.json")
    model = _scanned(models_dir)
    assert model.status is ModelStatus.unreadable
    assert str(outside / "e.json") in (model.error or "")
    assert "outside the Library folder" in (model.error or "")
    assert not (model.files or [])


def test_an_absolute_entry_through_a_link_makes_the_model_unreadable(
    models_dir: Path, outside: Path
) -> None:
    _link_folder(models_dir / "linked", outside)
    _provenance(models_dir, entry=str(models_dir / "linked" / "e.json"))
    model = _scanned(models_dir)
    assert model.status is ModelStatus.unreadable
    # The link is named, with where it points (library#20 review).
    assert f"is below {models_dir / 'linked'}, a link to" in (model.error or "")
    assert "add the model from there" in (model.error or "")


def test_a_file_reached_through_a_link_makes_the_model_unreadable(
    models_dir: Path, outside: Path
) -> None:
    _provenance(models_dir, files=[{"path": "link/x.bin"}])
    _link_folder(models_dir / "Strata-data" / "link", outside)
    model = _scanned(models_dir)
    assert model.status is ModelStatus.unreadable
    assert "x.bin" in (model.error or "")
    assert not _mentions(model, "x.bin")


def test_a_list_that_climbs_out_and_back_in_stays_listed(models_dir: Path) -> None:
    """LS7 writes `../` paths when a provenance file is adopted into another
    folder than its engine's: inside the Library folder, they are its files."""
    (models_dir / "pack.bin").write_bytes(b"p" * 5)
    _provenance(models_dir, files=[{"path": "e.json"}, {"path": "../pack.bin"}])
    model = _scanned(models_dir)
    assert model.status is ModelStatus.present, model.error
    assert models_dir / "pack.bin" in {Path(f.path) for f in model.files or []}


# --- where files are removed -----------------------------------------------------


def _untouched(*folders: Path) -> None:
    for folder in folders:
        assert not list(folder.rglob(f"{deletion.MOVED_ASIDE}*")), folder


def test_remove_refuses_a_path_outside_every_folder_and_moves_nothing(
    models_dir: Path, outside: Path
) -> None:
    mine = models_dir / "a.bin"
    mine.write_bytes(b"a")
    for stray in (
        str(outside / "x.bin"),
        os.path.join(str(models_dir), "..", "outside", "x.bin"),
        "x.bin",
    ):
        with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted") as refused:
            deletion.remove([str(mine), stray], [models_dir])
        assert "x.bin" in str(refused.value)
        assert mine.is_file() and (outside / "x.bin").is_file()
        _untouched(models_dir, outside)


def test_remove_refuses_a_folder_and_the_library_folder_itself(models_dir: Path) -> None:
    (models_dir / "sub").mkdir()
    (models_dir / "sub" / "kept.bin").write_bytes(b"k")
    for folder in (models_dir, models_dir / "sub"):
        with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted"):
            deletion.remove([str(folder)], [models_dir])
    assert (models_dir / "sub" / "kept.bin").is_file()
    _untouched(models_dir)


def test_remove_with_no_library_folder_moves_nothing(models_dir: Path) -> None:
    mine = models_dir / "a.bin"
    mine.write_bytes(b"a")
    with pytest.raises(deletion.DeleteRefused):
        deletion.remove([str(mine)], [])
    assert mine.is_file()


def test_remove_refuses_a_path_that_leaves_through_a_link(models_dir: Path, outside: Path) -> None:
    _link_folder(models_dir / "link", outside)
    mine = models_dir / "a.bin"
    mine.write_bytes(b"a")
    with pytest.raises(deletion.DeleteRefused, match="outside") as refused:
        deletion.remove([str(mine), str(models_dir / "link" / "x.bin")], [models_dir])
    assert "x.bin" in str(refused.value)
    assert mine.is_file() and (outside / "x.bin").is_file()
    _untouched(models_dir, outside)


def test_a_folder_that_becomes_a_link_mid_delete_puts_everything_back(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = models_dir / "a.bin", models_dir / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    real = deletion.held_by
    calls: list[str] = []

    def held_by(path: str, roots: Any) -> Path | None:
        calls.append(path)
        # Both pass the first check; the first file escapes just after its move.
        return None if len(calls) == 4 else real(path, roots)

    monkeypatch.setattr(deletion, "held_by", held_by)
    with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted"):
        deletion.remove([str(first), str(second)], [models_dir])
    assert first.is_file() and second.is_file()
    _untouched(models_dir)


def _stale(models_dir: Path, outside: Path) -> LibraryModel:
    """A model as a list built before this fix could hold it: one file
    outside the Library folder."""
    weights = models_dir / "a.gguf"
    weights.write_bytes(b"g" * 3)
    return LibraryModel(
        id=model_id(weights),
        path=str(weights),
        root=str(models_dir),
        format=ModelFormat.gguf,
        name="a",
        status=ModelStatus.present,
        files=[
            ModelFile(path=str(weights), role=ModelFileRole.weights, sizeBytes=3),
            ModelFile(path=str(outside / "x.bin"), role=ModelFileRole.other, sizeBytes=7),
        ],
    )


def test_the_plan_and_its_token_never_hold_an_outside_path(models_dir: Path, outside: Path) -> None:
    model = _stale(models_dir, outside)
    plan = deletion.plan(model, [model], roots=[models_dir])
    assert [Path(f.path) for f in plan.files] == [models_dir / "a.gguf"]
    assert not plan.kept
    assert plan.refusal and "x.bin" in plan.refusal and "outside" in plan.refusal
    inside_only = model.model_copy(update={"files": (model.files or [])[:1]})
    assert plan.token == deletion.plan(inside_only, [inside_only], roots=[models_dir]).token


def test_the_routes_refuse_a_model_listing_an_outside_file(
    configured_client: TestClient, models_dir: Path, outside: Path
) -> None:
    model = _stale(models_dir, outside)
    store = configured_client.app.state.state_store
    store.add_model(model, seen_at=datetime.now(tz=UTC))
    plan = configured_client.get(f"/v1/models/{model.id}/deletion").json()
    assert str(outside) not in json.dumps(plan["files"]) + json.dumps(plan.get("kept", []))
    assert "outside" in plan["refusal"]
    done = configured_client.post(f"/v1/models/{model.id}/delete", json={"token": plan["token"]})
    assert done.status_code == 409, done.text
    assert (outside / "x.bin").is_file() and (models_dir / "a.gguf").is_file()


def test_a_scanned_provenance_cannot_steer_a_delete_outside(
    configured_client: TestClient, models_dir: Path, outside: Path
) -> None:
    _provenance(models_dir, files=[{"path": "e.json"}, {"path": "../../outside/x.bin"}])
    configured_client.post("/v1/scan")
    for _ in range(400):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
    listed = {m["name"]: m for m in configured_client.get("/v1/models").json()["models"]}
    bad = listed["bad"]
    assert bad["status"] == "unreadable"
    plan = configured_client.get(f"/v1/models/{bad['id']}/deletion").json()
    assert "outside" not in json.dumps(plan["files"])
    assert plan["refusal"]
    configured_client.post(f"/v1/models/{bad['id']}/delete", json={"token": plan["token"]})
    assert (outside / "x.bin").is_file()


# --- where a provenance file is written --------------------------------------------


@pytest.mark.parametrize("listed", ["../../outside/x.bin", "ABSOLUTE", "link/x.bin"])
def test_the_agents_prepared_route_refuses_a_list_that_leaves(
    prepared_client: TestClient,  # noqa: F811
    models_dir: Path,
    outside: Path,
    listed: str,
) -> None:
    if listed == "ABSOLUTE":
        listed = str(outside / "x.bin")
    if listed.startswith("link/"):
        _link_folder(models_dir / "Strata-data" / "link", outside)
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    files = [{"path": "strata-iq2_xs.json"}, {"path": listed}]
    response = prepared_client.post(
        "/v1/run-operations/p/prepared", json=_body(lease, models_dir, files=files)
    )
    assert response.status_code in (400, 422), response.text
    assert "x.bin" in response.json()["detail"]
    assert not list((models_dir / "Strata-data").glob(f"*{prepared.SUFFIX}"))
    assert prepared_client.get("/v1/run-operations/p").json()["preparedFrom"] is None


def test_the_agents_prepared_route_refuses_an_entry_through_a_link(
    prepared_client: TestClient,  # noqa: F811
    models_dir: Path,
    outside: Path,
) -> None:
    _link_folder(models_dir / "linked", outside)
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    body = _body(lease, models_dir, entry=str(models_dir / "linked" / "e.json"))
    response = prepared_client.post("/v1/run-operations/p/prepared", json=body)
    assert response.status_code in (400, 422), response.text
    assert not list(outside.glob(f"*{prepared.SUFFIX}"))


def _adopt(client: TestClient, **body: Any) -> Any:
    request = {"name": "qwen-flash", **body}
    return client.post("/v1/models/prepared", json=request)


@pytest.mark.parametrize("listed", ["../outside/x.bin", "ABSOLUTE"])
def test_adopting_refuses_a_list_that_leaves(
    configured_client: TestClient, models_dir: Path, outside: Path, listed: str
) -> None:
    if listed == "ABSOLUTE":
        listed = str(outside / "x.bin")
    (models_dir / "x.json").write_text("{}", encoding="utf-8")
    answer = _adopt(
        configured_client,
        root=str(models_dir),
        provenance={"engine": "strata", "entry": "x.json", "files": [{"path": listed}]},
    )
    assert answer.status_code in (400, 422), answer.text
    assert "x.bin" in answer.text
    assert not list(models_dir.rglob(f"*{prepared.SUFFIX}"))


def test_adopting_refuses_a_relative_entry_that_leaves(
    configured_client: TestClient, models_dir: Path, outside: Path
) -> None:
    answer = _adopt(
        configured_client,
        root=str(models_dir),
        provenance={"engine": "strata", "entry": "../outside/e.json"},
    )
    assert answer.status_code in (400, 422), answer.text
    assert "e.json" in answer.text
    assert not list(models_dir.rglob(f"*{prepared.SUFFIX}"))
    assert not list(outside.rglob(f"*{prepared.SUFFIX}"))
