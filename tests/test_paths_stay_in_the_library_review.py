"""library#20, the review's second pass: another spelling of a file, a folder
swapped for a link while files move, refusals below a link, a linked
`subdirectory`, and control characters in a provenance file."""

from __future__ import annotations

import json
import os
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
from eugene_plexus_library.paths import model_id
from eugene_plexus_library.scanner import Scanner

from .test_paths_stay_in_the_library import _link_folder, _provenance, _untouched
from .test_run_preparation import _body, _submit, _to_preparing, prepared_client  # noqa: F401


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    (folder / "x.bin").write_bytes(b"x" * 7)
    (folder / "e.json").write_text("{}", encoding="utf-8")
    return folder


def _engine_folder(root: Path, name: str, files: list[str]) -> Path:
    data = root / f"{name}-data"
    data.mkdir(exist_ok=True)
    (data / prepared.ENGINE_FILES_MARKER).write_text("engine files", encoding="utf-8")
    (data / "e.json").write_text("{}", encoding="utf-8")
    body = {"engine": "strata", "entry": "e.json", "files": [{"path": f} for f in files]}
    (data / f"{name}{prepared.SUFFIX}").write_text(json.dumps(body), encoding="utf-8")
    return data


def _two_models(root: Path, listed: str, prep: Any = None) -> dict[str, LibraryModel]:
    """`victim` lists its pack; `other` lists `listed` (made by `prep`)."""
    victim = _engine_folder(root, "victim", ["e.json", "pack.bin"])
    (victim / "pack.bin").write_bytes(b"v" * 11)
    (root / "other-data").mkdir()
    if prep is not None:
        prep(root)
    _engine_folder(root, "other", ["e.json", listed])
    return {m.name: m for m in Scanner().scan([root]).models}


# --- 1. another spelling of a file -------------------------------------------------


@pytest.mark.parametrize(
    "listed",
    [
        "../victim-data/pack.bin.",
        "../victim-data/pack.bin ",
        "../victim-data./pack.bin",
        "../victim-data/pack.bin::$DATA",
        "../victim-data/victim.eugene-prepared.json.",
        ".../x.bin",
        ".. /x.bin",
    ],
)
def test_a_part_another_spelling_could_hide_behind_is_refused(
    models_dir: Path, listed: str
) -> None:
    models = _two_models(models_dir, listed)
    other = models["other"]
    assert other.status is ModelStatus.unreadable, other
    assert not (other.files or [])
    assert models["victim"].status is ModelStatus.present


def _kept_by_victim(models_dir: Path, models: dict[str, LibraryModel]) -> None:
    other = models["other"]
    assert other.status is ModelStatus.present, other.error
    plan = deletion.plan(other, list(models.values()), roots=[models_dir])
    assert [[u.name for u in k.usedBy] for k in plan.kept] == [["victim"]]
    assert plan.refusal is None
    deletion.remove([f.path for f in plan.files], [models_dir])
    assert (models_dir / "victim-data" / "pack.bin").is_file()
    assert (models_dir / "victim-data" / f"victim{prepared.SUFFIX}").is_file()


def test_a_file_another_model_lists_is_kept_through_a_folder_link(models_dir: Path) -> None:
    def link(root: Path) -> None:
        _link_folder(root / "other-data" / "j", root / "victim-data")

    _kept_by_victim(models_dir, _two_models(models_dir, "j/pack.bin", link))


def test_a_file_another_model_lists_is_kept_through_a_hard_link(models_dir: Path) -> None:
    def link(root: Path) -> None:
        os.link(root / "victim-data" / "pack.bin", root / "other-data" / "hl.bin")

    _kept_by_victim(models_dir, _two_models(models_dir, "hl.bin", link))


def test_a_file_another_model_lists_is_kept_in_another_case(models_dir: Path) -> None:
    probe = models_dir / "probe.bin"
    probe.write_bytes(b"p")
    folds = os.path.normcase("A") != "A"
    insensitive = (models_dir / "PROBE.BIN").exists()
    probe.unlink()
    if folds or not insensitive:
        pytest.skip("needs a case-insensitive file system that normcase does not fold (macOS)")
    _kept_by_victim(models_dir, _two_models(models_dir, "../victim-data/PACK.BIN"))


# --- 2. a folder swapped for a link while files move -------------------------------


def test_a_folder_swapped_for_a_link_during_the_move_is_put_back(
    models_dir: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = models_dir / "swapped"
    folder.mkdir()
    (folder / "x.bin").write_bytes(b"decoy")
    target = str(folder / "x.bin")
    real = os.replace
    swapped = {"done": False}

    def replace(src: Any, dst: Any) -> None:
        if not swapped["done"] and str(src) == target:
            swapped["done"] = True
            os.rename(folder, models_dir / "swapped.real")
            _link_folder(folder, outside)
        real(src, dst)

    monkeypatch.setattr(deletion.os, "replace", replace)
    with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted"):
        deletion.remove([target], [models_dir])
    monkeypatch.setattr(deletion.os, "replace", real)
    assert (outside / "x.bin").read_bytes() == b"x" * 7
    assert not [p.name for p in outside.iterdir() if p.name.startswith(deletion.MOVED_ASIDE)]


def test_the_last_check_before_removing_puts_everything_back(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = models_dir / "a.bin", models_dir / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    real = deletion.held_by
    aside: list[str] = []

    def held_by(path: str, roots: Any) -> Path | None:
        if deletion.MOVED_ASIDE in os.path.basename(path):
            aside.append(path)
            # The checks just after each move pass; the one before removing fails.
            if len(aside) == 3:
                return None
        return real(path, roots)

    monkeypatch.setattr(deletion, "held_by", held_by)
    with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted"):
        deletion.remove([str(first), str(second)], [models_dir])
    assert first.read_bytes() == b"a" and second.read_bytes() == b"b"
    _untouched(models_dir)


# --- 3. a refusal below a link says so -------------------------------------------------


def test_a_delete_refused_below_a_link_names_the_link_and_the_remedy(
    models_dir: Path, outside: Path
) -> None:
    (outside / "big").mkdir()
    (outside / "big" / "a.gguf").write_bytes(b"g" * 3)
    link = models_dir / "big"
    _link_folder(link, outside / "big")
    weights = link / "a.gguf"
    model = LibraryModel(
        id=model_id(weights),
        path=str(weights),
        root=str(models_dir),
        format=ModelFormat.gguf,
        name="a",
        status=ModelStatus.present,
        files=[ModelFile(path=str(weights), role=ModelFileRole.weights, sizeBytes=3)],
    )
    plan = deletion.plan(model, [model], roots=[models_dir])
    target = os.path.realpath(outside / "big")
    for said in (plan.refusal or "", _refused(str(weights), models_dir)):
        assert str(link) in said and target in said, said
        assert "as a Library folder" in said
        assert "Scan the Library again" not in said
    assert (outside / "big" / "a.gguf").is_file()


def _refused(path: str, root: Path) -> str:
    with pytest.raises(deletion.DeleteRefused) as refused:
        deletion.remove([path], [root])
    return str(refused.value)


def test_a_prepared_model_below_a_link_names_the_link(models_dir: Path, outside: Path) -> None:
    _provenance(models_dir, files=[{"path": "link/x.bin"}])
    link = models_dir / "Strata-data" / "link"
    _link_folder(link, outside)
    found = {m.name: m for m in Scanner().scan([models_dir]).models}["bad"]
    assert found.status is ModelStatus.unreadable
    assert str(link) in (found.error or "")
    assert os.path.realpath(outside) in (found.error or "")
    assert "as a Library folder" in (found.error or "")


# --- 5. a linked subdirectory when adopting ---------------------------------------------


def test_adopting_into_a_linked_subdirectory_is_refused(
    configured_client: TestClient, models_dir: Path, outside: Path
) -> None:
    (models_dir / "Strata-data").mkdir()
    entry = models_dir / "Strata-data" / "e.json"
    entry.write_text("{}", encoding="utf-8")
    _link_folder(models_dir / "link", outside)
    answer = configured_client.post(
        "/v1/models/prepared",
        json={
            "name": "via-link",
            "root": str(models_dir),
            "subdirectory": "link/made",
            "provenance": {"engine": "strata", "entry": str(entry)},
        },
    )
    assert answer.status_code == 400, answer.text
    assert sorted(p.name for p in outside.iterdir()) == ["e.json", "x.bin"]


# --- 6. control characters ----------------------------------------------------------------


def test_a_control_character_lists_the_model_unreadable_and_the_scan_goes_on(
    models_dir: Path,
) -> None:
    _engine_folder(models_dir, "nul", ["a\x00b"])
    _engine_folder(models_dir, "newline", ["e.json\n../../outside/x.bin"])
    tab = _engine_folder(models_dir, "tab", [])
    body = {"engine": "strata", "entry": "e\x00.json"}
    (tab / f"tab{prepared.SUFFIX}").write_text(json.dumps(body), encoding="utf-8")
    _engine_folder(models_dir, "good", ["e.json"])
    found = {m.name: m for m in Scanner().scan([models_dir]).models}
    for name in ("nul", "newline", "tab"):
        assert found[name].status is ModelStatus.unreadable, found[name]
        assert "control character" in (found[name].error or ""), found[name].error
        assert not (found[name].files or [])
    assert found["good"].status is ModelStatus.present


def test_remove_refuses_a_path_with_a_control_character(models_dir: Path) -> None:
    mine = models_dir / "a.bin"
    mine.write_bytes(b"a")
    assert deletion.held_by(str(mine) + "\x00x", [models_dir]) is None
    with pytest.raises(deletion.DeleteRefused, match="Nothing was deleted"):
        deletion.remove([str(mine), str(mine) + "\x00x"], [models_dir])
    assert mine.is_file()


def test_the_agents_prepared_route_refuses_a_control_character(
    prepared_client: TestClient,  # noqa: F811
    models_dir: Path,
) -> None:
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    files = [{"path": "strata-iq2_xs.json"}, {"path": "a\x00b"}]
    response = prepared_client.post(
        "/v1/run-operations/p/prepared", json=_body(lease, models_dir, files=files)
    )
    assert response.status_code == 400, response.text
    assert "control character" in response.json()["detail"]
    entry = str(models_dir / "Strata-data" / "strata-iq2_xs.json") + "\x00"
    response = prepared_client.post(
        "/v1/run-operations/p/prepared", json=_body(lease, models_dir, entry=entry)
    )
    assert response.status_code == 400, response.text
    assert not list((models_dir / "Strata-data").glob(f"*{prepared.SUFFIX}"))


def test_adopting_refuses_a_control_character_in_the_entry(
    configured_client: TestClient, models_dir: Path
) -> None:
    (models_dir / "e.json").write_text("{}", encoding="utf-8")
    for entry in ("e\x00.json", str(models_dir / "e.json") + "\x00"):
        answer = configured_client.post(
            "/v1/models/prepared",
            json={
                "name": "nul",
                "root": str(models_dir),
                "provenance": {"engine": "strata", "entry": entry},
            },
        )
        assert answer.status_code == 400, answer.text
    assert not list(models_dir.rglob(f"*{prepared.SUFFIX}"))
