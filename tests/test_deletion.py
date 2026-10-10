"""LS8: Delete a model from the Library, any format (Troy: a Delete for
every model). One rule for shared parts; all or nothing; refused while a
download or a run is working on it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import deletion, prepared
from eugene_plexus_library._generated.models import LibraryModel
from eugene_plexus_library.scanner import Scanner

from .conftest import qwen_like_kv, write_gguf, write_hf_model


def scan(root: Path) -> dict[str, LibraryModel]:
    return {m.name: m for m in prepared.link_sources(Scanner().scan([root]).models)}


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def strata_folder(root: Path) -> tuple[Path, Path]:
    """A GGUF, and two prepared models made from it in Strata-data sharing
    the MTP helper."""
    source = write_gguf(root / "repo" / "Flash-IQ2_XS.gguf", qwen_like_kv(name="F"))
    data = root / "Strata-data"
    _write(data / ".eugene-engine-files", b"marker")
    _write(data / "mtp" / "rt" / "experts.bin", b"m" * 20)
    _write(data / "mtp" / "tensors" / "big.bin", b"t" * 50)  # setup's, unrecorded
    for tag in ("a", "b"):
        _write(data / "packs" / tag / "dense.bin", b"d" * 30)
        (data / f"strata-{tag}.json").write_text("{}", encoding="utf-8")
        (data / f"{tag}{prepared.SUFFIX}").write_text(
            json.dumps(
                {
                    "engine": "strata",
                    "entry": f"strata-{tag}.json",
                    "source": {"path": str(source)},
                    "files": [
                        {"path": f"strata-{tag}.json"},
                        {"path": f"packs/{tag}/dense.bin"},
                        {"path": "mtp/rt/experts.bin", "shared": True},
                    ],
                }
            ),
            encoding="utf-8",
        )
    return source, data


def test_a_prepared_model_goes_with_its_own_files_and_keeps_the_shared_ones(
    models_dir: Path,
) -> None:
    source, data = strata_folder(models_dir)
    listed = scan(models_dir)
    plan = deletion.plan(listed["a"], list(listed.values()))
    removed = {Path(f.path) for f in plan.files}
    assert removed == {
        data / f"a{prepared.SUFFIX}",
        data / "strata-a.json",
        data / "packs" / "a" / "dense.bin",
    }
    # The MTP helper is b's too: kept, naming it. One rule, no Strata case.
    assert [(Path(k.path), [u.name for u in k.usedBy]) for k in plan.kept] == [
        (data / "mtp" / "rt" / "experts.bin", ["b"])
    ]
    assert plan.refusal is None
    deleted = deletion.remove([f.path for f in plan.files], [models_dir])
    assert {Path(p) for p in deleted} == removed
    assert not (data / "packs" / "a").exists()  # left empty, removed
    assert (data / "packs" / "b" / "dense.bin").is_file()
    assert (data / "mtp" / "rt" / "experts.bin").is_file()
    assert (data / "mtp" / "tensors" / "big.bin").is_file()  # unrecorded: stays
    assert (data / ".eugene-engine-files").is_file()
    assert source.is_file()


def test_the_last_user_of_a_shared_file_takes_it(models_dir: Path) -> None:
    _source, data = strata_folder(models_dir)
    listed = scan(models_dir)
    both = deletion.plan(listed["a"], list(listed.values()), also=[listed["b"].id])
    assert data / "mtp" / "rt" / "experts.bin" in {Path(f.path) for f in both.files}
    assert both.kept == []


def test_a_gguf_names_what_was_prepared_from_it(models_dir: Path) -> None:
    source, _data = strata_folder(models_dir)
    listed = scan(models_dir)
    gguf = next(m for m in listed.values() if m.path == str(source))
    plan = deletion.plan(gguf, list(listed.values()))
    assert sorted(m.name for m in plan.prepared_from) == ["a", "b"]
    assert {Path(f.path) for f in plan.files} == {source}
    answer = deletion.as_answer(plan, list(listed.values()))
    assert [d.name for d in answer.preparedFrom] == ["a", "b"]
    assert all(d.bytesFreed > 0 for d in answer.preparedFrom)


def test_a_split_gguf_goes_whole_and_a_shared_projector_stays(models_dir: Path) -> None:
    folder = models_dir / "vision"
    kv = qwen_like_kv(name="V") | {"split.count": 2}
    first = write_gguf(folder / "v-Q4_K_M-00001-of-00002.gguf", kv)
    second = _write(folder / "v-Q4_K_M-00002-of-00002.gguf", b"s" * 64)
    other = write_gguf(folder / "v-Q8_0.gguf", qwen_like_kv(name="V"))
    projector = write_gguf(folder / "mmproj-v-f16.gguf", {"general.architecture": "clip"})
    listed = scan(models_dir)
    q4 = next(m for m in listed.values() if m.path == str(first))
    plan = deletion.plan(q4, list(listed.values()))
    files = {Path(f.path) for f in plan.files}
    assert {first, second} <= files
    # Both quants pair with the lone projector: it stays for the other.
    assert any(Path(f.path) == projector for f in q4.files or [])
    assert projector not in files
    assert [Path(k.path) for k in plan.kept] == [projector]
    assert other.is_file()


def test_a_safetensors_folder_goes_whole(models_dir: Path) -> None:
    folder = write_hf_model(models_dir / "org" / "small")
    _write(folder / "README.md", b"readme")
    # Below the folder the scan lists nothing; Delete takes it all the same.
    _write(folder / "images" / "chart.png", b"png")
    listed = scan(models_dir)
    model = next(m for m in listed.values() if m.format.value == "safetensors")
    plan = deletion.plan(model, list(listed.values()))
    assert {folder / "README.md", folder / "images" / "chart.png"} <= {
        Path(f.path) for f in plan.files
    }
    deletion.remove([f.path for f in plan.files], [models_dir])
    assert not folder.exists()
    assert (models_dir / "org").exists() is False  # emptied up to the Library folder
    assert models_dir.is_dir()


def test_a_file_that_cannot_be_moved_leaves_everything(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, data = strata_folder(models_dir)
    listed = scan(models_dir)
    plan = deletion.plan(listed["a"], list(listed.values()))
    real = os.replace
    held = str(data / "packs" / "a" / "dense.bin")

    def replace(src: Any, dst: Any) -> None:
        if os.path.normcase(str(src)) == os.path.normcase(held):
            raise PermissionError(13, "The process cannot access the file")
        real(src, dst)

    monkeypatch.setattr(deletion.os, "replace", replace)
    with pytest.raises(deletion.DeleteRefused, match=r"dense.bin is in use.+Nothing was deleted"):
        deletion.remove([f.path for f in plan.files], [models_dir])
    for f in plan.files:
        assert Path(f.path).is_file(), f.path
    assert not list(data.rglob(f"{deletion.MOVED_ASIDE}*"))


class _Download:
    def __init__(self, state: str, target: str, model_id: str | None = None) -> None:
        from eugene_plexus_library._generated.models import DownloadState

        self.state = DownloadState(state)
        self.repo = "o/r"
        self.modelId = model_id

        class _File:
            destinationPath = target

        self.files = [_File()]


def test_a_download_or_a_run_working_on_it_refuses(models_dir: Path) -> None:
    source, _data = strata_folder(models_dir)
    listed = scan(models_dir)
    gguf = next(m for m in listed.values() if m.path == str(source))
    writing = deletion.plan(
        gguf, list(listed.values()), downloads=[_Download("downloading", str(source))]
    )
    assert writing.refusal and "download is writing into it" in writing.refusal
    finished = deletion.plan(
        gguf, list(listed.values()), downloads=[_Download("done", str(source))]
    )
    assert finished.refusal is None
    running = deletion.plan(
        gguf, list(listed.values()), operations=[{"step": "preparing", "model": {"id": gguf.id}}]
    )
    assert running.refusal and "Run is preparing on it" in running.refusal
    done = deletion.plan(
        gguf, list(listed.values()), operations=[{"step": "ready", "model": {"id": gguf.id}}]
    )
    assert done.refusal is None


def test_the_routes_plan_then_delete_and_the_model_leaves_the_library(
    configured_client: TestClient, models_dir: Path
) -> None:
    _source, data = strata_folder(models_dir)
    configured_client.post("/v1/scan")
    for _ in range(200):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
    models = {m["name"]: m for m in configured_client.get("/v1/models").json()["models"]}
    a = models["a"]
    plan = configured_client.get(f"/v1/models/{a['id']}/deletion").json()
    assert plan["bytesFreed"] > 0 and plan["kept"][0]["usedBy"][0]["name"] == "b"
    stale = configured_client.post(f"/v1/models/{a['id']}/delete", json={"token": "nope"})
    assert stale.status_code == 409 and (data / "strata-a.json").is_file()
    done = configured_client.post(f"/v1/models/{a['id']}/delete", json={"token": plan["token"]})
    assert done.status_code == 200, done.text
    assert done.json()["models"] == [a["id"]]
    assert not (data / "strata-a.json").exists()
    assert configured_client.get(f"/v1/models/{a['id']}").status_code == 404
    names = {m["name"] for m in configured_client.get("/v1/models").json()["models"]}
    assert "a" not in names and "b" in names


def _scanned(client: TestClient) -> dict[str, dict[str, Any]]:
    client.post("/v1/scan")
    for _ in range(200):
        if client.get("/v1/scan").json()["state"] != "scanning":
            break
    return {m["name"]: m for m in client.get("/v1/models").json()["models"]}


def test_a_refusal_holds_at_delete_time_too(
    configured_client: TestClient, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_library.routes import models as routes

    _source, data = strata_folder(models_dir)
    a = _scanned(configured_client)["a"]
    busy = {"downloads": [], "operations": [{"step": "starting", "model": {"id": a["id"]}}]}
    monkeypatch.setattr(routes, "_deletion_context", lambda _request: busy)
    plan = configured_client.get(f"/v1/models/{a['id']}/deletion").json()
    assert "Run is starting on it" in plan["refusal"]
    refused = configured_client.post(f"/v1/models/{a['id']}/delete", json={"token": plan["token"]})
    assert refused.status_code == 409 and "Run is starting on it" in refused.text
    assert (data / "strata-a.json").is_file()


def test_a_gguf_and_everything_prepared_from_it_go_together(
    configured_client: TestClient, models_dir: Path
) -> None:
    source, data = strata_folder(models_dir)
    models = _scanned(configured_client)
    gguf = next(m for m in models.values() if m["path"] == str(source))
    plan = configured_client.get(f"/v1/models/{gguf['id']}/deletion").json()
    both = [d["id"] for d in plan["preparedFrom"]]
    assert len(both) == 2
    done = configured_client.post(
        f"/v1/models/{gguf['id']}/delete", json={"token": plan["token"], "alsoDelete": both}
    )
    assert done.status_code == 200, done.text
    assert sorted(done.json()["models"]) == sorted([gguf["id"], *both])
    # The MTP helper had no user left among the models that stay.
    assert not (data / "mtp" / "rt" / "experts.bin").exists()
    assert not source.exists()
    assert (data / "mtp" / "tensors" / "big.bin").is_file()
    unknown = configured_client.post(
        f"/v1/models/{gguf['id']}/delete", json={"token": plan["token"], "alsoDelete": ["x"]}
    )
    assert unknown.status_code == 404
