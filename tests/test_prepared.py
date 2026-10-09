"""Prepared models are Library models (library-sources-and-engines.md §4.5, LS3).

A provenance file, `<name>.eugene-prepared.json`, makes an engine's own
files a model: the scan lists it, the judge offers it only to the engine it
was prepared for, and `POST /v1/models/prepared` writes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import eligibility, prepared
from eugene_plexus_library._generated.models import (
    EligibilityEngine,
    LibraryModel,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
)
from eugene_plexus_library.paths import model_id
from eugene_plexus_library.scanner import Scanner

from .conftest import qwen_like_kv, write_gguf, write_hf_model

STRATA = {
    "engine": "strata",
    "experimental": True,
    "accepts": [
        {
            "format": "gguf",
            "architectures": ["qwen4exp"],
            "preparation": {"recipe": "strata-prepare"},
            "preference": 50,
        },
        {"format": "prepared", "preparedFor": "strata", "preference": 50},
    ],
}
LLAMA = {"engine": "llama_cpp", "accepts": [{"format": "gguf", "preference": 10}]}


def engine(spec: dict[str, Any], *, available: bool = True) -> EligibilityEngine:
    return EligibilityEngine.model_validate({**spec, "available": available})


def provenance(folder: Path, name: str = "qwen-flash", **body: Any) -> Path:
    path = folder / f"{name}{prepared.SUFFIX}"
    path.write_text(json.dumps({"engine": "strata", **body}), encoding="utf-8")
    return path


def scan(root: Path) -> dict[str, LibraryModel]:
    return {m.name: m for m in Scanner().scan([root]).models}


# --- the scan ---------------------------------------------------------------


def test_the_scan_lists_a_prepared_model_from_its_provenance_file(models_dir: Path) -> None:
    (models_dir / "strata-qwen.json").write_text("{}", encoding="utf-8")
    path = provenance(
        models_dir, entry="strata-qwen.json", recipe="strata-prepare", recipeVersion="v0.1.39"
    )
    found = scan(models_dir)["qwen-flash"]
    assert found.format is ModelFormat.prepared
    assert found.status is ModelStatus.present
    assert found.path == str(path)
    assert found.id == model_id(path)
    # The engine's files are not read, so no size is claimed for them.
    assert found.sizeBytes is None
    assert found.prepared is not None
    assert found.prepared.engine.value == "strata"
    assert found.prepared.entry == "strata-qwen.json"
    assert Path(found.prepared.entryPath or "") == models_dir / "strata-qwen.json"
    assert found.prepared.entryFound is True
    assert found.prepared.recipe == "strata-prepare"
    assert {f.role for f in found.files or []} == {ModelFileRole.index, ModelFileRole.config}


def test_a_relative_entry_that_is_not_there_is_unreadable_and_named(models_dir: Path) -> None:
    provenance(models_dir, entry="gone.json")
    found = scan(models_dir)["qwen-flash"]
    assert found.status is ModelStatus.unreadable
    assert "gone.json" in (found.error or "")


def test_an_absolute_entry_on_another_node_is_listed_not_found_here(models_dir: Path) -> None:
    provenance(models_dir, entry=r"Z:\Strata\strata-qwen.json")
    found = scan(models_dir)["qwen-flash"]
    assert found.status is ModelStatus.present
    assert found.prepared is not None
    assert found.prepared.entryFound is False
    assert found.prepared.entryPath == r"Z:\Strata\strata-qwen.json"
    assert [f.role for f in found.files or []] == [ModelFileRole.index]


@pytest.mark.parametrize(
    "body,said",
    [
        ("not json", "not JSON"),
        ('{"entry": "x.json"}', "engine"),
        ('{"engine": "ninfer", "entry": "x.json"}', "engine"),
        ('{"engine": "strata"}', "entry"),
        ('{"formatVersion": 2, "engine": "strata", "entry": "x.json"}', "newer Eugene"),
    ],
)
def test_a_provenance_file_that_cannot_be_listed_says_why(
    models_dir: Path, body: str, said: str
) -> None:
    (models_dir / f"bad{prepared.SUFFIX}").write_text(body, encoding="utf-8")
    found = scan(models_dir)["bad"]
    assert found.format is ModelFormat.prepared
    assert found.status is ModelStatus.unreadable
    assert said in (found.error or "")


def test_fields_a_newer_eugene_adds_are_ignored(models_dir: Path) -> None:
    (models_dir / "c.json").write_text("{}", encoding="utf-8")
    provenance(models_dir, entry="c.json", files=["pack/0.bin"], formatVersion=1)
    assert scan(models_dir)["qwen-flash"].status is ModelStatus.present


def test_the_source_is_linked_when_the_library_lists_it(models_dir: Path) -> None:
    source = write_gguf(models_dir / "Flash-Next-IQ2_XS.gguf", qwen_like_kv(name="F"))
    (models_dir / "c.json").write_text("{}", encoding="utf-8")
    provenance(models_dir, entry="c.json", source={"path": str(source), "repoId": "o/r"})
    provenance(models_dir, name="orphan", entry="c.json", source={"path": "/gone.gguf"})
    found = scan(models_dir)
    assert found["qwen-flash"].prepared is not None
    assert found["qwen-flash"].prepared.sourceModelId == found["Flash-Next-IQ2_XS"].id
    assert found["orphan"].prepared is not None
    assert found["orphan"].prepared.sourceModelId is None
    assert found["orphan"].prepared.source is not None
    assert found["orphan"].prepared.source.path == "/gone.gguf"


def test_a_provenance_file_inside_a_safetensors_folder_is_found_too(models_dir: Path) -> None:
    folder = write_hf_model(models_dir / "hf")
    (folder / "e.json").write_text("{}", encoding="utf-8")
    provenance(folder, entry="e.json")
    found = scan(models_dir)
    assert found["qwen-flash"].format is ModelFormat.prepared
    assert found["hf"].format is ModelFormat.safetensors


# --- the judge ----------------------------------------------------------------


def prepared_model(engine_kind: str | None = "strata") -> LibraryModel:
    return LibraryModel.model_validate(
        {
            "id": "p",
            "path": "/models/p.eugene-prepared.json",
            "format": "prepared",
            "name": "p",
            "status": "present" if engine_kind else "unreadable",
            **({"prepared": {"engine": engine_kind, "entry": "c.json"}} if engine_kind else {}),
        }
    )


def verdicts(m: LibraryModel, *engines: EligibilityEngine) -> dict[str, tuple[str, str]]:
    judged = eligibility.judge(m, list(engines))
    return {str(v.engine): (str(v.verdict), v.reason) for v in judged.engines}


def test_a_prepared_model_runs_on_its_own_engine_only() -> None:
    answer = eligibility.judge(prepared_model(), [engine(STRATA), engine(LLAMA)])
    assert answer.level == "works_here"
    said = {str(v.engine): (str(v.verdict), v.reason) for v in answer.engines}
    assert said["strata"] == ("runs", "runs it as it is")
    assert said["llama_cpp"][0] == "no"
    assert "this one is prepared" in said["llama_cpp"][1]
    # Strata not installed but installable: another engine's dot.
    later = eligibility.judge(
        prepared_model(),
        [EligibilityEngine.model_validate({**STRATA, "available": False, "installable": True})],
    )
    assert later.level == "other_engine"


def test_absent_prepared_for_means_the_declaring_engine() -> None:
    own = {"engine": "strata", "accepts": [{"format": "prepared"}]}
    other = {"engine": "kev", "accepts": [{"format": "prepared"}]}
    said = verdicts(prepared_model(), engine(own), engine(other))
    assert said["strata"][0] == "runs"
    assert said["kev"] == (
        "no",
        "loads only models prepared for kev, and this one was prepared for strata",
    )


def test_prepared_for_is_data_naming_whose_preparation_an_engine_loads() -> None:
    own = {"engine": "kev", "accepts": [{"format": "prepared", "preparedFor": "kev"}]}
    assert verdicts(prepared_model(), engine(own))["kev"][0] == "no"
    # An engine may load another engine's preparation, by naming it.
    borrows = {"engine": "kev", "accepts": [{"format": "prepared", "preparedFor": "strata"}]}
    assert verdicts(prepared_model(), engine(borrows))["kev"][0] == "runs"


def test_an_unreadable_provenance_runs_nowhere_and_says_so() -> None:
    said = verdicts(prepared_model(None), engine(STRATA))
    assert said["strata"] == (
        "no",
        "loads prepared models, and this one's provenance file could not be read",
    )


def test_a_gguf_is_still_after_preparation_for_strata() -> None:
    flash = LibraryModel.model_validate(
        {
            "id": "g",
            "path": "/m/f.gguf",
            "format": "gguf",
            "name": "f",
            "status": "present",
            "architecture": "qwen4exp",
            "gguf": {},
        }
    )
    assert verdicts(flash, engine(STRATA))["strata"][0] == "after_preparation"


# --- adopting one through the route --------------------------------------------


def adopt(client: TestClient, **body: Any) -> Any:
    request = {"name": "qwen-flash", **body}
    request.setdefault("provenance", {"engine": "strata", "entry": "x.json"})
    return client.post("/v1/models/prepared", json=request)


def test_adopting_writes_the_file_beside_its_entry_and_lists_it_at_once(
    configured_client: TestClient, models_dir: Path
) -> None:
    folder = models_dir / "strata"
    folder.mkdir()
    entry = folder / "strata-qwen.json"
    entry.write_text("{}", encoding="utf-8")
    answer = adopt(configured_client, provenance={"engine": "strata", "entry": str(entry)})
    assert answer.status_code == 201, answer.text
    model = answer.json()
    written = folder / f"qwen-flash{prepared.SUFFIX}"
    assert model["path"] == str(written)
    assert model["format"] == "prepared"
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    # Relative, so the two move together.
    assert on_disk["entry"] == "strata-qwen.json"
    assert on_disk["formatVersion"] == 1
    assert "preparedAt" in on_disk
    listed = configured_client.get("/v1/models").json()["models"]
    assert [m["id"] for m in listed if m["format"] == "prepared"] == [model["id"]]
    # The next walk finds the same file and keeps it.
    configured_client.post("/v1/scan")
    for _ in range(200):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
    again = configured_client.get(f"/v1/models/{model['id']}").json()
    assert again["status"] == "present"
    assert again["prepared"]["entryFound"] is True


def test_an_entry_on_another_node_is_written_as_given(
    configured_client: TestClient, models_dir: Path
) -> None:
    answer = adopt(
        configured_client,
        root=str(models_dir),
        subdirectory="strata",
        provenance={"engine": "strata", "entry": r"Z:\Strata\strata-qwen.json"},
    )
    assert answer.status_code == 201, answer.text
    written = models_dir / "strata" / f"qwen-flash{prepared.SUFFIX}"
    assert json.loads(written.read_text(encoding="utf-8"))["entry"] == r"Z:\Strata\strata-qwen.json"
    assert answer.json()["prepared"]["entryFound"] is False


def test_adopting_refuses_and_writes_nothing_when_it_should(
    configured_client: TestClient, models_dir: Path, tmp_path: Path
) -> None:
    missing = adopt(
        configured_client, provenance={"engine": "strata", "entry": str(models_dir / "no.json")}
    )
    assert missing.status_code == 400
    assert "no.json" in missing.text
    elsewhere = adopt(configured_client, root=str(tmp_path / "other"))
    assert elsewhere.status_code == 400
    climbing = adopt(configured_client, root=str(models_dir), subdirectory="../out")
    assert climbing.status_code == 400
    assert not list(models_dir.rglob(f"*{prepared.SUFFIX}"))
    assert not list(tmp_path.rglob(f"*{prepared.SUFFIX}"))

    (models_dir / "x.json").write_text("{}", encoding="utf-8")
    assert adopt(configured_client, root=str(models_dir)).status_code == 201
    twice = adopt(configured_client, root=str(models_dir))
    assert twice.status_code == 409
    assert adopt(configured_client, name="../escape").status_code == 422


def test_fit_is_not_estimated_for_a_prepared_model(
    configured_client: TestClient, models_dir: Path
) -> None:
    (models_dir / "x.json").write_text("{}", encoding="utf-8")
    model = adopt(configured_client, root=str(models_dir)).json()
    answer = configured_client.get(f"/v1/models/{model['id']}/fit")
    assert answer.status_code == 422
    assert "Fit not estimated" in answer.text
