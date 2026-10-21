"""Re-point a Library entry whose file moved (row 22).

A model's id is the hash of its path, so a file that moved is a new entry and
the profiles stay on the old one, `missing`. These tests move real files
between two real folders, scan, and ask the HTTP surface to move the profiles.
Nothing here moves a profile except a call that names both entries.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library import fingerprint
from eugene_plexus_library.app import create_app
from eugene_plexus_library.profile_storage import StorageUnavailable
from eugene_plexus_library.settings import Settings

from .conftest import qwen_like_kv, wait_for_scan, write_gguf, write_hf_model

# Past the 1.56 MiB under which a file is read whole, so a block is a block.
BIG = 3 * 1024 * 1024


@pytest.fixture
def root_a(tmp_path: Path) -> Path:
    path = tmp_path / "root-a"
    path.mkdir()
    return path


@pytest.fixture
def root_b(tmp_path: Path) -> Path:
    path = tmp_path / "root-b"
    path.mkdir()
    return path


@pytest.fixture
def app(settings: Settings, root_a: Path, root_b: Path) -> FastAPI:
    settings.config_file.write_text(
        "modelRoots:\n"
        f"  - path: {root_a.as_posix()}\n    mounts: []\n"
        f"  - path: {root_b.as_posix()}\n    mounts: []\n"
        "scanOnStartup: false\n",
        encoding="utf-8",
    )
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _scan(client: TestClient) -> None:
    assert client.post("/v1/scan", json={"full": True}).status_code in (200, 202)
    wait_for_scan(client)


def _models(client: TestClient) -> dict[str, dict[str, Any]]:
    listed = client.get("/v1/models").json()["models"]
    return {m["path"]: m for m in listed}


def _profile(client: TestClient, model_id: str, name: str = "tuned", **fields: Any) -> dict:
    created = client.post(
        f"/v1/models/{model_id}/profiles", json={"name": name, "engine": "llama_cpp", **fields}
    )
    assert created.status_code == 201, created.text
    return created.json()


def _write(path: Path, *, size: int = BIG, **kv: Any) -> Path:
    return write_gguf(path, qwen_like_kv(**kv), payload=bytes(size))


def _move(client: TestClient, source: Path, target: Path) -> tuple[str, str]:
    """Move the file, scan, and give `(old id, new id)`; the old one is missing."""
    old = _models(client)[str(source)]["id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, target)
    _scan(client)
    return old, _models(client)[str(target)]["id"]


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


def _forget_fingerprint(app: FastAPI, model_id: str) -> None:
    """The entry as it was before fingerprints existed."""
    store = app.state.state_store
    with store._lock:
        store._models[model_id] = store._models[model_id].model_copy(update={"fingerprint": None})
        store._reindex_locked()  # the scan's cache reads the index


# --- the test the row exists for ----------------------------------------------


def test_moved_file_keeps_its_profiles(client: TestClient, root_a: Path, root_b: Path) -> None:
    """Scan a GGUF in root A and save a profile; move the file to root B and scan;
    relocate from the old id to the new one. The profile is on the new id and
    the old id is gone. Before the change the route did not exist (405)."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    old = _models(client)[str(source)]["id"]
    _profile(client, old, "long context", flags={"contextSize": 32768}, notes="tuned for 24 GB")

    old, new = _move(client, source, root_b / "m.gguf")
    assert client.get(f"/v1/models/{old}").json()["status"] == "missing"

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["moved"] == 1 and body["renamed"] == [] and body["warnings"] == []
    assert body["model"]["id"] == new and body["model"]["profileCount"] == 1
    moved = client.get(f"/v1/models/{new}/profiles").json()["profiles"]
    assert [(p["name"], p["flags"], p["notes"]) for p in moved] == [
        ("long context", {"contextSize": 32768}, "tuned for 24 GB")
    ]
    assert client.get(f"/v1/models/{old}").status_code == 404
    assert client.get(f"/v1/models/{old}/profiles").status_code == 404


def test_a_moved_file_is_recognised_but_nothing_moves_by_itself(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    """The scan says where the file went, and waits for a click: an unplugged
    drive makes its models missing, and profiles that moved to a copy would
    leave the original coming back without them."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])

    old, new = _move(client, source, root_b / "m.gguf")

    missing = client.get(f"/v1/models/{old}").json()
    assert missing["status"] == "missing"
    assert missing["movedTo"] == new
    assert missing["profileCount"] == 1
    assert client.get(f"/v1/models/{new}/profiles").json()["profiles"] == []
    found = client.get(f"/v1/models/{old}/relocate-candidates").json()["candidates"]
    assert [(c["model"]["id"], c["match"], c["differences"]) for c in found] == [(new, "same", [])]


def test_a_present_model_is_not_marked_moved(client: TestClient, root_a: Path) -> None:
    _write(root_a / "m.gguf")
    _scan(client)

    assert [m["movedTo"] for m in client.get("/v1/models").json()["models"]] == [None]


def test_two_copies_are_listed_and_neither_is_chosen(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    """With two files that match, the person picks."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    shutil.copy2(source, root_b / "copy-one.gguf")
    shutil.copy2(source, root_b / "copy-two.gguf")
    source.unlink()
    _scan(client)
    models = _models(client)
    old = next(m["id"] for m in models.values() if m["status"] == "missing")

    assert client.get(f"/v1/models/{old}").json()["movedTo"] is None
    found = client.get(f"/v1/models/{old}/relocate-candidates").json()["candidates"]
    assert sorted(c["model"]["name"] for c in found) == ["copy-one", "copy-two"]
    assert {c["match"] for c in found} == {"same"}


# --- a different file is refused -----------------------------------------------


def test_different_file_is_refused(client: TestClient, root_a: Path, root_b: Path) -> None:
    """One changed block, the same name and the same size: nothing moves."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    old_id = _models(client)[str(source)]["id"]
    _profile(client, old_id, "tuned")
    target = root_b / "m.gguf"
    old, new = _move(client, source, target)
    _flip_last_byte(target)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 409
    detail = answer.json()["detail"]["detail"]
    assert "is a different file" in detail and str(target) in detail
    assert client.get(f"/v1/models/{old}/profiles").json()["profiles"][0]["name"] == "tuned"
    assert client.get(f"/v1/models/{new}/profiles").json()["profiles"] == []
    assert client.get(f"/v1/models/{old}").status_code == 200


def test_a_file_of_another_size_is_refused_with_the_difference(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    old = _models(client)[str(source)]["id"]
    source.unlink()
    _write(root_b / "m.gguf", size=BIG + 4096)
    _scan(client)
    new = _models(client)[str(root_b / "m.gguf")]["id"]

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 409
    assert "weights file is" in answer.json()["detail"]["detail"]


def test_a_file_changed_since_the_scan_is_refused(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    """The target is read again at the click, not trusted from the scan."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    target = root_b / "m.gguf"
    old, new = _move(client, source, target)

    _flip_last_byte(target)  # after the scan, so the stored fingerprint is the old one

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 409
    assert "is a different file" in answer.json()["detail"]["detail"]


def test_a_target_that_is_gone_is_named(client: TestClient, root_a: Path, root_b: Path) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    target = root_b / "m.gguf"
    old, new = _move(client, source, target)
    target.unlink()  # after the scan: the entry is still `present`

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 409
    assert answer.json()["detail"]["detail"] == f"{target} is no longer there."


def test_a_target_that_cannot_be_read_names_the_cause(
    client: TestClient, root_a: Path, root_b: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    target = root_b / "m.gguf"
    old, new = _move(client, source, target)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(target))

    monkeypatch.setattr(fingerprint, "open", refuse, raising=False)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 409
    detail = answer.json()["detail"]["detail"]
    assert detail.startswith(f"Eugene could not read {target} to check it: ")
    assert "Permission denied" in detail


def test_a_model_that_is_still_there_has_nothing_to_move(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    shutil.copy2(source, root_b / "m.gguf")
    _scan(client)
    models = _models(client)
    old, other = models[str(source)]["id"], models[str(root_b / "m.gguf")]["id"]
    _profile(client, old)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": other})

    assert answer.status_code == 409
    assert answer.json()["detail"]["detail"] == (
        f"This model is still at {source}; there is nothing to move."
    )
    assert client.get(f"/v1/models/{old}/profiles").json()["profiles"] != []


def test_a_target_that_is_missing_too_is_refused(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    first = _write(root_a / "one.gguf")
    second = _write(root_a / "two.gguf", name="Other")
    _scan(client)
    models = _models(client)
    one, two = models[str(first)]["id"], models[str(second)]["id"]
    _profile(client, one)
    _profile(client, two)
    first.unlink()
    second.unlink()
    _scan(client)

    answer = client.post(f"/v1/models/{one}/relocate", json={"to": two})

    assert answer.status_code == 409
    assert "is no longer there" in answer.json()["detail"]["detail"]


def test_a_directory_model_does_not_take_a_gguf_profile(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    write_hf_model(root_b / "hf")
    _scan(client)
    models = _models(client)
    old = models[str(source)]["id"]
    other = models[str(root_b / "hf")]["id"]
    _profile(client, old)
    source.unlink()
    _scan(client)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": other})

    assert answer.status_code == 409
    assert "its format is safetensors, not gguf" in answer.json()["detail"]["detail"]


def test_unknown_ids_are_404(client: TestClient, root_a: Path) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    known = _models(client)[str(source)]["id"]

    assert client.post("/v1/models/nope/relocate", json={"to": known}).status_code == 404
    assert client.post(f"/v1/models/{known}/relocate", json={"to": "nope"}).status_code == 404
    assert client.get("/v1/models/nope/relocate-candidates").status_code == 404


# --- an entry from before fingerprints existed ------------------------------


def test_likely_needs_confirmation(
    app: FastAPI, client: TestClient, root_a: Path, root_b: Path
) -> None:
    """With no fingerprint to compare, a match on size and details is `likely`:
    Eugene says it cannot prove it, and moves nothing until the person confirms."""
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"], "tuned")
    old, new = _move(client, source, root_b / "m.gguf")
    _forget_fingerprint(app, old)

    candidates = client.get(f"/v1/models/{old}/relocate-candidates").json()["candidates"]
    assert [(c["model"]["id"], c["match"]) for c in candidates] == [(new, "likely")]
    assert candidates[0]["differences"] != []
    assert client.get(f"/v1/models/{old}").json()["movedTo"] is None

    refused = client.post(f"/v1/models/{old}/relocate", json={"to": new})
    assert refused.status_code == 409
    assert refused.json()["detail"]["detail"] == (
        "Eugene cannot prove these are the same file. Confirm to move anyway."
    )
    also = client.post(f"/v1/models/{old}/relocate", json={"to": new, "confirmLikely": False})
    assert also.status_code == 409
    assert client.get(f"/v1/models/{new}/profiles").json()["profiles"] == []

    moved = client.post(f"/v1/models/{old}/relocate", json={"to": new, "confirmLikely": True})

    assert moved.status_code == 200, moved.text
    assert [p["name"] for p in client.get(f"/v1/models/{new}/profiles").json()["profiles"]] == [
        "tuned"
    ]
    assert client.get(f"/v1/models/{old}").status_code == 404


def test_a_likely_match_still_needs_the_same_details(
    app: FastAPI, client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"])
    old = _models(client)[str(source)]["id"]
    source.unlink()
    _write(root_b / "m.gguf", context_length=8192)  # same size, other details
    _scan(client)
    new = _models(client)[str(root_b / "m.gguf")]["id"]
    _forget_fingerprint(app, old)

    assert client.get(f"/v1/models/{old}/relocate-candidates").json()["candidates"] == []
    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new, "confirmLikely": True})
    assert answer.status_code == 409
    assert "context length" in answer.json()["detail"]["detail"]


# --- what happens to the profiles --------------------------------------------


def test_a_name_the_target_uses_gets_moved_added_and_the_target_keeps_its_default(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    old = _models(client)[str(source)]["id"]
    _profile(client, old, "fast")  # the source's default
    _profile(client, old, "long", default=False)
    shutil.copy2(source, root_b / "m.gguf")  # the target exists before the move
    target = root_b / "m.gguf"
    _scan(client)
    new = _models(client)[str(target)]["id"]
    _profile(client, new, "fast", flags={"mine": True})  # the target's own default
    source.unlink()
    _scan(client)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["moved"] == 2
    assert body["renamed"] == [{"name": "fast", "renamedTo": "fast (moved)"}]
    profiles = client.get(f"/v1/models/{new}/profiles").json()["profiles"]
    assert [(p["name"], p["default"]) for p in profiles] == [
        ("fast", True),
        ("fast (moved)", False),
        ("long", False),
    ]
    assert profiles[0]["flags"] == {"mine": True}


def test_the_default_moves_when_the_target_has_none(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    old = _models(client)[str(source)]["id"]
    _profile(client, old, "a")
    _profile(client, old, "b", default=True)
    old, new = _move(client, source, root_b / "m.gguf")

    assert client.post(f"/v1/models/{old}/relocate", json={"to": new}).status_code == 200

    profiles = client.get(f"/v1/models/{new}/profiles").json()["profiles"]
    assert [(p["name"], p["default"]) for p in profiles] == [("b", True), ("a", False)]


def test_a_profile_that_names_the_old_path_moves_and_is_listed(
    client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    old = _models(client)[str(source)]["id"]
    _profile(client, old, "draft", extraArgs=["--model-draft", str(source)])
    _profile(client, old, "pinned", env={"MODEL_FILE": str(source).replace("\\", "/")})
    _profile(client, old, "plain", extraArgs=["--no-mmap"])
    old, new = _move(client, source, root_b / "m.gguf")

    body = client.post(f"/v1/models/{old}/relocate", json={"to": new}).json()

    assert body["moved"] == 3
    assert len(body["warnings"]) == 2
    assert any('"draft"' in w and "extra arguments" in w for w in body["warnings"])
    assert any('"pinned"' in w and "environment" in w for w in body["warnings"])
    draft = next(
        p
        for p in client.get(f"/v1/models/{new}/profiles").json()["profiles"]
        if p["name"] == "draft"
    )
    assert draft["extraArgs"] == ["--model-draft", str(source)]  # never edited


def test_the_stored_state_survives_a_restart(
    settings: Settings, client: TestClient, root_a: Path, root_b: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"], "tuned")
    old, new = _move(client, source, root_b / "m.gguf")
    assert client.post(f"/v1/models/{old}/relocate", json={"to": new}).status_code == 200

    with TestClient(create_app(settings=settings)) as again:
        assert again.get(f"/v1/models/{old}").status_code == 404
        profiles = again.get(f"/v1/models/{new}/profiles").json()["profiles"]
        assert [p["name"] for p in profiles] == ["tuned"]


def test_storage_that_cannot_be_written_is_a_503_and_nothing_moves(
    app: FastAPI, client: TestClient, root_a: Path, root_b: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    _profile(client, _models(client)[str(source)]["id"], "tuned")
    old, new = _move(client, source, root_b / "m.gguf")
    store = app.state.state_store

    def broken() -> None:
        raise StorageUnavailable("Saved profiles are unreadable: the disk is read-only.")

    monkeypatch.setattr(store, "_write_profiles_locked", broken)

    answer = client.post(f"/v1/models/{old}/relocate", json={"to": new})

    assert answer.status_code == 503
    assert "the disk is read-only" in answer.json()["detail"]
    monkeypatch.undo()
    assert client.get(f"/v1/models/{old}").status_code == 200
    assert [p["name"] for p in client.get(f"/v1/models/{old}/profiles").json()["profiles"]] == [
        "tuned"
    ]
    assert client.get(f"/v1/models/{new}/profiles").json()["profiles"] == []


# --- the fingerprint is taken by the scan ------------------------------------


def test_a_scan_fingerprints_what_it_finds_and_keeps_it_on_a_missing_entry(
    client: TestClient, root_a: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    model = _models(client)[str(source)]
    _profile(client, model["id"])
    assert model["fingerprint"].startswith("v1:")

    source.unlink()
    _scan(client)

    assert client.get(f"/v1/models/{model['id']}").json()["fingerprint"] == model["fingerprint"]


def test_a_rescan_reads_nothing_for_an_entry_that_has_its_fingerprint(
    client: TestClient, root_a: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(root_a / "m.gguf")
    _scan(client)
    reads: list[str] = []
    real = fingerprint.of
    monkeypatch.setattr(fingerprint, "of", lambda m: reads.append(m.path) or real(m))

    assert client.post("/v1/scan").status_code in (200, 202)  # incremental
    wait_for_scan(client)

    assert reads == []


def test_an_entry_scanned_before_fingerprints_gets_one_at_the_next_scan(
    app: FastAPI, client: TestClient, root_a: Path
) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    model_id = _models(client)[str(source)]["id"]
    _forget_fingerprint(app, model_id)
    assert client.get(f"/v1/models/{model_id}").json()["fingerprint"] is None

    assert client.post("/v1/scan").status_code in (200, 202)  # incremental: a cache hit
    wait_for_scan(client)

    assert client.get(f"/v1/models/{model_id}").json()["fingerprint"].startswith("v1:")


def test_a_changed_file_gets_a_new_fingerprint(client: TestClient, root_a: Path) -> None:
    source = _write(root_a / "m.gguf")
    _scan(client)
    before = _models(client)[str(source)]["fingerprint"]

    _flip_last_byte(source)
    _scan(client)

    after = _models(client)[str(source)]["fingerprint"]
    assert after.startswith("v1:") and after != before
