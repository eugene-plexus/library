"""LS10: the library writes a preparation's files itself (library-sources-and-engines.md §6.13).

The engine's node prepares the model in a folder of its own and sends the
files here a chunk at a time, under the run's lease (B104); the library
writes only inside an engine's own folder at the top of the run's Library
folder (B105), so a node never needs to be able to write in a Library folder.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import prepared, prepared_uploads
from eugene_plexus_library.prepared_uploads import UploadRefused, complete, state, target, write
from eugene_plexus_library.run_operations import Checkpoint

from .test_run_preparation import _body, _submit, _to_preparing, prepared_client  # noqa: F401

CONFIG = b'{"args": ["--pack", "packs/iq2_xs"]}'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- where the library writes (B105) ---------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "C:/Windows/x",
        "Strata-data\\x",
        "Strata-data/../ISTA-DASLab/x.gguf",
        "Strata-data/./x",
        "Strata-data//x",
        "Strata-data",
        "ISTA-DASLab/repo/x.gguf.done",
        "-data/x",
        "Strata-data/m.eugene-prepared.json",
        "Strata-data/x.bin.eugene-upload",
        "Strata-data/.eugene-engine-files",
        "Strata-data/CON.txt",
        "Strata-data/ x",
    ],
)
def test_a_preparation_writes_only_inside_an_engines_own_folder(tmp_path: Path, path: str):
    with pytest.raises(UploadRefused) as refused:
        target(tmp_path, path)
    assert refused.value.status == 400
    assert not any(tmp_path.iterdir())


def test_the_engines_own_folder_is_made_and_marked_when_there_is_none(tmp_path: Path):
    found = target(tmp_path, "Strata-data/packs/iq2_xs/dense.bin")
    assert write(found, 0, b"abc") == 3
    own = tmp_path / "Strata-data"
    assert (own / prepared.ENGINE_FILES_MARKER).is_file()
    # Bytes in flight never carry the file's own name.
    assert not found.path.exists()
    assert (own / "packs" / "iq2_xs" / "dense.bin.eugene-upload").read_bytes() == b"abc"


def test_a_folder_of_that_name_the_person_made_is_left_alone(tmp_path: Path):
    mine = tmp_path / "Strata-data"
    mine.mkdir()
    (mine / "notes.txt").write_text("mine", encoding="utf-8")
    found = target(tmp_path, "Strata-data/strata-iq2_xs.json")
    with pytest.raises(UploadRefused) as refused:
        write(found, 0, CONFIG)
    assert refused.value.status == 409
    assert "yours" in refused.value.detail
    assert sorted(p.name for p in mine.iterdir()) == ["notes.txt"]


def test_nothing_is_written_through_a_link(tmp_path: Path):
    own = tmp_path / "Strata-data"
    own.mkdir()
    (own / prepared.ENGINE_FILES_MARKER).write_text("engine files", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, own / "packs", target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi  # a junction needs no privilege

        _winapi.CreateJunction(str(outside), str(own / "packs"))
    found = target(tmp_path, "Strata-data/packs/iq2_xs/dense.bin")
    with pytest.raises(UploadRefused) as refused:
        write(found, 0, b"abc")
    assert refused.value.status == 409
    assert not any(outside.iterdir())


# --- how a file arrives (B104) ---------------------------------------------


def test_a_file_arrives_in_chunks_and_is_whole_under_its_own_name(tmp_path: Path):
    data = os.urandom(5000)
    found = target(tmp_path, "Strata-data/packs/iq2_xs/dense.bin")
    assert state(found) == {"sizeBytes": None, "sha256": None, "receivedBytes": 0}
    write(found, 0, data[:3000])
    assert state(found)["receivedBytes"] == 3000
    write(found, 3000, data[3000:])
    assert not found.path.exists()
    held = complete(found, len(data), _sha(data))
    assert held == {"sizeBytes": 5000, "sha256": _sha(data), "receivedBytes": 0}
    assert found.path.read_bytes() == data
    assert not found.partial.exists()
    # Said again (a lost answer): nothing changes.
    assert complete(found, len(data), _sha(data))["sizeBytes"] == 5000


def test_a_chunk_at_the_wrong_offset_says_where_to_send_from(tmp_path: Path):
    found = target(tmp_path, "Strata-data/x.bin")
    write(found, 0, b"12345")
    with pytest.raises(UploadRefused) as refused:
        write(found, 2, b"345")
    assert refused.value.status == 409
    assert "send from 5" in refused.value.detail
    # 0 starts it again.
    write(found, 0, b"ab")
    assert found.partial.read_bytes() == b"ab"


def test_a_file_that_arrived_damaged_or_short_is_not_put_in_place(tmp_path: Path):
    found = target(tmp_path, "Strata-data/x.bin")
    write(found, 0, b"12345")
    with pytest.raises(UploadRefused) as short:
        complete(found, 6, _sha(b"123456"))
    assert short.value.status == 409 and "send the rest" in short.value.detail
    with pytest.raises(UploadRefused) as damaged:
        complete(found, 5, _sha(b"54321"))
    assert damaged.value.status == 409 and "damaged" in damaged.value.detail
    assert not found.path.exists() and not found.partial.exists()
    with pytest.raises(UploadRefused) as nothing:
        complete(found, 5, _sha(b"12345"))
    assert "Nothing of x.bin has arrived" in nothing.value.detail


def test_a_file_sent_again_replaces_the_one_in_place(tmp_path: Path):
    found = target(tmp_path, "Strata-data/strata-iq2_xs.json")
    write(found, 0, b"old")
    complete(found, 3, _sha(b"old"))
    write(found, 0, CONFIG)
    complete(found, len(CONFIG), _sha(CONFIG))
    assert found.path.read_bytes() == CONFIG


# --- the routes, under the run's lease ---------------------------------------


def _send(client: TestClient, lease: str, path: str, data: bytes, *, chunk: int = 4) -> None:
    for offset in range(0, len(data), chunk):
        response = client.put(
            "/v1/run-operations/p/files",
            params={"path": path, "offset": offset, "lease": lease},
            content=data[offset : offset + chunk],
            headers={"content-type": "application/octet-stream"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["receivedBytes"] == min(offset + chunk, len(data))
    response = client.post(
        "/v1/run-operations/p/files/complete",
        json={"lease": lease, "path": path, "sizeBytes": len(data), "sha256": _sha(data)},
    )
    assert response.status_code == 200, response.text


def test_the_node_sends_the_files_and_the_run_lists_the_model(
    prepared_client: TestClient,  # noqa: F811
    models_dir: Path,
):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    held = prepared_client.post(
        "/v1/run-operations/p/files/state",
        json={"lease": lease, "path": "Strata-data/packs/iq2_xs/dense.bin"},
    )
    assert held.status_code == 200, held.text
    assert held.json() == {
        "path": "Strata-data/packs/iq2_xs/dense.bin",
        "sizeBytes": None,
        "sha256": None,
        "receivedBytes": 0,
    }
    _send(prepared_client, lease, "Strata-data/packs/iq2_xs/dense.bin", b"0123456789")
    _send(prepared_client, lease, "Strata-data/strata-iq2_xs.json", CONFIG)
    data = models_dir / "Strata-data"
    assert (data / "packs" / "iq2_xs" / "dense.bin").read_bytes() == b"0123456789"
    assert (data / "strata-iq2_xs.json").read_bytes() == CONFIG
    again = prepared_client.post(
        "/v1/run-operations/p/files/state",
        json={"lease": lease, "path": "Strata-data/strata-iq2_xs.json"},
    ).json()
    assert again["sizeBytes"] == len(CONFIG) and again["sha256"] == _sha(CONFIG)
    listed = prepared_client.post("/v1/run-operations/p/prepared", json=_body(lease, models_dir))
    assert listed.status_code == 200, listed.text
    assert listed.json()["model"]["format"] == "prepared"


def test_files_are_written_only_for_a_run_preparing_under_its_lease(
    prepared_client: TestClient,  # noqa: F811
    models_dir: Path,
):
    _submit(prepared_client)
    jobs = prepared_client.app.state.run_operations
    checking = jobs.claim("p", None)["lease"]
    put = {"path": "Strata-data/x.bin", "offset": 0}
    early = prepared_client.put(
        "/v1/run-operations/p/files", params={**put, "lease": checking}, content=b"x"
    )
    assert early.status_code == 409
    assert "not preparing" in early.json()["detail"]
    jobs.checkpoint("p", None, Checkpoint(lease=checking, step="preparing", engine="strata"))
    lease = jobs.claim("p", None)["lease"]
    stale = prepared_client.put(
        "/v1/run-operations/p/files", params={**put, "lease": checking}, content=b"x"
    )
    assert stale.status_code == 409
    assert "lease" in stale.json()["detail"]
    outside = prepared_client.put(
        "/v1/run-operations/p/files",
        params={"path": "ISTA-DASLab/repo/x.gguf.done", "offset": 0, "lease": lease},
        content=b"x",
    )
    assert outside.status_code == 400
    assert not (models_dir / "Strata-data" / "x.bin.eugene-upload").exists()
    assert not (models_dir / "ISTA-DASLab" / "repo" / "x.gguf.done").exists()


def test_a_chunk_larger_than_a_proxy_carries_is_refused(
    prepared_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    monkeypatch.setattr(prepared_uploads, "MAX_CHUNK_BYTES", 8)
    response = prepared_client.put(
        "/v1/run-operations/p/files",
        params={"path": "Strata-data/x.bin", "offset": 0, "lease": lease},
        content=b"123456789",
    )
    assert response.status_code == 413


def test_a_chunk_sent_without_its_length_is_counted_as_it_arrives(
    prepared_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    monkeypatch.setattr(prepared_uploads, "MAX_CHUNK_BYTES", 8)

    def unmeasured():
        yield b"12345"
        yield b"6789"

    response = prepared_client.put(
        "/v1/run-operations/p/files",
        params={"path": "Strata-data/x.bin", "offset": 0, "lease": lease},
        content=unmeasured(),
    )
    assert response.status_code == 413


def test_a_run_whose_library_folder_is_gone_writes_nothing(
    prepared_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _submit(prepared_client)
    lease = _to_preparing(prepared_client)
    other = tmp_path / "another-folder"
    other.mkdir()
    monkeypatch.setattr(prepared_client.app.state.config_store, "model_roots", lambda: [other])
    response = prepared_client.put(
        "/v1/run-operations/p/files",
        params={"path": "Strata-data/x.bin", "offset": 0, "lease": lease},
        content=b"x",
    )
    assert response.status_code == 409
    assert "not one of the Library's folders" in response.json()["detail"]
    assert not any(other.iterdir())
