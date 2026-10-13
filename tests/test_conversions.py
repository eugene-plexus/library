"""What v0.1 saved is converted once, kept as it was, and read in the current format only.

The fixtures are not look-alikes. `fixtures/v0_1/library-state.json` was written by
this library at the v0.1.0 pin (6e22230), and `fixtures/v0_1/library-config.yaml` by the config
store as it was before folders carried mounts (663189f~1), both run on Windows, so they
carry Windows line endings (the tests also try Unix ones). `fixtures/first_v0_2_build/`
was written by the code this change replaces (6bdd159): a version 2 state beside a
version 1 profile document, which is what the first v0.2 builds left behind.

Every conversion is held to the same six claims: converted correctly, a backup of the
old bytes kept, the new version recorded, a second start a no-op, a start that stopped
midway rerun cleanly, and a failure that degrades naming its cause.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_library import conversions, profile_storage
from eugene_plexus_library._generated.models import DownloadState, EngineKind, ModelProfileSpec
from eugene_plexus_library.app import create_app
from eugene_plexus_library.config import ConfigStore
from eugene_plexus_library.conversions import (
    BACKUP_SUFFIX,
    ConversionFailed,
    convert_config,
    convert_state,
)
from eugene_plexus_library.profile_storage import StorageUnavailable
from eugene_plexus_library.settings import Settings
from eugene_plexus_library.store import STATE_VERSION, StateStore

FIXTURES = Path(__file__).parent / "fixtures"
QWEN, GEMMA, TINY = "386a26e9af6dbb71", "4c26952e6cb539c3", "983b5bcf65063028"

newlines = pytest.mark.parametrize("newline", ["\r\n", "\n"], ids=["windows", "unix"])


class Crash(BaseException):
    """The process dying: nothing between the failing write and the next start runs."""


def _crash(*args: Any, **kwargs: Any) -> None:
    raise Crash


def _with_newlines(raw: bytes, newline: str) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\n", newline.encode())


def v0_1_state(directory: Path, newline: str = "\r\n") -> Path:
    path = directory / "library-state.json"
    path.write_bytes(
        _with_newlines((FIXTURES / "v0_1" / "library-state.json").read_bytes(), newline)
    )
    return path


def v0_1_config(directory: Path, newline: str = "\r\n") -> Path:
    path = directory / "config.yaml"
    path.write_bytes(
        _with_newlines((FIXTURES / "v0_1" / "library-config.yaml").read_bytes(), newline)
    )
    return path


def first_v0_2_build(directory: Path) -> Path:
    for name in ("library-state.json", "library-state.profiles.json"):
        (directory / name).write_bytes((FIXTURES / "first_v0_2_build" / name).read_bytes())
    return directory / "library-state.json"


def backups_of(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + BACKUP_SUFFIX + "*"))


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def profiles_in(state: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The profile document, by profile name, and the models it keeps."""
    profiles, identities = profile_storage.read(profile_storage.path_for(state))
    return {p["name"]: p for records in profiles.values() for p in records}, identities


class _Spy:
    """Fails the test if anything is written: what a no-op start must not do."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for owner, name in (
            (conversions, "write_private_text"),
            (conversions, "write_private_bytes"),
            (profile_storage, "write_private_text"),
        ):
            monkeypatch.setattr(owner, name, self._refuse)

    @staticmethod
    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"a second start wrote a file: {args[:1]}")


def settings_in(directory: Path) -> Settings:
    return Settings(
        config_file=directory / "config.yaml", state_file=directory / "library-state.json"
    )


def assert_no_path(text: str, directory: Path) -> None:
    """Nothing of where `directory` is: not as the system writes a path, not as
    `repr` escapes one (a Windows path in an `OSError`'s text has its backslashes
    doubled), and not even its own name, which any form of it carries."""
    assert str(directory) not in text, text
    assert repr(str(directory))[1:-1] not in text, text
    assert directory.name not in text, text


# --- the state of v0.1 --------------------------------------------------------


def test_the_versions_the_conversions_write_are_the_ones_this_release_reads() -> None:
    assert conversions.STATE_VERSION == STATE_VERSION
    assert conversions.PROFILES_VERSION == profile_storage.VERSION


@newlines
def test_the_state_v0_1_wrote_is_converted_and_nothing_is_lost(
    tmp_path: Path, newline: str
) -> None:
    state = v0_1_state(tmp_path, newline)
    before = state.read_bytes()
    original = json.loads(before)

    convert_state(state)

    # Version 2 is the record; the profiles left the file; everything else is as it was.
    converted = json.loads(state.read_text(encoding="utf-8"))
    assert converted["version"] == 2 and "profiles" not in converted
    assert {k: v for k, v in converted.items() if k != "version"} == {
        k: v for k, v in original.items() if k not in ("version", "profiles")
    }
    # The profiles are in their document, flash attention as On, Off or not set.
    by_name, identities = profiles_in(state)
    assert by_name["fast"]["flags"] == {
        "contextSize": 8192,
        "gpuLayers": 99,
        "flashAttention": "on",
    }
    assert by_name["cpu only"]["flags"] == {"gpuLayers": 0}, (
        "false is dropped, never turned into off"
    )
    assert by_name["built"]["flags"] == {
        "contextSize": 16384,
        "cacheType": "q8_0",
        "flashAttention": "on",
    }
    assert by_name["built"]["builtBy"]["flags"] == {
        "contextSize": 16384,
        "cacheType": "q8_0",
        "flashAttention": "on",
    }
    # Nothing else of a profile moved: identity, timestamps, notes, generation defaults.
    for old in (p for records in original["profiles"].values() for p in records):
        new = by_name[old["name"]]
        assert {k: v for k, v in new.items() if k not in ("flags", "builtBy")} == {
            k: v for k, v in old.items() if k not in ("flags", "builtBy")
        }
        if "builtBy" in old:
            assert {k: v for k, v in new["builtBy"].items() if k != "flags"} == {
                k: v for k, v in old["builtBy"].items() if k != "flags"
            }
    # The identities kept are those of the models that have profiles.
    assert {m["id"] for m in identities} == {QWEN, GEMMA}
    # The old bytes are kept exactly, and the document is the current version.
    assert backups_of(state) == [state.with_name(state.name + BACKUP_SUFFIX)]
    assert backups_of(state)[0].read_bytes() == before
    document = json.loads(profile_storage.path_for(state).read_text(encoding="utf-8"))
    assert document["version"] == 2


@newlines
def test_the_converted_state_loads_with_every_profile_and_download(
    tmp_path: Path, newline: str
) -> None:
    state = v0_1_state(tmp_path, newline)
    convert_state(state)

    store = StateStore(state)
    store.load()

    assert store.storage_error is None
    assert {m.id for m in store.list_models()} == {QWEN, GEMMA, TINY}
    assert [p.name for p in store.list_profiles(QWEN)] == ["fast", "cpu only"]
    fast, cpu = store.list_profiles(QWEN)
    assert fast.default and fast.flags == {
        "contextSize": 8192,
        "gpuLayers": 99,
        "flashAttention": "on",
    }
    assert cpu.flags == {"gpuLayers": 0}
    (built,) = store.list_profiles(GEMMA)
    assert built.builtBy is not None and built.builtBy.flags["flashAttention"] == "on"
    assert store.list_profiles(TINY) == []
    # A download in flight when it was written comes back paused, as it always did.
    (download,) = store.list_downloads()
    assert download.state == DownloadState.paused
    # And it is writable: a profile can be added, and the file stays version 2.
    store.delete_profile(QWEN, cpu.id)
    assert json.loads(state.read_text(encoding="utf-8"))["version"] == 2


def test_a_second_start_changes_nothing_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = v0_1_state(tmp_path)
    convert_state(state)
    kept = snapshot(tmp_path)

    _Spy(monkeypatch)
    convert_state(state)

    assert snapshot(tmp_path) == kept
    assert len(backups_of(state)) == 1


@pytest.mark.parametrize("stopped", ["after the backup", "after the profile document"])
def test_a_start_that_stopped_midway_reruns_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stopped: str
) -> None:
    state = v0_1_state(tmp_path)
    before = state.read_bytes()
    document = profile_storage.path_for(state)
    # The document is written by `profile_storage.write`, the state by the
    # converter's own write, which is the last thing it does.
    if stopped == "after the backup":
        monkeypatch.setattr(profile_storage, "write", _crash)
    else:
        monkeypatch.setattr(conversions, "write_private_text", _crash)
    with pytest.raises(Crash):
        convert_state(state)
    monkeypatch.undo()

    assert state.read_bytes() == before, "the state is only ever replaced by the finished one"
    assert document.exists() == (stopped == "after the profile document")
    left = document.read_bytes() if document.exists() else None

    convert_state(state)

    assert json.loads(state.read_text(encoding="utf-8"))["version"] == 2
    by_name, _ = profiles_in(state)
    assert by_name["fast"]["flags"]["flashAttention"] == "on"
    assert backups_of(state) == [state.with_name(state.name + BACKUP_SUFFIX)]
    assert backups_of(state)[0].read_bytes() == before
    if left is not None:
        assert document.read_bytes() == left, "the document an earlier start made is kept"
    store = StateStore(state)
    store.load()
    assert store.storage_error is None and len(store.list_profiles(QWEN)) == 2


def test_a_state_that_cannot_be_converted_is_left_as_it_was_and_the_cause_is_named(
    tmp_path: Path,
) -> None:
    state = tmp_path / "library-state.json"
    before = b'{"version": 1, "models": [], "profiles": ["unreadable tuning work"]}'
    state.write_bytes(before)

    with pytest.raises(ConversionFailed) as caught:
        convert_state(state)

    message = str(caught.value)
    assert str(state) in message and "its profiles are a list" in message
    assert "left as it was" in message and str(backups_of(state)[0]) in message
    assert state.read_bytes() == before
    assert backups_of(state)[0].read_bytes() == before
    assert not profile_storage.path_for(state).exists()
    # The store refuses writes with that cause, and says so on health.
    store = StateStore(state)
    store.load(blocked=message)
    assert store.storage_error == message
    with pytest.raises(StorageUnavailable, match="its profiles are a list"):
        store.replace_models([], scanned_at=datetime.now(UTC))


def test_a_backup_that_cannot_be_kept_stops_the_conversion_before_anything_changes(
    tmp_path: Path,
) -> None:
    state = v0_1_state(tmp_path)
    before = state.read_bytes()
    state.with_name(
        state.name + BACKUP_SUFFIX
    ).mkdir()  # something that is not a file is in the way

    with pytest.raises(ConversionFailed) as caught:
        convert_state(state)

    assert "left as it was" in str(caught.value) and BACKUP_SUFFIX in str(caught.value)
    assert state.read_bytes() == before
    assert not profile_storage.path_for(state).exists()


def test_a_write_that_fails_names_its_cause_and_the_copy_that_was_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = v0_1_state(tmp_path)
    before = state.read_bytes()

    def full(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(conversions, "write_private_text", full)
    with pytest.raises(ConversionFailed) as caught:
        convert_state(state)

    assert "No space left on device" in str(caught.value) and "OSError" in str(caught.value)
    assert str(backups_of(state)[0]) in str(caught.value)
    assert state.read_bytes() == before


def test_a_copy_that_says_something_else_is_never_written_over(tmp_path: Path) -> None:
    """It may be the only copy of what that was."""
    state = v0_1_state(tmp_path)
    before = state.read_bytes()
    earlier = state.with_name(state.name + BACKUP_SUFFIX)
    earlier.write_bytes(b"an older state, kept by an earlier conversion")

    convert_state(state)

    assert earlier.read_bytes() == b"an older state, kept by an earlier conversion"
    assert earlier.with_name(earlier.name + ".1").read_bytes() == before


# --- the profile document of the first v0.2 builds -------------------------------


def test_a_profile_document_of_version_1_is_converted_and_the_state_is_left_alone(
    tmp_path: Path,
) -> None:
    state = first_v0_2_build(tmp_path)
    document = profile_storage.path_for(state)
    state_before, document_before = state.read_bytes(), document.read_bytes()

    convert_state(state)

    assert state.read_bytes() == state_before, "a version 2 state is not rewritten"
    assert backups_of(state) == []
    assert backups_of(document) == [document.with_name(document.name + BACKUP_SUFFIX)]
    assert backups_of(document)[0].read_bytes() == document_before
    converted = json.loads(document.read_text(encoding="utf-8"))
    assert converted["version"] == 2
    by_name, _ = profiles_in(state)
    assert by_name["fast"]["flags"] == {
        "contextSize": 8192,
        "gpuLayers": 99,
        "flashAttention": "on",
    }
    assert by_name["cpu only"]["flags"] == {"gpuLayers": 0}
    # A profile that already says off still says off: only a checkbox is converted.
    assert by_name["three states"]["flags"] == {"contextSize": 16384, "flashAttention": "off"}
    assert by_name["three states"]["builtBy"]["flags"] == {
        "contextSize": 16384,
        "cacheType": "q8_0",
        "flashAttention": "on",
    }
    store = StateStore(state)
    store.load()
    assert store.storage_error is None
    assert [p.name for p in store.list_profiles(GEMMA)] == ["three states"]


def test_a_second_start_leaves_a_converted_profile_document_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = first_v0_2_build(tmp_path)
    convert_state(state)
    kept = snapshot(tmp_path)

    _Spy(monkeypatch)
    convert_state(state)

    assert snapshot(tmp_path) == kept


def test_a_profile_document_conversion_that_stopped_before_its_write_reruns_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = first_v0_2_build(tmp_path)
    document = profile_storage.path_for(state)
    before = document.read_bytes()
    monkeypatch.setattr(conversions, "write_private_text", _crash)
    with pytest.raises(Crash):
        convert_state(state)
    monkeypatch.undo()
    assert document.read_bytes() == before and len(backups_of(document)) == 1

    convert_state(state)

    assert json.loads(document.read_text(encoding="utf-8"))["version"] == 2
    assert len(backups_of(document)) == 1


def test_a_profile_document_that_cannot_be_converted_degrades_naming_the_cause(
    tmp_path: Path,
) -> None:
    state = first_v0_2_build(tmp_path)
    document = profile_storage.path_for(state)
    broken = {"version": 1, "profiles": {"a-model": "not a list of records"}, "models": []}
    document.write_text(json.dumps(broken), encoding="utf-8")
    before = document.read_bytes()

    with pytest.raises(ConversionFailed) as caught:
        convert_state(state)

    assert "the profiles of 'a-model' are not records" in str(caught.value)
    assert document.read_bytes() == before
    store = StateStore(state)
    store.load(blocked=str(caught.value))
    # The cache that was readable is still there to read, and nothing can be written.
    assert {m.id for m in store.list_models()} == {QWEN, GEMMA}
    assert store.storage_error == str(caught.value)
    assert "unsupported profile storage version" not in (store.storage_error or "")
    with pytest.raises(StorageUnavailable):
        store.create_profile(QWEN, ModelProfileSpec(name="x", engine=EngineKind.llama_cpp))


def test_a_state_of_version_1_beside_a_document_keeps_the_document(tmp_path: Path) -> None:
    """A start that made the document and stopped before it rewrote the state: the
    document is the truth, and what is embedded is never put over it."""
    state = v0_1_state(tmp_path)
    document = profile_storage.path_for(state)
    mine = {
        "id": "mine",
        "createdAt": None,
        "updatedAt": None,
        "name": "mine",
        "default": True,
        "engine": "llama_cpp",
        "flags": {"flashAttention": "off"},
    }
    profile_storage.write(document, {QWEN: [mine]}, [])
    before = document.read_bytes()

    convert_state(state)

    assert document.read_bytes() == before
    assert "profiles" not in json.loads(state.read_text(encoding="utf-8"))
    store = StateStore(state)
    store.load()
    assert [p.name for p in store.list_profiles(QWEN)] == ["mine"]


# --- what is not converted --------------------------------------------------------


def test_files_that_are_already_current_are_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = v0_1_state(tmp_path)
    convert_state(state)
    for path in backups_of(state):
        path.unlink()
    kept = snapshot(tmp_path)

    _Spy(monkeypatch)
    convert_state(state)

    assert snapshot(tmp_path) == kept and backups_of(state) == []


@pytest.mark.parametrize(
    "content",
    [b"{ this is not json", b'{"version": 999, "futureData": "keep"}', b"[1, 2, 3]", b""],
    ids=["unreadable", "newer", "not an object", "empty"],
)
def test_a_state_that_is_not_version_1_is_left_for_the_store_to_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    state = tmp_path / "library-state.json"
    state.write_bytes(content)

    _Spy(monkeypatch)
    convert_state(state)

    assert state.read_bytes() == content and backups_of(state) == []
    store = StateStore(state)
    store.load()
    assert store.storage_error is not None


def test_a_fresh_install_has_nothing_to_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Spy(monkeypatch)
    convert_state(tmp_path / "library-state.json")
    convert_config(tmp_path / "config.yaml")
    assert list(tmp_path.iterdir()) == []


# --- the library starts on the files v0.1 left ----------------------------------------


def test_the_library_converts_what_v0_1_saved_when_it_starts(tmp_path: Path) -> None:
    state = v0_1_state(tmp_path)
    config = v0_1_config(tmp_path)

    with TestClient(create_app(settings_in(tmp_path))) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        profiles = client.get(f"/v1/models/{QWEN}/profiles").json()["profiles"]
        assert {p["name"]: p["flags"] for p in profiles} == {
            "fast": {"contextSize": 8192, "gpuLayers": 99, "flashAttention": "on"},
            "cpu only": {"gpuLayers": 0},
        }
        assert client.get("/v1/config").json()["modelRoots"] == [
            {"path": "/data/models", "mounts": []},
            {"path": "D:\\gguf", "mounts": []},
        ]

    assert json.loads(state.read_text(encoding="utf-8"))["version"] == 2
    assert len(backups_of(state)) == 1 and len(backups_of(config)) == 1
    # A restart converts nothing more: no second copy, and the files that were converted
    # are not rewritten (the state is, by a download that was in flight and is parked).
    kept = snapshot(tmp_path)
    with TestClient(create_app(settings_in(tmp_path))) as client:
        assert client.get("/healthz").json()["status"] == "ok"
    after = snapshot(tmp_path)
    assert sorted(after) == sorted(kept)
    for name in ("config.yaml", "library-state.profiles.json"):
        assert after[name] == kept[name]


def test_a_failed_state_conversion_degrades_and_the_library_says_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state = tmp_path / "library-state.json"
    before = b'{"version": 1, "models": [], "profiles": {"m": {"not": "a list"}}}'
    state.write_bytes(before)
    caplog.set_level(logging.ERROR)

    with TestClient(create_app(settings_in(tmp_path))) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        cause = health["details"]["storageError"]
        assert "the profiles of 'm' are not records" in cause and str(state) in cause
        assert client.get("/v1/models").json()["models"] == []
        assert client.get("/v1/config/schema").status_code == 200

    assert state.read_bytes() == before
    assert "the profiles of 'm' are not records" in caplog.text


# --- the config of v0.1 ---------------------------------------------------------------


@newlines
def test_a_config_of_bare_strings_is_rewritten_as_folders(tmp_path: Path, newline: str) -> None:
    config = v0_1_config(tmp_path, newline)
    before = config.read_bytes()
    original = yaml.safe_load(before)
    assert original["modelRoots"] == ["/data/models", "D:\\gguf"], "the fixture is the old shape"

    convert_config(config)

    converted = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert converted["modelRoots"] == [
        {"path": "/data/models", "mounts": []},
        {"path": "D:\\gguf", "mounts": []},
    ]
    assert {k: v for k, v in converted.items() if k != "modelRoots"} == {
        k: v for k, v in original.items() if k != "modelRoots"
    }
    assert backups_of(config) == [config.with_name(config.name + BACKUP_SUFFIX)]
    assert backups_of(config)[0].read_bytes() == before
    store = ConfigStore(config)
    store.load()
    assert store.model_roots() == [Path("/data/models"), Path("D:\\gguf")]
    assert [f.model_dump() for f in store.library_folders()] == converted["modelRoots"]


def test_a_second_start_leaves_a_converted_config_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = v0_1_config(tmp_path)
    convert_config(config)
    kept = snapshot(tmp_path)

    _Spy(monkeypatch)
    convert_config(config)

    assert snapshot(tmp_path) == kept


def test_only_the_bare_strings_of_a_config_are_converted(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "scanOnStartup: false\n"
        "modelRoots:\n"
        "  - /models\n"
        "  - path: /archive\n"
        "    mounts: [/mnt/archive, 'Z:\\archive']\n"
        "  - '   '\n"
        "  - 42\n"
        "  - ' /padded '\n"
        "catalogueSources:\n"
        "  - id: hf\n"
        "    kind: hf_hub\n"
        "    token: {v: 1, n: bm9uY2U, c: c2VhbGVk}\n",
        encoding="utf-8",
    )
    original = yaml.safe_load(config.read_text(encoding="utf-8"))

    convert_config(config)

    converted = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert converted["modelRoots"] == [
        {"path": "/models", "mounts": []},
        {"path": "/archive", "mounts": ["/mnt/archive", "Z:\\archive"]},
        42,
        {"path": "/padded", "mounts": []},
    ]
    # A sealed token, and every other key, comes through the rewrite as it was.
    assert converted["catalogueSources"] == original["catalogueSources"]
    assert converted["scanOnStartup"] is False


@pytest.mark.parametrize(
    "content",
    [
        "modelRoots:\n  - path: /models\n    mounts: []\n",
        "modelRoots: []\nscanOnStartup: true\n",
        "scanOnStartup: true\n",
        "modelRoots: [unclosed\n",
        "- just\n- a list\n",
        "",
    ],
    ids=["folders", "empty list", "no roots", "unparseable", "not a mapping", "empty file"],
)
def test_a_config_with_no_bare_strings_is_not_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(content, encoding="utf-8")

    _Spy(monkeypatch)
    convert_config(config)

    assert config.read_text(encoding="utf-8") == content and backups_of(config) == []


def test_a_config_conversion_that_stopped_before_its_write_reruns_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = v0_1_config(tmp_path)
    before = config.read_bytes()
    monkeypatch.setattr(conversions, "write_private_text", _crash)
    with pytest.raises(Crash):
        convert_config(config)
    monkeypatch.undo()
    assert config.read_bytes() == before and len(backups_of(config)) == 1

    convert_config(config)

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["modelRoots"][0] == {
        "path": "/data/models",
        "mounts": [],
    }
    assert len(backups_of(config)) == 1 and backups_of(config)[0].read_bytes() == before


def test_a_config_that_cannot_be_converted_runs_on_defaults_and_names_the_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = v0_1_config(tmp_path)
    before = config.read_bytes()
    config.with_name(config.name + BACKUP_SUFFIX).mkdir()
    caplog.set_level(logging.ERROR)

    with pytest.raises(ConversionFailed) as caught:
        convert_config(config)
    assert str(config) in str(caught.value) and "left as it was" in str(caught.value)

    with TestClient(create_app(settings_in(tmp_path))) as client:
        # Degraded, not dead: the library is up and the endpoints that repair it answer.
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/config").json()["modelRoots"] == []
        result = client.patch("/v1/config", json={"modelRoots": ["/data/models"]}).json()
        assert result["applied"] == ["modelRoots"]

    assert str(config) in caplog.text and "Running on defaults" in caplog.text
    assert before != config.read_bytes(), "the repair is what replaced it"


def test_a_failed_config_conversion_shows_on_healthz_naming_the_file_and_not_where_it_is(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The library runs on defaults with no folders, and `/healthz` used to say `ok`.

    The reproduction: the state is already version 2, and the copy of the config cannot be
    kept because something else is at its name."""
    convert_state(v0_1_state(tmp_path))
    config = v0_1_config(tmp_path)
    before = config.read_bytes()
    config.with_name(config.name + BACKUP_SUFFIX).mkdir()
    caplog.set_level(logging.ERROR)

    with TestClient(create_app(settings_in(tmp_path))) as client:
        answer = client.get("/healthz")
        assert answer.status_code == 200, "degraded, not dead"
        health = answer.json()
        assert health["status"] == "degraded"
        details = health["details"]
        assert details["rootsConfigured"] == 0, "this is the library that runs on defaults"
        assert "storageError" not in details, "the state converted: this is the config's alone"
        error = details["configError"]
        # The file by its name, what was observed, and what to do; never where it is.
        assert error.startswith("config.yaml: what v0.1 saved could not be converted (")
        assert "config.yaml.v0.1.bak" in error, "the cause names the file in the way"
        assert "It is left as it was." in error and error.endswith("Fix the cause and restart.")
        assert_no_path(error, tmp_path)
        # The endpoints that repair it are up.
        assert client.get("/v1/config/schema").status_code == 200

    # The log keeps the whole path, once, with the same words.
    [line] = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert str(config) in line and line.endswith(
        "Running on defaults; fix it via /v1/config and restart."
    )
    assert config.read_bytes() == before, "a start that could not convert does not touch the file"


def test_a_start_that_converted_everything_reports_no_config_error(tmp_path: Path) -> None:
    v0_1_state(tmp_path)
    v0_1_config(tmp_path)

    with TestClient(create_app(settings_in(tmp_path))) as client:
        health = client.get("/healthz").json()

    assert health["status"] == "ok" and "configError" not in health["details"]


def test_a_start_with_nothing_to_convert_reports_no_config_error(tmp_path: Path) -> None:
    with TestClient(create_app(settings_in(tmp_path))) as client:
        health = client.get("/healthz").json()

    assert health["status"] == "ok" and "configError" not in health["details"]


def test_the_plain_sentence_names_files_by_name_where_the_log_names_them_by_path(
    tmp_path: Path,
) -> None:
    config = v0_1_config(tmp_path)
    config.with_name(config.name + BACKUP_SUFFIX).mkdir()

    with pytest.raises(ConversionFailed) as caught:
        convert_config(config)

    assert str(config) in str(caught.value), "the log's sentence is as it was"
    plain = caught.value.plain
    assert plain.startswith("config.yaml: what v0.1 saved could not be converted (")
    assert_no_path(plain, tmp_path)


def test_the_plain_sentence_for_a_failure_after_the_copy_names_the_copy_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = v0_1_config(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError(13, "Permission denied", str(config))

    monkeypatch.setattr(conversions, "write_private_text", refuse)
    with pytest.raises(ConversionFailed) as caught:
        convert_config(config)

    assert caught.value.plain == (
        "config.yaml: what v0.1 saved could not be converted "
        "(PermissionError: Permission denied (config.yaml)). It is left as it was. "
        "The copy from before is config.yaml.v0.1.bak. Fix the cause and restart."
    )
    assert str(tmp_path) in str(caught.value)
    assert_no_path(caught.value.plain, tmp_path)


def test_the_plain_sentence_for_the_state_names_it_by_name_too(tmp_path: Path) -> None:
    state = tmp_path / "library-state.json"
    state.write_bytes(b'{"version": 1, "models": [], "profiles": {"m": {"not": "a list"}}}')

    with pytest.raises(ConversionFailed) as caught:
        convert_state(state)

    assert caught.value.plain.startswith("library-state.json: what v0.1 saved could not be")
    assert "the profiles of 'm' are not records" in caught.value.plain
    assert "The copy from before is library-state.json.v0.1.bak." in caught.value.plain
    assert_no_path(caught.value.plain, tmp_path)
    assert str(state) in str(caught.value)


def test_safe_mode_does_not_convert_a_config_it_ignores(tmp_path: Path) -> None:
    config = v0_1_config(tmp_path)
    before = config.read_bytes()
    settings = settings_in(tmp_path)
    settings.safe_mode = True

    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/config").json()["modelRoots"] == []

    assert config.read_bytes() == before and backups_of(config) == []
