"""The model directories can default from the environment.

`EUGENE_PLEXUS_LIBRARY_DEFAULT_MODEL_ROOTS` is what a container image
sets, because it knows its models volume is at `/models` before anyone
has opened Config. The claims under test: it is a *default* of the
`modelRoots` field and never an override (the schema reports it, an
explicit list replaces it, clearing returns to it), it is never written
to the config file, and an install whose file predates the variable
picks it up -- which is every container that came up before this
existed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_library._generated.models import ConfigUpdateRequest
from eugene_plexus_library.app import create_app
from eugene_plexus_library.config import DEFAULT_ROOTS_VARIABLE, ConfigStore, as_schema
from eugene_plexus_library.settings import Settings

from .conftest import qwen_like_kv, write_gguf

# --- Settings ---------------------------------------------------------------


def test_the_variable_is_pathsep_separated_like_path_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """`os.pathsep`, because a Windows path has a colon in it."""
    monkeypatch.setenv(DEFAULT_ROOTS_VARIABLE, os.pathsep.join(["/models", " /srv/more ", ""]))

    assert Settings().default_roots() == ["/models", "/srv/more"]


def test_unset_means_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_ROOTS_VARIABLE, raising=False)

    assert Settings().default_roots() == []
    assert Settings(default_model_roots="").default_roots() == []


# --- ConfigStore ------------------------------------------------------------


def _patch(store: ConfigStore, **fields: object) -> None:
    store.apply_patch(ConfigUpdateRequest.model_validate(fields))


def _on_disk(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_a_fresh_install_uses_the_default_and_does_not_write_it_down(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()

    assert store.model_roots() == [Path("/models")]
    # The wire carries the object form, whatever shape the default was given in.
    assert store.as_document().model_dump()["modelRoots"] == [{"path": "/models", "mounts": []}]
    assert store.roots_are_defaulted()
    # The file records what the operator chose, which is nothing yet.
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == []


def test_a_file_written_before_the_variable_existed_picks_it_up(tmp_path: Path) -> None:
    """Every container that came up before this default did has
    `modelRoots: []` on disk from its first boot. That must not pin it
    to an empty library."""
    (tmp_path / "config.yaml").write_text("modelRoots: []\nscanOnStartup: true\n", encoding="utf-8")
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()

    assert store.model_roots() == [Path("/models")]
    assert store.roots_are_defaulted()


def test_directories_the_operator_set_win(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("modelRoots:\n  - /data/mine\n", encoding="utf-8")
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()

    assert store.model_roots() == [Path("/data/mine")]
    assert not store.roots_are_defaulted()


def test_an_explicit_list_replaces_the_default_and_is_persisted(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()

    _patch(store, modelRoots=["/data/mine"])

    assert store.model_roots() == [Path("/data/mine")]
    assert not store.roots_are_defaulted()
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == [
        {"path": "/data/mine", "mounts": []}
    ]


@pytest.mark.parametrize("cleared", [None, []], ids=["null", "empty-list"])
def test_clearing_the_list_returns_to_the_default(tmp_path: Path, cleared: object) -> None:
    """`null` clears to default everywhere in this protocol. Here `[]`
    does too: with the environment naming the directories, "none" is not
    a state this install can be in, only "these others"."""
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()
    _patch(store, modelRoots=["/data/mine"])

    _patch(store, modelRoots=cleared)

    assert store.model_roots() == [Path("/models")]
    assert store.roots_are_defaulted()
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == []


def test_without_a_default_an_empty_list_is_an_empty_list(tmp_path: Path) -> None:
    """The behaviour every bare-metal install has had since M2 is
    unchanged when the variable is unset."""
    store = ConfigStore(tmp_path / "config.yaml")
    store.load()
    _patch(store, modelRoots=["/data/mine"])

    _patch(store, modelRoots=[])

    assert store.model_roots() == []
    assert not store.roots_are_defaulted()
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == []


def test_the_default_survives_a_reload_after_an_unrelated_write(tmp_path: Path) -> None:
    """A PATCH to another field rewrites the file; the default must not
    be baked in by that write."""
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()
    _patch(store, scanOnStartup=False)
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == []

    later = ConfigStore(tmp_path / "config.yaml", default_roots=["/mnt/elsewhere"])
    later.load()

    assert later.model_roots() == [Path("/mnt/elsewhere")]


def test_the_schema_reports_this_installs_default(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    field = next(f for f in store.schema().fields if f.key == "modelRoots")

    assert field.default == [{"path": "/models", "mounts": []}]
    assert DEFAULT_ROOTS_VARIABLE in (field.description or "")
    assert field.valueType.value == "library_folders"


def test_the_schema_without_a_default_is_unchanged() -> None:
    field = next(f for f in as_schema().fields if f.key == "modelRoots")

    assert field.default == []
    assert DEFAULT_ROOTS_VARIABLE not in (field.description or "")


# --- through the app --------------------------------------------------------


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
        **overrides,  # type: ignore[arg-type]
    )


def _client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as client:
        yield client


def _wait_for_scan(client: TestClient) -> dict[str, object]:
    for _ in range(200):
        body = client.get("/v1/scan").json()
        if body["state"] != "scanning":
            return body
    raise AssertionError("scan never finished")


def test_a_default_directory_is_scanned_with_nobody_having_opened_config(
    tmp_path: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container's claim end to end: mount the directory, and the
    model in it is catalogued at startup."""
    write_gguf(models_dir / "a-Q4_K_M.gguf", qwen_like_kv(name="A"))
    monkeypatch.setenv(DEFAULT_ROOTS_VARIABLE, str(models_dir))

    for client in _client(_settings(tmp_path)):
        _wait_for_scan(client)
        assert client.get("/v1/config").json()["modelRoots"] == [
            {"path": str(models_dir), "mounts": []}
        ]
        health = client.get("/healthz").json()
        assert health["status"] == "ok"
        assert health["details"]["rootsConfigured"] == 1
        assert [m["name"] for m in client.get("/v1/models").json()["models"]] == ["a-Q4_K_M"]
        schema = client.get("/v1/config/schema").json()
        assert next(f for f in schema["fields"] if f["key"] == "modelRoots")["default"] == [
            {"path": str(models_dir), "mounts": []}
        ]


def test_a_default_directory_that_is_not_there_is_reported_not_invented(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing mounted at /models: the root is configured, the scan says
    it is missing, health goes degraded naming it, and the boot log says
    what that means in a container. Explained, not predicted."""
    missing = tmp_path / "not-mounted"
    caplog.set_level(logging.INFO, logger="eugene_plexus_library.app")

    for client in _client(_settings(tmp_path, default_model_roots=str(missing))):
        _wait_for_scan(client)
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["details"]["rootsConfigured"] == 1
        assert health["details"]["unreadableRoots"] == [str(missing)]

    assert "nothing is mounted there" in caplog.text
    assert DEFAULT_ROOTS_VARIABLE in caplog.text


def test_the_operators_own_directories_silence_the_default(
    tmp_path: Path, models_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "config.yaml").write_text(
        f"modelRoots:\n  - {models_dir.as_posix()}\n", encoding="utf-8"
    )
    caplog.set_level(logging.INFO, logger="eugene_plexus_library.app")

    for client in _client(_settings(tmp_path, default_model_roots="/models")):
        assert client.get("/v1/config").json()["modelRoots"] == [
            {"path": models_dir.as_posix(), "mounts": []}
        ]

    assert "is not in use" in caplog.text
