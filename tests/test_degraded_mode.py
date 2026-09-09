"""Bad config never crashes the component.

The rule: the config endpoints are how an operator repairs a broken
install, so refusing to start over a broken config locks them out of the
fix. Every case here has to end with a live HTTP surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library.app import create_app
from eugene_plexus_library.settings import Settings

from .conftest import qwen_like_kv, write_gguf


def _client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as client:
        yield client


@pytest.fixture
def settings_at(tmp_path: Path) -> Settings:
    return Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
    )


def test_a_config_file_that_is_not_a_mapping_still_starts(settings_at: Settings) -> None:
    settings_at.config_file.write_text("- just\n- a list\n", encoding="utf-8")

    for client in _client(settings_at):
        assert client.get("/healthz").status_code == 200
        # And the endpoint that fixes it is reachable.
        assert client.get("/v1/config/schema").status_code == 200
        assert client.patch("/v1/config", json={"scanOnStartup": False}).status_code == 200


def test_unparseable_yaml_still_starts(settings_at: Settings) -> None:
    settings_at.config_file.write_text("modelRoots: [unclosed\n", encoding="utf-8")

    for client in _client(settings_at):
        assert client.get("/healthz").status_code == 200


def test_a_root_that_is_not_a_string_does_not_stop_the_scan(
    settings_at: Settings, models_dir: Path
) -> None:
    """A hand-edited config file must not be able to wedge the walk."""
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    settings_at.config_file.write_text(
        f"modelRoots:\n  - 42\n  - {models_dir.as_posix()}\n", encoding="utf-8"
    )

    for client in _client(settings_at):
        for _ in range(200):
            body = client.get("/v1/scan").json()
            if body["state"] != "scanning":
                break
        assert client.get("/healthz").status_code == 200
        assert len(client.get("/v1/models").json()["models"]) == 1


def test_a_corrupt_state_file_still_starts(settings_at: Settings) -> None:
    """Losing a scan cache is free. Refusing to boot over one is not."""
    settings_at.state_file.write_text("{{{ not json", encoding="utf-8")

    for client in _client(settings_at):
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").json()["models"] == []


def test_safe_mode_ignores_the_config_file(settings_at: Settings, models_dir: Path) -> None:
    """Safe mode boots from defaults so a config that breaks startup can
    be repaired through the API."""
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    settings_at.config_file.write_text(
        f"modelRoots:\n  - {models_dir.as_posix()}\n", encoding="utf-8"
    )
    settings_at.safe_mode = True

    for client in _client(settings_at):
        body = client.get("/healthz").json()
        assert body["safeMode"] is True
        assert body["status"] == "degraded"
        # No roots were loaded, so nothing was scanned.
        assert client.get("/v1/models").json()["models"] == []
        assert client.get("/v1/config").json()["modelRoots"] == []


def test_safe_mode_still_writes_repairs_through_to_disk(
    settings_at: Settings, models_dir: Path
) -> None:
    """The repair has to survive the next boot, or safe mode is a
    read-only dead end."""
    settings_at.safe_mode = True

    for client in _client(settings_at):
        assert client.patch("/v1/config", json={"modelRoots": [str(models_dir)]}).status_code == 200

    assert str(models_dir) in settings_at.config_file.read_text(encoding="utf-8")
