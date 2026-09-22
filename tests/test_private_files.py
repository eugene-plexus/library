"""The config file -- which can hold the Hugging Face token -- is 0600.

With no master key the token is on disk in plaintext (a standalone
library, or the window before the operator unlocks), so the file's mode
is the only thing between the token and every other account on the host.

Two kinds of test, because CI is Linux and this box is Windows. The
POSIX one reads the mode off the finished file -- the property itself.
The platform-independent ones watch `os.open` and assert the file was
*created* asking for `0600` through an exclusive temp; before the fix
they fail on both, because `Path.open` never calls `os.open` at all.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_library import _private_files
from eugene_plexus_library.settings import Settings

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")

TOKEN = "hf_thisIsTheOperatorsToken"


@pytest.fixture(autouse=True)
def _no_ambient_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own installed library must not shape these tests."""
    monkeypatch.delenv("EUGENE_PLEXUS_LIBRARY_DEFAULT_MODEL_ROOTS", raising=False)


class _OpenSpy:
    def __init__(self, real: Any) -> None:
        self.real = real
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, path: Any, flags: int, mode: int = 0o777, *args: Any, **kw: Any) -> int:
        self.calls.append((os.fspath(path), flags, mode))
        return self.real(path, flags, mode, *args, **kw)


@pytest.fixture
def stock_umask() -> Iterator[None]:
    """The umask a stock Linux account has, under which `open()` gives
    `0644` -- so the test cannot pass by inheriting a strict one."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _save_a_token(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"hfToken": TOKEN})
    assert response.status_code == 200, response.text
    assert "hfToken" in response.json()["applied"]


@posix_only
@pytest.mark.usefixtures("stock_umask")
def test_the_config_holding_the_token_is_0600(client: TestClient, settings: Settings) -> None:
    _save_a_token(client)

    path = settings.config_file
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["hfToken"] == TOKEN
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_config_is_created_asking_for_0600_through_an_exclusive_temp(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _OpenSpy(os.open)
    monkeypatch.setattr(os, "open", spy)

    _save_a_token(client)

    prefix = f".{settings.config_file.name}."
    temps = [(flags, mode) for path, flags, mode in spy.calls if Path(path).name.startswith(prefix)]
    assert temps, "the config was never written through a private temp"
    assert all(mode == 0o600 and flags & os.O_EXCL for flags, mode in temps)
    assert yaml.safe_load(settings.config_file.read_text(encoding="utf-8"))["hfToken"] == TOKEN
    leftovers = [p.name for p in settings.config_file.parent.iterdir() if p.name.startswith(prefix)]
    assert leftovers == []


def test_a_failed_replace_leaves_the_old_config_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("hfToken: old\n", encoding="utf-8")

    def refuse(*_: Any) -> None:
        raise OSError("disk said no")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="disk said no"):
        _private_files.write_private_text(target, "hfToken: new\n")

    assert target.read_text(encoding="utf-8") == "hfToken: old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml"]
