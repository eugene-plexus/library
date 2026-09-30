"""Settings never lie (Troy, 2026-09-29: fundamental).

For the library's config trio:

- **A null in the file is the default**: `catalogueEnabled: null` turned
  the catalogue off while the schema said it was on.
- **An empty token is no token**: it read `"<redacted>"`, so a UI said a
  token was saved that was not. An empty hub address is the hub.
- **A restart is pending only for this PATCH's keys**, while they differ
  from what the process runs on, and the schema says which (`inEffect`).
- **The environment's default roots say where they come from**
  (`defaultSource`) rather than in a sentence appended to the description.
- **A saved token reaches downloads at once**, not only the next search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eugene_plexus_library._generated.models import ConfigUpdateRequest
from eugene_plexus_library.config import DEFAULT_ROOTS_VARIABLE, ConfigStore


def _store(tmp_path: Path, content: dict[str, Any] | None = None, **kwargs: Any) -> ConfigStore:
    path = tmp_path / "library.yaml"
    if content is not None:
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
    store = ConfigStore(path, **kwargs)
    store.load()
    return store


def _patch(store: ConfigStore, body: dict[str, Any]):  # type: ignore[no-untyped-def]
    return store.apply_patch(ConfigUpdateRequest.model_validate(body))


def _field(store: ConfigStore, key: str):  # type: ignore[no-untyped-def]
    return next(f for f in store.schema().fields if f.key == key)


def test_a_null_in_the_file_is_the_default(tmp_path: Path) -> None:
    store = _store(tmp_path, {"catalogueEnabled": None, "scanOnStartup": None})
    doc = store.as_document().model_dump()
    assert doc["catalogueEnabled"] is True and doc["scanOnStartup"] is True


def test_an_empty_token_is_no_token(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _patch(store, {"hfToken": "", "catalogueBaseUrl": ""})
    doc = store.as_document().model_dump()
    assert doc.get("hfToken") is None, "an empty token read as saved"
    assert doc["catalogueBaseUrl"] == "https://huggingface.co"
    _patch(store, {"hfToken": "hf_real"})
    assert store.as_document().model_dump()["hfToken"] == "<redacted>"


def test_a_restart_is_pending_only_while_the_value_differs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _patch(store, {"logLevel": "DEBUG"})
    assert first.requiresRestart is True and first.pendingRestart == ["logLevel"]
    field = _field(store, "logLevel")
    assert field.pendingRestart is True and field.inEffect == "INFO"
    second = _patch(store, {"maxConcurrentDownloads": 2})
    assert second.requiresRestart is False and second.pendingRestart == []
    _patch(store, {"logLevel": "INFO"})
    assert store.pending_restart() == {}


def test_the_default_roots_say_where_they_come_from(tmp_path: Path) -> None:
    store = _store(tmp_path, default_roots=["/models"])
    field = _field(store, "modelRoots")
    assert field.default == [{"path": "/models", "mounts": []}]
    assert field.defaultSource and DEFAULT_ROOTS_VARIABLE in field.defaultSource
    plain = _field(_store(tmp_path / "other"), "modelRoots")
    assert plain.defaultSource is None and plain.unsetMeans


def test_a_saved_token_reaches_the_hub_client_at_once(client: Any) -> None:
    saved = client.patch("/v1/config", json={"hfToken": "hf_new"})
    assert saved.status_code == 200, saved.text
    hub = client.app.state.hub_client
    assert hub._token == "hf_new"
