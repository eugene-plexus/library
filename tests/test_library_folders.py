"""Library folders carry their mounts (2026-09-14).

The claims: a bare string is a folder with no mounts wherever a folder is
expected (file, PATCH, default variable) and the wire always answers in
the object form; a mount must be absolute and shaped, and two mounts of
one shape are refused because a node takes the first of its shape; the
folders' own paths are what the scanner and the Test button use, never a
mount; and the M2-era config file a script writes by hand still loads.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from eugene_plexus_library._generated.models import ConfigUpdateRequest, ConfigUpdateResult
from eugene_plexus_library.config import ConfigStore
from eugene_plexus_library.folders import coerce_folders, folder_paths, validate_folders


def _patch(store: ConfigStore, **fields: object) -> ConfigUpdateResult:
    return store.apply_patch(ConfigUpdateRequest.model_validate(fields))


def _on_disk(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- coercion ---------------------------------------------------------------


def test_a_bare_string_is_a_folder_with_no_mounts() -> None:
    assert coerce_folders(["/models", " D:\\gguf "]) == [
        {"path": "/models", "mounts": []},
        {"path": "D:\\gguf", "mounts": []},
    ]


def test_objects_keep_their_mounts_and_junk_is_skipped_not_raised() -> None:
    value = [
        {"path": "/models", "mounts": ["/mnt/models", "\\\\NAS\\models", 7, ""]},
        {"mounts": ["/x"]},  # no path
        42,
        "",
    ]
    assert coerce_folders(value) == [
        {"path": "/models", "mounts": ["/mnt/models", "\\\\NAS\\models"]},
    ]
    assert folder_paths(value) == ["/models"]
    assert coerce_folders("not a list") == []
    assert coerce_folders(None) == []


# --- validation -------------------------------------------------------------


def test_validation_accepts_strings_and_objects_together() -> None:
    assert (
        validate_folders(
            [
                "/models",
                {"path": "/archive", "mounts": ["/mnt/archive", "Z:\\archive"]},
                {"path": "D:\\local"},
            ]
        )
        is None
    )


def test_a_mount_must_be_absolute_and_shaped() -> None:
    error = validate_folders([{"path": "/models", "mounts": ["models"]}])
    assert error is not None
    assert "absolute" in error and "'models'" in error


def test_two_mounts_of_one_shape_are_refused_because_only_the_first_could_be_used() -> None:
    error = validate_folders([{"path": "/models", "mounts": ["/mnt/a", "/mnt/b"]}])
    assert error is not None
    assert "POSIX-shaped" in error
    error = validate_folders([{"path": "/models", "mounts": ["Z:\\a", "\\\\nas\\b"]}])
    assert error is not None
    assert "Windows-shaped" in error


def test_validation_names_what_else_is_wrong() -> None:
    assert "list" in (validate_folders("/models") or "")
    assert "unknown key" in (validate_folders([{"path": "/m", "to": "/x"}]) or "")
    assert "`path`" in (validate_folders([{"mounts": []}]) or "")
    assert "`mounts` must be a list" in (validate_folders([{"path": "/m", "mounts": "/x"}]) or "")
    assert "expected a path or an object" in (validate_folders([3]) or "")


def test_duplicate_folders_are_refused_across_both_shapes(tmp_path: Path) -> None:
    error = validate_folders([str(tmp_path), {"path": str(tmp_path / "."), "mounts": []}])
    assert error is not None and "duplicates" in error


# --- the store ----------------------------------------------------------------


def test_a_file_written_for_path_list_loads_as_folders_and_is_written_back_as_such(
    tmp_path: Path,
) -> None:
    """`m11-acceptance.sh` writes `modelRoots:` as bare strings by hand;
    so did every install before this. It still loads, and the next write
    holds one shape."""
    (tmp_path / "config.yaml").write_text("modelRoots:\n  - /data/mine\n", encoding="utf-8")
    store = ConfigStore(tmp_path / "config.yaml")
    store.load()

    assert store.model_roots() == [Path("/data/mine")]
    assert [f.model_dump() for f in store.library_folders()] == [
        {"path": "/data/mine", "mounts": []}
    ]

    _patch(store, scanOnStartup=False)
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == [
        {"path": "/data/mine", "mounts": []}
    ]


def test_mounts_persist_and_the_paths_are_what_the_scanner_walks(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml")
    store.load()

    result = _patch(
        store,
        modelRoots=[{"path": "/models", "mounts": ["/mnt/models", "\\\\NAS\\models"]}],
    )

    assert result.applied == ["modelRoots"]
    assert store.model_roots() == [Path("/models")]
    assert store.library_folders()[0].mounts == ["/mnt/models", "\\\\NAS\\models"]
    on_disk = _on_disk(tmp_path / "config.yaml")
    assert on_disk["modelRoots"][0]["mounts"] == ["/mnt/models", "\\\\NAS\\models"]  # type: ignore[index]


def test_adding_mounts_to_the_environments_default_replaces_it_with_an_explicit_folder(
    tmp_path: Path,
) -> None:
    """The container says `/models`; the operator adds where the nodes
    mount it. That is an explicit list now -- same path, with mounts --
    and it is written down, because the mounts are the operator's."""
    store = ConfigStore(tmp_path / "config.yaml", default_roots=["/models"])
    store.load()
    assert store.roots_are_defaulted()

    _patch(store, modelRoots=[{"path": "/models", "mounts": ["\\\\NAS\\models"]}])

    assert not store.roots_are_defaulted()
    assert store.library_folders()[0].mounts == ["\\\\NAS\\models"]
    assert _on_disk(tmp_path / "config.yaml")["modelRoots"] == [
        {"path": "/models", "mounts": ["\\\\NAS\\models"]}
    ]


# --- the routes ---------------------------------------------------------------


def test_a_bad_mount_is_rejected_at_patch_with_the_reason(
    client: TestClient, models_dir: Path
) -> None:
    result = client.patch(
        "/v1/config", json={"modelRoots": [{"path": str(models_dir), "mounts": ["relative"]}]}
    ).json()

    assert result["applied"] == []
    assert result["rejected"][0]["key"] == "modelRoots"
    assert "absolute" in result["rejected"][0]["message"]


def test_the_test_button_stats_the_folders_own_path_and_never_a_mount(
    client: TestClient, models_dir: Path
) -> None:
    """A mount is a path on another machine; only that machine can stat
    it. Here it does not exist and must not be reported as a problem."""
    body = {
        "overrides": {"modelRoots": [{"path": str(models_dir), "mounts": ["/definitely/not/here"]}]}
    }
    result = client.post("/v1/config/test", json=body).json()

    assert result["ok"] is True
    assert "1 model directory is readable" in result["summary"]


def test_the_folders_are_a_resource_a_service_token_can_read(
    client: TestClient, models_dir: Path
) -> None:
    """A node's agent inherits its path rules from here, with a service
    token; the config trio it mirrors is operator-only."""
    client.patch(
        "/v1/config",
        json={"modelRoots": [{"path": str(models_dir), "mounts": ["\\\\NAS\\models"]}]},
    )

    body = client.get("/v1/folders").json()

    assert body == {"folders": [{"path": str(models_dir), "mounts": ["\\\\NAS\\models"]}]}
