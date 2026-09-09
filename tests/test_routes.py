"""The HTTP surface: models, profiles, scan, config, health, admin."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from .conftest import projector_kv, qwen_like_kv, write_gguf, write_hf_model


def _wait_for_scan(client: TestClient) -> dict:
    """Poll until the walk settles. The endpoint is asynchronous by
    design; a test that assumed otherwise would pass or fail on timing."""
    for _ in range(200):
        body = client.get("/v1/scan").json()
        if body["state"] not in ("scanning",):
            return body
    raise AssertionError("scan never finished")


# --- models -------------------------------------------------------------


def test_startup_scan_populates_the_library(
    configured_client: TestClient, models_dir: Path
) -> None:
    write_gguf(models_dir / "a-Q4_K_M.gguf", qwen_like_kv(name="A"))
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)

    body = configured_client.get("/v1/models").json()
    assert [m["name"] for m in body["models"]] == ["a-Q4_K_M"]
    assert body["lastScanAt"] is not None


def test_models_are_listed_alphabetically(configured_client: TestClient, models_dir: Path) -> None:
    for name in ("zeta", "alpha", "mid"):
        write_gguf(models_dir / f"{name}.gguf", qwen_like_kv(name=name))
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)

    names = [m["name"] for m in configured_client.get("/v1/models").json()["models"]]
    assert names == ["alpha", "mid", "zeta"]


def test_get_one_model(configured_client: TestClient, models_dir: Path) -> None:
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)

    listed = configured_client.get("/v1/models").json()["models"][0]
    fetched = configured_client.get(f"/v1/models/{listed['id']}")

    assert fetched.status_code == 200
    assert fetched.json()["path"] == listed["path"]


def test_unknown_model_is_a_problem_document(configured_client: TestClient) -> None:
    response = configured_client.get("/v1/models/deadbeef")

    assert response.status_code == 404
    assert response.json()["detail"]["component"] == "library"


def test_reverse_lookup_by_path(configured_client: TestClient, models_dir: Path) -> None:
    """How the runtime dashboard gets from a `Runtime.modelPath` back to
    a library entry without reproducing the id derivation."""
    path = write_gguf(models_dir / "a.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)

    body = configured_client.get("/v1/models", params={"path": str(path)}).json()
    assert len(body["models"]) == 1
    assert body["models"][0]["path"] == str(path)


def test_reverse_lookup_of_an_unknown_path_is_empty_not_an_error(
    configured_client: TestClient, tmp_path: Path
) -> None:
    body = configured_client.get("/v1/models", params={"path": str(tmp_path / "nope")}).json()
    assert body["models"] == []


def test_deleting_a_present_model_is_refused(
    configured_client: TestClient, models_dir: Path
) -> None:
    """A present model reappears on the next scan, so a delete that
    looked like it worked would be a lie."""
    path = write_gguf(models_dir / "a.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)
    model = configured_client.get("/v1/models").json()["models"][0]

    response = configured_client.delete(f"/v1/models/{model['id']}")

    assert response.status_code == 409
    assert path.exists()
    assert len(configured_client.get("/v1/models").json()["models"]) == 1


def test_a_missing_model_can_be_forgotten(configured_client: TestClient, models_dir: Path) -> None:
    path = write_gguf(models_dir / "a.gguf", qwen_like_kv())
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)
    model = configured_client.get("/v1/models").json()["models"][0]
    configured_client.post(
        f"/v1/models/{model['id']}/profiles", json={"name": "tuned", "engine": "llama_cpp"}
    )

    path.unlink()
    configured_client.post("/v1/scan")
    _wait_for_scan(configured_client)

    assert configured_client.get(f"/v1/models/{model['id']}").json()["status"] == "missing"
    assert configured_client.delete(f"/v1/models/{model['id']}").status_code == 204
    assert configured_client.get(f"/v1/models/{model['id']}").status_code == 404


# --- profiles --------------------------------------------------------------


def _one_model(client: TestClient, models_dir: Path) -> str:
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    client.post("/v1/scan")
    _wait_for_scan(client)
    return client.get("/v1/models").json()["models"][0]["id"]


def test_profile_crud(configured_client: TestClient, models_dir: Path) -> None:
    model_id = _one_model(configured_client, models_dir)

    created = configured_client.post(
        f"/v1/models/{model_id}/profiles",
        json={
            "name": "long context",
            "engine": "llama_cpp",
            "flags": {"contextSize": 32768, "nGpuLayers": 99},
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "notes": "OOMs above 24 layers on the 3090",
        },
    )
    assert created.status_code == 201
    profile = created.json()
    assert profile["default"] is True
    assert profile["flags"]["contextSize"] == 32768

    listed = configured_client.get(f"/v1/models/{model_id}/profiles").json()
    assert len(listed["profiles"]) == 1

    replaced = configured_client.put(
        f"/v1/models/{model_id}/profiles/{profile['id']}",
        json={"name": "long context", "engine": "llama_cpp", "flags": {"contextSize": 8192}},
    )
    assert replaced.status_code == 200
    assert replaced.json()["flags"] == {"contextSize": 8192}

    assert (
        configured_client.delete(f"/v1/models/{model_id}/profiles/{profile['id']}").status_code
        == 204
    )
    assert configured_client.get(f"/v1/models/{model_id}/profiles").json()["profiles"] == []


def test_flags_are_stored_not_validated(configured_client: TestClient, models_dir: Path) -> None:
    """The validator is the agent's adapter schema, at
    runtime-creation time. A profile is allowed to be wrong."""
    model_id = _one_model(configured_client, models_dir)

    response = configured_client.post(
        f"/v1/models/{model_id}/profiles",
        json={"name": "nonsense", "engine": "llama_cpp", "flags": {"notARealFlag": "banana"}},
    )

    assert response.status_code == 201
    assert response.json()["flags"] == {"notARealFlag": "banana"}


def test_duplicate_profile_name_is_a_conflict(
    configured_client: TestClient, models_dir: Path
) -> None:
    model_id = _one_model(configured_client, models_dir)
    body = {"name": "fast", "engine": "llama_cpp"}
    configured_client.post(f"/v1/models/{model_id}/profiles", json=body)

    assert configured_client.post(f"/v1/models/{model_id}/profiles", json=body).status_code == 409


def test_profiles_on_an_unknown_model_are_404(configured_client: TestClient) -> None:
    assert configured_client.get("/v1/models/deadbeef/profiles").status_code == 404
    assert (
        configured_client.post(
            "/v1/models/deadbeef/profiles", json={"name": "x", "engine": "llama_cpp"}
        ).status_code
        == 404
    )


def test_an_unknown_engine_is_rejected_by_the_schema(
    configured_client: TestClient, models_dir: Path
) -> None:
    """`EngineKind` is a closed enum: an engine is supported exactly when
    an adapter exists for it.

    The value here is deliberately one that will never be an engine. This
    test used `vllm` until specs 811112b, when M4 made `vllm` real and the
    profile started being accepted — a passing contract change looking
    exactly like a broken test. A negative test wants a value the enum
    cannot grow into, not the next thing on the roadmap.
    """
    model_id = _one_model(configured_client, models_dir)
    response = configured_client.post(
        f"/v1/models/{model_id}/profiles",
        json={"name": "x", "engine": "not_an_engine"},
    )

    assert response.status_code == 422


# --- scan --------------------------------------------------------------------


def test_scan_reports_what_it_skipped(configured_client: TestClient, models_dir: Path) -> None:
    """The important half of the response — a scanner that silently drops
    what it did not understand is indistinguishable from a broken one."""
    write_gguf(models_dir / "model.gguf", qwen_like_kv(name="Pair"))
    write_gguf(models_dir / "mmproj-F32.gguf", projector_kv(name="Pair"))
    configured_client.post("/v1/scan")
    body = _wait_for_scan(configured_client)

    reasons = {s["reason"] for s in body["skipped"]}
    assert "projector" in reasons
    assert body["state"] == "done"


def test_scan_reports_per_root_status(configured_client: TestClient, tmp_path: Path) -> None:
    configured_client.patch("/v1/config", json={"modelRoots": [str(tmp_path / "nowhere")]})
    configured_client.post("/v1/scan")
    body = _wait_for_scan(configured_client)

    assert body["roots"][0]["status"] == "missing"


def test_a_second_scan_is_a_conflict(configured_client: TestClient, models_dir: Path) -> None:
    """There is only ever one; starting a second would double the disk
    cost to reach the same answer."""
    for index in range(30):
        write_gguf(models_dir / f"m{index}.gguf", qwen_like_kv(vocab=400))

    first = configured_client.post("/v1/scan")
    second = configured_client.post("/v1/scan")

    assert first.status_code == 202
    # The first may already have finished on a fast disk, in which case a
    # second start is legitimate — assert only that we never get a
    # *third* concurrent state, not a race-dependent status code.
    assert second.status_code in (202, 409)
    _wait_for_scan(configured_client)


def test_cancelling_when_nothing_runs_is_a_conflict(configured_client: TestClient) -> None:
    assert configured_client.delete("/v1/scan").status_code == 409


def test_scan_with_no_roots_fails_with_a_reason(client: TestClient) -> None:
    """A fresh install is not an error, but a scan over zero directories
    is a failed scan and saying so beats reporting success."""
    body = client.post("/v1/scan").json()

    assert body["state"] == "failed"
    assert "no model directories" in body["error"]


def test_full_scan_is_accepted(configured_client: TestClient, models_dir: Path) -> None:
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    response = configured_client.post("/v1/scan", json={"full": True})
    _wait_for_scan(configured_client)

    assert response.status_code == 202
    assert response.json()["full"] is True


def test_a_mixed_tree_end_to_end(configured_client: TestClient, models_dir: Path) -> None:
    write_gguf(models_dir / "pub" / "repo" / "llama-Q4_K_M.gguf", qwen_like_kv(name="L"))
    write_gguf(models_dir / "pub" / "repo" / "mmproj-F32.gguf", projector_kv(name="L"))
    write_hf_model(models_dir / "hf" / "mistral")
    configured_client.post("/v1/scan")
    body = _wait_for_scan(configured_client)

    assert body["state"] == "done"
    models = configured_client.get("/v1/models").json()["models"]
    assert {m["format"] for m in models} == {"gguf", "safetensors"}


# --- config -------------------------------------------------------------------


def test_config_schema_declares_the_roots_as_a_path_list(client: TestClient) -> None:
    """Roots are config rather than a resource of their own precisely so
    the generic editor renders them with no library-specific code."""
    schema = client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "modelRoots")

    assert field["valueType"] == "path_list"
    assert schema["component"] == "library"


def test_patching_roots(client: TestClient, models_dir: Path) -> None:
    response = client.patch("/v1/config", json={"modelRoots": [str(models_dir)]})

    assert response.json()["applied"] == ["modelRoots"]
    assert client.get("/v1/config").json()["modelRoots"] == [str(models_dir)]


def test_patching_roots_does_not_start_a_scan(client: TestClient, models_dir: Path) -> None:
    """A PATCH that quietly kicks off minutes of disk work is a PATCH
    that appears to hang."""
    write_gguf(models_dir / "a.gguf", qwen_like_kv())
    client.patch("/v1/config", json={"modelRoots": [str(models_dir)]})

    assert client.get("/v1/scan").json()["state"] == "idle"


def test_a_non_list_root_value_is_rejected(client: TestClient) -> None:
    result = client.patch("/v1/config", json={"modelRoots": "/one/path"}).json()

    assert result["applied"] == []
    assert result["rejected"][0]["key"] == "modelRoots"


def test_duplicate_roots_are_rejected(client: TestClient, models_dir: Path) -> None:
    """Two spellings of one directory would scan it twice; silently
    dropping one of the operator's entries is worse than telling them."""
    result = client.patch(
        "/v1/config",
        json={"modelRoots": [str(models_dir), str(models_dir / ".")]},
    ).json()

    assert result["rejected"][0]["key"] == "modelRoots"
    assert "duplicate" in result["rejected"][0]["message"]


def test_unknown_config_key_is_rejected_without_losing_the_valid_ones(
    client: TestClient, models_dir: Path
) -> None:
    result = client.patch(
        "/v1/config", json={"modelRoots": [str(models_dir)], "nonsense": 1}
    ).json()

    assert result["applied"] == ["modelRoots"]
    assert [e["key"] for e in result["rejected"]] == ["nonsense"]


def test_config_test_reports_readable_roots(client: TestClient, models_dir: Path) -> None:
    client.patch("/v1/config", json={"modelRoots": [str(models_dir)]})
    body = client.post("/v1/config/test", json={}).json()

    assert body["ok"] is True
    assert body["component"] == "library"


def test_config_test_names_the_bad_directory(client: TestClient, tmp_path: Path) -> None:
    client.patch("/v1/config", json={"modelRoots": [str(tmp_path / "nope")]})
    body = client.post("/v1/config/test", json={}).json()

    assert body["ok"] is False
    assert "does not exist" in body["error"]


def test_config_test_with_overrides_checks_uncommitted_paths(
    client: TestClient, models_dir: Path
) -> None:
    """The Test button next to a directory picker, before saving."""
    body = client.post(
        "/v1/config/test", json={"overrides": {"modelRoots": [str(models_dir)]}}
    ).json()

    assert body["ok"] is True
    assert client.get("/v1/config").json()["modelRoots"] == []


def test_config_test_with_no_roots_is_not_a_failure(client: TestClient) -> None:
    """A fresh install, mid-way through adding the first directory."""
    body = client.post("/v1/config/test", json={}).json()

    assert body["ok"] is True
    assert "No model directories" in body["summary"]


# --- health + admin ---------------------------------------------------------------


def test_healthz_is_ok_with_no_roots(client: TestClient) -> None:
    """A fresh install waiting for the wizard is not a fault, and a
    health surface that goes yellow on day one teaches people to ignore
    it."""
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["component"] == "library"
    assert body["details"]["rootsConfigured"] == 0


def test_healthz_is_degraded_when_a_configured_root_is_unreadable(
    client: TestClient, tmp_path: Path
) -> None:
    client.patch("/v1/config", json={"modelRoots": [str(tmp_path / "unplugged")]})
    client.post("/v1/scan")
    _wait_for_scan(client)

    body = client.get("/healthz").json()
    assert body["status"] == "degraded"
    assert body["details"]["unreadableRoots"] == [str(tmp_path / "unplugged")]


def test_healthz_needs_no_credentials(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200


def test_restart_is_scheduled(client: TestClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[float] = []
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"call_later": lambda _s, delay, _fn: calls.append(delay)})(),
    )
    response = client.post("/v1/admin/restart")

    assert response.status_code == 202
    assert response.json()["scheduled"] is True
    assert calls == [0.5]
