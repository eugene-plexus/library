"""Bearer auth: reads take a service token, writes do not.

Verify-only role — the watchdog issues tokens, this component validates
them. Tests mint JWTs directly against a known signing key, standing in
for the watchdog.

The split under test is the one that matters here: the runtime dashboard
resolving a `modelPath` back to a library entry arrives with a service
token, so reads must accept one. Everything that writes is operator-only
**including profile edits** — a profile is the operator's tuning work,
and a compromised peer should not be able to rewrite the flags a model
launches with.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library.app import create_app
from eugene_plexus_library.auth_state import AuthState, load_auth_state
from eugene_plexus_library.settings import Settings

_JWT_ALG = "HS256"


def _issue(*, signing_key: bytes, sub: str, aud: str, ttl_seconds: int = 60) -> str:
    """Mint a JWT exactly the way the watchdog would."""
    issued_at = int(time.time())
    claims = {"sub": sub, "aud": aud, "iat": issued_at, "exp": issued_at + ttl_seconds}
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def authed_client(tmp_path: Path, signing_key: bytes) -> Iterator[TestClient]:
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
    )
    app: FastAPI = create_app(settings=settings)
    app.state.auth_state = AuthState(signing_key=signing_key, service_token=None, master_key=None)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def operator_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="operator", aud="operator")


@pytest.fixture
def service_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="gateway", aud="service:gateway")


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- the open door -------------------------------------------------------


def test_healthz_needs_no_token(authed_client: TestClient) -> None:
    """Supervisors and load balancers probe it without credentials."""
    assert authed_client.get("/healthz").status_code == 200


def test_a_missing_token_is_rejected(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["detail"]["component"] == "library"


def test_a_token_signed_with_the_wrong_key_is_rejected(authed_client: TestClient) -> None:
    other = _issue(signing_key=secrets.token_bytes(32), sub="operator", aud="operator")
    assert authed_client.get("/v1/models", headers=_auth(other)).status_code == 401


def test_an_expired_token_is_rejected(authed_client: TestClient, signing_key: bytes) -> None:
    stale = _issue(signing_key=signing_key, sub="operator", aud="operator", ttl_seconds=-10)
    assert authed_client.get("/v1/models", headers=_auth(stale)).status_code == 401


def test_an_unknown_audience_is_rejected(authed_client: TestClient, signing_key: bytes) -> None:
    odd = _issue(signing_key=signing_key, sub="someone", aud="somebody-else")
    assert authed_client.get("/v1/models", headers=_auth(odd)).status_code == 401


# --- reads ----------------------------------------------------------------


def test_a_service_token_can_read_models(authed_client: TestClient, service_token: str) -> None:
    """The runtime dashboard's reverse lookup arrives this way."""
    assert authed_client.get("/v1/models", headers=_auth(service_token)).status_code == 200


def test_a_service_token_can_read_the_scan_state(
    authed_client: TestClient, service_token: str
) -> None:
    assert authed_client.get("/v1/scan", headers=_auth(service_token)).status_code == 200


def test_an_operator_token_can_read(authed_client: TestClient, operator_token: str) -> None:
    assert authed_client.get("/v1/models", headers=_auth(operator_token)).status_code == 200


# --- writes ------------------------------------------------------------------


def test_a_service_token_cannot_start_a_scan(authed_client: TestClient, service_token: str) -> None:
    assert authed_client.post("/v1/scan", headers=_auth(service_token)).status_code == 401


def test_a_service_token_cannot_forget_a_model(
    authed_client: TestClient, service_token: str
) -> None:
    assert (
        authed_client.delete("/v1/models/deadbeef", headers=_auth(service_token)).status_code == 401
    )


def test_a_service_token_cannot_write_a_profile(
    authed_client: TestClient, service_token: str
) -> None:
    """A profile is the operator's tuning work."""
    response = authed_client.post(
        "/v1/models/deadbeef/profiles",
        json={"name": "x", "engine": "llama_cpp"},
        headers=_auth(service_token),
    )
    assert response.status_code == 401


def test_a_service_token_cannot_edit_config(authed_client: TestClient, service_token: str) -> None:
    response = authed_client.patch(
        "/v1/config", json={"modelRoots": []}, headers=_auth(service_token)
    )
    assert response.status_code == 401


def test_an_operator_token_can_write(authed_client: TestClient, operator_token: str) -> None:
    response = authed_client.patch(
        "/v1/config", json={"scanOnStartup": False}, headers=_auth(operator_token)
    )
    assert response.status_code == 200


# --- posture ---------------------------------------------------------------------


def test_no_signing_key_means_auth_disabled(client: TestClient) -> None:
    """The dev / standalone path. Production spawns via the watchdog
    always supply the key."""
    assert client.get("/v1/models").status_code == 200


def test_a_partial_auth_configuration_is_refused() -> None:
    """A missing piece is almost always a wiring bug, and one that would
    otherwise present as everything mysteriously 401-ing."""
    with pytest.raises(ValueError, match="partially-auth"):
        load_auth_state(
            signing_key_b64=None,
            service_token="a-token",
            master_key_b64=None,
        )


def test_a_misshaped_key_is_refused() -> None:
    import base64

    with pytest.raises(ValueError, match="32 bytes"):
        load_auth_state(
            signing_key_b64=base64.b64encode(b"too short").decode(),
            service_token=None,
            master_key_b64=None,
        )
