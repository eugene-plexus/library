"""Bearer auth: reads take a service token, writes do not.

Verify-only role -- the library holds no key and checks every bearer
against the trust bundle its agent keeps (per-node token keys,
2026-09-25). `FakeInstall` stands in for the agent and the root with a
real bundle on disk.

The split under test is the one that matters here: the runtime dashboard
resolving a `modelPath` back to a library entry arrives with a service
token, so reads must accept one. Everything that writes is operator-only
**including profile edits** — a profile is the operator's tuning work,
and a compromised peer should not be able to rewrite the flags a model
launches with.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library import tokens
from eugene_plexus_library.app import create_app
from eugene_plexus_library.auth_state import load_auth_state
from eugene_plexus_library.settings import Settings

from .conftest import FakeInstall


@pytest.fixture
def authed_client(tmp_path: Path, install: FakeInstall) -> Iterator[TestClient]:
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "state.json",
    )
    app: FastAPI = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def operator_token(install: FakeInstall) -> str:
    return install.session()


@pytest.fixture
def service_token(install: FakeInstall) -> str:
    """This machine's gateway, beside the library."""
    return install.service("gateway")


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


def test_a_token_signed_by_a_key_the_bundle_does_not_list_is_rejected(
    authed_client: TestClient, install: FakeInstall
) -> None:
    stranger = tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    other, _ = stranger.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=[install.recipient], ttl_seconds=60
    )
    assert authed_client.get("/v1/models", headers=_auth(other)).status_code == 401


def test_an_expired_token_is_rejected(authed_client: TestClient, install: FakeInstall) -> None:
    # Expired past the 300 s clock-skew leeway, not merely past `exp`.
    stale = install.session(ttl=60, now=int(time.time()) - 1000)
    assert authed_client.get("/v1/models", headers=_auth(stale)).status_code == 401


def test_a_session_for_another_machine_is_rejected(
    authed_client: TestClient, install: FakeInstall
) -> None:
    odd = install.session(aud=["node:elsewhere"])
    assert authed_client.get("/v1/models", headers=_auth(odd)).status_code == 401


def test_an_agent_or_gateway_on_another_machine_can_read(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """A GPU node's agent resolving the model it is about to launch, and
    a gateway elsewhere reading a profile -- the two remote readers D5
    names. The gateway only from a machine the operator granted one."""
    install.far_grants = ("gateway",)
    install.publish()
    for sub in ("agent", "gateway"):
        token = install.foreign_service(sub)
        assert authed_client.get("/v1/models", headers=_auth(token)).status_code == 200, sub


def test_a_gateway_token_from_a_machine_without_the_grant_is_refused(
    authed_client: TestClient, install: FakeInstall
) -> None:
    token = install.foreign_service("gateway")
    assert authed_client.get("/v1/models", headers=_auth(token)).status_code == 401


def test_another_machines_driver_reads_nothing(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """No `service:*` wildcard. A driver on another machine has no reason
    to read the library, so a leaked worker key cannot spend one here.

    Another machine's key cannot even sign one -- the verifier's grant
    rule refuses it first. The root's key can sign any `sub`, so it is
    what shows the library's own rule: off this machine, `agent` and
    `gateway` read, and nothing else does, whoever signed it."""
    for sub in ("inference-driver", "library", "control"):
        from_far = install.raw(
            install.far, tokens.TYP_SERVICE, sub=sub, aud=[install.recipient], jti="x"
        )
        assert authed_client.get("/v1/models", headers=_auth(from_far)).status_code == 401, sub
        from_root, _ = install.root.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[install.recipient], ttl_seconds=60
        )
        assert authed_client.get("/v1/models", headers=_auth(from_root)).status_code == 401, sub
    for sub in ("agent", "gateway"):
        from_root, _ = install.root.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[install.recipient], ttl_seconds=60
        )
        assert authed_client.get("/v1/models", headers=_auth(from_root)).status_code == 200, sub


def test_a_remote_agent_token_cannot_write(authed_client: TestClient, install: FakeInstall) -> None:
    response = authed_client.patch(
        "/v1/config", json={"scanOnStartup": False}, headers=_auth(install.foreign_service())
    )
    assert response.status_code == 401


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


def test_a_service_token_cannot_claim_a_download(
    authed_client: TestClient, service_token: str
) -> None:
    """Taking responsibility for launching something is not a read, and
    a claim is the one write on a download record that is not about the
    transfer."""
    assert (
        authed_client.post("/v1/downloads/whatever/claim", headers=_auth(service_token)).status_code
        == 401
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


def test_no_trust_bundle_means_auth_disabled(client: TestClient) -> None:
    """The dev / standalone path. Production spawns via the agent
    always supply one."""
    assert client.get("/v1/models").status_code == 200


@pytest.mark.parametrize(
    "missing", ["trust_bundle_file", "trust_authority", "auth_recipient", "service_token"]
)
def test_a_partial_auth_configuration_is_refused(install: FakeInstall, missing: str) -> None:
    """A missing piece is almost always a wiring bug, and one that would
    otherwise present as everything mysteriously 401-ing."""
    env: dict[str, str | None] = {
        "trust_bundle_file": str(install.bundle_path),
        "trust_authority": install.authority,
        "auth_recipient": install.recipient,
        "service_token": install.service("library"),
        "master_key_b64": None,
    }
    env[missing] = None
    with pytest.raises(ValueError, match="missing"):
        load_auth_state(**env)  # type: ignore[arg-type]


def test_a_misshaped_master_key_is_refused(install: FakeInstall) -> None:
    import base64

    with pytest.raises(ValueError, match="32 bytes"):
        install.auth_state(master_key_b64=base64.b64encode(b"too short").decode())
