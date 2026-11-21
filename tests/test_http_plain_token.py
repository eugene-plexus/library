"""agent#38 in this component's copy of `_http` (specs#135): no token of this
install goes to another machine in plain HTTP unless the machine table says
that machine speaks nothing else.

The copy here was left at the module from before agent#38 until specs#135.
Each test records what reached the plain transport, so "refused" means
nothing was sent, and "sent" means the request left as it would in an
install whose agent wrote this process a table.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from eugene_plexus_library import _http


def _token(typ: str = "ep-service+jwt") -> str:
    head = base64.urlsafe_b64encode(json.dumps({"alg": "EdDSA", "typ": typ}).encode())
    return head.decode().rstrip("=") + ".e30.c2ln"


TOKEN = {"Authorization": "Bearer " + _token()}
OLDER = "http://10.0.0.7:8079"  # a machine on a version with no machine port
NEW = "http://10.0.0.8:8079"  # a machine that joined a moment ago
OWN = "http://10.0.0.2:8079"  # this machine's own agent under its LAN address


@pytest.fixture(autouse=True)
def _restore_table(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(_http.MACHINE_PINS_ENV, raising=False)
    yield
    _http.set_machine_table(None)


def _install(document: dict | None) -> None:
    _http.set_machine_table(None if document is None else _http.parse_machine_table(document))


def _table(**extra: object) -> dict:
    return {"routes": {OLDER: None}, "local": [OWN], "required": False, **extra}


def _recording(client: httpx.AsyncClient | httpx.Client) -> list[httpx.Request]:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, text="ok")

    transport = client._transport
    assert isinstance(transport, _http.MachineTransport | _http.SyncMachineTransport)
    transport._plain = httpx.MockTransport(answer)  # type: ignore[assignment]
    return sent


async def _get(url: str, headers: dict[str, str] | None = None) -> list[httpx.Request]:
    async with _http.internal_client(timeout=5) as client:
        sent = _recording(client)
        await client.get(url, headers=headers)
    return sent


async def test_a_token_to_a_machine_the_table_does_not_name_is_not_sent() -> None:
    _install(_table())
    async with _http.internal_client(timeout=5) as client:
        sent = _recording(client)
        with pytest.raises(_http.MachineRefused) as caught:
            await client.get(NEW + "/v1/components", headers=TOKEN)
    assert sent == [], "a token went to a machine not known yet in plain HTTP"
    assert "10.0.0.8" in str(caught.value)


def test_the_synchronous_client_refuses_it_too() -> None:
    _install(_table())
    with _http.sync_client_for(NEW, timeout=5) as client:
        sent = _recording(client)
        with pytest.raises(_http.MachineRefused):
            client.get(NEW + "/v1/runtimes/x", headers=TOKEN)
    assert sent == []


async def test_a_client_key_counts_as_a_token() -> None:
    _install(_table())
    with pytest.raises(_http.MachineRefused):
        await _get(NEW + "/v1/models", {"x-api-key": _token("ep-client+jwt")})


async def test_a_machine_on_an_older_version_is_still_reached_in_plain_http() -> None:
    _install(_table())
    sent = await _get(OLDER + "/v1/components", TOKEN)
    assert [str(r.url) for r in sent] == [OLDER + "/v1/components"]


async def test_loopback_and_this_machines_own_address_are_not_the_rules_business() -> None:
    _install(_table())
    assert len(await _get("http://127.0.0.1:8079/v1/node/operations-log", TOKEN)) == 1
    assert len(await _get("http://localhost:8079/v1/node", TOKEN)) == 1
    assert len(await _get(OWN + "/v1/node", TOKEN)) == 1


async def test_a_request_without_a_token_of_this_install_goes_as_before() -> None:
    _install(_table())
    assert len(await _get("http://10.0.0.9:11434/api/tags")) == 1
    other = {"Authorization": "Bearer sk-a-provider-key"}
    assert len(await _get("http://10.0.0.9:8000/v1/models", other)) == 1


async def test_an_entry_that_does_not_hold_together_is_refused_without_a_token_too() -> None:
    _install(_table(routes={OLDER: None, "http://10.0.0.5:8079": False}))
    with pytest.raises(_http.MachineRefused):
        await _get("http://10.0.0.5:8079/v1/node")


async def test_a_process_with_no_table_behaves_as_before() -> None:
    _install(None)
    assert _http.machine_table() is _http.NO_TABLE
    assert len(await _get(NEW + "/v1/components", TOKEN)) == 1


async def test_a_table_file_that_names_no_machine_yet_still_holds_the_token_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a component sees when its agent wrote the table before any other
    machine joined: the file is there and empty, which is a table."""
    path = tmp_path / "machine_pins.json"
    path.write_text(json.dumps({"routes": {}, "required": False}), encoding="utf-8")
    monkeypatch.setenv(_http.MACHINE_PINS_ENV, str(path))
    assert _http.machine_table().present
    with pytest.raises(_http.MachineRefused):
        await _get(NEW + "/v1/components", TOKEN)


# --- this component's own call with a token ---------------------------------


async def test_the_operations_report_still_reaches_this_machines_agent() -> None:
    """The Library's call with its service token goes to its own agent at
    the loopback address the agent hands it (`AGENT_URL`), which the rule
    leaves alone, table or not."""
    from types import SimpleNamespace

    from eugene_plexus_library import operations

    _install(_table())
    _http.set_shared_client(operations.CLIENT_KEY, None)
    client = _http.shared_internal_client(operations.CLIENT_KEY, timeout=httpx.Timeout(5.0))
    sent = _recording(client)
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(agent_url="http://127.0.0.1:8079"),
            auth_state=SimpleNamespace(service_token=_token()),
        )
    )
    try:
        await operations.report(app, {"action": "profile.update"})
    finally:
        _http.set_shared_client(operations.CLIENT_KEY, None)
        await client.aclose()
    assert [r.url.path for r in sent] == ["/v1/node/operations-log"]
