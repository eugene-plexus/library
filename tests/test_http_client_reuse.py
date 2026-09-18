"""One context, one clock — the library's half, and the one place that
must KEEP the user's proxy.

The hub is one of exactly two things in this product that talk to the
public internet (engine acquisition on the agent is the other). A
blanket `trust_env=False` — which is how the adversarial review phrased
the fix — would have made this component unable to reach
huggingface.co for every user behind a corporate proxy, turning a
performance fix into a total loss of discovery and download. So the rule
is per site, and these are the checks that hold that line in both
directions.

What the hub client does gain is the process-wide SSL context: building
a client without `verify=` parses certifi's PEM bundle at ~104 ms of
synchronous CPU on the event loop.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from eugene_plexus_library import _http
from eugene_plexus_library.hub import HubClient


def test_the_hub_client_keeps_the_users_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """**Egress, deliberately.** For a user behind a corporate proxy
    this is how they reach the model hub at all."""
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    client = HubClient()
    assert client._client.trust_env is True
    assert client._client._mounts, "the hub lost the user's proxy"


def test_the_hub_is_not_classified_as_internal() -> None:
    assert not _http.is_internal("https://huggingface.co")
    assert not _http.is_internal("https://raw.githubusercontent.com")


def test_a_loopback_url_is_still_internal() -> None:
    """The other direction of the same rule: this component also talks
    to its own node, and that traffic must not take a proxy."""
    assert _http.is_internal("http://127.0.0.1:8079")
    assert _http.is_internal("http://192.168.16.252:8283")


def test_the_ssl_context_is_built_once_per_process() -> None:
    assert _http.ssl_context() is _http.ssl_context()


def test_building_the_hub_client_does_not_reparse_the_bundle() -> None:
    _http.ssl_context()
    started = time.perf_counter()
    HubClient()
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 50, f"the hub client cost {elapsed_ms:.0f} ms; a certifi parse is ~104 ms"


def test_no_duration_in_this_component_is_measured_with_monotonic() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_library"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"
