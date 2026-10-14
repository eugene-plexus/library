"""The scan fixtures wait for their subject even when a worker starts slowly."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library.scan_manager import ScanManager

from .test_routes import _wait_for_scan


@pytest.fixture
def slow_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ScanManager._run

    async def delayed(self: ScanManager, *args: Any, **kwargs: Any) -> None:
        import asyncio

        await asyncio.sleep(0.2)
        await original(self, *args, **kwargs)

    monkeypatch.setattr(ScanManager, "_run", delayed)


@pytest.mark.usefixtures("slow_startup")
def test_configured_fixture_has_finished_its_startup_scan(configured_client: TestClient) -> None:
    assert configured_client.delete("/v1/scan").status_code == 409


def test_scan_wait_is_limited_by_elapsed_time_not_poll_count() -> None:
    class Scan:
        polls = 0
        status_code = 200
        text = ""

        def get(self, path: str) -> Scan:
            self.polls += 1
            return self

        def json(self) -> dict[str, str]:
            return {"state": "scanning" if self.polls <= 210 else "idle"}

    began = time.perf_counter()
    assert _wait_for_scan(Scan())["state"] == "idle"  # type: ignore[arg-type]
    assert time.perf_counter() - began < 30
