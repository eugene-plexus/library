"""Transport model edits must accompany a reviewed, versioned contract update."""

import json
from pathlib import Path

from fastapi import FastAPI

from eugene_plexus_library.routes.run_operations import router


def test_run_protocol_matches_the_vendored_contract():
    app = FastAPI()
    app.include_router(router)
    actual = app.openapi()
    canonical = json.loads(
        (Path(__file__).resolve().parents[1] / "contracts/run-operations.json").read_text(
            encoding="utf-8"
        )
    )
    assert actual["paths"] == canonical["paths"]
    assert actual["components"] == canonical["components"]
