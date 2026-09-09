"""Opt-in: drive the catalogue against the real HuggingFace hub.

Mock transports agree with whatever the client believes. The live hub
disagrees when it is wrong — which is how five of this component's
behaviours were established in the first place, and how the two label
defects in the first version of the candidate grouping were found (four
files reading as `UD-Q6_K`, and others falling back to their whole
filename). No fixture would have surfaced either.

Enable it explicitly, because it spends someone else's bandwidth:

    EUGENE_PLEXUS_LIBRARY_LIVE_HUB=1 pytest tests/test_live_hub.py

Skipped when unset, so CI stays green and stays off the network. One run
transfers about 13 MB: a preflight of a large-vocab GGUF's header, plus
one small real download.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LIVE = os.environ.get("EUGENE_PLEXUS_LIBRARY_LIVE_HUB")

pytestmark = pytest.mark.skipif(
    not LIVE, reason="set EUGENE_PLEXUS_LIBRARY_LIVE_HUB=1 to call the real hub"
)

# A repo chosen because it is awkward in every way that matters: 30
# `.gguf` files of which 25 are choices, a split candidate, two vision
# projectors, an imatrix file, an MTP draft model, and a hybrid
# attention/SSM architecture whose KV cache the naive formula gets wrong
# by 4.1x.
REPO = "unsloth/Qwen3.8-27B-GGUF"
QUANT = "Qwen3.8-27B-UD-Q4_K_M.gguf"
SMALL_FILE = "imatrix_unsloth.gguf"  # 13.6 MB, the smallest thing in the repo

GIB = 1024**3


@pytest.fixture
def live_client(configured_client: TestClient) -> TestClient:
    return configured_client


def test_search_returns_repos_without_sizes(live_client: TestClient) -> None:
    body = live_client.get(
        "/v1/catalogue/search", params={"q": "qwen3", "format": "gguf", "limit": 5}
    ).json()
    assert body["results"]
    assert all(r["repo"] for r in body["results"])
    # Upstream's search response carries no sizes, which is why fit
    # verdicts live on the detail screen.
    assert all("sizeBytes" not in r for r in body["results"])
    assert body.get("nextCursor")


def test_a_real_repo_groups_into_choices(live_client: TestClient) -> None:
    body = live_client.get(
        "/v1/catalogue/model", params={"repo": REPO, "contextLength": 32768}
    ).json()

    assert len(body["candidates"]) == 25
    assert len(body["projectors"]) == 2
    labels = [c["label"] for c in body["candidates"]]
    assert len(set(labels)) == len(labels), "every choice needs a distinct label"
    assert {"Q4_0", "Q8_0", "BF16", "UD-Q4_K_M", "UD-Q6_K_XL"} <= set(labels)

    split = [c for c in body["candidates"] if len(c["files"]) > 1]
    assert len(split) == 1
    assert split[0]["sizeBytes"] > 50 * GIB, "a split candidate is summed, not sampled"

    bf16 = next(c for c in body["candidates"] if c["label"] == "BF16")
    assert abs(bf16["bitsPerWeight"] - 16.0) < 0.01, "the parameter count is real"

    assert body["recommended"]["reason"]
    assert any("projector" in w for w in body["warnings"])


def test_a_remote_header_reads_for_a_fraction_of_the_file(live_client: TestClient) -> None:
    """The milestone's best result: ~11 MB of a 16.5 GB file gives the
    machine-readable quant and the real layer shape."""
    started = time.monotonic()
    body = live_client.get(
        "/v1/catalogue/model/preflight",
        params={"repo": REPO, "file": QUANT, "contextLength": 32768},
    ).json()
    elapsed = time.monotonic() - started

    assert body["fileType"] == 15, "general.file_type, not the filename"
    assert body["quantization"] == "Q4_K_M"
    assert body["agreesWithFilename"] is True
    assert body["blockCount"] == 65
    assert body["attentionLayers"] == 16, "hybrid: every fourth layer holds KV"
    assert body["vocabSize"] == 248_320
    assert body["bytesRead"] < 0.01 * 16.4e9, "under 1% of the file"
    assert body["fit"]["basis"] == "metadata"
    # 32768 x 16 x 4 x 512 x 2 = 2 GiB. The naive reading would be 8.12.
    assert abs(body["fit"]["kvCacheBytes"] - 2 * GIB) < 0.01 * GIB
    assert elapsed < 60


def test_a_gated_repo_browses_and_warns(live_client: TestClient) -> None:
    """Metadata, sizes and digests are public; only the bytes are not."""
    response = live_client.get(
        "/v1/catalogue/model", params={"repo": "meta-llama/Llama-3.3-70B-Instruct"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gated"] in ("auto", "manual")
    assert any("gated" in w.lower() for w in body["warnings"])


def test_a_real_download_verifies_and_lands_under_its_own_name(
    live_client: TestClient, models_dir: Path
) -> None:
    """The whole loop: resolve, pin a commit, transfer, verify, rename,
    then find the library entry it became."""
    started = live_client.post("/v1/downloads", json={"repo": REPO, "files": [SMALL_FILE]})
    assert started.status_code == 202, started.text
    record = started.json()

    assert record["destinationDirectory"], "resolved before a byte moved"
    assert record["resolvedCommit"], "pinned to a commit, not a moving branch"

    for _ in range(240):
        record = live_client.get(f"/v1/downloads/{record['id']}").json()
        if record["state"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.25)

    assert record["state"] == "done", record.get("error")
    entry = record["files"][0]
    assert entry["verified"] is True, "the digest was checked before the rename"

    landed = Path(entry["destinationPath"])
    assert landed.exists()
    assert landed.name == SMALL_FILE, "plainly named, exactly as upstream has it"
    assert not landed.with_name(landed.name + ".part").exists()
    # <root>/<publisher>/<repo>/<file>
    assert landed.parent.name == REPO.split("/")[1]
    assert landed.parent.parent.name == REPO.split("/")[0]
    assert landed.parent.parent.parent == models_dir
    assert record["modelId"], "the post-download scan found it"


def test_the_air_gap_switch_refuses_rather_than_hanging(live_client: TestClient) -> None:
    live_client.patch("/v1/config", json={"catalogueEnabled": False})
    try:
        response = live_client.get("/v1/catalogue/search", params={"q": "qwen"})
        assert response.status_code == 409
        assert "catalogueEnabled" in response.json()["detail"]["detail"]
    finally:
        live_client.patch("/v1/config", json={"catalogueEnabled": True})
