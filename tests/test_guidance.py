"""The quant table, hardware detection, and the guidance endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_library import hardware, quants
from eugene_plexus_library._generated.models import Arch, Os

from .conftest import qwen_like_kv, write_gguf

# -- the quant table ----------------------------------------------------


def test_every_tier_explains_the_scheme_not_the_model() -> None:
    """The line this component does not cross. Size, bits per weight and
    fit are arithmetic; which quant family is *better* on a given model
    is upstream research and would have to be invented."""
    table = quants.table()
    assert len(table.tiers) >= 15
    for tier in table.tiers:
        assert tier.summary
        assert tier.guidance
        text = f"{tier.summary} {tier.guidance}".lower()
        assert "recommended for" not in text
        assert "★" not in text


def test_the_default_is_named_outright() -> None:
    """ "If you do not know what to pick, pick this" is the answer the
    source complaint was asking for."""
    q4 = quants.describe("Q4_K_M")
    assert q4 is not None
    assert q4.guidance is not None
    assert "pick this" in q4.guidance


def test_a_publishers_decorated_tier_still_resolves() -> None:
    """`UD-Q6_K_XL` is a repacking of the `Q6_K` family. Reporting a tier
    we do describe as unknown left a third of one real repo's candidates
    with no reference text at all."""
    assert quants.describe("UD-Q6_K_XL") is not None
    assert quants.describe("UD-Q6_K_XL").tier == "Q6_K"  # type: ignore[union-attr]
    assert quants.describe("UD-Q4_K_M").tier == "Q4_K_M"  # type: ignore[union-attr]
    assert quants.describe("Q4_K_XL").tier == "Q4_K_M"  # type: ignore[union-attr]
    # A legacy 4-bit tier resolves to the legacy family, not to the
    # K-quant that merely shares its first three characters.
    assert quants.describe("Q4_1").tier == "Q4_0"  # type: ignore[union-attr]


def test_an_unrecognisable_tier_is_none_not_a_nearest_guess() -> None:
    """The same rule as an unrecognised `general.file_type`: report what
    is known, never map onto a plausible neighbour."""
    assert quants.describe("Q9_ULTRA_PLUS") is None
    assert quants.describe(None) is None
    assert quants.describe("") is None


def test_the_naming_decorations_are_explained() -> None:
    notes = quants.suffix_notes("UD-Q4_K_XL")
    assert any("Unsloth Dynamic" in note for note in notes)
    assert any("mixture" in note for note in notes)


# -- hardware -----------------------------------------------------------


def test_this_host_reports_memory_and_says_what_it_could_not_see() -> None:
    """Never raises: a probe that fails becomes a warning, so a machine
    with a broken vendor tool still gets a verdict that admits it."""
    detected = hardware.detect()
    assert detected.hostname
    assert detected.os in (Os.windows, Os.linux, Os.macos)
    assert detected.arch in (Arch.x64, Arch.arm64)
    assert detected.cpuCount and detected.cpuCount >= 1
    assert detected.ramTotalBytes and detected.ramTotalBytes > 0
    assert detected.detectedAt is not None
    # Either an accelerator was found, or the absence is explained.
    assert detected.gpus or detected.warnings


def test_available_memory_is_below_total() -> None:
    """The whole reason both are reported: a third of RAM is in use on an
    ordinary desktop, and scoring against total promises a fit that
    swaps."""
    total, available = hardware.host_memory()
    assert total is not None and available is not None
    assert 0 < available <= total


def test_a_missing_vendor_tool_is_not_an_error() -> None:
    """Most machines have exactly one of nvidia-smi / rocm-smi / xpu-smi,
    so absence is the common case."""
    assert hardware._run(["definitely-not-a-real-tool-9271", "--version"]) is None


# -- the endpoints ------------------------------------------------------


def test_hardware_endpoint_answers(configured_client: TestClient) -> None:
    body = configured_client.get("/v1/hardware").json()
    assert body["hostname"]
    assert "gpus" in body


def test_quants_endpoint_answers(configured_client: TestClient) -> None:
    body = configured_client.get("/v1/quants").json()
    assert len(body["tiers"]) >= 15
    assert body["updatedAt"]


def test_fit_reads_the_shape_back_off_a_scanned_model(
    configured_client: TestClient, models_dir: Path
) -> None:
    """The scan keeps the architecture-prefixed KV pairs as an escape
    hatch, and this is what that pays for: the layer and head counts come
    back out by suffix, so the fit is arithmetic rather than a guess."""
    write_gguf(models_dir / "hybrid.gguf", qwen_like_kv(vocab=64))
    configured_client.post("/v1/scan")
    for _ in range(200):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break

    models = configured_client.get("/v1/models").json()["models"]
    assert models
    model_id = models[0]["id"]

    body = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 4096}
    ).json()
    assert body["modelId"] == model_id
    assert body["fit"]["verdict"] in ("fits", "tight", "split", "no")
    assert body["fit"]["contextLength"] == 4096
    assert body["fit"]["notes"]


def test_fit_accepts_a_budget_for_another_host(
    configured_client: TestClient, models_dir: Path
) -> None:
    """The only honest answer when the GPU is in a different building."""
    write_gguf(models_dir / "m.gguf", qwen_like_kv(vocab=64))
    configured_client.post("/v1/scan")
    for _ in range(200):
        if configured_client.get("/v1/scan").json()["state"] != "scanning":
            break
    model_id = configured_client.get("/v1/models").json()["models"][0]["id"]

    body = configured_client.get(
        f"/v1/models/{model_id}/fit",
        params={"vramBytes": 24 * 1024**3, "ramBytes": 64 * 1024**3},
    ).json()
    assert body["fit"]["budget"]["source"] == "override"
    assert body["fit"]["budget"]["vramFreeBytes"] == 24 * 1024**3
    assert any("caller-supplied" in note for note in body["fit"]["notes"])


def test_fit_on_an_unknown_model_is_a_404(configured_client: TestClient) -> None:
    response = configured_client.get("/v1/models/deadbeef/fit")
    assert response.status_code == 404
