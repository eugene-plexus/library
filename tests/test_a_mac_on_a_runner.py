"""A4: what a Mac on a GitHub runner showed the library getting wrong.

Measured 2026-09-30 on GitHub's macOS runners (docs/acceptance/
a4-macos-runner-run.md in specs):

* Metal's `recommendedMaxWorkingSetSize` is two thirds of RAM there
  (5,010,800,640 of 7,516,192,768 bytes), exactly MLX's own figure, while
  the library scored every fit against `0.75 * hw.memsize` and so called a
  model that fits one Metal will not hold. The same fix is in the agent;
  `gpu_probe.py` is the same file in both.
* `mlx-community/Qwen3-0.6B-4bit` read as 93,188,096 parameters, because
  summed shapes count MLX's packed uint32 words.
* The same model read as having no chat template: it keeps it in
  `tokenizer_config.json`, which the library never looked in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eugene_plexus_library import gpu_probe, hardware
from eugene_plexus_library._generated.models import Vendor
from eugene_plexus_library.formats import safetensors
from eugene_plexus_library.scanner import Scanner

from .conftest import write_safetensors

RAM = 7_516_192_768
WORKING_SET = 5_010_800_640


def test_the_budget_is_metals_working_set() -> None:
    warnings: list[str] = []
    metal = gpu_probe.MetalDevice(
        name="Apple M2 Pro", working_set_bytes=WORKING_SET, unified_memory=True
    )
    [gpu] = hardware._apple_gpu(RAM, warnings, metal=lambda: metal)
    assert gpu.vramTotalBytes == WORKING_SET
    assert gpu.vramTotalBytes != int(RAM * 0.75)
    assert gpu.vendor is Vendor.apple
    assert gpu.name == "Apple M2 Pro"
    assert gpu.vramFreeBytes is None
    assert warnings == []


def test_with_no_metal_it_falls_back_to_two_thirds_and_says_so(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(hardware, "_apple_chip", lambda: "Apple M1 (Virtual)")
    warnings: list[str] = []
    [gpu] = hardware._apple_gpu(RAM, warnings, metal=lambda: None)
    assert gpu.vramTotalBytes == int(RAM * 2 / 3)
    assert gpu.name == "Apple M1 (Virtual)"
    assert any("Metal could not be asked" in w for w in warnings), warnings


def test_metal_is_not_asked_off_a_mac() -> None:
    if sys.platform == "darwin":
        pytest.skip("this is the off-Mac half")
    assert gpu_probe.metal_device() is None


def test_gpu_probe_is_the_agents_file_byte_for_byte() -> None:
    agent = (
        Path(__file__).resolve().parents[2] / "agent/src/eugene_plexus_agent/engines/gpu_probe.py"
    )
    if not agent.is_file():
        pytest.skip("no agent checkout beside this one")
    ours = Path(gpu_probe.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert ours == agent.read_bytes().replace(b"\r\n", b"\n")


# --------------------------------------------------------------------------- #
# an MLX-converted directory, as the scanner reads it
# --------------------------------------------------------------------------- #

V, H = 64, 128  # vocabulary and hidden size of the synthetic model
REAL_PARAMETERS = V * H + H * H + H  # embeddings, one q_proj, one norm


def _mlx_model(directory: Path, *, template_in: str = "tokenizer_config.json") -> Path:
    """A 4-bit MLX conversion: packed uint32 weights beside BF16 scales and
    biases, one module quantized at a group size of its own, a norm left
    unquantized, and the chat template wherever `template_in` says."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "max_position_embeddings": 4096,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.self_attn.q_proj": {"group_size": 32, "bits": 4},
        },
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    packed = H * 4 // 32
    write_safetensors(
        directory / "model.safetensors",
        {
            "model.embed_tokens.weight": ("U32", [V, packed]),
            "model.embed_tokens.scales": ("BF16", [V, H // 64]),
            "model.embed_tokens.biases": ("BF16", [V, H // 64]),
            "model.layers.0.self_attn.q_proj.weight": ("U32", [H, packed]),
            "model.layers.0.self_attn.q_proj.scales": ("BF16", [H, H // 32]),
            "model.layers.0.self_attn.q_proj.biases": ("BF16", [H, H // 32]),
            "model.norm.weight": ("BF16", [H]),
        },
    )
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    if template_in == "tokenizer_config.json":
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": template}), encoding="utf-8"
        )
    elif template_in == "chat_template.jinja":
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (directory / "chat_template.jinja").write_text(template, encoding="utf-8")
    else:
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return directory


def test_an_mlx_model_counts_parameters_not_packed_words(models_dir: Path) -> None:
    _mlx_model(models_dir / "qwen-mlx")
    [model] = Scanner().scan([models_dir]).models
    stored = sum(safetensors.read_header(Path(model.path) / "model.safetensors").elements.values())
    assert stored < REAL_PARAMETERS  # what the library used to report
    assert model.parameters == REAL_PARAMETERS
    assert model.safetensors is not None and model.safetensors.mlxQuantization is not None


def test_a_group_size_that_cannot_be_read_leaves_the_count_absent() -> None:
    elements = {"m.weight": 16, "m.scales": 2, "m.biases": 2}
    assert safetensors.mlx_parameters(elements, {"bits": 4}) is None


def test_an_unquantized_directory_is_counted_as_stored(models_dir: Path) -> None:
    directory = models_dir / "plain"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )
    write_safetensors(
        directory / "model.safetensors", {"w": ("BF16", [10, 20]), "n": ("BF16", [5])}
    )
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    [model] = Scanner().scan([models_dir]).models
    assert model.parameters == 205


@pytest.mark.parametrize(
    ("where", "expected"),
    [("tokenizer_config.json", True), ("chat_template.jinja", True), ("nowhere", False)],
)
def test_the_chat_template_is_found_where_models_keep_it(
    models_dir: Path, where: str, expected: bool
) -> None:
    _mlx_model(models_dir / "qwen-mlx", template_in=where)
    [model] = Scanner().scan([models_dir]).models
    assert model.capabilities is not None
    assert model.capabilities.chatTemplate is expected


def test_a_wired_limit_someone_set_is_the_working_set() -> None:
    """After `sudo sysctl iogpu.wired_limit_mb=5973` a new process's Metal
    device reported 6,263,144,448 bytes (5973 MiB exactly), and a running
    one kept its old figure, so the sysctl is read on every call."""
    assert gpu_probe.working_set_bytes(5_010_800_640, 5973) == 6_263_144_448


def test_the_default_wired_limit_means_metals_own_figure() -> None:
    assert gpu_probe.working_set_bytes(5_010_800_640, 0) == 5_010_800_640
    assert gpu_probe.working_set_bytes(5_010_800_640, None) == 5_010_800_640
