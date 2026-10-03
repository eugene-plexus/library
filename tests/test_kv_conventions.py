"""Three GGUF conventions the KV estimate did not read.

From the upstream drift audit of 2026-10-03
(`specs/docs/maintenance/upstream-drift-2026-10-03.md`, llama.cpp section).
llama.cpp reads each of them and this library did not, so each made the
estimate too large: safe, but a model that fits could be told it does not.
The semantics are llama.cpp's own at tag b11375, read in its source:

(a) **A scalar `attention.sliding_window_pattern` is a period.**
    `llama_model_base::load_swa_pattern` takes the array if there is one,
    else reads the scalar into `n_pattern` and calls
    `llama_hparams::set_swa_pattern(n_pattern, dense_first)`: every
    `n_pattern`-th layer is full attention, the rest slide; 0 is every
    layer sliding, 1 none. `dense_first` is the architecture's own, so
    the table is per architecture (`src/models/*.cpp`). Layers past
    `block_count - nextn_predict_layers` are full.
(b) **`attention.shared_kv_layers`.** gemma4 sets
    `n_layer_kv_from_start = block_count - shared_kv_layers`; a layer at or
    past it has no cache of its own (`llama_hparams::has_kv`) and reuses
    an earlier one (the `reuse` callback in `llama_model::create_memory`).
    gemma3n does the same with `n_layer_kv_from_start = 20` written into
    its loader, and does not read the key.
(c) **MLA.** With `attention.key_length_mla` and `attention.value_length_mla`
    both set (`llama_hparams::is_mla`), the cache allocates K and no V
    (`has_v = !is_mla` in `llama_kv_cache`). The converter has already
    written `key_length` as `kv_lora_rank + qk_rope_head_dim`,
    `value_length` as `kv_lora_rank` and `head_count_kv` as 1, so K is the
    compressed latent and the arithmetic must not add a V beside it.

Each test writes a GGUF with the keys llama.cpp's converter writes for
that architecture and reads it back through the one shape builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import fit, preflight
from eugene_plexus_library._generated.models import KvCacheType
from eugene_plexus_library.formats import gguf

from .conftest import qwen_like_kv, write_gguf
from .test_one_fit_path import scanned

F16 = KvCacheType.f16
CTX = 32768


def shape_of(tmp_path: Path, kv: dict[str, Any]) -> fit.ModelShape:
    path = write_gguf(tmp_path / "m.gguf", kv)
    return preflight.shape_from_gguf(gguf.read_metadata(path))


def windows(shape: fit.ModelShape) -> list[int | None]:
    assert shape.layers is not None, "the per-layer form was not built"
    return [layer.window for layer in shape.layers]


# -- (a) a scalar sliding_window_pattern is a period -------------------------


def swa_model(arch: str, *, blocks: int, pattern: Any, window: int = 2048, **extra: Any) -> dict:
    """The keys `conversion/plamo.py` writes: `sliding_window` and the HF
    config's `sliding_window_pattern`, a scalar, beside the usual ones."""
    return {
        "general.architecture": arch,
        f"{arch}.block_count": blocks,
        f"{arch}.context_length": 65536,
        f"{arch}.embedding_length": 2048,
        f"{arch}.attention.head_count": 16,
        f"{arch}.attention.head_count_kv": 4,
        f"{arch}.attention.key_length": 128,
        f"{arch}.attention.value_length": 128,
        f"{arch}.attention.sliding_window": window,
        f"{arch}.attention.sliding_window_pattern": pattern,
        **extra,
    }


def test_a_scalar_pattern_is_a_period_with_the_full_layer_last(tmp_path: Path) -> None:
    """plamo3: `load_swa_pattern(ml, 8)`, so a scalar 8 makes layers 7 and
    15 full and the other fourteen slide over the window."""
    shape = shape_of(tmp_path, swa_model("plamo3", blocks=16, pattern=8))
    full = {7, 15}
    assert windows(shape) == [None if i in full else 2048 for i in range(16)]
    # Two layers grow with the context; fourteen stop at 2,048 tokens.
    per_token = 4 * (128 + 128)
    assert shape.kv_bytes(CTX, F16) == (2 * CTX + 14 * 2048) * per_token * 2


def test_a_dense_first_architecture_puts_the_full_layer_first(tmp_path: Path) -> None:
    """cohere2moe: `load_swa_pattern(ml, 4, true)`, so layers 0 and 4 are full."""
    shape = shape_of(tmp_path, swa_model("cohere2moe", blocks=8, pattern=4))
    assert windows(shape) == [None, 2048, 2048, 2048, None, 2048, 2048, 2048]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [(0, [2048] * 6), (1, [None] * 6)],
    ids=["0-is-every-layer-sliding", "1-is-no-layer-sliding"],
)
def test_a_scalar_of_zero_or_one(tmp_path: Path, pattern: int, expected: list) -> None:
    shape = shape_of(tmp_path, swa_model("gpt-oss", blocks=6, pattern=pattern))
    assert windows(shape) == expected


def test_prediction_layers_are_never_sliding(tmp_path: Path) -> None:
    """`set_swa_pattern` walks `n_layer()`, which is `block_count` less
    `nextn_predict_layers`, and marks the rest full."""
    kv = swa_model("gpt-oss", blocks=5, pattern=2, **{"gpt-oss.nextn_predict_layers": 1})
    assert windows(shape_of(tmp_path, kv)) == [2048, None, 2048, None, None]


@pytest.mark.parametrize(
    "kv",
    [
        # No loader of this architecture reads the key as a period.
        swa_model("llama", blocks=8, pattern=4),
        # llama4's loader overrides the window with 8192, so the file's
        # own window is not the one in effect.
        swa_model("llama4", blocks=8, pattern=4),
        # A pattern with no window says a layer slides but not how far.
        {
            k: v
            for k, v in swa_model("plamo3", blocks=8, pattern=4).items()
            if not k.endswith("sliding_window")
        },
    ],
    ids=["unread-architecture", "window-not-the-files", "no-window"],
)
def test_a_scalar_that_does_not_apply_leaves_the_scalars(tmp_path: Path, kv: dict) -> None:
    shape = shape_of(tmp_path, kv)
    assert shape.layers is None
    blocks = kv[f"{kv['general.architecture']}.block_count"]
    assert shape.kv_bytes(CTX, F16) == CTX * blocks * 4 * 256 * 2


# -- (b) attention.shared_kv_layers ------------------------------------------


def gemma_e_model(arch: str, *, blocks: int, shared: int) -> dict:
    """The keys `conversion/gemma.py` writes for an E-model: a bool
    pattern, full and sliding head sizes, a window, and the share count."""
    return {
        "general.architecture": arch,
        f"{arch}.block_count": blocks,
        f"{arch}.context_length": 131072,
        f"{arch}.embedding_length": 1536,
        f"{arch}.attention.head_count": 8,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.key_length": 512,
        f"{arch}.attention.value_length": 512,
        f"{arch}.attention.key_length_swa": 256,
        f"{arch}.attention.value_length_swa": 256,
        f"{arch}.attention.sliding_window": 512,
        f"{arch}.attention.sliding_window_pattern": [(i % 5) != 4 for i in range(blocks)],
        f"{arch}.attention.shared_kv_layers": shared,
    }


def own_cache(layers_with_kv: int) -> int:
    """K and V bytes at `CTX` for the first `layers_with_kv` layers of
    `gemma_e_model`, counted layer by layer."""
    total = 0
    for i in range(layers_with_kv):
        slides = (i % 5) != 4
        tokens = min(CTX, 512) if slides else CTX
        total += tokens * 1 * ((256 + 256) if slides else (512 + 512))
    return total * 2


def test_gemma4_shared_layers_hold_no_cache_of_their_own(tmp_path: Path) -> None:
    """35 blocks, 20 shared: the first 15 hold a cache, the last 20 reuse."""
    shape = shape_of(tmp_path, gemma_e_model("gemma4", blocks=35, shared=20))
    assert shape.kv_bytes(CTX, F16) == own_cache(15)
    assert shape.attention_layers == 15
    # The context-solving form agrees: the shared layers add no slope.
    slope, fixed = shape.kv_terms(F16) or (0.0, 0)
    assert slope * CTX + fixed == own_cache(15)


@pytest.mark.parametrize(
    "shared",
    [0, 29],
    # 29 of 30 leaves one layer with a cache, a file llama.cpp refuses to
    # load (`GGML_ASSERT(n_layer_kv_from_start >= 2)`), so nothing is
    # assumed shared rather than everything.
    ids=["nothing-shared", "a-boundary-llama.cpp-refuses"],
)
def test_gemma4_with_nothing_shared_caches_every_layer(tmp_path: Path, shared: int) -> None:
    shape = shape_of(tmp_path, gemma_e_model("gemma4", blocks=30, shared=shared))
    assert shape.kv_bytes(CTX, F16) == own_cache(30)
    assert shape.attention_layers == 30


@pytest.mark.parametrize(
    ("blocks", "shared"),
    [(30, 10), (35, 15), (30, 0)],
    ids=["e2b", "e4b", "the-key-is-not-what-llama.cpp-reads"],
)
def test_gemma3n_caches_its_first_twenty_layers(tmp_path: Path, blocks: int, shared: int) -> None:
    """gemma3n's loader writes `n_layer_kv_from_start = 20` and never reads
    the key; both published E-models (30 - 10, 35 - 15) agree with it."""
    shape = shape_of(tmp_path, gemma_e_model("gemma3n", blocks=blocks, shared=shared))
    assert shape.kv_bytes(CTX, F16) == own_cache(20)
    assert shape.attention_layers == 20


def test_shared_layers_on_another_architecture_are_not_read(tmp_path: Path) -> None:
    shape = shape_of(tmp_path, gemma_e_model("gemma3", blocks=30, shared=10))
    assert shape.kv_bytes(CTX, F16) == own_cache(30)


# -- (c) MLA: the cache is the compressed latent -----------------------------


def deepseek_mla(**extra: Any) -> dict:
    """What `conversion/deepseek.py` writes for an MLA model, with
    DeepSeek-V3's numbers: kv_lora_rank 512, qk_rope 64, qk_nope 128,
    v_head_dim 128, 128 heads converted into one MQA head of 576."""
    arch = "deepseek2"
    return {
        "general.architecture": arch,
        f"{arch}.block_count": 61,
        f"{arch}.context_length": 163840,
        f"{arch}.embedding_length": 7168,
        f"{arch}.attention.head_count": 128,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.kv_lora_rank": 512,
        f"{arch}.attention.key_length": 512 + 64,
        f"{arch}.attention.value_length": 512,
        f"{arch}.attention.key_length_mla": 128 + 64,
        f"{arch}.attention.value_length_mla": 128,
        f"{arch}.rope.dimension_count": 64,
        **extra,
    }


def test_an_mla_cache_holds_the_latent_and_no_value(tmp_path: Path) -> None:
    shape = shape_of(tmp_path, deepseek_mla())
    # 576 per token per layer, K only. Read as K plus V it was 1,088.
    assert shape.kv_bytes(CTX, F16) == CTX * 61 * 576 * 2
    slope, fixed = shape.kv_terms(F16) or (0.0, 0)
    assert (slope, fixed) == (61 * 576 * 2, 0)


@pytest.mark.parametrize(
    ("kv", "per_token"),
    [
        # A file converted before MLA was: no `_mla` keys, 128 full heads,
        # and llama.cpp caches K and V for every one of them.
        (
            {
                **{
                    k: v
                    for k, v in deepseek_mla().items()
                    if not k.endswith(("_mla", "head_count_kv", "key_length", "value_length"))
                },
                "deepseek2.attention.head_count_kv": 128,
                "deepseek2.attention.key_length": 192,
                "deepseek2.attention.value_length": 128,
            },
            128 * (192 + 128),
        ),
        # `is_mla` needs both lengths; one is not MLA.
        (
            {k: v for k, v in deepseek_mla().items() if not k.endswith("value_length_mla")},
            576 + 512,
        ),
        # The keys on an architecture whose loader never reads them.
        (
            {
                **{k.replace("deepseek2.", "llama."): v for k, v in deepseek_mla().items()},
                "general.architecture": "llama",
            },
            576 + 512,
        ),
    ],
    ids=["converted-before-mla", "one-mla-length", "unread-architecture"],
)
def test_without_mla_both_k_and_v_are_cached(
    tmp_path: Path, kv: dict[str, Any], per_token: int
) -> None:
    shape = shape_of(tmp_path, kv)
    assert shape.kv_bytes(CTX, F16) == CTX * 61 * per_token * 2


def test_the_on_disk_route_reads_mla_too(configured_client: TestClient, models_dir: Path) -> None:
    """The stored entry keeps every scalar key, so the Library's own fit
    route reaches the same reader: one number for one file."""
    kv = qwen_like_kv(architecture="deepseek2", vocab=64)
    kv.update(deepseek_mla())
    model_id = scanned(configured_client, models_dir, kv)
    body = configured_client.get(f"/v1/models/{model_id}/fit", params={"contextLength": CTX}).json()
    assert body["fit"]["basis"] == "metadata"
    assert body["fit"]["kvCacheBytes"] == CTX * 61 * 576 * 2
