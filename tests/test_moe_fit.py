"""MoE-aware fit (moe-aware-fit A3b): expert bytes from the tensor table, and what moves."""

from __future__ import annotations

import struct
from pathlib import Path

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library._generated.models import FitOffload, FitVerdict, HostHardware
from eugene_plexus_library.formats import gguf

from .conftest import qwen_like_kv, write_gguf

GIB = 2**30
# ggml type ids and sizes, restated rather than imported: a fixture that
# shares constants with the code under test can agree with it about
# something wrong.
F32, Q8_0, Q4_K, UNKNOWN = 0, 8, 12, 99


def _tensor(name: str, dims: list[int], ggml_type: int, offset: int = 0) -> bytes:
    raw = name.encode()
    out = struct.pack("<Q", len(raw)) + raw + struct.pack("<I", len(dims))
    out += b"".join(struct.pack("<Q", d) for d in dims)
    return out + struct.pack("<I", ggml_type) + struct.pack("<Q", offset)


def _model(tmp_path: Path, tensors: list[tuple[str, list[int], int]], name="m.gguf") -> Path:
    payload = b"".join(_tensor(n, d, t) for n, d, t in tensors)
    return write_gguf(tmp_path / name, qwen_like_kv(), tensor_count=len(tensors), payload=payload)


def test_the_tensor_table_gives_expert_bytes_and_never_needs_the_weights(tmp_path):
    path = _model(
        tmp_path,
        [
            ("token_embd.weight", [2048, 1000], Q8_0),  # 2,048,000 / 32 * 34
            ("blk.0.attn_q.weight", [2048, 2048], Q4_K),  # 4,194,304 / 256 * 144
            ("blk.0.ffn_up_exps.weight", [2048, 768, 128], Q4_K),
            ("blk.0.ffn_down_chexps.weight", [768, 2048, 128], Q4_K),
            ("blk.0.ffn_up_shexp.weight", [2048, 768], Q4_K),  # shared: stays
            ("output_norm.weight", [2048], F32),
        ],
    )
    meta = gguf.read_metadata(path, tensors=True)
    embd = 2048 * 1000 // 32 * 34
    attn = 2048 * 2048 // 256 * 144
    exps = 2048 * 768 * 128 // 256 * 144
    shared = 2048 * 768 // 256 * 144
    assert meta.expert_bytes == 2 * exps
    assert meta.tensor_bytes == embd + attn + 2 * exps + shared + 2048 * 4
    # Off by default, so a remote preflight's window is unchanged.
    plain = gguf.read_metadata(path)
    assert plain.expert_bytes is None and plain.tensor_bytes is None


def test_a_type_the_reader_does_not_know_is_unknown_not_guessed(tmp_path):
    path = _model(
        tmp_path,
        [("blk.0.ffn_up_exps.weight", [256, 256], Q4_K), ("x.weight", [256], UNKNOWN)],
    )
    meta = gguf.read_metadata(path, tensors=True)
    assert meta.expert_bytes is None and meta.tensor_bytes is None


def test_a_dense_model_has_zero_expert_bytes(tmp_path):
    path = _model(tmp_path, [("blk.0.ffn_up.weight", [2048, 2048], Q4_K)])
    assert gguf.read_metadata(path, tensors=True).expert_bytes == 0


# --------------------------------------------------------------------------- #
# What moves, on design §0's measured numbers
# --------------------------------------------------------------------------- #

MOE_WEIGHTS = int(18.56e9)  # Qwen3-30B-A3B Q4_K_M, file size
MOE_EXPERTS = int(16.348 * GIB)
DENSE_WEIGHTS = int(15.401 * GIB)  # Qwen3.6-27B Q4_K_M


def _budget(vram_gib: float, ram_gib: float):
    hardware = HostHardware.model_validate(
        {"hostname": "t", "os": "windows", "arch": "x64", "gpus": [], "ramTotalBytes": 64 * GIB}
    )
    return fit_mod.budget_from_hardware(
        hardware, vram_override=int(vram_gib * GIB), ram_override=int(ram_gib * GIB)
    )


def _shape():
    # 48 layers, 4 KV heads of 128: the 30B-A3B's cache arithmetic.
    return fit_mod.ModelShape(
        block_count=48, head_count_kv=4, key_length=128, value_length=128, context_length=262144
    )


def test_an_8gb_card_moves_experts_for_the_moe_model_and_layers_for_the_dense_one():
    budget = _budget(7.5, 24)
    moe = fit_mod.compute(
        weights_bytes=MOE_WEIGHTS,
        budget=budget,
        context_length=16384,
        shape=_shape(),
        expert_bytes=MOE_EXPERTS,
    )
    dense = fit_mod.compute(
        weights_bytes=DENSE_WEIGHTS,
        budget=budget,
        context_length=16384,
        shape=_shape(),
        expert_bytes=0,
    )
    # One verdict word for both, as design §0 M3 measured; the new field
    # is what tells 46.5 tok/s from 4.9.
    assert moe.verdict is FitVerdict.split and dense.verdict is FitVerdict.split
    assert moe.offload is FitOffload.experts
    assert dense.offload is FitOffload.layers
    assert moe.expertBytes == MOE_EXPERTS
    assert any("system memory" in n for n in moe.notes or [])


def test_offload_is_null_where_there_is_nothing_to_move_or_nothing_known():
    unknown = fit_mod.compute(
        weights_bytes=MOE_WEIGHTS, budget=_budget(7.5, 24), context_length=16384, shape=_shape()
    )
    assert unknown.offload is None  # expert share unknown: no description
    fits = fit_mod.compute(
        weights_bytes=MOE_WEIGHTS,
        budget=_budget(30, 24),
        context_length=16384,
        shape=_shape(),
        expert_bytes=MOE_EXPERTS,
    )
    assert fits.verdict is FitVerdict.fits and fits.offload is None


def test_a_card_too_small_even_for_the_non_expert_part_moves_layers():
    tiny = fit_mod.compute(
        weights_bytes=MOE_WEIGHTS,
        budget=_budget(1.5, 60),
        context_length=65536,
        shape=_shape(),
        expert_bytes=MOE_EXPERTS,
    )
    assert tiny.verdict is FitVerdict.split and tiny.offload is FitOffload.layers


def test_the_context_with_experts_in_ram_is_offered_where_nothing_else_is():
    budget = _budget(7.5, 24)
    common = {"weights_bytes": MOE_WEIGHTS, "budget": budget, "shape": _shape()}
    # Entirely on the card: nothing, which is why a small card got no context.
    assert fit_mod.max_context_that_fits(**common) is None
    offered = fit_mod.max_context_experts_in_ram(expert_bytes=MOE_EXPERTS, **common)
    assert offered is not None and offered >= 32768 and offered % 256 == 0
    # A dense model, or an unknown share, is not offered one.
    assert fit_mod.max_context_experts_in_ram(expert_bytes=0, **common) is None
    assert fit_mod.max_context_experts_in_ram(expert_bytes=None, **common) is None
    # Experts that do not fit in RAM and the card together: none.
    small_ram = {**common, "budget": _budget(7.5, 4)}
    assert fit_mod.max_context_experts_in_ram(expert_bytes=MOE_EXPERTS, **small_ram) is None


def test_a_sharded_models_experts_are_summed_across_its_parts(tmp_path):
    from eugene_plexus_library.scanner import Scanner

    root = tmp_path / "root"
    parts = []
    for index in (1, 2):
        payload = _tensor(f"blk.{index}.ffn_up_exps.weight", [256, 256, 8], Q4_K)
        kv = qwen_like_kv() | {"split.no": index - 1, "split.count": 2}
        parts.append(
            write_gguf(root / f"moe-0000{index}-of-00002.gguf", kv, tensor_count=1, payload=payload)
        )
    result = Scanner().scan([root])
    model = next(m for m in result.models if m.path == str(parts[0]))
    assert model.gguf.shardCount == 2
    assert model.gguf.expertBytes == 2 * (256 * 256 * 8 // 256 * 144)


def test_the_fit_route_says_experts_move_and_offers_their_context(configured_client, models_dir):
    import time

    experts = 256 * 256 * 64 // 256 * 144  # 1,179,648 bytes of experts
    payload = _tensor("blk.0.ffn_up_exps.weight", [256, 256, 64], Q4_K)
    payload += _tensor("blk.0.attn_q.weight", [256, 256], Q4_K)
    kv = qwen_like_kv(vocab=64)
    kv["llama.block_count"] = 4
    kv["llama.attention.head_count_kv"] = 2
    kv["llama.attention.key_length"] = 64
    kv["llama.attention.value_length"] = 64
    # Pad the file past its declared tensors so weights exceed the experts,
    # as a real file's do.
    write_gguf(models_dir / "moe.gguf", kv, tensor_count=2, payload=payload + bytes(2 * experts))

    def scan():
        while configured_client.get("/v1/scan").json()["state"] == "scanning":
            time.sleep(0.01)
        assert configured_client.post("/v1/scan").status_code == 202
        deadline = time.perf_counter() + 5
        while configured_client.get("/v1/scan").json()["state"] == "scanning":
            assert time.perf_counter() < deadline
            time.sleep(0.01)

    scan()
    model = next(m for m in configured_client.get("/v1/models").json()["models"])
    assert model["gguf"]["expertBytes"] == experts
    # Ask once with room to spare for the route's own arithmetic, then give
    # the card just less than the whole model but more than all but experts.
    roomy = configured_client.get(
        f"/v1/models/{model['id']}/fit", params={"vramBytes": 64 * GIB, "ramBytes": 64 * GIB}
    ).json()["fit"]
    everything = roomy["requiredBytes"]
    card = everything - experts // 2
    answer = configured_client.get(
        f"/v1/models/{model['id']}/fit", params={"vramBytes": card, "ramBytes": 64 * GIB}
    ).json()
    assert answer["fit"]["verdict"] in ("tight", "split")
    assert answer["fit"]["offload"] == "experts"
    assert answer["fit"]["expertBytes"] == experts
    assert answer["maxContextExpertsInRam"] is not None


def test_the_scanner_records_expert_bytes_and_rereads_an_entry_cached_without_them(tmp_path):
    from eugene_plexus_library.scanner import Scanner, cache_key

    path = _model(
        tmp_path / "root",
        [
            ("blk.0.ffn_up_exps.weight", [256, 256, 8], Q4_K),
            ("blk.0.attn_q.weight", [256, 256], Q4_K),
        ],
    )
    first = Scanner().scan([tmp_path / "root"])
    model = next(m for m in first.models if m.path == str(path))
    assert model.gguf.expertBytes == 256 * 256 * 8 // 256 * 144

    stale = model.model_copy(update={"gguf": model.gguf.model_copy(update={"expertBytes": None})})
    key = cache_key(path, path.stat())
    second = Scanner(cache_lookup=lambda k: stale if k == key else None).scan([tmp_path / "root"])
    again = next(m for m in second.models if m.path == str(path))
    assert again.gguf.expertBytes == model.gguf.expertBytes
