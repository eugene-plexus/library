"""Row 68: facts from headers, never model names in a catalogue listing."""

from pathlib import Path

import pytest

from eugene_plexus_library import fit, model_fit, preflight
from eugene_plexus_library._generated.models import (
    EngineKind,
    FitOffload,
    FitVerdict,
    KvCacheType,
    MemoryBudget,
)
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.scanner import Scanner

from .conftest import pad_gguf, write_gguf
from .test_moe_fit import _tensor
from .test_preflight import SlicedFile, client_for

GIB = 2**30


def header(**extra: object) -> dict:
    return {
        "general.architecture": "deepseek4",
        "general.name": "DeepSeek-V4-Flash",
        "deepseek4.block_count": 44,
        "deepseek4.nextn_predict_layers": 1,
        "deepseek4.context_length": 1048576,
        "deepseek4.attention.head_count_kv": 1,
        "deepseek4.attention.key_length": 576,
        "deepseek4.attention.value_length": 512,
        "deepseek4.attention.key_length_mla": 192,
        "deepseek4.attention.value_length_mla": 128,
        **extra,
    }


def model_file(path: Path, *, head: bool = True, **extra: object) -> Path:
    names = ["blk.0.attn_norm.weight", "blk.0.ffn_up_exps.weight"]
    if head:
        names.append("blk.43.nextn.eh_proj.weight")
    table = b"".join(_tensor(n, [32], 0) for n in names)
    return pad_gguf(write_gguf(path, header(**extra), tensor_count=len(names), payload=table), 128)


def test_v4_mla_does_not_charge_for_a_value_cache(tmp_path: Path) -> None:
    meta = gguf.read_metadata(model_file(tmp_path / "m.gguf"), tensors=True)
    shape = preflight.shape_from_gguf(meta)
    assert shape.layers is not None
    assert all(layer.value_length == 0 for layer in shape.layers)
    assert shape.kv_bytes(8192, KvCacheType.f16) == 44 * 8192 * 576 * 2


def test_v4_fit_keeps_experts_in_ram_and_uses_the_selected_cache(tmp_path: Path) -> None:
    model = Scanner().scan([model_file(tmp_path / "m.gguf").parent]).models[0]
    assert model.gguf is not None and model.gguf.expertBytes == 128
    budget = MemoryBudget(
        vramFreeBytes=12 * GIB,
        vramTotalBytes=12 * GIB,
        ramAvailableBytes=128 * GIB,
        ramTotalBytes=128 * GIB,
        gpuCount=1,
    )
    results = [
        fit.compute(
            weights_bytes=110 * GIB,
            expert_bytes=105 * GIB,
            budget=budget,
            context_length=8192,
            shape=model_fit.shape_of(model),
            kv_cache_type=cache,
        )
        for cache in (KvCacheType.f16, KvCacheType.q8_0)
    ]
    for result in results:
        assert result.verdict is FitVerdict.split and result.offload is FitOffload.experts
    assert results[1].kvCacheBytes < results[0].kvCacheBytes
    assert results[0].kvCacheBytes == 44 * 8192 * 576 * 2


@pytest.mark.parametrize("head", [True, False])
def test_v4_fallback_and_own_head_are_independent_header_facts(tmp_path: Path, head: bool) -> None:
    model_file(tmp_path / "renamed.gguf", head=head)
    model = Scanner().scan([tmp_path]).models[0]
    assert model.capabilities is not None and model.capabilities.chatTemplate is False
    assert model.gguf is not None and model.gguf.speculation is not None
    assert model.gguf.speculation.mtpHead is head
    fallback = model.gguf.chatTemplateFallback
    assert fallback is not None
    assert fallback.id == "deepseek-v4"
    assert fallback.name == "DeepSeek V4"
    assert fallback.engines == [EngineKind.llama_cpp]
    assert "/b11719/" in str(fallback.sourceUrl)


@pytest.mark.parametrize(
    "extra",
    [
        {"tokenizer.chat_template": "{{ messages }}"},
        {"general.architecture": "llama"},
        {"general.name": "DeepSeek-V4.1-Flash"},
        {"general.name": "DeepSeek-V4-Flash-0731"},
        {"general.name": "Unknown"},
    ],
)
def test_no_fallback_for_embedded_or_unreviewed_variants(tmp_path: Path, extra: dict) -> None:
    model_file(tmp_path / "DeepSeek-V4-Flash.gguf", **extra)
    model = Scanner().scan([tmp_path]).models[0]
    assert model.gguf is not None and model.gguf.chatTemplateFallback is None


async def test_remote_preflight_uses_the_same_header_evidence(tmp_path: Path) -> None:
    served = SlicedFile(model_file(tmp_path / "m.gguf"))
    result = await preflight.preflight(
        client_for(served),
        repo="org/renamed",
        revision="pinned",
        path="m.gguf",
        budget=MemoryBudget(),
        context_length=8192,
    )
    assert result.capabilities is not None and result.capabilities.chatTemplate is False
    assert result.chatTemplateFallback is not None
    assert result.chatTemplateFallback.id == "deepseek-v4"


def test_a_cached_pre_row68_header_gains_the_fallback(tmp_path: Path) -> None:
    model_file(tmp_path / "m.gguf")
    scanner = Scanner()
    old = scanner.scan([tmp_path]).models[0]
    assert old.gguf is not None
    old = old.model_copy(
        update={"gguf": old.gguf.model_copy(update={"chatTemplateFallback": None})}
    )
    model = Scanner(cache_lookup=lambda _: old).scan([tmp_path]).models[0]
    assert model.gguf is not None and model.gguf.chatTemplateFallback is not None


def test_a_split_v4_keeps_the_head_in_a_later_shard_and_refreshes_cached_facts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "flash-00001-of-00002.gguf"
    last = tmp_path / "flash-00002-of-00002.gguf"
    for index, path, tensor in [
        (0, first, "blk.0.attn_norm.weight"),
        (1, last, "blk.43.nextn.eh_proj.weight"),
    ]:
        pad_gguf(
            write_gguf(
                path,
                header(**{"split.count": 2, "split.no": index}),
                tensor_count=1,
                payload=_tensor(tensor, [32], 0),
            ),
            128,
        )
    model = Scanner().scan([tmp_path]).models[0]
    assert model.gguf.speculation.mtpHead is True
    old = model.model_copy(
        update={
            "gguf": model.gguf.model_copy(
                update={"speculation": model.gguf.speculation.model_copy(update={"mtpHead": False})}
            )
        }
    )
    refreshed = Scanner(cache_lookup=lambda _: old).scan([tmp_path]).models[0]
    assert refreshed.gguf.speculation.mtpHead is True
