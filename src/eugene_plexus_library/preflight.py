"""Read a remote candidate's real metadata before committing to the download.

The milestone's best result, and the reason quant guidance here is not
filename guesswork.

## Measured, on a 16.5 GB GGUF

    Range: bytes=0-12582911     12.6 MB in 0.8 s
    kv_block = 10,945,379 B     0.066% of the file

    general.file_type              = 15   -> Q4_K_M, machine-read
    qwen35.block_count             = 65
    qwen35.full_attention_interval = 4    -> 16 layers hold KV, not 65
    qwen35.attention.head_count_kv = 4
    qwen35.attention.key_length    = 256
    qwen35.attention.value_length  = 256
    qwen35.context_length          = 262144

The KV block sits at the *front* of a GGUF, so the whole of what the fit
arithmetic needs is reachable with one ranged read. Safetensors is two
orders of magnitude cheaper: eight bytes give the header length, a
second read gives the header, and the parameter count in it is exact.

## Why the window grows instead of being guessed

The KV block's size is not knowable in advance — it is dominated by the
tokenizer, which is 10.9 MB on a 248k-vocab model and 1,743 bytes on a
projector. So the probe starts small and the parser says how far it
needed to get: a 1 MB probe of the Qwen file failed cleanly with
*"wanted to reach offset 1,048,581"*, and one more request finished it.

Each grow **appends** rather than re-reading from zero, and the second
probe jumps straight to 12 MB — the measured worst case in the wild —
instead of doubling. Both of those were earned by running it: re-reading
cost 1+2+4+8+16 MB of transfer to reach an 11 MB block, and doubling
from 1 MB took four requests where two will do.

## Explicit, never automatic

It spends someone else's bandwidth per call. Firing it across a listing
would be ~270 MB for one repo, so it is an endpoint an operator reaches
for on the one candidate they have settled on — not something a UI does
on hover.
"""

from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath

from . import fit as fit_mod
from ._generated.models import (
    CataloguePreflight,
    KvCacheType,
    MemoryBudget,
    ModelCapabilities,
    ModelFormat,
    RecommendedSampling,
)
from .catalogue import quant_label
from .formats import gguf, safetensors
from .hub import HubClient, HubError

log = logging.getLogger(__name__)

INITIAL_WINDOW = 1024 * 1024
"""First probe. Covers a projector (1,743 B of KV) and most small
models outright; a large-vocab model needs one more request."""

TYPICAL_KV_WINDOW = 12 * 1024 * 1024
"""Second probe, when the first was short.

Jumped to rather than doubled into. The largest KV block measured in the
wild is 10,945,379 bytes, so this covers a current large-vocab model in
one more request -- where doubling from 1 MB took four, because the
truncation offset a failed parse reports is where it stopped inside the
vocabulary walk and not the size of the whole block."""

MAX_WINDOW = 64 * 1024 * 1024
"""Ceiling on how much of someone else's file we will pull to read a
header. The largest KV block measured in the wild is ~11 MB, so this is
six times the worst real case — and a file that wants more than this is
malformed or is not what it claims, which is worth refusing rather than
chasing."""


def attention_layers(meta: gguf.GgufMetadata) -> int | None:
    """How many layers actually hold a KV cache.

    **Not** the layer count on a hybrid attention/SSM model. A verified
    current release declares `block_count: 65` with
    `full_attention_interval: 4`, so 16 layers cache and the naive
    arithmetic over-estimates the cache by 4.1x at every context length
    — at that model's own trained context, the difference between "fits
    on a 5090" and "no candidate in this repo is runnable".

    Nor on a model whose later layers reuse an earlier layer's cache
    (`kv_layers_from_start`): those hold none of their own.

    Only the conventions actually seen are honoured, and a model using a
    different one falls through to the block count. That errs
    pessimistic, which is the right direction: an over-estimate costs an
    operator a second look, an under-estimate costs them an OOM they
    were promised would not happen.
    """
    blocks = meta.arch_key("block_count")
    if not isinstance(blocks, int) or blocks <= 0:
        return None

    own = kv_layers_from_start(meta)
    if own is not None:
        return own

    interval = meta.arch_key("full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        return max(blocks // interval, 1)

    # Some hybrids list the attention layers outright instead.
    indices = meta.arch_key("attention.layer_indices")
    if isinstance(indices, list) and indices:
        return len(indices)

    return blocks


# Three conventions llama.cpp reads and this module once did not, each
# making the estimate too large (upstream drift audit, 2026-10-03). Every
# rule below is llama.cpp's own at tag b11375, read in its source, and is
# applied only where that source applies it: an architecture outside a
# table keeps the plain reading, which errs pessimistic.

_SWA_PERIOD_DENSE_FIRST: dict[str, bool] = {
    "afmoe": False,
    "cohere2": False,
    "cohere2moe": True,
    "exaone-moe": False,
    "gemma2": False,
    "gemma3": False,
    "gemma3n": False,
    "gpt-oss": False,
    "laguna": True,
    "mellum": False,
    "muse-glimmer": False,
    "olmo2": False,
    "plamo3": False,
}
"""Architectures whose loader reads a scalar `sliding_window_pattern` as
a period, mapped to that loader's `dense_first`.

`llama_model_base::load_swa_pattern` takes the per-layer array when there
is one, else reads the scalar into `n_pattern` and calls
`llama_hparams::set_swa_pattern(n_pattern, dense_first)` (ggml-org/
llama.cpp#29042: "the loaders read a scalar as a period"). Each name here
is a `general.architecture` whose `src/models/*.cpp` calls it with the
window taken from the file's own `attention.sliding_window`.

Left out on purpose, so a scalar there keeps the plain reading: `llama4`
and `smallthinker` (their loaders overwrite the window with 8192 and 4096,
so the file's window is not the one in effect), `exaone4` (it slides only
when the model has 64 layers), and `gemma-embedding` and `modern-bert`
(encoders with symmetric windows). The architectures that read only the
array (`gemma4`, `step35`, `mimo2` and others) are not here either: the
array is already read below.

A file of a listed architecture with **no** pattern key gets the period
its loader defaults to (6 for `gemma3`, 2 for `gpt-oss`) from llama.cpp,
and the plain reading here; reading that default is a separate change."""


def swa_period(meta: gguf.GgufMetadata) -> list[bool] | None:
    """Which layers slide, from a scalar `sliding_window_pattern`.

    `set_swa_pattern`: with `dense_first` every `n`-th layer starting at 0
    is full, else every `n`-th layer ending a period is; 0 is every layer
    sliding and 1 none; and the layers past `n_layer()`, which is
    `block_count` less `nextn_predict_layers`, are full. `None` when the
    file has no scalar, the architecture does not read one as a period,
    or there is no window to slide over (most of those loaders turn
    sliding off without one).
    """
    raw = meta.arch_key("attention.sliding_window_pattern")
    # A GGUF bool is a Python bool, which is also an int.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    pattern: int = raw
    dense_first = _SWA_PERIOD_DENSE_FIRST.get(meta.architecture or "")
    blocks = meta.arch_key("block_count")
    window = meta.arch_key("attention.sliding_window")
    if dense_first is None or not isinstance(blocks, int) or blocks <= 0:
        return None
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    nextn = meta.arch_key("nextn_predict_layers")
    layers = blocks - (nextn if isinstance(nextn, int) and 0 < nextn <= blocks else 0)

    def slides(index: int) -> bool:
        if index >= layers:
            return False
        if pattern == 0:
            return True
        if dense_first:
            return index % pattern != 0
        return index % pattern < pattern - 1

    return [slides(index) for index in range(blocks)]


def kv_layers_from_start(meta: gguf.GgufMetadata) -> int | None:
    """How many leading layers hold a cache of their own, when the rest
    reuse one; `None` when every layer holds its own.

    llama.cpp's `n_layer_kv_from_start`: `llama_hparams::has_kv` gives a
    layer before it a cache and none to a layer at or after it, which
    `create_memory`'s `reuse` callback points at the layer two before the
    boundary if it slides and one before if not. Set by two loaders only:

    * `gemma4`: `block_count - attention.shared_kv_layers` (the key
      optional, 0 when absent);
    * `gemma3n`: 20, written into the loader. It does not read
      `shared_kv_layers` at all, and both published E-models agree
      (30 - 10 and 35 - 15).

    A boundary at or past the last layer shares nothing. One below 2 is
    a file llama.cpp refuses to load (it asserts `>= 2`), so nothing is
    assumed shared for it either.
    """
    blocks = meta.arch_key("block_count")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        return None
    arch = meta.architecture
    if arch == "gemma3n":
        start = 20
    elif arch == "gemma4":
        shared = meta.arch_key("attention.shared_kv_layers")
        start = blocks - (shared if isinstance(shared, int) and not isinstance(shared, bool) else 0)
    else:
        return None
    if start < 2 or start >= blocks:
        return None
    return start


_MLA_ARCHITECTURES = frozenset(
    {
        "bailingmoe3",
        "deepseek2",
        "deepseek32",
        "dots3note",
        "glm-dsa",
        "glm5-next",
        "hy_v4",
        "kimi-k3",
        "kimi-linear",
    }
)
"""The architectures whose loader reads `attention.key_length_mla` and
`attention.value_length_mla` (`src/models/*.cpp` at b11375)."""


def caches_values(meta: gguf.GgufMetadata) -> bool:
    """False for multi-head latent attention, whose cache holds K alone.

    `llama_hparams::is_mla` is both `_mla` lengths set, and `llama_kv_cache`
    then allocates K and no V for every layer (`has_v = !is_mla`). K is
    the compressed latent already: the converter writes `key_length` as
    `kv_lora_rank + qk_rope_head_dim` (576 on DeepSeek-V3), `value_length`
    as `kv_lora_rank` and `head_count_kv` as 1, so the plain reading of K
    plus V counted the latent twice (1,088 elements a token a layer where
    llama.cpp stores 576). A file converted before MLA has neither `_mla`
    key, 128 full heads, and both caches, which the plain reading gets
    right.
    """
    if meta.architecture not in _MLA_ARCHITECTURES:
        return True

    def positive(suffix: str) -> bool:
        value = meta.arch_key(suffix)
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    return not (positive("attention.key_length_mla") and positive("attention.value_length_mla"))


def per_layer_kv(meta: gguf.GgufMetadata) -> tuple[fit_mod.LayerKV, ...] | None:
    """One `LayerKV` per layer, when the file says enough to build them.

    **`attention.head_count_kv` is not always an integer.** On a current
    mainstream 12B it is an array of 48 -- eight heads on five layers of
    every six, one on the sixth -- and the scalar reader below returns
    `None` for it and falls back to `head_count`, which is the
    pre-grouped-query assumption. Together with the sliding-window
    layers that array goes with, the scalar arithmetic reported
    **24.0 GiB** of KV at 16k where the truth is **0.56 GiB**.

    `attention.sliding_window_pattern` is a bool array, `True` where the
    layer slides; `key_length_swa` / `value_length_swa` are that layer's
    own dimensions, which are shorter. A **scalar** pattern is a period,
    read the way llama.cpp's loader for that architecture reads it
    (`swa_period`). Absent either, a declared `sliding_window` with no
    per-layer mask is not applied at all -- guessing which layers slide
    would be inventing the number this module exists to stop inventing.

    Two more conventions make layers unlike each other: later layers that
    reuse an earlier layer's cache hold none of their own
    (`kv_layers_from_start`; a `LayerKV` with no heads), and a multi-head
    latent attention cache holds K and no V (`caches_values`; a value
    length of 0).

    `None` means "use the scalars", which is most files and every older
    one.
    """

    def integer(suffix: str) -> int | None:
        value = meta.arch_key(suffix)
        return int(value) if isinstance(value, int) else None

    blocks = integer("block_count")
    if not blocks:
        return None

    heads_kv = meta.arch_key("attention.head_count_kv")
    pattern = meta.arch_key("attention.sliding_window_pattern")
    if not isinstance(pattern, list):
        pattern = swa_period(meta)
    own_cache = kv_layers_from_start(meta)
    values_cached = caches_values(meta)
    if not isinstance(heads_kv, list) and pattern is None and own_cache is None and values_cached:
        # Nothing per-layer to say. The scalars describe this file.
        return None

    key_length = integer("attention.key_length")
    value_length = integer("attention.value_length")
    if key_length is None or value_length is None:
        embedding = integer("embedding_length")
        head_count = integer("attention.head_count")
        if not embedding or not head_count:
            return None
        key_length = key_length or embedding // head_count
        value_length = value_length or embedding // head_count

    key_swa = integer("attention.key_length_swa") or key_length
    value_swa = integer("attention.value_length_swa") or value_length
    window = integer("attention.sliding_window")

    def head_at(index: int) -> int | None:
        if isinstance(heads_kv, list):
            value = heads_kv[index] if index < len(heads_kv) else None
            return int(value) if isinstance(value, int) else None
        if isinstance(heads_kv, int):
            return heads_kv
        return integer("attention.head_count")

    layers: list[fit_mod.LayerKV] = []
    for index in range(blocks):
        if own_cache is not None and index >= own_cache:
            # Reuses an earlier layer's cache and allocates none.
            layers.append(fit_mod.LayerKV(0, key_length, value_length, None))
            continue
        heads = head_at(index)
        if not heads:
            return None
        slides = (
            bool(pattern[index]) if isinstance(pattern, list) and index < len(pattern) else False
        )
        if slides and not window:
            # It says a layer slides but not how far. Charging it the
            # full context is the pessimistic read, and pessimistic is
            # the direction this module errs in.
            slides = False
        value = value_swa if slides else value_length
        layers.append(
            fit_mod.LayerKV(
                head_count_kv=heads,
                key_length=key_swa if slides else key_length,
                value_length=value if values_cached else 0,
                window=window if slides else None,
            )
        )
    return tuple(layers)


# The keys whose absence-as-a-length means "this file has per-layer
# attention and we did not keep it". Read as names rather than inferred,
# because the point is to notice a term we know we needed.
_PER_LAYER_KEYS = ("attention.head_count_kv", "attention.sliding_window_pattern")


def per_layer_dropped(meta: gguf.GgufMetadata) -> bool:
    """Did this file declare a per-layer term that was stepped over?

    `array_lengths` records the length of every array the reader walked
    past instead of keeping, so a key present there and absent from `kv`
    is one we know existed and cannot use. That is a different situation
    from a file that declares the simple scalar form, and the difference
    is exactly what `basis` is supposed to tell a person.
    """
    arch = meta.architecture
    if arch is None:
        return False
    return any(
        f"{arch}.{suffix}" in meta.array_lengths and meta.arch_key(suffix) is None
        for suffix in _PER_LAYER_KEYS
    )


def shape_from_gguf(meta: gguf.GgufMetadata) -> fit_mod.ModelShape:
    """The KV-cache terms, read off the file's own KV block.

    **The one place a shape is built from a GGUF.** It was not, for a
    while: `routes/guidance.py` grew its own reader at M3 and the 43x
    per-layer fix landed here and not there, so `GET
    /v1/models/{id}/fit` answered 4,864 where the starter set answered
    262,144 for the same file -- both saying `basis: metadata` (review
    §6.1 #4). That route delegates here now, and the rule is that a
    second shape builder is the bug rather than the fix.
    """

    def integer(suffix: str) -> int | None:
        value = meta.arch_key(suffix)
        return int(value) if isinstance(value, int) else None

    blocks = integer("block_count")
    layers = per_layer_kv(meta)
    return fit_mod.ModelShape(
        block_count=blocks,
        attention_layers=attention_layers(meta),
        head_count_kv=integer("attention.head_count_kv"),
        key_length=integer("attention.key_length"),
        value_length=integer("attention.value_length"),
        embedding_length=integer("embedding_length"),
        head_count=integer("attention.head_count"),
        context_length=meta.context_length,
        layers=layers,
        per_layer_unavailable=layers is None and per_layer_dropped(meta),
    )


async def read_gguf_header(
    client: HubClient, *, repo: str, revision: str, path: str
) -> tuple[gguf.GgufMetadata, int]:
    """`(metadata, bytes_read)` for a remote GGUF, over HTTP Range.

    Grows the window on `GgufTruncated` rather than guessing: the
    exception carries the offset the parser wanted, so the retry asks
    for exactly that much.
    """
    window = INITIAL_WINDOW
    buffer = b""
    while True:
        # Append rather than re-read from zero. Growing by re-reading
        # cost 1+2+4+8+16 MB of transfer to reach an 11 MB KV block,
        # which is twice the file we actually needed and was visible as
        # a 2.3-second preflight on a fast link.
        chunk = await client.read_range(
            repo=repo, revision=revision, path=path, first=len(buffer), last=window - 1
        )
        buffer += chunk
        if not chunk:
            raise HubError(
                f"{path} stopped returning bytes at offset {len(buffer)} while its "
                "header was still incomplete.",
                status=502,
            )
        try:
            # The tensor table too, for expert bytes; it follows the KV
            # block, so the window grows to it the same way.
            meta = gguf.read_metadata_stream(
                io.BytesIO(buffer), name=PurePosixPath(path).name, tensors=True
            )
        except gguf.GgufTruncated as exc:
            if len(buffer) < window:
                # The file itself is shorter than the window, so there
                # is no more to fetch: this is a genuinely truncated or
                # non-GGUF file, not a small probe.
                raise HubError(
                    f"{path} is {len(buffer)} bytes and its header is incomplete. The "
                    "upload is truncated, or the file is not a GGUF.",
                    status=502,
                ) from exc
            target = max(
                exc.needed, TYPICAL_KV_WINDOW if window < TYPICAL_KV_WINDOW else window * 2
            )
            needed = min(target, MAX_WINDOW)
            if needed <= window:
                raise HubError(
                    f"{path}'s header needs more than {MAX_WINDOW // (1024 * 1024)} MB to "
                    "read, which no real model does. Refusing to pull more of the file to "
                    "parse a header.",
                    status=502,
                ) from exc
            log.info("preflight %s: header needs %d bytes, refetching", path, needed)
            window = needed
            continue
        except gguf.GgufError as exc:
            raise HubError(f"{path} is not a readable GGUF: {exc}", status=502) from exc
        return meta, len(buffer)


async def read_safetensors_header(
    client: HubClient, *, repo: str, revision: str, path: str
) -> tuple[safetensors.SafetensorsHeader, int]:
    """`(header, bytes_read)` for a remote safetensors file.

    Two requests and about 11 KB: the first eight bytes are the header
    length, which is the whole trick.
    """
    prefix = await client.read_range(repo=repo, revision=revision, path=path, first=0, last=7)
    try:
        length = safetensors.header_length(prefix, name=PurePosixPath(path).name)
    except safetensors.SafetensorsError as exc:
        raise HubError(f"{path} is not a readable safetensors file: {exc}", status=502) from exc

    body = await client.read_range(
        repo=repo, revision=revision, path=path, first=8, last=8 + length - 1
    )
    try:
        header = safetensors.parse_header(body, name=PurePosixPath(path).name)
    except safetensors.SafetensorsError as exc:
        raise HubError(f"{path} is not a readable safetensors file: {exc}", status=502) from exc
    return header, 8 + len(body)


async def preflight(
    client: HubClient,
    *,
    repo: str,
    revision: str,
    path: str,
    budget: MemoryBudget,
    context_length: int,
    kv_cache_type: KvCacheType = KvCacheType.f16,
    candidate_size: int | None = None,
) -> CataloguePreflight:
    """Read one remote file's metadata and re-score its fit from it.

    `candidate_size` is the whole candidate's size when the caller knows
    it — a split model's shards sum to more than the file being probed,
    and scoring the probe's own size would understate it by every shard
    but the first.
    """
    resolved = await client.head_file(repo=repo, revision=revision, path=path)
    name = PurePosixPath(path).name
    weights_bytes = candidate_size or resolved.size or 0

    if name.lower().endswith(".gguf"):
        meta, bytes_read = await read_gguf_header(client, repo=repo, revision=revision, path=path)
        shape = shape_from_gguf(meta)
        metadata_quant = meta.quantization
        filename_quant = quant_label(name)
        sampling = meta.recommended_sampling

        return CataloguePreflight(
            repo=repo,
            resolvedCommit=resolved.commit,
            file=path,
            format=ModelFormat.gguf,
            bytesRead=bytes_read,
            quantization=metadata_quant or filename_quant,
            fileType=meta.file_type,
            # Both are reported and neither silently preferred when they
            # disagree: a requantized file keeps its old name more often
            # than a metadata field is wrong, but not always.
            agreesWithFilename=(
                None
                if metadata_quant is None or filename_quant is None
                else filename_quant.endswith(metadata_quant)
            ),
            architecture=meta.architecture,
            blockCount=shape.block_count,
            attentionLayers=shape.attention_layers,
            contextLength=meta.context_length,
            vocabSize=meta.vocab_size,
            shardCount=meta.shard_count,
            capabilities=ModelCapabilities(
                chat=not meta.is_embedding,
                embedding=meta.is_embedding,
                vision=meta.is_projector,
                chatTemplate=meta.has_chat_template,
            ),
            recommendedSampling=(
                RecommendedSampling(
                    temperature=sampling.temperature,
                    topK=sampling.top_k,
                    topP=sampling.top_p,
                )
                if sampling
                else None
            ),
            fit=fit_mod.compute(
                weights_bytes=weights_bytes,
                budget=budget,
                context_length=context_length,
                shape=shape,
                kv_cache_type=kv_cache_type,
                # One part's table is not a sharded model's: unknown then.
                expert_bytes=meta.expert_bytes if (meta.shard_count or 1) == 1 else None,
            ),
        )

    if name.lower().endswith(".safetensors"):
        header, bytes_read = await read_safetensors_header(
            client, repo=repo, revision=revision, path=path
        )
        # No layer metadata in a safetensors header — the shape lives in
        # `config.json` beside it, which is a separate file and a
        # separate decision. So the fit stays an estimate here, and says
        # so, rather than pretending the parameter count implies a KV
        # cache size.
        shape = fit_mod.ModelShape(parameters=header.parameters)
        return CataloguePreflight(
            repo=repo,
            resolvedCommit=resolved.commit,
            file=path,
            format=ModelFormat.safetensors,
            bytesRead=bytes_read,
            parameters=header.parameters,
            dtype=header.dominant_dtype,
            fit=fit_mod.compute(
                weights_bytes=weights_bytes,
                budget=budget,
                context_length=context_length,
                shape=shape,
                kv_cache_type=kv_cache_type,
            ),
        )

    raise HubError(
        f"{path} is neither a .gguf nor a .safetensors file, so there is no header to "
        "read. Preflight a candidate's weights file.",
        status=400,
    )
