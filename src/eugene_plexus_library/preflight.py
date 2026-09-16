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

    Only the conventions actually seen are honoured, and a model using a
    different one falls through to the block count. That errs
    pessimistic, which is the right direction: an over-estimate costs an
    operator a second look, an under-estimate costs them an OOM they
    were promised would not happen.
    """
    blocks = meta.arch_key("block_count")
    if not isinstance(blocks, int) or blocks <= 0:
        return None

    interval = meta.arch_key("full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        return max(blocks // interval, 1)

    # Some hybrids list the attention layers outright instead.
    indices = meta.arch_key("attention.layer_indices")
    if isinstance(indices, list) and indices:
        return len(indices)

    return blocks


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
    own dimensions, which are shorter. Absent the pattern, a declared
    `sliding_window` with no per-layer mask is not applied at all --
    guessing which layers slide would be inventing the number this
    module exists to stop inventing.

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
    if not isinstance(heads_kv, list) and not isinstance(pattern, list):
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
        layers.append(
            fit_mod.LayerKV(
                head_count_kv=heads,
                key_length=key_swa if slides else key_length,
                value_length=value_swa if slides else value_length,
                window=window if slides else None,
            )
        )
    return tuple(layers)


def shape_from_gguf(meta: gguf.GgufMetadata) -> fit_mod.ModelShape:
    """The KV-cache terms, read off the file's own KV block."""

    def integer(suffix: str) -> int | None:
        value = meta.arch_key(suffix)
        return int(value) if isinstance(value, int) else None

    blocks = integer("block_count")
    return fit_mod.ModelShape(
        block_count=blocks,
        attention_layers=attention_layers(meta),
        head_count_kv=integer("attention.head_count_kv"),
        key_length=integer("attention.key_length"),
        value_length=integer("attention.value_length"),
        embedding_length=integer("embedding_length"),
        head_count=integer("attention.head_count"),
        context_length=meta.context_length,
        layers=per_layer_kv(meta),
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
            meta = gguf.read_metadata_stream(io.BytesIO(buffer), name=PurePosixPath(path).name)
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
