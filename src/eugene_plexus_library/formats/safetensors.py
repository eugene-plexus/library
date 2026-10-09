"""A safetensors header reader, and what a HuggingFace model directory is.

Reads the JSON header at the front of a `.safetensors` file and the
`config.json` beside it. Never a tensor.

## The layout

    u64  header length, little-endian
    JSON header of that length: {tensor_name: {dtype, shape, data_offsets},
                                 "__metadata__": {...}}
    tensor data

Which makes this the mirror image of GGUF's cost profile: the whole
header for a real MiniLM-L6 is **11,416 bytes** and gives an exact
parameter count (22,713,728) and dtype by summing declared shapes. GGUF
gives no parameter count at all and charges ~10 MB to say so.

That asymmetry is why `parameters` is optional on the wire and
`sizeLabel` is a display string: exact-or-absent is the honest shape,
and a schema requiring a count would be unfillable for every GGUF in
existence.

## A model is a directory here

Not a file. `config.json` plus one or more weight files plus tokenizer
files, all loaded together — none of them is "the" file the way a
`.gguf` is. Two things in that directory are traps:

* An **adapter** (`adapter_config.json` + `adapter_model.safetensors`,
  no base weights) is a LoRA, not a model.
* A **HuggingFace cache** stores one snapshot directory per revision,
  so a recursive scan finds the same model once per revision ever
  fetched. `refs/main` names the current one. `blobs/` and `refs/` are
  infrastructure, and `.no_exist/` is a *negative* cache full of
  zero-byte files with real-looking names — a naive "is there an
  `adapter_config.json`?" probe finds one there.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
INDEX_SUFFIX = ".safetensors.index.json"

# HuggingFace cache directory names that are never a model. `.no_exist`
# is the one that bites: it holds zero-byte files named after things
# that were checked for and found absent.
HF_INFRASTRUCTURE = frozenset({"blobs", "refs", ".no_exist"})

HF_REPO_DIR_RE = re.compile(r"^models--(?P<org>.+?)--(?P<name>.+)$")

# Dataset caches share the hub directory with model caches and are
# never models. Pruned by name rather than walked and rejected: one
# hub can hold dozens, each contributing three infrastructure
# directories of pure noise to the skip list.
HF_DATASET_DIR_RE = re.compile(r"^datasets--")

# `model-00001-of-00003.safetensors`
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")

_MAX_HEADER = 1 << 26  # 64 MiB. Real headers are kilobytes.

# Bytes per element, by safetensors dtype name.
DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# Integer dtypes are bookkeeping — token type ids, position indices —
# not weights, and counting them toward "the dominant dtype" would
# report an F32 model as mixed. Verified on MiniLM: 512 I64 elements
# beside 22,713,216 F32 ones.
_WEIGHT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"})


class SafetensorsError(Exception):
    """The file is not readable as safetensors. Becomes the
    `unreadable_header` skip reason."""


@dataclass
class SafetensorsHeader:
    """One `.safetensors` file's declared contents."""

    header_bytes: int
    tensor_count: int
    parameters: int
    """Exact — the sum of declared shapes. Not an estimate."""

    dtype_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    elements: dict[str, int] = field(default_factory=dict)
    """Stored elements per tensor, which is what `mlx_parameters` needs to
    tell packed storage from parameters."""

    @property
    def dominant_dtype(self) -> str | None:
        """The dtype most of this file's weights are in."""
        return dominant_dtype(self.dtype_counts)


def mlx_parameters(elements: dict[str, int], quantization: dict[str, Any]) -> int | None:
    """Model parameters of an MLX-quantized directory, from its packed storage.

    **Why (A4, 2026-09-30).** The library reported
    `mlx-community/Qwen3-0.6B-4bit` as 93,188,096 parameters, on a GitHub
    macOS runner: summed shapes count storage elements, and MLX's
    `QuantizedLinear` stores `<module>.weight` as uint32 words holding
    32/bits values each, beside `<module>.scales` and `<module>.biases`,
    one of each per group of `group_size` inputs. So a quantized module
    holds `scales x group_size` parameters, whatever width packed them,
    and its scales and biases are bookkeeping. A module the conversion
    left unquantized (a norm, say) has no `.scales` and counts as stored.

    `quantization` is `config.json`'s top-level block: its `group_size`,
    and a per-module entry wherever the conversion chose differently.
    None when a quantized module's group size cannot be read, since an
    absent count is honest and a wrong one is not.
    """
    default_group = quantization.get("group_size")
    total = 0
    for name, count in elements.items():
        if name.endswith(".scales"):
            module = name[: -len(".scales")]
            override = quantization.get(module)
            group = override.get("group_size") if isinstance(override, dict) else default_group
            if not isinstance(group, int) or isinstance(group, bool) or group <= 0:
                return None
            total += count * group
        elif (
            name.endswith((".weight", ".biases")) and f"{name.rsplit('.', 1)[0]}.scales" in elements
        ):
            continue
        else:
            total += count
    return total


def dominant_dtype(counts: dict[str, int]) -> str | None:
    """The dtype most of the weights are in, given element counts.

    Takes a mapping rather than being only a property because a sharded
    model's answer is over the sum across every shard, not any one file.
    """
    weights = {k: v for k, v in counts.items() if k in _WEIGHT_DTYPES}
    pool = weights or counts
    return max(pool, key=lambda k: pool[k]) if pool else None


def read_header(path: Path) -> SafetensorsHeader:
    """Read one safetensors file's header.

    Raises `SafetensorsError` on anything malformed; lets `OSError`
    through so a permission problem is not reported as a corrupt file.
    """
    with path.open("rb") as fh:
        raw_length = fh.read(8)
        length = header_length(raw_length, name=path.name)
        raw = fh.read(length)
        if len(raw) != length:
            raise SafetensorsError(
                f"{path.name}: truncated header (wanted {length}, got {len(raw)})"
            )
    return parse_header(raw, name=path.name)


def header_length(raw_length: bytes, *, name: str = "<stream>") -> int:
    """Decode the 8-byte length prefix that starts every safetensors file.

    Split out because it is the whole of the first read a *remote*
    header needs: eight bytes over HTTP Range say how many more to
    fetch, which is why reading a safetensors header off the hub costs
    ~11 KB and two requests where a GGUF costs ~11 MB.
    """
    if len(raw_length) != 8:
        raise SafetensorsError(f"{name}: truncated before header length")
    (length,) = struct.unpack("<Q", raw_length)
    if length == 0 or length > _MAX_HEADER:
        raise SafetensorsError(f"{name}: implausible header length {length}")
    return int(length)


def parse_header(raw: bytes, *, name: str = "<stream>") -> SafetensorsHeader:
    """Parse the JSON header block, given exactly its bytes.

    Takes bytes rather than a file so the same parse serves a local file
    and a ranged read of a remote one.
    """
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise SafetensorsError(f"{name}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{name}: header is not a JSON object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}

    parameters = 0
    dtype_counts: dict[str, int] = {}
    elements: dict[str, int] = {}
    for tensor_name, entry in header.items():
        if not isinstance(entry, dict):
            raise SafetensorsError(f"{name}: tensor {tensor_name!r} is not an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise SafetensorsError(f"{name}: tensor {tensor_name!r} has no shape/dtype")
        count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise SafetensorsError(f"{name}: tensor {tensor_name!r} has a bad shape")
            count *= dimension
        parameters += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        elements[tensor_name] = count

    return SafetensorsHeader(
        header_bytes=8 + len(raw),
        tensor_count=len(header),
        parameters=parameters,
        dtype_counts=dtype_counts,
        metadata=metadata,
        elements=elements,
    )


def has_chat_template(directory: Path, config: dict[str, Any]) -> bool:
    """Whether a HuggingFace directory carries a chat template.

    **Three places, and the commonest was missed (A4, 2026-09-30).** A
    template lives in `tokenizer_config.json` for most models, as
    `chat_template`; newer exports write `chat_template.jinja` (or
    `chat_template.json`) beside it; `config.json` rarely holds one. The
    library read only the last two, so `mlx-community/Qwen3-0.6B-4bit`,
    whose template is in its tokenizer config, read as having none.
    """
    if "chat_template" in config:
        return True
    if (directory / "chat_template.jinja").exists() or (directory / "chat_template.json").exists():
        return True
    try:
        tokenizer = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(tokenizer, dict) and bool(tokenizer.get("chat_template"))


@dataclass
class ModelConfig:
    """The parts of `config.json` worth surfacing."""

    architecture: str | None
    context_length: int | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mlx_quantization(self) -> tuple[int | None, int | None] | None:
        """`(bits, group_size)` from an MLX-style top-level `quantization`
        block, or None when the directory carries no such marker.

        The one positive signal that a safetensors directory was
        prepared by `mlx_lm.convert` for the MLX loader: vanilla HF
        exports spell their quantization `quantization_config`, a
        different key, and an MLX-quantized directory packs weights as
        integer tensors no other engine here can load. Absence means
        unknown, not incompatible — an unquantized MLX conversion writes
        no block at all.
        """
        block = self.raw.get("quantization")
        if not isinstance(block, dict):
            return None
        bits = block.get("bits")
        group = block.get("group_size")
        return (
            bits if isinstance(bits, int) and not isinstance(bits, bool) else None,
            group if isinstance(group, int) and not isinstance(group, bool) else None,
        )

    @property
    def is_embedding(self) -> bool:
        """Encoder architectures are embedding models. `BertModel` and
        friends have no LM head, so serving one as a chat model produces
        nothing useful."""
        arch = (self.architecture or "").lower()
        return arch.endswith("model") and any(
            token in arch for token in ("bert", "roberta", "xlm", "e5", "gte")
        )


def read_config(path: Path) -> ModelConfig:
    """Read a `config.json`. Raises `SafetensorsError` if unparseable."""
    return parse_config(path.read_bytes(), name=path.name)


def parse_config(data: bytes, *, name: str = "config.json") -> ModelConfig:
    """Parse a `config.json` from its bytes: a local file, or a remote one
    the catalogue read before download (LS2), by the same rules."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise SafetensorsError(f"{name}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise SafetensorsError(f"{name}: not a JSON object")

    architectures = raw.get("architectures")
    architecture: str | None = None
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        architecture = architectures[0]
    elif isinstance(raw.get("model_type"), str):
        architecture = str(raw["model_type"])

    # `max_position_embeddings` is the conventional field. Some configs
    # only carry `n_positions` (GPT-2 lineage) or a nested text config.
    context: int | None = None
    for key in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            context = value
            break

    return ModelConfig(architecture=architecture, context_length=context, raw=raw)


def shard_position(path: Path) -> tuple[str, int, int] | None:
    """`(stem, index, total)` for a sharded weight filename, else None."""
    stem = path.name.removesuffix(".safetensors")
    match = SHARD_RE.match(stem)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index")), int(match.group("total"))


def is_adapter_dir(directory: Path, names: set[str]) -> bool:
    """A LoRA rather than a model: an `adapter_config.json` and no base
    `config.json`.

    Ordering matters. Some adapter directories ship a copy of the base
    model's `config.json`, and some full models ship an adapter
    alongside; requiring the *absence* of base weights is what separates
    them. Not modelled as a first-class thing at M2 — `--lora` in
    `extraArgs` covers anyone who needs one now.
    """
    if ADAPTER_CONFIG_NAME not in names:
        return False
    has_base_weights = any(
        n.endswith(".safetensors") and not n.startswith("adapter_model") for n in names
    )
    return not has_base_weights


def hf_repo_id(directory: Path) -> str | None:
    """`…/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<rev>`
    -> `sentence-transformers/all-MiniLM-L6-v2`.

    The org name may itself contain `-`, which `models--a--b--c` makes
    ambiguous; HuggingFace's own encoding splits on the first `--`, so
    that is what this does.
    """
    for parent in directory.parents:
        match = HF_REPO_DIR_RE.match(parent.name)
        if match:
            return f"{match.group('org')}/{match.group('name')}"
    return None


def hf_model_name(directory: Path) -> str | None:
    """The name to show for a model living in a HuggingFace cache.

    A snapshot directory is named after the revision hash, so the
    directory name — the right answer everywhere else — reads as
    `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The repo's own name is
    what the operator downloaded and what they will look for.
    """
    repo = hf_repo_id(directory)
    return repo.split("/", 1)[1] if repo else None


def hf_revision(directory: Path) -> str | None:
    """The snapshot revision, when this directory is one."""
    parent = directory.parent
    if parent.name != "snapshots":
        return None
    return directory.name


def hf_current_revision(directory: Path) -> str | None:
    """What `refs/main` points at for the cache this snapshot lives in.

    Returns None when there is no cache, no `refs/main`, or it cannot be
    read — in which case the caller keeps every revision rather than
    guessing which to drop. Dropping a model because a ref file was
    unreadable is a worse failure than listing it twice.
    """
    parent = directory.parent
    if parent.name != "snapshots":
        return None
    ref = parent.parent / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
