"""A GGUF metadata reader.

Reads the header and the key/value block at the front of a `.gguf` file
and nothing else — never a tensor, never the weights. Hand-written
against the documented container layout rather than taken from a
dependency, because the whole job is a few hundred lines and the
alternative is pulling a tensor-loading stack into a component that
never loads a tensor.

## The layout

    magic     4 bytes, b"GGUF"
    version   u32       (3 for everything current)
    n_tensors u64
    n_kv      u64
    then n_kv times:
        key     length-prefixed UTF-8 string (u64 length, no NUL)
        type    u32, indexing GGUF_TYPE below
        value   per type; ARRAY is (element type u32, count u64, items)

## What makes this expensive, and why the code looks the way it does

Measured on a real 27B Q4_K_M: **10,942,419 bytes of KV block**, because
`tokenizer.ggml.tokens` (248,320 strings) and `tokenizer.ggml.merges`
(247,587 more) live in it. Three consequences drive the implementation:

1. **Array values are skipped, never materialised.** Nothing here needs
   a vocabulary, and holding one costs ~10 MB per model scanned.
2. **A string array cannot be skipped with one seek.** Each string's
   length is inline, so the only way past 248,320 of them is to step
   them one at a time. That loop, not the I/O, is the cost — reading
   the whole header into memory first saves about 12%, not 90%, so this
   reads through a buffered file object and keeps the memory flat.
3. **You cannot stop early.** `general.file_type` — the quant tier — is
   the *last* KV pair in that same file, on the far side of the vocab.
   KV order is not specified by the format and the interesting fields
   are not conveniently at the front.

## What it refuses to guess

`file_type` is an enum whose families churn upstream (imatrix variants,
MXFP4, whatever is next). Values known here are mapped to a label;
anything else is reported as the raw integer with `label=None`, and the
caller falls back to the filename. Mapping an unrecognised number onto a
plausible neighbour would put a wrong quant tier in front of an operator
choosing a model, which is worse than admitting ignorance — and M3's
hardware-fit guidance is built on this field.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TypeGuard

MAGIC = b"GGUF"

# GGUF value type tags, in the order the format defines them.
(
    T_UINT8,
    T_INT8,
    T_UINT16,
    T_INT16,
    T_UINT32,
    T_INT32,
    T_FLOAT32,
    T_BOOL,
    T_STRING,
    T_ARRAY,
    T_UINT64,
    T_INT64,
    T_FLOAT64,
) = range(13)

_SCALAR = {
    T_UINT8: ("<B", 1),
    T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4),
    T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4),
    T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

# Arrays at or below this length are kept; longer ones are stepped over.
# 64 covers the small structural arrays that carry meaning
# (`rope.dimension_sections`, `vision.image_mean`, per-layer flags) and
# excludes every vocabulary.
_INLINE_ARRAY_LIMIT = 64

# Sanity ceilings. A corrupt or truncated file can present a u64 length
# as something like 2^61; without a bound the reader tries to allocate
# it. These are far above anything real — the largest vocabulary in the
# wild is a few hundred thousand tokens, and no key is a kilobyte.
_MAX_KV = 1 << 16
_MAX_STRING = 1 << 24
_MAX_ARRAY = 1 << 28

# `general.file_type`. These are llama.cpp's LLAMA_FTYPE_MOSTLY_* values.
# 14/15/32/0 are verified against real files whose names agree; the rest
# follow upstream's enum. Absent entries are reported raw on purpose.
FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}

# Quant tiers as they appear in filenames. Longest-first so `Q4_K_M`
# wins over a hypothetical `Q4_K` prefix match.
_FILENAME_QUANTS = sorted(
    {*FILE_TYPES.values(), "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "MXFP4"},
    key=len,
    reverse=True,
)
_FILENAME_QUANT_RE = re.compile(
    r"(?:^|[._-])(" + "|".join(re.escape(q) for q in _FILENAME_QUANTS) + r")(?:$|[._-])",
    re.IGNORECASE,
)

# `…-00001-of-00005.gguf`, produced by upstream's `llama-gguf-split`.
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")


class GgufError(Exception):
    """The file is not readable as GGUF. Carries a message intended for
    an operator: it becomes the `unreadable_header` skip reason, which
    is the only thing they will see explaining a missing model."""


def _is_number(value: Any) -> TypeGuard[int | float]:
    """`bool` is an `int` in Python, and a stray `True` in a numeric
    metadata field would otherwise arrive downstream as `1.0`."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Sampling:
    """The author's suggested sampling parameters, if the file carries
    any. A typed carrier rather than a dict so the mapping onto the wire
    schema is checked — splatting an untyped dict into a model
    constructor is where a renamed field goes unnoticed until runtime."""

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def __bool__(self) -> bool:
        return any(v is not None for v in (self.temperature, self.top_k, self.top_p))


@dataclass
class GgufMetadata:
    """What the KV block said, plus what we concluded from it."""

    version: int
    tensor_count: int
    kv_count: int
    header_bytes: int
    """Where the tensor data starts — also, usefully, the size of what
    had to be walked to read this."""

    kv: dict[str, Any] = field(default_factory=dict)
    """Scalar and small-array values, verbatim. Vocabulary arrays are
    recorded as `None` with their length in `array_lengths`."""

    array_lengths: dict[str, int] = field(default_factory=dict)
    """Element counts for arrays that were stepped over rather than
    read. `tokenizer.ggml.tokens` here is the vocabulary size."""

    @property
    def architecture(self) -> str | None:
        value = self.kv.get("general.architecture")
        return str(value) if isinstance(value, str) else None

    @property
    def general_type(self) -> str | None:
        """`general.type` — `mmproj` for a vision projector, `model` for
        a model. **Optional**: verified absent on a real embedding model,
        so absence means "model" and must not be treated as unknown."""
        value = self.kv.get("general.type")
        return str(value) if isinstance(value, str) else None

    @property
    def is_projector(self) -> bool:
        """A vision projector belongs to the model beside it and is not
        an entry of its own. Decided by metadata rather than by the
        `mmproj-` filename convention, so a renamed file still tells the
        truth — and these are 0.9-1.8 GB each, so being wrong is loud."""
        return self.general_type == "mmproj" or self.architecture == "clip"

    def arch_key(self, suffix: str) -> Any:
        """Read an architecture-prefixed key, e.g. `context_length` ->
        `qwen35.context_length`. There is no fixed key set in GGUF: the
        prefix is whatever `general.architecture` says, so a reader with
        hardcoded key names goes blind on every architecture released
        after it was written."""
        arch = self.architecture
        if arch is None:
            return None
        return self.kv.get(f"{arch}.{suffix}")

    @property
    def context_length(self) -> int | None:
        value = self.arch_key("context_length")
        return int(value) if isinstance(value, int) else None

    @property
    def vocab_size(self) -> int | None:
        return self.array_lengths.get("tokenizer.ggml.tokens")

    @property
    def size_label(self) -> str | None:
        """`general.size_label`, e.g. `"27B"`. A **string** the uploader
        typed, not a parameter count — GGUF carries no numeric one, and
        inventing a number from this would be a fabrication."""
        value = self.kv.get("general.size_label")
        return str(value) if isinstance(value, str) else None

    @property
    def name(self) -> str | None:
        value = self.kv.get("general.name")
        return str(value) if isinstance(value, str) else None

    @property
    def file_type(self) -> int | None:
        value = self.kv.get("general.file_type")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def quantization(self) -> str | None:
        """Label for `file_type`, or None when the value is one this
        reader does not know. None is a real answer — see the module
        docstring on refusing to guess."""
        ft = self.file_type
        return None if ft is None else FILE_TYPES.get(ft)

    @property
    def has_chat_template(self) -> bool:
        return bool(self.kv.get("tokenizer.chat_template"))

    @property
    def is_embedding(self) -> bool:
        """A dedicated embedding model. Detected by a pooling type or
        explicitly non-causal attention.

        Worth detecting because the failure is silent: llama-server's
        own help says `--embedding` is "for use only with dedicated
        embedding models", and launching one of these as a chat model
        does not error — it starts, serves, and returns nonsense.
        """
        if self.arch_key("pooling_type") is not None:
            return True
        causal = self.arch_key("attention.causal")
        return causal is False

    @property
    def recommended_sampling(self) -> Sampling:
        """`general.sampling.*` — what the model's author suggests.

        Reported, never applied. The gateway owns every parameter that
        affects output and no component substitutes a local default;
        this exists so the operator does not have to go read the model
        card to find out the file had an opinion.
        """
        temp = self.kv.get("general.sampling.temp")
        top_k = self.kv.get("general.sampling.top_k")
        top_p = self.kv.get("general.sampling.top_p")
        return Sampling(
            temperature=float(temp) if _is_number(temp) else None,
            top_k=int(top_k) if _is_int(top_k) else None,
            top_p=float(top_p) if _is_number(top_p) else None,
        )

    @property
    def shard_count(self) -> int | None:
        """`split.count`, written by `llama-gguf-split`. The filename
        also carries it; this is the metadata's own answer."""
        value = self.kv.get("split.count")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def public_kv(self) -> dict[str, Any]:
        """KV pairs safe to hand to a client: scalars and small arrays,
        with the stepped-over arrays represented by their length. The
        escape hatch for architecture-specific keys this reader has no
        named property for."""
        out: dict[str, Any] = {k: v for k, v in self.kv.items() if v is not None}
        for key, length in self.array_lengths.items():
            out[f"{key}.length"] = length
        return out


class _Reader:
    """Cursor over the header. Every read is bounded; a truncated file
    raises `GgufError` rather than an EOF surprise three frames up."""

    def __init__(self, fh: BinaryIO, path: Path) -> None:
        self._fh = fh
        self._path = path

    def _fail(self, message: str) -> GgufError:
        return GgufError(f"{self._path.name}: {message}")

    def raw(self, count: int) -> bytes:
        data = self._fh.read(count)
        if len(data) != count:
            raise self._fail(f"truncated: wanted {count} bytes, got {len(data)}")
        return data

    def scalar(self, tag: int) -> Any:
        fmt, size = _SCALAR[tag]
        return struct.unpack(fmt, self.raw(size))[0]

    def length(self, *, limit: int, what: str) -> int:
        value = int(self.scalar(T_UINT64))
        if value < 0 or value > limit:
            raise self._fail(f"implausible {what} length {value}")
        return value

    def string(self) -> str:
        size = self.length(limit=_MAX_STRING, what="string")
        return self.raw(size).decode("utf-8", errors="replace")

    def skip(self, count: int) -> None:
        self._fh.seek(count, 1)

    def value(self, tag: int, *, key: str, lengths: dict[str, int]) -> Any:
        if tag == T_STRING:
            return self.string()
        if tag == T_ARRAY:
            return self._array(key=key, lengths=lengths)
        if tag not in _SCALAR:
            raise self._fail(f"unknown value type {tag} for key {key!r}")
        return self.scalar(tag)

    def _array(self, *, key: str, lengths: dict[str, int]) -> Any:
        element_tag = int(self.scalar(T_UINT32))
        count = self.length(limit=_MAX_ARRAY, what="array")
        lengths[key] = count

        if count <= _INLINE_ARRAY_LIMIT:
            return [self.value(element_tag, key=key, lengths={}) for _ in range(count)]

        # Too big to keep. Step over it — see the module docstring for
        # why a string array costs one read per element.
        if element_tag == T_STRING:
            for _ in range(count):
                self.skip(self.length(limit=_MAX_STRING, what="string"))
        elif element_tag in _SCALAR:
            self.skip(count * _SCALAR[element_tag][1])
        else:
            raise self._fail(f"unknown array element type {element_tag} for key {key!r}")
        return None

    def tell(self) -> int:
        return self._fh.tell()


def read_metadata(path: Path) -> GgufMetadata:
    """Read one GGUF file's header and KV block.

    Raises `GgufError` for anything that is not a readable GGUF —
    including a file that merely has the extension. `OSError` is left to
    propagate: a permission error or a vanished file is the caller's to
    classify, and folding it into `GgufError` would report a locked file
    as a malformed one.
    """
    with path.open("rb") as fh:
        reader = _Reader(fh, path)

        magic = reader.raw(4)
        if magic != MAGIC:
            raise GgufError(f"{path.name}: not a GGUF file (magic {magic!r})")

        version = int(reader.scalar(T_UINT32))
        if version < 2 or version > 3:
            # v1 predates the current KV encoding; anything above 3 does
            # not exist yet and would be parsed on a guess.
            raise GgufError(f"{path.name}: unsupported GGUF version {version}")

        tensor_count = reader.length(limit=1 << 32, what="tensor count")
        kv_count = reader.length(limit=_MAX_KV, what="kv count")

        kv: dict[str, Any] = {}
        lengths: dict[str, int] = {}
        for _ in range(kv_count):
            key = reader.string()
            tag = int(reader.scalar(T_UINT32))
            kv[key] = reader.value(tag, key=key, lengths=lengths)

        return GgufMetadata(
            version=version,
            tensor_count=tensor_count,
            kv_count=kv_count,
            header_bytes=reader.tell(),
            kv=kv,
            array_lengths=lengths,
        )


def strip_shard_suffix(stem: str) -> str:
    """`model-00001-of-00005` -> `model`. Unchanged if not a shard."""
    match = SHARD_RE.match(stem)
    return match.group("stem") if match else stem


def shard_position(path: Path) -> tuple[str, int, int] | None:
    """`(stem, index, total)` for a shard filename, else None.

    Grouping is by filename rather than by metadata because it has to
    work without opening every file: with five 8 GB shards, the point is
    to read the first one's header and stat the rest.
    """
    match = SHARD_RE.match(path.stem)
    if match is None:
        return None
    return match.group("stem"), int(match.group("index")), int(match.group("total"))


def quant_from_filename(name: str) -> str | None:
    """Pull a quant tier out of a filename, e.g.
    `Qwen3.6-27B-Q4_K_M.gguf` -> `Q4_K_M`.

    The label an operator actually speaks, and the fallback when
    `file_type` is a value this reader does not know. Also the
    cross-check: when both exist and disagree, both get reported and
    neither is silently preferred — a requantized file keeps its old
    name more often than a metadata field is wrong, but not always.
    """
    match = _FILENAME_QUANT_RE.search(name)
    return match.group(1).upper() if match else None
