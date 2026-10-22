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

# `<arch>.pooling_type` is llama.cpp's `llama_pooling_type`: 0 none, 1 mean,
# 2 cls, 3 last, 4 rank. Rank is what a reranker is converted with.
POOLING_RANK = 4

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
# It covers the small structural arrays that carry meaning
# (`rope.dimension_sections`, `vision.image_mean`, per-layer flags) and
# excludes every vocabulary.
#
# **It was 64, and a per-layer array is as long as the block count.**
# The shipped starter classes declare 32, 42, 48 and **65** blocks, so
# the limit missed by one on a model already in the product's own
# starter file: `attention.head_count_kv` was stepped over, the fit fell
# back to the same-every-layer scalars, and the answer came out up to
# 43x too large while still reporting `basis: metadata` (roadmap R1.3,
# from review §6.1 #4's neighbourhood).
#
# 512 rather than 66, because the number to clear is "how many layers
# will a model have", which only goes up, and the thing being excluded
# is a vocabulary -- three orders of magnitude away, so there is no
# squeeze between the two. A per-layer array of int32 at this limit is
# 2 KB.
_INLINE_ARRAY_LIMIT = 512
_TARGET_ARRAYS = frozenset({"dflash.target_layers", "eagle3.target_layers"})

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


class GgufTruncated(GgufError):
    """The header ran past the end of what was available.

    Distinct from a malformed file, because a remote header read over
    HTTP Range cannot know the KV block's size in advance: a 1 MB probe
    of a 248k-vocab model is short by ~10 MB, and that is a signal to
    fetch more rather than a broken file. `needed` is the offset the
    parser was reaching for when it ran out, so one more ranged request
    finishes the job.
    """

    def __init__(self, message: str, *, needed: int) -> None:
        super().__init__(message)
        self.needed = needed


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

    tensor_bytes: int | None = None
    """Every tensor's data size summed from the tensor table, or None when
    the table was not read or named a type this reader does not know."""

    expert_bytes: int | None = None
    """The MoE expert tensors' share of `tensor_bytes` (0 for a dense
    model), under the same conditions. See `read_tensor_table`."""

    has_classifier_tensors: bool = False
    """The tensor table names a `cls.*` classification head."""

    nextn_blocks: frozenset[int] | None = None
    """Blocks whose `nextn.eh_proj` tensor the table names: where an MTP
    head is. None when the table was not read."""

    has_first_block: bool | None = None
    """The table names a `blk.0.` tensor. A separate MTP file holds only
    the last block, so this is what tells it from a model that carries its
    own head (SD1). None when the table was not read."""

    has_markov_head: bool = False
    """The tensor table names `markov_w1.weight`: a DFlash file is DSpark."""

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

    @property
    def mtp_tensors(self) -> bool:
        """The last block carries an MTP head: llama.cpp's own test
        (`common_speculative_types_from_gguf`, b11719) for a file that
        drafts as `draft-mtp`. `<arch>.nextn_predict_layers` alone is not
        enough, because a separate MTP file declares it too."""
        blocks = self.arch_key("block_count")
        if not isinstance(blocks, int) or self.nextn_blocks is None:
            return False
        return blocks - 1 in self.nextn_blocks

    @property
    def mtp_head(self) -> bool:
        """This file is a whole model with its own MTP head."""
        return self.mtp_tensors and self.has_first_block is True

    def draft_kind(self, filename: str) -> str | None:
        """The kind when this file is a companion rather than a model of
        its own, else None. DFlash, DSpark and EAGLE3 are always companions.

        A separate MTP head is told by its tensors: the last block's
        `nextn` tensors and no first block, so it cannot run alone. A whole
        small model is a companion only when it is named as one
        (`draft-*.gguf`), because any small model could otherwise be taken
        for a draft.
        """
        if self.architecture == "dflash":
            return "dspark" if self.has_markov_head else "dflash"
        if self.architecture == "eagle3":
            return "eagle3"
        if self.is_projector or self.nextn_blocks is None:
            return None
        if self.mtp_tensors and self.has_first_block is False:
            return "mtp"
        if self.has_first_block and DRAFT_NAME_RE.search(filename):
            # llama.cpp drafts with any file that has the head as `draft-mtp`.
            return "mtp" if self.mtp_tensors else "simple"
        return None

    @property
    def target_layers(self) -> list[int] | None:
        value = self.arch_key("target_layers")
        if (
            not isinstance(value, list)
            or not 1 <= len(value) <= _INLINE_ARRAY_LIMIT
            or any(not _is_int(layer) or layer < 0 for layer in value)
        ):
            return None
        return value

    @property
    def target_hidden_size(self) -> int | None:
        value = self.arch_key("target_hidden_size")
        return value if _is_int(value) and value > 0 else None

    @property
    def draft_target_problem(self) -> str | None:
        if self.target_hidden_size is None:
            return f"missing or invalid {self.architecture}.target_hidden_size"
        layers = self.target_layers
        if layers is None:
            return f"missing or invalid {self.architecture}.target_layers (1 to 512 layer ids)"
        if self.architecture == "eagle3" and len(layers) != 3:
            return "eagle3.target_layers must name exactly 3 target layers"
        return None

    @property
    def target_min_layers(self) -> int | None:
        layers = self.target_layers
        return max(layers) + (self.architecture != "eagle3") if layers else None

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
        does not error — it starts, and answers neither embeddings (a
        501) nor chat (a 500, or text that means nothing).

        **True for a reranker too**: rank pooling is a pooling type. A
        caller that needs to tell them apart asks `is_rerank` first, or
        `is_search_embedding`, which already has.
        """
        if self.arch_key("pooling_type") is not None:
            return True
        causal = self.arch_key("attention.causal")
        return causal is False

    @property
    def is_rerank(self) -> bool:
        """Rank pooling, classifier labels, or a non-causal classifier head.

        It scores how well a passage answers a question, and needs
        `--reranking` the way an embedder needs `--embeddings`. The
        agent reads the same markers when it starts a model with `auto`
        (`engines/run_mode.py`), so what the console predicts from this
        and what the agent decides agree.

        The gpustack bge-reranker-v2-m3 names neither metadata key, but
        carries a non-causal classifier head in its tensor table.
        """
        if self.arch_key("pooling_type") == POOLING_RANK:
            return True
        if self.arch_key("attention.causal") is False and self.has_classifier_tensors:
            return True
        labels = self.arch_key("classifier.output_labels")
        if isinstance(labels, list):
            return len(labels) > 0
        # A label list long enough to have been stepped over rather than kept
        # is still a list of labels.
        arch = self.architecture
        return (
            arch is not None and self.array_lengths.get(f"{arch}.classifier.output_labels", 0) > 0
        )

    @property
    def is_search_embedding(self) -> bool:
        """An embedding model that is not a reranker: what `capabilities.embedding` says."""
        return self.is_embedding and not self.is_rerank

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
        # Measured once, not per skip. The header cannot grow while it
        # is being read, and re-measuring inside `skip` would cost two
        # seeks per stepped-over string -- 248,320 of them on a real
        # large-vocab model, each one discarding the buffered read-ahead
        # this reader depends on.
        self._end = fh.seek(0, 2)
        fh.seek(0)
        # The offset is tracked here rather than asked of the stream.
        # `raw` runs once per key, per scalar and per stepped-over
        # string -- half a million times on a 248k-vocab model -- and a
        # `tell()` in that loop measurably slowed the whole scan when it
        # was tried. This is also exact, which a buffered `tell()` after
        # a short read is not.
        self._pos = 0
        #: Arrays read whole whatever their size: only what a caller asks
        #: for by key (the vocabulary, for `special_tokens`).
        self.keep: frozenset[str] = frozenset()

    def _fail(self, message: str) -> GgufError:
        return GgufError(f"{self._path.name}: {message}")

    def raw(self, count: int) -> bytes:
        data = self._fh.read(count)
        if len(data) != count:
            raise GgufTruncated(
                f"{self._path.name}: truncated: wanted {count} bytes at offset "
                f"{self._pos}, got {len(data)}",
                needed=self._pos + count,
            )
        self._pos += count
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
        """Step over a value without reading it.

        Bounds-checked rather than a bare relative `seek`: seeking past
        the end of an in-memory buffer succeeds silently, so a short
        probe window would surface as a truncation at some later,
        misleading offset instead of at the array that overran.
        """
        target = self._pos + count
        if target > self._end:
            raise GgufTruncated(
                f"{self._path.name}: truncated: wanted to reach offset {target}, "
                f"stream ends at {self._end}",
                needed=target,
            )
        self._fh.seek(target)
        self._pos = target

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

        if (count <= _INLINE_ARRAY_LIMIT or key in self.keep) and (
            key not in _TARGET_ARRAYS or count <= _INLINE_ARRAY_LIMIT
        ):
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
        return self._pos


def read_metadata(path: Path, *, tensors: bool = False) -> GgufMetadata:
    """Read one GGUF file's header and KV block, and optionally its tensor table.

    Raises `GgufError` for anything that is not a readable GGUF —
    including a file that merely has the extension. `OSError` is left to
    propagate: a permission error or a vanished file is the caller's to
    classify, and folding it into `GgufError` would report a locked file
    as a malformed one.
    """
    with path.open("rb") as fh:
        return read_metadata_stream(fh, name=path.name, tensors=tensors)


# ggml's own type table: (elements per block, bytes per block), by the
# `ggml_type` id stored for each tensor. Checked 2026-09-30 by summing the
# table for two real files against their data regions (Qwen3-30B-A3B
# Q4_K_M and UD-Q3_K_XL): equal to within alignment padding. An id not
# here is a type this reader will not size, and says so with None rather
# than guessing -- the module's rule.
GGML_TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
}
# MoE expert tensors: `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`,
# `ffn_gate_up_exps`, and the chunked `_chexps` forms llama.cpp's own fit
# names in its placements. A shared expert (`_shexp`) is always active and
# stays with the rest.
_EXPERT_TENSOR = re.compile(r"_(?:ch)?exps(?:\.|$)")
_MAX_TENSORS = 1 << 20
_MAX_DIMS = 8


#: The tensor that holds an MTP head, by block (llama.cpp's
#: `LLM_TENSOR_NEXTN_EH_PROJ`).
_NEXTN_RE = re.compile(r"^blk\.(\d+)\.nextn\.eh_proj\.weight$")
#: A small whole model named as a draft, as the catalogue sorts them.
DRAFT_NAME_RE = re.compile(r"^draft[-_.]", re.IGNORECASE)


@dataclass(frozen=True)
class _TensorTable:
    total: int | None
    experts: int | None
    classifier: bool
    nextn_blocks: frozenset[int]
    has_first_block: bool
    has_markov_head: bool


def is_expert_tensor(name: str) -> bool:
    return _EXPERT_TENSOR.search(name) is not None


def _read_tensor_table(reader: _Reader, count: int) -> _TensorTable:
    """Sum every tensor's bytes and the experts' share, from the table alone,
    and note where any MTP head is."""
    total = 0
    experts = 0
    known = True
    classifier = False
    nextn: set[int] = set()
    first_block = False
    markov_head = False
    for _ in range(min(count, _MAX_TENSORS)):
        name = reader.string()
        classifier |= name.startswith("cls.")
        first_block |= name.startswith("blk.0.")
        markov_head |= name == "markov_w1.weight"
        head = _NEXTN_RE.match(name)
        if head is not None:
            nextn.add(int(head.group(1)))
        dims = int(reader.scalar(T_UINT32))
        if dims < 1 or dims > _MAX_DIMS:
            raise reader._fail(f"implausible dimension count {dims} for tensor {name!r}")
        elements = 1
        for _ in range(dims):
            elements *= int(reader.scalar(T_UINT64))
        ggml_type = int(reader.scalar(T_UINT32))
        reader.scalar(T_UINT64)  # offset
        size = GGML_TYPE_SIZES.get(ggml_type)
        if size is None or elements % size[0]:
            known = False
            continue
        nbytes = elements // size[0] * size[1]
        total += nbytes
        if is_expert_tensor(name):
            experts += nbytes
    return _TensorTable(
        total if known else None,
        experts if known else None,
        classifier,
        frozenset(nextn),
        first_block,
        markov_head,
    )


def read_metadata_stream(
    fh: BinaryIO,
    *,
    name: str = "<stream>",
    tensors: bool = False,
    keep_arrays: frozenset[str] = frozenset(),
) -> GgufMetadata:
    """The same parse, over any seekable binary stream.

    Exists so a *remote* header can be read: M3's preflight fetches the
    first few megabytes of a file over HTTP Range and parses them from a
    buffer, which is what lets quant guidance be built on
    `general.file_type` rather than on a filename. A buffer that stops
    inside the KV block raises `GgufTruncated` carrying the offset it
    wanted, so the caller fetches exactly that much and retries instead
    of guessing a window size.
    """
    reader = _Reader(fh, Path(name))
    reader.keep = keep_arrays | _TARGET_ARRAYS

    magic = reader.raw(4)
    if magic != MAGIC:
        raise GgufError(f"{name}: not a GGUF file (magic {magic!r})")

    version = int(reader.scalar(T_UINT32))
    if version < 2 or version > 3:
        # v1 predates the current KV encoding; anything above 3 does
        # not exist yet and would be parsed on a guess.
        raise GgufError(f"{name}: unsupported GGUF version {version}")

    tensor_count = reader.length(limit=1 << 32, what="tensor count")
    kv_count = reader.length(limit=_MAX_KV, what="kv count")

    kv: dict[str, Any] = {}
    lengths: dict[str, int] = {}
    for _ in range(kv_count):
        key = reader.string()
        tag = int(reader.scalar(T_UINT32))
        kv[key] = reader.value(tag, key=key, lengths=lengths)
    header_bytes = reader.tell()

    # The tensor table follows the KV block: names, shapes, types and
    # offsets, a few kilobytes, never a weight. Read on request only, so a
    # remote preflight's window is unchanged unless it asks for it.
    tensor_bytes: int | None = None
    expert_bytes: int | None = None
    classifier = False
    nextn_blocks: frozenset[int] | None = None
    first_block: bool | None = None
    markov_head = False
    architecture = kv.get("general.architecture")
    non_causal = kv.get(f"{architecture}.attention.causal") is False
    if tensors or non_causal:
        table = _read_tensor_table(reader, tensor_count)
        classifier = table.classifier
        markov_head = table.has_markov_head
        if tensors:
            tensor_bytes, expert_bytes = table.total, table.experts
            nextn_blocks, first_block = table.nextn_blocks, table.has_first_block

    return GgufMetadata(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        header_bytes=header_bytes,
        kv=kv,
        array_lengths=lengths,
        tensor_bytes=tensor_bytes,
        expert_bytes=expert_bytes,
        has_classifier_tensors=classifier,
        nextn_blocks=nextn_blocks,
        has_first_block=first_block,
        has_markov_head=markov_head,
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


#: `tokenizer.ggml.token_type` values llama.cpp parses as special when it
#: tokenizes text with special-token parsing on: unknown, control and
#: user-defined (`llama-vocab.cpp`, the special-token cache). Normal (1),
#: unused (5) and byte (6) tokens are not.
_SPECIAL_TYPES = frozenset({2, 3, 4})
#: What a markup spelling holds: `< > [ ] |`, or DeepSeek's full-width bar.
_MARKUP = frozenset("<>[]|\uff5c")
_VOCAB = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})
#: Turn and end-of-generation spellings llama.cpp treats as control tokens at
#: load whatever type a GGUF gives them (`llama-vocab.cpp`, the end-of-turn
#: override): a community GGUF that types them normal is still read with
#: them as specials. Listed whenever the vocabulary holds them.
_TURN_SPELLINGS = frozenset(
    {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|end|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|return|>",
        "<|call|>",
        "<start_of_turn>",
        "<end_of_turn>",
    }
)


def special_tokens(path: Path) -> list[str]:
    """The spellings this model's engine reads as special tokens in text (DI4).

    Reads the whole vocabulary, which `read_metadata` never does, so it is
    for one model on request. Only spellings that look like markup are
    kept: three characters or more, not all whitespace, holding one of
    `< > [ ] |` or the full-width bar. A whitespace or word token a
    vocabulary marks user-defined is left out, because breaking it would
    change ordinary text. Longest first, each once."""
    with path.open("rb") as fh:
        meta = read_metadata_stream(fh, name=path.name, keep_arrays=_VOCAB)
    tokens = meta.kv.get("tokenizer.ggml.tokens")
    types = meta.kv.get("tokenizer.ggml.token_type")
    if not isinstance(tokens, list) or not isinstance(types, list):
        return []
    found = {
        token
        for token, kind in zip(tokens, types, strict=False)
        if isinstance(token, str)
        and (
            token in _TURN_SPELLINGS
            or (
                kind in _SPECIAL_TYPES
                and len(token) >= 3
                and not token.isspace()
                and any(c in _MARKUP for c in token)
            )
        )
    }
    return sorted(found, key=lambda t: (-len(t), t))
