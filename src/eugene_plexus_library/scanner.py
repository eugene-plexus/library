"""The walk: turn directories of files into a list of launchable models.

Most of the difficulty in this component is here, and almost none of it
is in reading a header. A directory of `.gguf` files is not a list of
models, and the gap between those two things is where a library either
earns trust or quietly loses a model somebody downloaded.

## What is one model

| Format | The model is | Not a model |
|---|---|---|
| GGUF | one `.gguf` file | the `mmproj-*` projector; an MTP or draft companion; shards 2..N |
| GGUF, split | the **first** shard | the other shards individually |
| safetensors | the **directory** | `blobs/`, `refs/`, `.no_exist/`, adapters |
| prepared | its `<name>.eugene-prepared.json` | nothing: the engine's files are not read |

## Everything skipped is reported

`skipped[]` carries a path and a reason for every file the walk saw and
decided against. This is not diagnostics padding — it is the feature. A
scanner that silently drops what it did not understand is
indistinguishable from a broken one, and "why isn't my model showing
up" with no answer is how a tool gets uninstalled.

## Cost, and the shape it forces

Reading a large-vocab GGUF header is ~50-60 ms warm (see
`formats.gguf`), so a header is opened at most **once** per scan and
only when the cache misses. Two rules keep that true and are easy to
break:

* The cache is consulted from the `(path, size, mtime_ns)` stat, before
  anything is opened.
* Metadata read during projector detection is carried forward into the
  entry rather than re-read. Reading the same 10 MB block twice in one
  scan is invisible in a unit test and doubles the wall clock on a real
  library.

A projector header, by contrast, is ~1 KB and 0.04 ms — measured — so
projectors are re-read every scan without caching and it costs nothing.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import prepared
from ._generated.models import (
    DraftFile,
    DraftKind,
    GgufDetail,
    GgufSpeculation,
    KevCheckpointDetail,
    LibraryModel,
    MlxQuantization,
    ModelCapabilities,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    OllamaDetail,
    RecommendedSampling,
    SafetensorsDetail,
    SkippedPath,
    SkipReason,
)
from ._generated.models import Status1 as ScanRootStatus
from .formats import gguf, kev, safetensors
from .formats import ollama as ollama_store
from .paths import display_name, model_id, normalize

log = logging.getLogger(__name__)

GGUF_SUFFIX = ".gguf"
SAFETENSORS_SUFFIX = ".safetensors"

# Partial downloads. M3's downloader writes these; having the reason
# before then beats surfacing a half-fetched 40 GB model as broken.
PARTIAL_SUFFIXES = (".part", ".partial", ".download", ".tmp", ".incomplete")

# Formats with no reader yet. Named so an MLX user is told "not
# supported" rather than left wondering why the directory looks empty.
UNSUPPORTED_SUFFIXES = frozenset({".npz", ".mlx", ".bin", ".pt", ".pth", ".ckpt"})

# Directories never worth descending into.
SKIP_DIR_NAMES = frozenset({".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv"})

TOKENIZER_NAMES = frozenset({"vocab.txt", "merges.txt", "vocab.json", "spiece.model"})


@dataclass
class ScanCounters:
    """Running totals, read by the scan endpoint while a walk is live.

    Mutated from the worker thread and read from the event loop without
    a lock: every field is a single int or a str rebind, both atomic
    under CPython, and a progress counter that is momentarily one behind
    is not worth a lock on the hot path.
    """

    files_scanned: int = 0
    models_found: int = 0
    current_path: str | None = None


@dataclass
class RootResult:
    """Outcome for one configured root."""

    path: str
    status: ScanRootStatus
    models_found: int = 0
    files_scanned: int = 0
    error: str | None = None


@dataclass
class ScanResult:
    models: list[LibraryModel] = field(default_factory=list)
    skipped: list[SkippedPath] = field(default_factory=list)
    roots: list[RootResult] = field(default_factory=list)
    files_scanned: int = 0


# A cached entry is reusable when the file it came from is unchanged.
CacheKey = tuple[str, int, int]
CacheLookup = Callable[[CacheKey], LibraryModel | None]


def cache_key(path: Path, stat: os.stat_result) -> CacheKey:
    """`(normalized path, size, mtime_ns)` — the incremental-rescan key.

    `mtime_ns` rather than `mtime`: whole-second resolution misses a
    file rewritten within the same second as its last scan, which is
    exactly what a re-quantize-and-overwrite loop does.
    """
    return (normalize(path), stat.st_size, stat.st_mtime_ns)


def _mlx_quantization(config: safetensors.ModelConfig) -> MlxQuantization | None:
    """The MLX conversion marker, when `config.json` carries one.

    Load-bearing for engine choice: an MLX-quantized directory packs its
    weights as integer tensors that only the MLX loader reads, so the UI
    should offer the mlx engine and not vLLM for it. Absence means
    unknown, not incompatible — see the schema's own caveats, including
    why `parameters` must not feed per-parameter arithmetic here.
    """
    marker = config.mlx_quantization
    if marker is None:
        return None
    bits, group = marker
    return MlxQuantization(bits=bits, groupSize=group)


KEV_HEAD_NAME = kev.HEAD_NAME


def _is_kev_checkpoint(names: set[str]) -> bool:
    """`adapter_config.json` + `head.pt`: the decision head is the one
    positive marker separating a Kev checkpoint from an ordinary LoRA."""
    return kev.is_checkpoint(names)


def _kev_role(name: str) -> ModelFileRole:
    return kev.file_role(name)


def _timestamp(stat: os.stat_result) -> datetime:
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _path_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _stat_size(entry: os.DirEntry[str]) -> int | None:
    try:
        return entry.stat().st_size
    except OSError:
        return None


def _lacks_speculation(model: LibraryModel) -> bool:
    """A GGUF entry cached before SD1 (2026-10-18), re-read once so it says
    whether it carries its own MTP head. A bound drafter cached as a model
    before row 17 must also be reclassified from its header."""
    return (
        model.format is ModelFormat.gguf
        and model.gguf is not None
        and model.status is ModelStatus.present
        and (model.gguf.speculation is None or model.architecture in {"dflash", "eagle3"})
    )


@dataclass(frozen=True)
class _Draft:
    """A speculative-decoding companion found in a model folder."""

    entry: os.DirEntry[str]
    metadata: gguf.GgufMetadata
    kind: DraftKind
    size: int


def _with_drafts(model: LibraryModel, drafts: list[_Draft], paired: set[str]) -> LibraryModel:
    """The model with the companions beside it it can draft with, and only
    those: its files' `draft` entries and `sizeBytes` are rebuilt from this
    scan, so a companion deleted since the entry was cached is gone.

    MTP and simple companions pair by architecture. Bound drafters pair by
    target hidden size and layer ids. Every kind also checks vocabulary
    size when both files declare one.
    """
    if model.format is not ModelFormat.gguf or model.gguf is None:
        return model
    detail = model.gguf
    mine: list[_Draft] = []
    for draft in drafts:
        if draft.kind in {DraftKind.dflash, DraftKind.dspark, DraftKind.eagle3}:
            meta = draft.metadata
            if meta.draft_target_problem is not None:
                continue
            target = detail.metadata or {}
            hidden = target.get(f"{model.architecture}.embedding_length")
            blocks = target.get(f"{model.architecture}.block_count")
            minimum = meta.target_min_layers
            if (
                not isinstance(hidden, int)
                or isinstance(hidden, bool)
                or hidden != meta.target_hidden_size
                or not isinstance(blocks, int)
                or isinstance(blocks, bool)
                or minimum is None
                or blocks < minimum
            ):
                continue
        elif draft.metadata.architecture != model.architecture:
            continue
        theirs = draft.metadata.vocab_size
        if theirs is not None and detail.vocabSize is not None and theirs != detail.vocabSize:
            continue
        mine.append(draft)
        paired.add(draft.entry.path)
    old = [f for f in model.files or [] if f.role is ModelFileRole.draft]
    if not mine and not old:
        return model
    kept = [f for f in model.files or [] if f.role is not ModelFileRole.draft]
    files = kept + [
        ModelFile(path=d.entry.path, role=ModelFileRole.draft, sizeBytes=d.size) for d in mine
    ]
    size = model.sizeBytes
    if size is not None:
        size = size - sum(f.sizeBytes or 0 for f in old) + sum(d.size for d in mine)
    speculation = GgufSpeculation(
        mtpHead=detail.speculation.mtpHead if detail.speculation is not None else False,
        draftFiles=[DraftFile(path=d.entry.path, kind=d.kind, sizeBytes=d.size) for d in mine],
    )
    return model.model_copy(
        update={
            "files": files,
            "fileCount": len(files),
            "sizeBytes": size,
            "gguf": detail.model_copy(update={"speculation": speculation}),
        }
    )


def _draft_skip_detail(draft: _Draft, paired: set[str]) -> str:
    labels = {
        DraftKind.mtp: "MTP head (draft-mtp)",
        DraftKind.simple: "draft model (draft-simple)",
        DraftKind.dflash: "DFlash drafter (draft-dflash)",
        DraftKind.dspark: "DSpark drafter (draft-dspark)",
        DraftKind.eagle3: "EAGLE3 drafter (draft-eagle3)",
    }
    what = labels[draft.kind]
    if draft.entry.path in paired:
        return f"{what} for speculative decoding with the model beside it"
    if draft.kind in {DraftKind.mtp, DraftKind.simple}:
        return (
            f"{what} for speculative decoding, but no model of the same architecture is beside it"
        )
    meta = draft.metadata
    problem = meta.draft_target_problem
    if problem is not None:
        return f"{what} cannot pair: {problem}"
    return (
        f"{what}, but no model it was made for is beside it: it needs hidden size "
        f"{meta.target_hidden_size} and at least {meta.target_min_layers} layers"
    )


def _lacks_expert_bytes(model: LibraryModel) -> bool:
    """A GGUF entry cached before the tensor table was read (2026-09-30).

    Re-read once rather than served: without `expertBytes` a MoE model's
    fit cannot say that its experts, not its layers, would move, and a
    library that never re-scans would never learn it. One header read per
    model, after which the entry carries the number and is cached again.
    """
    return (
        model.format is ModelFormat.gguf
        and model.gguf is not None
        and model.gguf.expertBytes is None
    )


class Scanner:
    """Walks roots and produces models.

    Holds no state between runs beyond the cache lookup handed in, so a
    cancelled scan leaves nothing half-updated — the caller simply keeps
    the previous result.
    """

    def __init__(
        self,
        *,
        cache_lookup: CacheLookup | None = None,
        should_cancel: Callable[[], bool] | None = None,
        counters: ScanCounters | None = None,
        follow_symlinks: bool = False,
    ) -> None:
        self._cache_lookup = cache_lookup or (lambda _key: None)
        self._should_cancel = should_cancel or (lambda: False)
        self.counters = counters or ScanCounters()
        self._follow_symlinks = follow_symlinks
        self._stores: set[str] = set()

    # -- entry point ----------------------------------------------------

    def scan(self, roots: list[Path], *, ollama_roots: Sequence[Path] = ()) -> ScanResult:
        """Walk `roots`. Those also in `ollama_roots` are Ollama stores, read
        through their manifests rather than walked (CX9)."""
        result = ScanResult()
        seen: set[str] = set()
        stores = {normalize(r) for r in ollama_roots}
        self._stores = stores
        for root in roots:
            if self._should_cancel():
                break
            result.roots.append(
                self._scan_root(root, result, seen, store=normalize(root) in stores)
            )
        result.models = prepared.link_sources(result.models)
        return result

    def _scan_root(
        self, root: Path, result: ScanResult, seen: set[str], *, store: bool = False
    ) -> RootResult:
        text = str(root)
        try:
            if not root.exists():
                return RootResult(
                    path=text, status=ScanRootStatus.missing, error="path does not exist"
                )
            if not root.is_dir():
                return RootResult(
                    path=text, status=ScanRootStatus.unreadable, error="path is not a directory"
                )
            # Cheapest possible readability probe: if the directory
            # cannot be listed, nothing below it matters.
            with os.scandir(root) as probe:
                next(iter(probe), None)
        except PermissionError as exc:
            return RootResult(
                path=text, status=ScanRootStatus.unreadable, error=f"permission denied ({exc})"
            )
        except OSError as exc:
            return RootResult(path=text, status=ScanRootStatus.unreadable, error=str(exc))

        models_before = len(result.models)
        files_before = result.files_scanned

        if store:
            self._scan_ollama_store(root, result, seen)
        else:
            for directory in self._walk(root, result):
                if self._should_cancel():
                    break
                self._scan_directory(directory, root, result, seen)

        return RootResult(
            path=text,
            status=ScanRootStatus.ok,
            models_found=len(result.models) - models_before,
            files_scanned=result.files_scanned - files_before,
        )

    # -- walking --------------------------------------------------------

    def _walk(self, root: Path, result: ScanResult) -> Iterator[Path]:
        """Yield directories worth looking at, depth-first.

        Prunes rather than filtering afterwards: a HuggingFace `blobs/`
        can hold tens of gigabytes across thousands of hash-named files,
        and descending into it to reject each one individually would
        dominate the scan.
        """
        stack: list[tuple[Path, frozenset[tuple[int, int]]]] = [(root, frozenset())]
        while stack:
            if self._should_cancel():
                return
            current, ancestors = stack.pop()
            try:
                stat = current.stat()
            except OSError as exc:
                log.warning("cannot stat directory %s: %s", current, exc)
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in ancestors:
                result.skipped.append(
                    SkippedPath(
                        path=str(current),
                        reason=SkipReason.not_a_model,
                        detail="directory link cycle: target is an ancestor",
                    )
                )
                continue
            ancestors = ancestors | {identity}
            yield current
            if (current / prepared.ENGINE_FILES_MARKER).is_file():
                # An engine's own folder: nothing below it is a model (B44).
                result.skipped.append(
                    SkippedPath(
                        path=str(current),
                        reason=SkipReason.not_a_model,
                        detail="an engine's own prepared files; only provenance files are listed",
                    )
                )
                continue

            try:
                with os.scandir(current) as it:
                    children = sorted(
                        (e for e in it if self._descend_into(e)),
                        key=lambda e: e.name,
                    )
            except OSError as exc:
                log.warning("cannot list %s: %s", current, exc)
                continue

            for entry in children:
                name = entry.name
                if name in SKIP_DIR_NAMES:
                    continue
                if normalize(entry.path) in self._stores:
                    # An Ollama store inside this folder is read through its manifests
                    # as its own folder (CX9), never walked as files.
                    result.skipped.append(
                        SkippedPath(
                            path=entry.path,
                            reason=SkipReason.not_a_model,
                            detail="an Ollama model store, listed as its own folder",
                        )
                    )
                    continue
                if safetensors.HF_DATASET_DIR_RE.match(name):
                    result.skipped.append(
                        SkippedPath(
                            path=entry.path,
                            reason=SkipReason.not_a_model,
                            detail="HuggingFace dataset cache",
                        )
                    )
                    continue
                if name in safetensors.HF_INFRASTRUCTURE:
                    detail = "HuggingFace cache infrastructure"
                    if name == ".no_exist":
                        detail += " — a negative cache of zero-byte placeholder files"
                    result.skipped.append(
                        SkippedPath(path=entry.path, reason=SkipReason.not_a_model, detail=detail)
                    )
                    continue
                stack.append((Path(entry.path), ancestors))

    def _descend_into(self, entry: os.DirEntry[str]) -> bool:
        try:
            if not self._follow_symlinks and (entry.is_symlink() or Path(entry.path).is_junction()):
                return False
            return entry.is_dir()
        except OSError as exc:
            log.warning("cannot inspect directory entry %s: %s", entry.path, exc)
            return False

    def _scan_directory(
        self, directory: Path, root: Path, result: ScanResult, seen: set[str]
    ) -> None:
        entries: list[os.DirEntry[str]] = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            entries.append(entry)
                    except OSError as exc:
                        result.skipped.append(
                            SkippedPath(
                                path=entry.path,
                                reason=SkipReason.unreadable_header,
                                detail=f"cannot inspect file: {exc}",
                            )
                        )
        except OSError as exc:
            log.warning("cannot list %s: %s", directory, exc)
            return

        entries.sort(key=lambda entry: entry.name)
        names = {e.name for e in entries}

        # A prepared model is its provenance file, wherever it sits: beside a
        # GGUF, inside a safetensors folder, or alone. The engine's own files
        # are not read (`prepared`), so this comes before any format's rules.
        for entry in entries:
            if prepared.is_provenance(entry.name):
                result.files_scanned += 1
                self.counters.files_scanned += 1
                self.counters.current_path = entry.path
                model = prepared.model_from(Path(entry.path), root)
                if model.id not in seen:
                    seen.add(model.id)
                    result.models.append(model)
                    self.counters.models_found += 1
        if prepared.ENGINE_FILES_MARKER in names:
            return  # the engine's own files (B44)

        # A safetensors model *is* the directory, so that is decided
        # before any file in it is considered on its own.
        if self._handle_safetensors_dir(directory, root, entries, names, result, seen):
            return

        self._handle_loose_files(directory, root, entries, result, seen)

    # -- an Ollama store ------------------------------------------------

    def _scan_ollama_store(self, root: Path, result: ScanResult, seen: set[str]) -> None:
        """List what an Ollama store's manifests name (CX9), read-only.

        One model per GGUF blob, named by its first `model:tag`, with the
        header read as for any GGUF (and cached the same way). What Ollama
        keeps that llama.cpp cannot load is skipped with its reason.
        """
        if not ollama_store.is_store(root):
            result.skipped.append(
                SkippedPath(
                    path=str(root),
                    reason=SkipReason.not_a_model,
                    detail="not an Ollama model store: it needs `blobs` and `manifests` folders",
                )
            )
            return
        found, skipped = ollama_store.read_store(root)
        for skip in skipped:
            result.skipped.append(
                SkippedPath(
                    path=skip.path, reason=SkipReason.unsupported_format, detail=skip.detail
                )
            )
        for entry in found:
            if self._should_cancel():
                return
            result.files_scanned += 1
            self.counters.files_scanned += 1
            self.counters.current_path = str(entry.blob)
            model = self._ollama_model(entry, root, result)
            if model is not None and model.id not in seen:
                seen.add(model.id)
                result.models.append(model)
                self.counters.models_found += 1

    def _ollama_model(
        self, entry: ollama_store.StoreModel, root: Path, result: ScanResult
    ) -> LibraryModel | None:
        try:
            stat = entry.blob.stat()
        except OSError as exc:
            result.skipped.append(
                SkippedPath(
                    path=str(entry.blob),
                    reason=SkipReason.unreadable_header,
                    detail=f"cannot stat: {exc}",
                )
            )
            return None
        cached = self._cache_lookup(cache_key(entry.blob, stat))
        built: LibraryModel | None = None
        if (
            cached is not None
            and cached.ollama is not None
            and not _lacks_expert_bytes(cached)
            and not _lacks_speculation(cached)
        ):
            built = cached
        else:
            metadata = self._read_gguf(entry.blob, result)
            if metadata is None:
                return None
            built = self._build_gguf_model(entry.blob, root, [], metadata, [])
        if built is None:
            return None
        projector_file = (
            ModelFile(
                path=str(entry.projector),
                role=ModelFileRole.projector,
                sizeBytes=_path_size(entry.projector),
            )
            if entry.projector is not None
            else None
        )
        files = [f for f in built.files or [] if f.role is not ModelFileRole.projector]
        total = built.sizeBytes
        if projector_file is not None:
            files.append(projector_file)
            if built.sizeBytes is not None:
                total = built.sizeBytes + (projector_file.sizeBytes or 0)
        update: dict[str, object] = {
            "name": entry.tags[0],
            "displayName": None,
            "files": files,
            "fileCount": len(files),
            "sizeBytes": total,
            "ollama": OllamaDetail(
                tags=entry.tags,
                needsMigration=(
                    True
                    if entry.legacy_layout
                    and (built.architecture or "") in ollama_store.MIGRATED_ARCHITECTURES
                    else None
                ),
            ),
        }
        if built.gguf is not None:
            update["gguf"] = built.gguf.model_copy(
                update={
                    "projectorPath": str(entry.projector) if entry.projector is not None else None
                }
            )
        if built.capabilities is not None:
            update["capabilities"] = built.capabilities.model_copy(
                update={"vision": entry.projector is not None}
            )
        return built.model_copy(update=update)

    # -- safetensors -----------------------------------------------------

    def _handle_safetensors_dir(
        self,
        directory: Path,
        root: Path,
        entries: list[os.DirEntry[str]],
        names: set[str],
        result: ScanResult,
        seen: set[str],
    ) -> bool:
        """True if this directory is a safetensors model, or a deliberate
        rejection of one — either way its files are not considered
        individually afterwards."""
        weight_names = sorted(n for n in names if n.endswith(SAFETENSORS_SUFFIX))
        if not weight_names:
            return False

        if _is_kev_checkpoint(names):
            # An adapter directory WITH a decision head is not a LoRA to
            # skip — it is a launchable Kev decision checkpoint, and
            # `head.pt` is what tells the two apart. Decided before the
            # adapter skip below or every checkpoint would vanish as
            # "a LoRA, not a model" (measured layout: jaredpalmer/kev-0.8b
            # is adapter_config.json + adapter_model.safetensors + head.pt
            # + tokenizer + provenance.json, with NO base config.json).
            model = self._kev_model(directory, root, entries, names)
            if model is not None and model.path not in seen:
                seen.add(model.path)
                result.models.append(model)
                self.counters.models_found += 1
            return True

        if safetensors.is_adapter_dir(directory, names):
            result.skipped.append(
                SkippedPath(
                    path=str(directory),
                    reason=SkipReason.adapter,
                    detail="adapter_config.json with no base weights — a LoRA, not a model",
                )
            )
            return True

        if safetensors.CONFIG_NAME not in names:
            result.skipped.append(
                SkippedPath(
                    path=str(directory),
                    reason=SkipReason.not_a_model,
                    detail=f"{SAFETENSORS_SUFFIX} files with no {safetensors.CONFIG_NAME}",
                )
            )
            return True

        # A HuggingFace cache keeps one snapshot directory per revision.
        # Keep the one refs/main names and report the rest; otherwise the
        # same model is listed once per revision ever fetched.
        revision = safetensors.hf_revision(directory)
        if revision is not None:
            current = safetensors.hf_current_revision(directory)
            if current is not None and current != revision:
                result.skipped.append(
                    SkippedPath(
                        path=str(directory),
                        reason=SkipReason.older_revision,
                        detail=(
                            f"HuggingFace snapshot {revision[:12]}; refs/main is {current[:12]}"
                        ),
                    )
                )
                return True

        model = self._build_safetensors_model(directory, root, entries, weight_names, revision)
        if model is not None and model.id not in seen:
            seen.add(model.id)
            result.models.append(model)
            self.counters.models_found += 1
        result.files_scanned += len(weight_names)
        self.counters.files_scanned += len(weight_names)
        return True

    def _build_safetensors_model(
        self,
        directory: Path,
        root: Path,
        entries: list[os.DirEntry[str]],
        weight_names: list[str],
        revision: str | None,
    ) -> LibraryModel | None:
        primary = directory / weight_names[0]
        self.counters.current_path = str(primary)

        try:
            primary_stat = primary.stat()
        except OSError as exc:
            return self._unreadable(directory, root, ModelFormat.safetensors, str(exc))

        # Keyed on the primary weight file. A tokenizer edited without
        # touching the weights will not invalidate the entry — acceptable,
        # because nothing surfaced here comes from a tokenizer file, and
        # `full: true` re-reads everything when that assumption breaks.
        cached = self._cache_lookup(cache_key(primary, primary_stat))
        if cached is not None:
            return cached

        config_path = directory / safetensors.CONFIG_NAME
        try:
            config = safetensors.read_config(config_path)
        except (safetensors.SafetensorsError, OSError) as exc:
            return self._unreadable(directory, root, ModelFormat.safetensors, str(exc))

        parameters = 0
        dtype_counts: dict[str, int] = {}
        elements: dict[str, int] = {}
        for name in weight_names:
            try:
                header = safetensors.read_header(directory / name)
            except (safetensors.SafetensorsError, OSError) as exc:
                return self._unreadable(directory, root, ModelFormat.safetensors, str(exc))
            parameters += header.parameters
            for dtype, count in header.dtype_counts.items():
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            elements.update(header.elements)
        quantization = config.raw.get("quantization")
        if isinstance(quantization, dict):
            # An MLX conversion stores packed words, not parameters (A4).
            parameters = safetensors.mlx_parameters(elements, quantization) or 0

        files = self._safetensors_files(entries, weight_names[0])
        shards = sum(1 for n in weight_names if safetensors.shard_position(Path(n)) is not None)

        return LibraryModel(
            id=model_id(directory),
            path=str(directory),
            root=str(root),
            format=ModelFormat.safetensors,
            # A cache snapshot directory is named after the revision
            # hash, so fall back to the repo's own name there.
            name=safetensors.hf_model_name(directory) or display_name(directory),
            status=ModelStatus.present,
            sizeBytes=sum(f.sizeBytes or 0 for f in files),
            fileCount=len(files),
            architecture=config.architecture,
            contextLength=config.context_length,
            parameters=parameters or None,
            capabilities=ModelCapabilities(
                chat=not config.is_embedding,
                embedding=config.is_embedding,
                vision=False,
                chatTemplate=safetensors.has_chat_template(directory, config.raw),
            ),
            files=files,
            safetensors=SafetensorsDetail(
                mlxQuantization=_mlx_quantization(config),
                dtype=safetensors.dominant_dtype(dtype_counts),
                shardCount=max(shards, 1),
                repoId=safetensors.hf_repo_id(directory),
                revision=revision,
                configPath=str(config_path),
            ),
            modifiedAt=_timestamp(primary_stat),
        )

    def _kev_model(
        self,
        directory: Path,
        root: Path,
        entries: list[os.DirEntry[str]],
        names: set[str],
    ) -> LibraryModel | None:
        """One Kev decision checkpoint, files and provenance included.

        The point of recording it is reproducibility: the checkpoint is
        only half a runnable model — Kev downloads the base named in
        `adapter_config.json` separately — so `kev.baseModel` is what an
        offline restore needs to pre-fetch. Every artifact the loader
        reads (adapter, decision head, tokenizer, calibration) is in
        `files` with a role. Decision-only: `capabilities.decision`,
        never `chat` — offering it a conversation fails clearly at the
        driver instead of producing prose.
        """
        try:
            raw = json.loads(
                (directory / safetensors.ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            raw = {}
        base_model = raw.get("base_model_name_or_path") if isinstance(raw, dict) else None

        files: list[ModelFile] = []
        for entry in entries:
            if not entry.is_file():
                continue
            role = _kev_role(entry.name)
            files.append(
                ModelFile(path=str(directory / entry.name), role=role, sizeBytes=_stat_size(entry))
            )

        try:
            primary_stat = directory.stat()
        except OSError:
            return None
        revision = safetensors.hf_revision(directory)
        return LibraryModel(
            id=model_id(directory),
            path=str(directory),
            root=str(root),
            format=ModelFormat.kev_checkpoint,
            name=safetensors.hf_model_name(directory) or display_name(directory),
            status=ModelStatus.present,
            sizeBytes=sum(f.sizeBytes or 0 for f in files),
            fileCount=len(files),
            capabilities=ModelCapabilities(
                chat=False,
                embedding=False,
                vision=False,
                decision=True,
            ),
            files=files,
            kev=KevCheckpointDetail(
                baseModel=str(base_model) if base_model else None,
                repoId=safetensors.hf_repo_id(directory),
                revision=revision,
            ),
            modifiedAt=_timestamp(primary_stat),
        )

    def _safetensors_files(
        self, entries: list[os.DirEntry[str]], primary_name: str
    ) -> list[ModelFile]:
        out: list[ModelFile] = []
        for entry in entries:
            name = entry.name
            if name.endswith(SAFETENSORS_SUFFIX):
                role = ModelFileRole.weights if name == primary_name else ModelFileRole.shard
            elif name.endswith(".index.json"):
                role = ModelFileRole.index
            elif name == safetensors.CONFIG_NAME or name.endswith("_config.json"):
                role = ModelFileRole.config
            elif name.startswith("tokenizer") or name in TOKENIZER_NAMES:
                role = ModelFileRole.tokenizer
            else:
                role = ModelFileRole.other
            out.append(ModelFile(path=entry.path, role=role, sizeBytes=_stat_size(entry)))
        return out

    # -- GGUF and other loose files ---------------------------------------

    def _handle_loose_files(
        self,
        directory: Path,
        root: Path,
        entries: list[os.DirEntry[str]],
        result: ScanResult,
        seen: set[str],
    ) -> None:
        ggufs: list[os.DirEntry[str]] = []

        for entry in entries:
            lower = entry.name.lower()
            if lower.endswith(PARTIAL_SUFFIXES):
                result.skipped.append(
                    SkippedPath(
                        path=entry.path,
                        reason=SkipReason.incomplete_download,
                        detail="partial download",
                    )
                )
            elif lower.endswith(GGUF_SUFFIX):
                ggufs.append(entry)
            elif Path(lower).suffix in UNSUPPORTED_SUFFIXES:
                result.skipped.append(
                    SkippedPath(
                        path=entry.path,
                        reason=SkipReason.unsupported_format,
                        detail=f"no reader for {Path(entry.name).suffix} files",
                    )
                )

        if ggufs:
            self._handle_ggufs(root, ggufs, result, seen)

    def _handle_ggufs(
        self,
        root: Path,
        entries: list[os.DirEntry[str]],
        result: ScanResult,
        seen: set[str],
    ) -> None:
        """Group a directory's `.gguf` files into models.

        Shard grouping is by filename, deliberately: with five 8 GB parts
        the point is to read the first one's header and merely stat the
        rest.
        """
        shard_groups: dict[str, dict[int, os.DirEntry[str]]] = {}
        singles: list[os.DirEntry[str]] = []

        for entry in entries:
            position = gguf.shard_position(Path(entry.name))
            if position is None:
                singles.append(entry)
            else:
                stem, index, _total = position
                shard_groups.setdefault(stem, {})[index] = entry

        # (entry, other shards, metadata-or-None-if-cache-hit)
        candidates: list[tuple[os.DirEntry[str], list[os.DirEntry[str]]]] = []
        cached_models: list[LibraryModel] = []
        metadata_by_path: dict[str, gguf.GgufMetadata] = {}
        projectors: list[tuple[os.DirEntry[str], gguf.GgufMetadata]] = []
        drafts: list[_Draft] = []

        def consider(entry: os.DirEntry[str], shards: list[os.DirEntry[str]]) -> None:
            """Cache first, header only on a miss — see the module
            docstring. A projector is never a cached model, so a hit
            always means "this is a model, use the stored entry"."""
            result.files_scanned += 1
            self.counters.files_scanned += 1
            self.counters.current_path = entry.path
            path = Path(entry.path)
            try:
                stat = path.stat()
            except OSError as exc:
                result.skipped.append(
                    SkippedPath(
                        path=entry.path,
                        reason=SkipReason.unreadable_header,
                        detail=f"cannot stat: {exc}",
                    )
                )
                return

            cached = self._cache_lookup(cache_key(path, stat))
            # The first part's stat cannot detect a copy interrupted in a later part.
            if cached is not None and shards:
                sizes = {f.path: f.sizeBytes for f in cached.files or []}
                if any(sizes.get(e.path) != _stat_size(e) for e in shards):
                    cached = None
            if (
                cached is not None
                and not _lacks_expert_bytes(cached)
                and not _lacks_speculation(cached)
                # V4's head can live in another shard. Old cached scans
                # recorded only the first shard's drafting tensors.
                and not (cached.architecture == "deepseek4" and shards)
            ):
                # Earlier scans kept the complete header's scalar metadata.
                # Refresh the observation without re-reading a large vocabulary.
                if (
                    cached.architecture == "deepseek4"
                    and cached.gguf is not None
                    and cached.capabilities is not None
                    and cached.capabilities.chatTemplate is False
                ):
                    from .chat_templates import fallback_for

                    raw = cached.gguf.metadata or {}
                    observed = gguf.GgufMetadata(
                        version=cached.gguf.ggufVersion or 0,
                        tensor_count=0,
                        kv_count=len(raw),
                        header_bytes=0,
                        kv=raw,
                    )
                    cached = cached.model_copy(
                        update={
                            "gguf": cached.gguf.model_copy(
                                update={"chatTemplateFallback": fallback_for(observed)}
                            )
                        }
                    )
                cached_models.append(cached)
                return

            metadata = self._read_gguf(path, result)
            if metadata is None:
                return
            # Validate every part, even when an unknown type prevents sizing
            # the whole model. Carry the experts' sum forward without re-reading.
            for shard in shards:
                part = self._read_gguf(Path(shard.path), result)
                if part is None:
                    return
                metadata.expert_bytes = (
                    metadata.expert_bytes + part.expert_bytes
                    if metadata.expert_bytes is not None and part.expert_bytes is not None
                    else None
                )
                if metadata.architecture == "deepseek4":
                    metadata.nextn_blocks = (
                        metadata.nextn_blocks | part.nextn_blocks
                        if metadata.nextn_blocks is not None
                        and part.nextn_blocks is not None
                        and part.architecture == metadata.architecture
                        and part.arch_key("block_count") == metadata.arch_key("block_count")
                        else None
                    )
                    metadata.has_first_block = bool(
                        metadata.has_first_block or part.has_first_block
                    )
            if metadata.is_projector:
                projectors.append((entry, metadata))
                return
            kind = (
                metadata.draft_kind(entry.name)
                if not shards or metadata.architecture in {"dflash", "eagle3"}
                else None
            )
            if kind is not None:
                drafts.append(_Draft(entry, metadata, DraftKind(kind), stat.st_size))
                return
            metadata_by_path[entry.path] = metadata
            candidates.append((entry, shards))

        for entry in singles:
            consider(entry, [])

        for stem, parts in sorted(shard_groups.items()):
            first = parts.get(1)
            if first is None:
                # Shards with no part 1: the model is incomplete. Report
                # every part rather than inventing an entry from part 2.
                for entry in parts.values():
                    result.skipped.append(
                        SkippedPath(
                            path=entry.path,
                            reason=SkipReason.shard_member,
                            detail=f"shard of {stem!r}, but part 00001 is missing",
                        )
                    )
                continue
            rest = [e for index, e in sorted(parts.items()) if index != 1]
            for entry in rest:
                result.skipped.append(
                    SkippedPath(
                        path=entry.path,
                        reason=SkipReason.shard_member,
                        detail=f"part of the split model {stem!r}, listed under its first shard",
                    )
                )
            consider(first, rest)

        for entry, metadata in projectors:
            result.skipped.append(
                SkippedPath(
                    path=entry.path,
                    reason=SkipReason.projector,
                    detail=(
                        f"vision projector (general.type={metadata.general_type or 'clip'}) "
                        "for the model beside it"
                    ),
                )
            )

        found: list[LibraryModel] = list(cached_models)
        for entry, shards in candidates:
            built = self._build_gguf_model(
                Path(entry.path), root, shards, metadata_by_path[entry.path], projectors
            )
            if built is not None:
                found.append(built)

        # Drafts are paired on every scan, cached models included, because a
        # companion downloaded beside a model leaves the model's own file and
        # so its cache entry unchanged.
        paired: set[str] = set()
        for model in found:
            model = _with_drafts(model, drafts, paired)
            if model.id not in seen:
                seen.add(model.id)
                result.models.append(model)
                self.counters.models_found += 1

        for draft in drafts:
            result.skipped.append(
                SkippedPath(
                    path=draft.entry.path,
                    reason=SkipReason.draft,
                    detail=_draft_skip_detail(draft, paired),
                )
            )

    def _read_gguf(self, path: Path, result: ScanResult) -> gguf.GgufMetadata | None:
        try:
            # The tensor table too, for `expertBytes` (moe-aware-fit §1.2).
            return gguf.read_metadata(path, tensors=True)
        except gguf.GgufError as exc:
            detail = str(exc)
        except OSError as exc:
            detail = f"cannot read: {exc}"
        result.skipped.append(
            SkippedPath(path=str(path), reason=SkipReason.unreadable_header, detail=detail)
        )
        return None

    def _build_gguf_model(
        self,
        path: Path,
        root: Path,
        shards: list[os.DirEntry[str]],
        metadata: gguf.GgufMetadata,
        projectors: list[tuple[os.DirEntry[str], gguf.GgufMetadata]],
    ) -> LibraryModel | None:
        from .chat_templates import fallback_for

        try:
            stat = path.stat()
        except OSError as exc:
            return self._unreadable(path, root, ModelFormat.gguf, str(exc))

        projector = self._match_projector(metadata, projectors)

        files = [ModelFile(path=str(path), role=ModelFileRole.weights, sizeBytes=stat.st_size)]
        total = stat.st_size
        for entry in shards:
            size = _stat_size(entry)
            total += size or 0
            files.append(ModelFile(path=entry.path, role=ModelFileRole.shard, sizeBytes=size))
        if projector is not None:
            size = _stat_size(projector)
            total += size or 0
            files.append(
                ModelFile(path=projector.path, role=ModelFileRole.projector, sizeBytes=size)
            )

        from_metadata = metadata.quantization
        from_filename = gguf.quant_from_filename(path.name)
        disagrees = (
            from_metadata is not None
            and from_filename is not None
            and from_metadata.upper() != from_filename.upper()
        )

        name = display_name(path)
        declared = metadata.name
        sampling = metadata.recommended_sampling
        expert_bytes = metadata.expert_bytes

        return LibraryModel(
            id=model_id(path),
            path=str(path),
            root=str(root),
            format=ModelFormat.gguf,
            name=name,
            displayName=declared if declared and declared != name else None,
            status=ModelStatus.present,
            sizeBytes=total,
            fileCount=len(files),
            architecture=metadata.architecture,
            contextLength=metadata.context_length,
            # GGUF carries no parameter count — only `sizeLabel`, which
            # is a string the uploader typed. See formats.gguf.
            parameters=None,
            sizeLabel=metadata.size_label,
            capabilities=ModelCapabilities(
                chat=not (metadata.is_embedding or metadata.is_rerank),
                embedding=metadata.is_search_embedding,
                rerank=metadata.is_rerank,
                vision=projector is not None,
                chatTemplate=metadata.has_chat_template,
            ),
            files=files,
            gguf=GgufDetail(
                quantization=from_metadata or from_filename,
                fileType=metadata.file_type,
                quantizationDisagrees=disagrees or None,
                ggufVersion=metadata.version,
                shardCount=len(shards) + 1,
                vocabSize=metadata.vocab_size,
                expertBytes=expert_bytes,
                chatTemplateFallback=fallback_for(metadata),
                projectorPath=projector.path if projector is not None else None,
                speculation=GgufSpeculation(mtpHead=metadata.mtp_head, draftFiles=[]),
                recommendedSampling=(
                    RecommendedSampling(
                        temperature=sampling.temperature,
                        topK=sampling.top_k,
                        topP=sampling.top_p,
                    )
                    if sampling
                    else None
                ),
                metadata=metadata.public_kv(),
            ),
            modifiedAt=_timestamp(stat),
        )

    def _match_projector(
        self,
        metadata: gguf.GgufMetadata,
        projectors: list[tuple[os.DirEntry[str], gguf.GgufMetadata]],
    ) -> os.DirEntry[str] | None:
        """Pair a model with the projector sitting beside it.

        Same directory plus a matching `general.name` is the reliable
        signal — verified on a real pair, where the projector repeats the
        model's name exactly. Failing that, a lone projector in a
        directory with one model is the overwhelmingly common shape and
        is taken; refusing would mislabel a vision model as text-only.
        """
        if not projectors:
            return None
        name = metadata.name
        if name:
            for entry, projector_metadata in projectors:
                if projector_metadata.name == name:
                    return entry
        return projectors[0][0] if len(projectors) == 1 else None

    # -- helpers -----------------------------------------------------------

    def _unreadable(
        self, path: Path, root: Path, model_format: ModelFormat, error: str
    ) -> LibraryModel:
        """An entry for something clearly meant to be a model and clearly
        not readable. Surfaced rather than skipped: a file the operator
        can see and we cannot explain is the worst of the three states."""
        return LibraryModel(
            id=model_id(path),
            path=str(path),
            root=str(root),
            format=model_format,
            name=display_name(path),
            status=ModelStatus.unreadable,
            error=error,
        )
