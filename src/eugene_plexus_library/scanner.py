"""The walk: turn directories of files into a list of launchable models.

Most of the difficulty in this component is here, and almost none of it
is in reading a header. A directory of `.gguf` files is not a list of
models, and the gap between those two things is where a library either
earns trust or quietly loses a model somebody downloaded.

## What is one model

| Format | The model is | Not a model |
|---|---|---|
| GGUF | one `.gguf` file | the `mmproj-*` projector; shards 2..N |
| GGUF, split | the **first** shard | the other shards individually |
| safetensors | the **directory** | `blobs/`, `refs/`, `.no_exist/`, adapters |

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

import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ._generated.models import (
    GgufDetail,
    LibraryModel,
    ModelCapabilities,
    ModelFile,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    RecommendedSampling,
    SafetensorsDetail,
    SkippedPath,
    SkipReason,
)
from ._generated.models import Status1 as ScanRootStatus
from .formats import gguf, safetensors
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


def _timestamp(stat: os.stat_result) -> datetime:
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _stat_size(entry: os.DirEntry[str]) -> int | None:
    try:
        return entry.stat().st_size
    except OSError:
        return None


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
    ) -> None:
        self._cache_lookup = cache_lookup or (lambda _key: None)
        self._should_cancel = should_cancel or (lambda: False)
        self.counters = counters or ScanCounters()

    # -- entry point ----------------------------------------------------

    def scan(self, roots: list[Path]) -> ScanResult:
        result = ScanResult()
        seen: set[str] = set()
        for root in roots:
            if self._should_cancel():
                break
            result.roots.append(self._scan_root(root, result, seen))
        return result

    def _scan_root(self, root: Path, result: ScanResult, seen: set[str]) -> RootResult:
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
        stack = [root]
        while stack:
            if self._should_cancel():
                return
            current = stack.pop()
            yield current

            try:
                with os.scandir(current) as it:
                    children = sorted(
                        (e for e in it if e.is_dir(follow_symlinks=False)),
                        key=lambda e: e.name,
                    )
            except OSError as exc:
                log.warning("cannot list %s: %s", current, exc)
                continue

            for entry in children:
                name = entry.name
                if name in SKIP_DIR_NAMES:
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
                stack.append(Path(entry.path))

    def _scan_directory(
        self, directory: Path, root: Path, result: ScanResult, seen: set[str]
    ) -> None:
        try:
            with os.scandir(directory) as it:
                entries = sorted(
                    (e for e in it if e.is_file(follow_symlinks=False)),
                    key=lambda e: e.name,
                )
        except OSError as exc:
            log.warning("cannot list %s: %s", directory, exc)
            return

        names = {e.name for e in entries}

        # A safetensors model *is* the directory, so that is decided
        # before any file in it is considered on its own.
        if self._handle_safetensors_dir(directory, root, entries, names, result, seen):
            return

        self._handle_loose_files(directory, root, entries, result, seen)

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
        for name in weight_names:
            try:
                header = safetensors.read_header(directory / name)
            except (safetensors.SafetensorsError, OSError) as exc:
                return self._unreadable(directory, root, ModelFormat.safetensors, str(exc))
            parameters += header.parameters
            for dtype, count in header.dtype_counts.items():
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count

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
                chatTemplate=(
                    "chat_template" in config.raw or (directory / "chat_template.jinja").exists()
                ),
            ),
            files=files,
            safetensors=SafetensorsDetail(
                dtype=safetensors.dominant_dtype(dtype_counts),
                shardCount=max(shards, 1),
                repoId=safetensors.hf_repo_id(directory),
                revision=revision,
                configPath=str(config_path),
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
            if cached is not None:
                cached_models.append(cached)
                return

            metadata = self._read_gguf(path, result)
            if metadata is None:
                return
            if metadata.is_projector:
                projectors.append((entry, metadata))
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

        for model in cached_models:
            if model.id not in seen:
                seen.add(model.id)
                result.models.append(model)
                self.counters.models_found += 1

        for entry, shards in candidates:
            built = self._build_gguf_model(
                Path(entry.path), root, shards, metadata_by_path[entry.path], projectors
            )
            if built is not None and built.id not in seen:
                seen.add(built.id)
                result.models.append(built)
                self.counters.models_found += 1

    def _read_gguf(self, path: Path, result: ScanResult) -> gguf.GgufMetadata | None:
        try:
            return gguf.read_metadata(path)
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
                chat=not metadata.is_embedding,
                embedding=metadata.is_embedding,
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
                projectorPath=projector.path if projector is not None else None,
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
