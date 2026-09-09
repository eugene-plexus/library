"""Turning an upstream repo into choices an operator can act on.

## The work this does that a file listing does not

Verified on `unsloth/Qwen3.8-27B-GGUF`: 35 tree entries, 30 of them
`.gguf`, and **25 actual choices**.

    24  single-file quant candidates      Qwen3.8-27B-UD-Q4_K_M.gguf
     1  split candidate, two shards       BF16/…-00001-of-00002.gguf + …-00002…
     2  vision projectors                 mmproj-BF16.gguf, mmproj-F16.gguf
     1  imatrix calibration file          imatrix_unsloth.gguf
     1  MTP draft model                   MTP/mtp-Qwen3.8-27B-Q4_0.gguf

List the `.gguf` files and you offer four things that cannot be launched
and one that is half a model. The rules are the scanner's, applied to a
remote listing instead of a local directory — which is the reason they
are worth having in one shape: a projector is a projector whether it is
on disk or on the hub.

## Sizes are per candidate, never per file

`BF16/…-00001-of-00002.gguf` is 46.55 GiB and the model is 50.90 GiB.
Score the file named on the launch line and you understate the candidate
by the size of every other shard.

## The quant label is filename-derived here, and says so

The hub exposes no per-file quant, so at listing time the name is all
there is — the very thing a local scan refuses to trust. Hence
`quantSource: filename`, and hence `preflight`, which reads
`general.file_type` over HTTP Range and upgrades it to `metadata`. Fit
is computed from the size either way, which is authoritative, so a
mislabelled file changes what is displayed and never what is promised.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import fit as fit_mod
from . import quants
from ._generated.models import (
    AlreadyOwned,
    Basis,
    CatalogueCandidate,
    CatalogueFile,
    CatalogueModel,
    CatalogueRecommendation,
    CatalogueSearchResult,
    Fit,
    FitVerdict,
    GateKind,
    KvCacheType,
    MatchedOn,
    MemoryBudget,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    QuantSource,
)
from .formats import gguf
from .hub import FileMetadata, RepoInfo
from .store import StateStore

log = logging.getLogger(__name__)

LOW_QUALITY_BPW = 4.0
"""Below this, say so. Recommending a 1.81-bits-per-weight quant without
comment is how a first impression becomes "this thing is stupid", and
the failure is ours rather than the model's."""

_SAFETENSORS_SIDECARS = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "chat_template.jinja",
}

_DOC_NAMES = {"readme.md", "license", "license.txt", ".gitattributes", "notice"}

_PROJECTOR_RE = re.compile(r"(^|[/_-])mmproj", re.IGNORECASE)
_IMATRIX_RE = re.compile(r"imatrix", re.IGNORECASE)
_DRAFT_RE = re.compile(r"(^|/)(mtp|draft)[-_/]|(^|[/_-])(mtp|draft)\.", re.IGNORECASE)
# Does a label look like a quant tier at all? Loose on purpose: the
# point is to tell `Q4_K_XL` and `UD-IQ2_S` (tiers we have never heard
# of) from `instruct` or a bare model name, not to validate a list.
_TIER_SHAPED_RE = re.compile(r"(?:^|[-_.])(?:I?Q\d|BF16|F16|F32|MXFP4|TQ\d)", re.IGNORECASE)


@dataclass
class _Group:
    """Files that make up one candidate, before it is scored."""

    label: str
    format: ModelFormat
    files: list[FileMetadata] = field(default_factory=list)
    quant: str | None = None
    shard_total: int | None = None

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _role_for(path: str, fmt: ModelFormat, *, index: int) -> ModelFileRole:
    name = PurePosixPath(path).name.lower()
    if fmt is ModelFormat.gguf:
        return ModelFileRole.weights if index == 0 else ModelFileRole.shard
    if name.endswith(".safetensors"):
        return ModelFileRole.weights if index == 0 else ModelFileRole.shard
    if name.endswith(".index.json"):
        return ModelFileRole.index
    if "tokenizer" in name or name in ("vocab.json", "merges.txt"):
        return ModelFileRole.tokenizer
    if name.endswith(".json") or name.endswith(".jinja"):
        return ModelFileRole.config
    return ModelFileRole.other


def _to_catalogue_file(meta: FileMetadata, role: ModelFileRole) -> CatalogueFile:
    return CatalogueFile(
        path=meta.path,
        sizeBytes=meta.size,
        sha256=meta.sha256,
        gitBlobSha1=meta.git_blob_sha1,
        role=role,
        lfs=meta.lfs,
    )


def _common_prefix(stems: list[str]) -> str:
    """The longest prefix every candidate stem shares, cut at a separator.

    Within one repo the stems are `Qwen3.8-27B-UD-Q4_K_M`,
    `Qwen3.8-27B-Q4_0`, `Qwen3.8-27B-BF16` — so the shared prefix is the
    model's own name and what remains is exactly what distinguishes one
    choice from another. Cut at a separator so a shared `Q` between
    `Q4_0` and `Q8_0` does not eat into the tier itself.
    """
    if len(stems) < 2:
        return ""
    shortest = min(stems, key=len)
    limit = 0
    for index in range(len(shortest)):
        if any(stem[index] != shortest[index] for stem in stems):
            break
        limit = index + 1
    prefix = shortest[:limit]
    cut = max(prefix.rfind("-"), prefix.rfind("_"), prefix.rfind("."))
    return prefix[: cut + 1] if cut >= 0 else ""


def label_from_stem(stem: str, prefix: str) -> str:
    """What to show in the list for one candidate.

    Deliberately **not** matched against llama.cpp's `file_type` names.
    Publishers invent tiers constantly — `Q4_K_XL`, `UD-Q8_K_XL`,
    whatever the next importance-matrix family is called — and a fixed
    list produced two visible failures on the first real repo: four
    different files all reading `UD-Q6_K` because only the `Q6_K` part
    matched, and other files falling back to their whole filename
    because nothing matched at all. Four identical rows on a screen
    whose whole job is "pick one" is the worst thing this could do.
    """
    remainder = stem[len(prefix) :].strip("-_.") if prefix else stem
    return remainder or stem


def quant_from_label(label: str) -> str | None:
    """The label, when it looks like a quant tier rather than a name.

    Loose by design: this decides whether to *call* the thing a quant,
    and the reference text for it is looked up separately by
    `quants.describe`, which tolerates decorations it does not know.
    """
    return label if _TIER_SHAPED_RE.search(label) else None


def quant_label(filename: str) -> str | None:
    """The tier from one filename alone, for callers with no repo context.

    Used by the preflight, which is handed a single path. Falls back to
    the `file_type`-derived family, so it can be shorter than the
    catalogue's label for the same file.
    """
    quant = gguf.quant_from_filename(filename)
    if quant is None:
        return None
    if re.search(r"(?:^|[._-])UD[._-]", filename, re.IGNORECASE):
        return f"UD-{quant}"
    return quant


def _gguf_groups(
    files: Iterable[FileMetadata],
) -> tuple[list[_Group], list[FileMetadata], list[FileMetadata]]:
    """`(candidates, projectors, other)` for the `.gguf` half of a repo."""
    candidates: dict[str, _Group] = {}
    projectors: list[FileMetadata] = []
    other: list[FileMetadata] = []

    for meta in files:
        path = PurePosixPath(meta.path)
        if path.suffix.lower() != ".gguf":
            continue

        # A projector belongs to the model beside it, is 0.9-1.8 GB, and
        # is never a candidate on its own. A repo commonly ships two
        # precisions of one and the operator picks.
        if _PROJECTOR_RE.search(meta.path):
            projectors.append(meta)
            continue
        # The calibration matrix a publisher used to make the IQ quants.
        # A `.gguf` that is not a model at all.
        if _IMATRIX_RE.search(meta.path):
            other.append(meta)
            continue
        # Multi-token-prediction / speculative-draft weights, usually in
        # their own subdirectory. Launchable only as a `--model-draft`
        # alongside something else, which is not a choice this screen is
        # offering.
        if _DRAFT_RE.search(meta.path):
            other.append(meta)
            continue

        shard = gguf.shard_position(PurePosixPath(path.name))  # type: ignore[arg-type]
        if shard is not None:
            stem, _index, total = shard
            key = f"{path.parent}/{stem}"
            group = candidates.get(key)
            if group is None:
                group = _Group(
                    label=stem,
                    format=ModelFormat.gguf,
                    shard_total=total,
                )
                candidates[key] = group
            group.files.append(meta)
            continue

        candidates[meta.path] = _Group(
            label=path.stem,
            format=ModelFormat.gguf,
        )
        candidates[meta.path].files.append(meta)

    # Labels are relative to the repo, so they can only be decided once
    # every candidate is known.
    prefix = _common_prefix([g.label for g in candidates.values()])
    for group in candidates.values():
        group.label = label_from_stem(group.label, prefix)
        group.quant = quant_from_label(group.label)
        # Shard 1 first: it is the file that goes on the launch line, and
        # the only one whose header carries the metadata.
        group.files.sort(key=lambda m: m.path)

    return list(candidates.values()), projectors, other


def _safetensors_groups(
    files: Iterable[FileMetadata], *, repo: str
) -> tuple[list[_Group], list[FileMetadata]]:
    """`(candidates, other)` for the safetensors half of a repo.

    One candidate per directory holding weights, because a repo can
    publish a base model at the root and variants in subdirectories. The
    sidecars come with it: `config.json` and the tokenizer files are not
    optional, and a download that fetches only the `.safetensors` files
    produces a directory nothing can load.
    """
    by_directory: dict[str, list[FileMetadata]] = {}
    other: list[FileMetadata] = []

    for meta in files:
        path = PurePosixPath(meta.path)
        name = path.name.lower()
        if (
            name.endswith(".safetensors")
            or name in _SAFETENSORS_SIDECARS
            or name.endswith(".index.json")
        ):
            by_directory.setdefault(str(path.parent), []).append(meta)
        elif name in _DOC_NAMES:
            other.append(meta)

    candidates: list[_Group] = []
    for directory, members in by_directory.items():
        weights = [m for m in members if m.path.lower().endswith(".safetensors")]
        if not weights:
            # Sidecars with no weights beside them: a tokenizer-only
            # directory, or a config for a variant published elsewhere.
            other.extend(members)
            continue
        label = repo.split("/")[-1] if directory in (".", "") else PurePosixPath(directory).name
        members.sort(key=lambda m: (not m.path.lower().endswith(".safetensors"), m.path))
        candidates.append(
            _Group(
                label=label,
                format=ModelFormat.safetensors,
                files=members,
                shard_total=len(weights) if len(weights) > 1 else None,
            )
        )
    return candidates, other


def _owned(group: _Group, store: StateStore) -> AlreadyOwned | None:
    """Is this candidate already on the disk?

    The join only this component can make, and what stops a 16 GB
    re-download of a file the operator already has. Matching is by
    filename and total size: the library deliberately never hashes model
    content — a 40 GB read per scan is not on the table — so `matchedOn`
    labels this as the strong guess it is rather than a certainty.
    """
    weights = group.files[0] if group.files else None
    if weights is None:
        return None
    wanted = PurePosixPath(weights.path).name.lower()
    for model in store.list_models():
        if model.status is not ModelStatus.present:
            continue
        if PurePosixPath(model.path.replace("\\", "/")).name.lower() != wanted:
            continue
        if model.sizeBytes and abs(model.sizeBytes - group.size) > 1024 * 1024:
            # Same name, materially different size: a requantized
            # re-upload, or a truncated copy. Not the same file, and
            # claiming it were would talk an operator out of a download
            # they need.
            continue
        return AlreadyOwned(modelId=model.id, path=model.path, matchedOn=MatchedOn.name_and_size)
    return None


def _shape_from_repo(info: RepoInfo) -> fit_mod.ModelShape:
    """What the repo-level metadata can contribute to the KV term.

    Almost nothing, and that is the point: the hub reports a repo's
    architecture, parameter count and trained context, but nothing about
    layers or attention heads. So a catalogue fit is `basis: estimate`
    until someone preflights the file — which is exactly the distinction
    the field exists to draw.
    """
    return fit_mod.ModelShape(
        context_length=info.context_length,
        parameters=info.parameters,
    )


def score(
    group: _Group,
    *,
    budget: MemoryBudget,
    context_length: int,
    shape: fit_mod.ModelShape,
    kv_cache_type: KvCacheType,
) -> Fit:
    return fit_mod.compute(
        weights_bytes=group.size,
        budget=budget,
        context_length=context_length,
        shape=shape,
        kv_cache_type=kv_cache_type,
    )


def _sort_key(candidate: CatalogueCandidate) -> tuple[float, int]:
    """Smallest first, by measured width where it is known.

    Bits per weight rather than raw size so a repo holding two model
    sizes still reads as an ordered list of quality steps; size breaks
    the tie and covers the candidates with no parameter count.
    """
    return (candidate.bitsPerWeight or 0.0, candidate.sizeBytes)


def recommend(
    candidates: list[CatalogueCandidate], *, context_length: int, budget: MemoryBudget
) -> tuple[CatalogueRecommendation | None, list[str]]:
    """The largest candidate that fully fits, and what to say about it.

    Not the largest that *runs*: partial offload is materially slower and
    is a decision the operator should make knowingly rather than inherit
    from a recommendation.
    """
    warnings: list[str] = []
    fitting = [c for c in candidates if c.fit and c.fit.verdict is FitVerdict.fits]

    if not fitting:
        runnable = [
            c for c in candidates if c.fit and c.fit.verdict in (FitVerdict.tight, FitVerdict.split)
        ]
        if runnable:
            best = max(runnable, key=_sort_key)
            free = fit_mod.format_bytes(budget.vramFreeBytes)
            warnings.append(
                f"Nothing here fits entirely in {free} of free GPU memory at "
                f"{context_length:,} tokens of context. The closest is {best.label} "
                f"({fit_mod.format_bytes(best.sizeBytes)}), which would need "
                f"{fit_mod.format_bytes(best.fit.requiredBytes if best.fit else None)}. "
                "Lower the context, accept partial CPU offload, or pick a smaller model."
            )
        else:
            warnings.append(
                "No candidate in this repo fits this machine at any of the offered sizes. "
                "A smaller model is the answer, not a smaller quant."
            )
        return None, warnings

    best = max(fitting, key=_sort_key)
    assert best.fit is not None
    reason_parts = [
        f"{best.label} is the largest option that fits entirely in GPU memory at "
        f"{context_length:,} tokens: {fit_mod.format_bytes(best.sizeBytes)} of weights plus "
        f"{fit_mod.format_bytes(best.fit.kvCacheBytes)} of KV cache, "
        f"{fit_mod.format_bytes(best.fit.requiredBytes)} in total against "
        f"{fit_mod.format_bytes(budget.vramFreeBytes)} free."
    ]
    if best.bitsPerWeight:
        reason_parts.append(f"That works out to {best.bitsPerWeight:.2f} bits per weight.")
    if best.fit.basis is Basis.estimate:
        reason_parts.append(
            "The KV-cache figure is an estimate — preflight this file for the real one."
        )

    low_quality = None
    if best.bitsPerWeight is not None and best.bitsPerWeight < LOW_QUALITY_BPW:
        low_quality = (
            f"{best.label} is only {best.bitsPerWeight:.2f} bits per weight. Below 4 bits "
            "quality is noticeably worse and below 3 it is severe, so this is a "
            "compromise rather than a recommendation: a shorter context, partial CPU "
            "offload, or a smaller model at a higher quant are all likely to serve you "
            "better."
        )
        warnings.append(low_quality)

    return (
        CatalogueRecommendation(
            label=best.label,
            reason=" ".join(reason_parts),
            lowQualityWarning=low_quality,
        ),
        warnings,
    )


def build_model(
    *,
    info: RepoInfo,
    files: list[FileMetadata],
    revision: str,
    store: StateStore,
    budget: MemoryBudget,
    context_length: int,
    kv_cache_type: KvCacheType = KvCacheType.f16,
) -> CatalogueModel:
    """Assemble the detail response: candidates, guidance, warnings."""
    gguf_groups, projectors, gguf_other = _gguf_groups(files)
    st_groups, st_other = _safetensors_groups(files, repo=info.repo)

    shape = _shape_from_repo(info)
    parameters = info.parameters

    candidates: list[CatalogueCandidate] = []
    for group in [*gguf_groups, *st_groups]:
        roles = [_role_for(f.path, group.format, index=i) for i, f in enumerate(group.files)]
        candidates.append(
            CatalogueCandidate(
                label=group.label,
                format=group.format,
                files=[_to_catalogue_file(f, r) for f, r in zip(group.files, roles, strict=True)],
                sizeBytes=group.size,
                quantization=group.quant,
                quantSource=QuantSource.filename if group.quant else None,
                bitsPerWeight=fit_mod.bits_per_weight(group.size, parameters),
                fit=score(
                    group,
                    budget=budget,
                    context_length=context_length,
                    shape=shape,
                    kv_cache_type=kv_cache_type,
                ),
                alreadyOwned=_owned(group, store),
                gated=info.gated is not GateKind.open,
            )
        )

    candidates.sort(key=_sort_key)
    recommendation, warnings = recommend(candidates, context_length=context_length, budget=budget)

    if info.gated is GateKind.manual:
        warnings.insert(
            0,
            "This model is gated and needs manual approval from its publisher, which can "
            "take days. Browsing and sizing work now; the download will fail with a 403 "
            "until access is granted and `hfToken` is set.",
        )
    elif info.gated is GateKind.auto:
        warnings.insert(
            0,
            "This model is gated. Accept its licence on the model's own page and set "
            "`hfToken` in this component's config; the download will fail with a 403 "
            "until both are done.",
        )

    if projectors:
        warnings.append(
            f"This is a multimodal model: {len(projectors)} vision "
            f"{'projector is' if len(projectors) == 1 else 'projectors are'} published "
            "separately, and one has to be downloaded alongside the quant for the model "
            "to see images. They are listed under `projectors`."
        )

    formats: list[ModelFormat] = []
    if gguf_groups:
        formats.append(ModelFormat.gguf)
    if st_groups:
        formats.append(ModelFormat.safetensors)

    other = [_to_catalogue_file(f, ModelFileRole.other) for f in [*gguf_other, *st_other]]

    return CatalogueModel(
        repo=info.repo,
        owner=info.owner,
        name=info.name,
        revision=revision,
        resolvedCommit=info.sha,
        gated=info.gated,
        private=bool(info.raw.get("private")),
        downloads=info.raw.get("downloads"),
        likes=info.raw.get("likes"),
        trendingScore=info.raw.get("trendingScore"),
        license=info.license,
        tags=info.tags,
        pipelineTag=info.raw.get("pipeline_tag"),
        libraryName=info.raw.get("library_name"),
        createdAt=info.raw.get("createdAt"),
        lastModified=info.raw.get("lastModified"),
        formats=formats,
        parameters=parameters,
        architecture=info.architecture,
        contextLength=info.context_length,
        chatTemplate=info.chat_template,
        candidates=candidates,
        projectors=[_to_catalogue_file(f, ModelFileRole.projector) for f in projectors],
        otherFiles=other,
        totalSizeBytes=sum(f.size for f in files),
        recommended=recommendation,
        warnings=warnings,
    )


def build_search_result(entry: dict) -> CatalogueSearchResult:
    """One search row. No sizes, and deliberately so.

    Upstream's search response carries filenames without sizes, so a fit
    verdict on a row would cost one extra call per row — fifty API
    requests for one keystroke, against a budget of 500 per five
    minutes. Sizes and guidance live on the detail response.
    """
    repo = str(entry.get("id") or entry.get("modelId") or "")
    tags = [t for t in (entry.get("tags") or []) if isinstance(t, str)]

    formats: list[ModelFormat] = []
    if "gguf" in tags or entry.get("library_name") == "gguf":
        formats.append(ModelFormat.gguf)
    if "safetensors" in tags or entry.get("library_name") == "transformers":
        formats.append(ModelFormat.safetensors)

    gated_raw = entry.get("gated")
    gated = (
        GateKind.auto
        if gated_raw == "auto"
        else GateKind.manual
        if gated_raw == "manual"
        else GateKind.open
    )

    license_tag = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)

    return CatalogueSearchResult(
        repo=repo,
        owner=entry.get("author") or (repo.split("/")[0] if "/" in repo else None),
        name=repo.split("/")[-1] if repo else None,
        formats=formats,
        gated=gated,
        private=bool(entry.get("private")),
        downloads=entry.get("downloads"),
        likes=entry.get("likes"),
        trendingScore=entry.get("trendingScore"),
        pipelineTag=entry.get("pipeline_tag"),
        libraryName=entry.get("library_name"),
        license=license_tag,
        tags=tags,
        createdAt=entry.get("createdAt"),
        lastModified=entry.get("lastModified"),
    )


def candidate_size(files: list[FileMetadata], path: str) -> int | None:
    """The whole candidate's size, given one of its files.

    A split model's shards sum to more than the file on the launch line,
    so a preflight that scored the file it probed would understate the
    candidate by every other shard — 46.55 GiB against a real 50.90 GiB
    on the one verified example.
    """
    groups, _, _ = _gguf_groups(files)
    for group in groups:
        if any(f.path == path for f in group.files):
            return group.size
    return None


def quant_notes(tier: str | None) -> list[str]:
    """Reference prose for one tier, for a UI that wants it inline."""
    described = quants.describe(tier)
    notes = []
    if described is not None and described.guidance:
        notes.append(described.guidance)
    notes.extend(quants.suffix_notes(tier))
    return notes
