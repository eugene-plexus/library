"""Each engine's fit, by that engine's own fit model (LS6, Troy's L11).

An engine's adapter declares how it uses memory beside what it loads
(`EngineFitModel`); this applies it to one model on one node:

* `spill` (llama.cpp): `fit.compute`, the arithmetic since M3.
* `reserved_share` (vLLM): `fit.compute_reserved_share`, its share of each card.
* `engine_table` (Strata): the engine's own answer for the file, which its
  adapter worked out for the node from the engine's setup and sent along.
  Nothing here knows Strata's rules.

An engine with no fit model, a table without the file, or a model whose
size is not known yet, is *Fit not estimated*: never another engine's
number in its place (library-sources-and-engines.md §6.1). Engines are
named by kind only; the console words them (B2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from . import fit as fit_mod
from . import model_fit
from ._generated.models import (
    Basis,
    EligibilityCandidate,
    EligibilityEngine,
    EngineFit,
    EngineFitModel,
    EngineTableFit,
    Fit,
    FitModelKind,
    FitOffload,
    FitQuestion,
    FitVerdict,
    HostHardware,
    LibraryModel,
    MemoryBudget,
    ModelFormat,
    Source,
)


@dataclass(frozen=True)
class Sizing:
    """What an engine's fit is computed from, wherever the model is."""

    #: The bytes an engine loads; None when not known (a search row).
    weights_bytes: int | None
    shape: fit_mod.ModelShape = field(default_factory=fit_mod.ModelShape)
    expert_bytes: int | None = None
    #: The file an engine's table names: a GGUF's first shard, or the file a
    #: prepared model was made from. Without its folder.
    file: str | None = None
    #: The facts were a guess (a search row's): the answer says so.
    approximate: bool = False


@dataclass(frozen=True)
class Question:
    """`FitQuestion` resolved: the budget and the context to score at."""

    budget: MemoryBudget
    context_length: int


def _name(path: str | None) -> str | None:
    if not path:
        return None
    # Either separator: a path as some host spells it.
    return PureWindowsPath(PurePosixPath(path).name).name or None


def sizing_of_model(model: LibraryModel) -> Sizing:
    if model.format is ModelFormat.prepared:
        source = model.prepared.source if model.prepared else None
        return Sizing(weights_bytes=None, file=_name(source.file if source else None))
    return Sizing(
        weights_bytes=model_fit.weights_of(model) if model.sizeBytes else None,
        shape=model_fit.shape_of(model),
        expert_bytes=model_fit.expert_bytes_of(model),
        file=_name(model.path),
    )


def sizing_of_candidate(candidate: EligibilityCandidate) -> Sizing:
    return Sizing(
        weights_bytes=candidate.sizeBytes,
        file=candidate.file,
        # A search row's facts are a guess; a version's were read off the hub.
        approximate=bool(candidate.approximate),
    )


def has_numbers(question: FitQuestion) -> bool:
    return any(
        value is not None
        for value in (
            question.vramFreeBytes,
            question.vramTotalBytes,
            question.ramAvailableBytes,
            question.ramTotalBytes,
            question.gpuCount,
        )
    )


def budget_of(question: FitQuestion, detected: HostHardware | None) -> MemoryBudget:
    """The node's memory as the caller measured it, or this host's when the
    caller sent no numbers (`detected`)."""
    if not has_numbers(question):
        assert detected is not None
        return fit_mod.budget_from_hardware(detected, unified_override=question.unifiedMemory)
    vram_free = question.vramFreeBytes or 0
    vram_total = max(question.vramTotalBytes or vram_free, vram_free)
    cards = question.gpuCount
    if cards is None:
        cards = 1 if vram_total > 0 else 0
    ram_available = question.ramAvailableBytes
    ram_total = question.ramTotalBytes
    return MemoryBudget(
        vramFreeBytes=vram_free,
        vramTotalBytes=vram_total,
        largestGpuFreeBytes=vram_free // cards if cards > 0 else 0,
        ramAvailableBytes=ram_available if ram_available is not None else (ram_total or 0),
        ramTotalBytes=ram_total if ram_total is not None else (ram_available or 0),
        gpuCount=cards,
        unifiedMemory=bool(question.unifiedMemory),
        source=Source.override,
    )


def not_estimated(reason: str, kind: FitModelKind | None = None) -> EngineFit:
    return EngineFit(estimated=False, model=kind, reason=reason)


def _gib(count: int | None) -> str:
    return fit_mod.format_bytes(count)


def _spill_words(result: Fit, max_context: int | None) -> str:
    budget = result.budget or MemoryBudget()
    need = f"about {_gib(result.requiredBytes)} at {result.contextLength:,} tokens"
    free = budget.vramFreeBytes or 0
    no_card = (budget.vramTotalBytes or 0) == 0 and not budget.unifiedMemory
    verdict = result.verdict
    if verdict is FitVerdict.unknown:
        return (
            "a graphics card here did not say how much memory it has, so there is "
            "nothing to compare against"
        )
    if no_card:
        where = f"{_gib(budget.ramAvailableBytes)} of system memory free"
        if verdict is FitVerdict.fits:
            return f"fits in system memory: {need}, {where}"
        if verdict is FitVerdict.tight:
            return f"fits this machine's memory, not what is free now: {need}, {where}"
        return f"too large here: {need}, more than this machine's memory"
    if verdict is FitVerdict.fits:
        return f"fits in free graphics memory: {need}, {_gib(free)} free"
    if verdict is FitVerdict.tight:
        return f"fits an idle card: {need}, {_gib(free)} of {_gib(budget.vramTotalBytes)} free now"
    if verdict is FitVerdict.split:
        how = (
            "its experts in system memory"
            if result.offload is FitOffload.experts
            else "part of it in system memory (partial offload)"
            if result.offload is FitOffload.layers
            else "system memory as well"
        )
        return f"runs with {how}: {need}, {_gib(free)} free on the card"
    longest = f"; it fits up to {max_context:,} tokens" if max_context else ""
    return f"too large here: {need}, more than graphics and system memory together{longest}"


def _share_words(result: Fit, utilization: float, max_context: int | None) -> str:
    budget = result.budget or MemoryBudget()
    share = int((budget.vramTotalBytes or 0) * utilization)
    need = f"about {_gib(result.requiredBytes)} at {result.contextLength:,} tokens"
    takes = f"the {_gib(share)} it takes ({utilization:.0%} of the cards' memory)"
    verdict = result.verdict
    if verdict is FitVerdict.unknown:
        return "no graphics card memory here to take a share of"
    if verdict is FitVerdict.fits:
        return f"fits its share: {need} of {takes}"
    if verdict is FitVerdict.tight:
        return (
            f"fits its share, {need} of {takes}, but only {_gib(budget.vramFreeBytes)} is "
            "free now, and it does not start with less than its share free"
        )
    longest = f"; its share holds up to {max_context:,} tokens" if max_context else ""
    return (
        f"too large for its share: {need}, more than {takes}; nothing moves to "
        f"system memory{longest}"
    )


def _table_row(table: Sequence[EngineTableFit], file: str | None) -> EngineTableFit | None:
    if not file:
        return None
    wanted = file.lower()
    return next((row for row in table if row.file.lower() == wanted), None)


def engine_fit(sizing: Sizing, engine: EligibilityEngine, question: Question) -> EngineFit:
    """One engine's fit for one model on the node the question describes."""
    declared: EngineFitModel | None = engine.fit
    if declared is None:
        return not_estimated("this engine has no fit estimate in Eugene yet")
    kind = declared.kind
    if kind is FitModelKind.engine_table:
        row = _table_row(declared.table or [], sizing.file)
        if row is None:
            return not_estimated("the engine's own table has no row for this model's file", kind)
        answer = row.fit.model_copy(update={"model": kind})
        if sizing.approximate and not answer.approximate:
            answer = answer.model_copy(update={"approximate": True})
        return answer
    if sizing.weights_bytes is None:
        return not_estimated("its size is not known yet: pick a version for a fit", kind)
    context = question.context_length
    if kind is FitModelKind.reserved_share:
        utilization = declared.gpuMemoryUtilization
        if utilization is None:
            return not_estimated("the engine declares no share of the card it takes", kind)
        result = fit_mod.compute_reserved_share(
            weights_bytes=sizing.weights_bytes,
            budget=question.budget,
            context_length=context,
            utilization=utilization,
            shape=sizing.shape,
        )
        longest = fit_mod.max_context_reserved_share(
            weights_bytes=sizing.weights_bytes,
            budget=question.budget,
            shape=sizing.shape,
            utilization=utilization,
        )
        reason = _share_words(result, utilization, longest)
        share = int((question.budget.vramTotalBytes or 0) * utilization)
    else:
        # A spill engine under a caller's budget compares free memory, as
        # `GET /v1/models/{id}/fit` does with `vramBytes` alone: the total is
        # for a share-taking engine, and it must not turn this page's
        # `split` into the popover's `tight`.
        spill_budget = (
            question.budget.model_copy(update={"vramTotalBytes": question.budget.vramFreeBytes})
            if question.budget.source is Source.override
            else question.budget
        )
        result = fit_mod.compute(
            weights_bytes=sizing.weights_bytes,
            budget=spill_budget,
            context_length=context,
            shape=sizing.shape,
            expert_bytes=sizing.expert_bytes,
        )
        longest = fit_mod.max_context_that_fits(
            weights_bytes=sizing.weights_bytes, budget=question.budget, shape=sizing.shape
        )
        reason = _spill_words(result, longest)
        share = None
    return EngineFit(
        estimated=True,
        verdict=result.verdict,
        model=kind,
        reason=reason,
        requiredBytes=result.requiredBytes,
        shareBytes=share,
        contextLength=result.contextLength,
        maxContextLength=longest,
        approximate=sizing.approximate or result.basis is Basis.estimate,
    )


def counts_as_no(answer: EngineFit | None) -> bool:
    """Only a measured `no` counts against a model; never *not estimated*
    or `unknown` (§4.3)."""
    return bool(answer and answer.estimated and answer.verdict is FitVerdict.no)
