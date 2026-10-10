"""The one judge of model against engine (library-sources-and-engines.md, LS1).

Each engine's adapter declares what it loads as data (`ModelRequirement`);
this matches a model's own facts against it. The console's Library page
and the agent's Run both ask here, so no rule is written twice: the MLX
rule used to be, once in the console and once in the agent.

Since LS2 it judges models not downloaded yet too (`EligibilityCandidate`):
Discover's search rows, a repo's versions and the starter set, by the
facts the catalogue answers carry. The rules are the same; what differs is
what an absent fact means. On a library model the library read the files,
so an absent fact is *unreadable* and fails a term that needs it. On a
candidate nobody has read them yet, so an absent fact is *not known* and
the term is assumed, at best `may_run`, and said.

Pure: the facts are the library's own, the engines are the caller's, and
nothing is fetched. Since LS6 it answers each engine's fit too, when asked,
by the engine's own fit model (`engine_fit`). Engines are named by kind
only; how an engine is called in words is the console's business, not the
library's.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath

from . import engine_fit
from ._generated.models import (
    EligibilityCandidate,
    EligibilityEngine,
    EligibilityLevel,
    EngineVerdict,
    EngineVerdictKind,
    LibraryModel,
    MlxQuantizationRule,
    ModelEligibility,
    ModelRequirement,
    ModelRequirementAuthority,
)

_RANK = {
    EngineVerdictKind.runs: 0,
    EngineVerdictKind.may_run: 1,
    EngineVerdictKind.after_preparation: 2,
    EngineVerdictKind.no: 3,
}
_LAST = 1_000_000

# Past this many names a list is counted, not read out: llama.cpp declares
# every architecture its build knows, well over a hundred.
_NAMES_READ_OUT = 4


@dataclass(frozen=True)
class Facts:
    """What the judge reads of a model, wherever the model is."""

    id: str
    format: str
    architecture: str | None
    quantization: str | None
    #: None only on a candidate: not known yet.
    mlx_quantized: bool | None
    #: True for a library model: the files were read, so an absent fact is
    #: unreadable. False for a candidate: an absent fact is not known yet.
    read: bool
    approximate: bool = False
    #: The engine a `prepared` model was prepared for (its provenance file);
    #: None for every other format, and for a provenance nobody could read.
    prepared_for: str | None = None
    #: The file's name without its folder: a GGUF's first shard (LS4). None
    #: on a candidate that names a repo rather than a file.
    file: str | None = None


def facts_of_model(model: LibraryModel) -> Facts:
    return Facts(
        id=model.id,
        format=_value(model.format),
        architecture=model.architecture,
        quantization=model.gguf.quantization if model.gguf else None,
        mlx_quantized=bool(model.safetensors and model.safetensors.mlxQuantization is not None),
        read=True,
        prepared_for=_value(model.prepared.engine) if model.prepared else None,
        # Either separator: a path as the library's own host spells it.
        file=PureWindowsPath(model.path).name if model.path else None,
    )


def facts_of_candidate(candidate: EligibilityCandidate) -> Facts:
    return Facts(
        id=candidate.id,
        format=_value(candidate.format),
        architecture=candidate.architecture,
        quantization=candidate.quantization,
        mlx_quantized=candidate.mlxQuantized,
        read=False,
        approximate=bool(candidate.approximate),
        file=candidate.file,
    )


def _value(item: object) -> str:
    return str(getattr(item, "value", item))


def _either(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _arch_failure(names: Sequence[str], it: str | None) -> str:
    if len(names) > _NAMES_READ_OUT:
        count = f"loads {len(names)} named architectures"
        if it is None:
            return f"{count}, and this library could not read this one's"
        return f"{count}, and {it} is not one of them"
    return (
        f"loads only the {_either(names)} architecture, and this one is "
        f"{it or 'of an architecture this library could not read'}"
    )


def _arch_assumed(names: Sequence[str]) -> str:
    if len(names) > _NAMES_READ_OUT:
        return f"its architecture is one of the {len(names)} it loads"
    return f"its architecture is {_either(names)}"


def _named(file: str | None, names: Sequence[str]) -> bool:
    """A file name against a list, ignoring case: Windows and the hub both
    keep a name's case, and a person's copy may not."""
    return file is not None and file.casefold() in {n.casefold() for n in names}


def _files_failure(names: Sequence[str], it: str | None) -> str:
    if len(names) > _NAMES_READ_OUT:
        return (
            f"runs only the {len(names)} files on its own list, and {it or 'this one'} is not one"
        )
    return f"runs only {_either(names)}, and this one is {it or 'not named'}"


def _files_assumed(names: Sequence[str]) -> str:
    if len(names) > _NAMES_READ_OUT:
        return f"it is one of the {len(names)} files on its own list"
    return f"it is {_either(names)}"


def _check(facts: Facts, need: ModelRequirement, engine: str) -> tuple[str | None, list[str]]:
    """`(why not, naming the term; the terms assumed for want of a fact)`.

    `engine` is the kind declaring `need`: a `prepared` requirement with no
    `preparedFor` means models prepared for that engine itself (LS3)."""
    if facts.format != _value(need.format):
        return f"loads {_value(need.format)} models, and this one is {facts.format}", []
    assumed: list[str] = []
    if facts.format == "prepared":
        wanted = _value(need.preparedFor) if need.preparedFor is not None else engine
        if facts.prepared_for is None and not facts.read:
            assumed.append(f"it was prepared for {wanted}")
        elif facts.prepared_for is None:
            return "loads prepared models, and this one's provenance file could not be read", []
        elif facts.prepared_for != wanted:
            return (
                f"loads only models prepared for {wanted}, and this one was prepared for "
                f"{facts.prepared_for}"
            ), []
    if need.architectures:
        if facts.architecture is None and not facts.read:
            assumed.append(_arch_assumed(need.architectures))
        elif facts.architecture not in need.architectures:
            return _arch_failure(need.architectures, facts.architecture), []
    if need.quantizations:
        if facts.quantization is None and not facts.read:
            assumed.append(f"it is {_either(need.quantizations)}")
        elif facts.quantization not in need.quantizations:
            it = facts.quantization or "of a quantization this library could not read"
            return f"loads only {_either(need.quantizations)}, and this one is {it}", []
    if need.files:
        if facts.file is None and not facts.read:
            assumed.append(_files_assumed(need.files))
        elif not _named(facts.file, need.files):
            return _files_failure(need.files, facts.file), []
    if need.mlxQuantization == MlxQuantizationRule.required:
        if facts.mlx_quantized is None:
            assumed.append("it is MLX-quantized")
        elif not facts.mlx_quantized:
            return "loads only MLX-quantized folders, and this one is not", []
    if need.mlxQuantization == MlxQuantizationRule.forbidden:
        if facts.mlx_quantized is None:
            assumed.append("it is not MLX-quantized")
        elif facts.mlx_quantized:
            return "cannot load MLX-quantized weights; only MLX reads them", []
    return None, assumed


def _verdict_of(need: ModelRequirement, assumed: Sequence[str]) -> EngineVerdictKind:
    if need.preparation is not None:
        return EngineVerdictKind.after_preparation
    if need.authority == ModelRequirementAuthority.engine or assumed:
        return EngineVerdictKind.may_run
    return EngineVerdictKind.runs


def _words(verdict: EngineVerdictKind, need: ModelRequirement, assumed: Sequence[str]) -> str:
    unless = f" if {' and '.join(assumed)}, which is not known yet" if assumed else ""
    if verdict == EngineVerdictKind.after_preparation:
        made = need.preparation.note if need.preparation and need.preparation.note else None
        return "runs it after preparing it" + (f" ({made})" if made else "") + unless
    if need.authority == ModelRequirementAuthority.engine:
        said = need.note or "may run it; only the engine can tell, when it loads"
        return said + (f"; and only{unless}" if assumed else "")
    if assumed:
        return "runs it" + unless + (f"; {need.note}" if need.note else "")
    return "runs it as it is" + (f"; {need.note}" if need.note else "")


def _preference(need: ModelRequirement) -> int:
    return need.preference if need.preference is not None else 100


def _judge_engine(facts: Facts, engine: EligibilityEngine) -> tuple[EngineVerdict, bool]:
    """One model against one engine, and whether the verdict rests on a guess."""
    met: list[tuple[ModelRequirement, list[str]]] = []
    for need in engine.accepts:
        failure, assumed = _check(facts, need, _value(engine.engine))
        if failure is None:
            met.append((need, assumed))
    best = (
        min(met, key=lambda m: (_RANK[_verdict_of(*m)], len(m[1]), _preference(m[0])))
        if met
        else None
    )
    if best is None:
        verdict, reason, assumed = EngineVerdictKind.no, _why_not(facts, engine), []
    else:
        need, assumed = best
        verdict = _verdict_of(need, assumed)
        reason = _words(verdict, need, assumed)
    return (
        EngineVerdict(
            engine=engine.engine,
            available=engine.available,
            installable=bool(engine.installable),
            experimental=bool(engine.experimental),
            verdict=verdict,
            reason=reason,
            preparation=best[0].preparation if best else None,
            preference=_preference(best[0]) if best else None,
        ),
        bool(assumed),
    )


def judge_engine(model: LibraryModel, engine: EligibilityEngine) -> EngineVerdict:
    """One library model against one engine: the best requirement it meets, or why none."""
    return _judge_engine(facts_of_model(model), engine)[0]


def _why_not(facts: Facts, engine: EligibilityEngine) -> str:
    """The most telling reason: a requirement of this model's own format
    fails on a finer term, which says more than "wrong format" does."""
    if not engine.accepts:
        return "loads no model from the library"
    same_format = [n for n in engine.accepts if _value(n.format) == facts.format]
    if same_format:
        return _check(facts, same_format[0], _value(engine.engine))[0] or "does not load this model"
    formats = sorted({_value(n.format) for n in engine.accepts})
    return f"loads {_either(formats)} models, and this one is {facts.format}"


def level_of(verdicts: Iterable[EngineVerdict]) -> EligibilityLevel:
    """Troy's three-level dot (L5).

    An engine whose own fit says `no` counts as one that cannot run the
    model (LS6, §4.3): a model too large for llama.cpp that Strata runs
    after preparing it is *other engine*, and one that fits no engine that
    would load it is *not here*. Only a measured `no` counts, never a fit
    that is not estimated or `unknown`.
    """
    verdicts = [
        v.model_copy(update={"verdict": EngineVerdictKind.no})
        if engine_fit.counts_as_no(v.fit)
        else v
        for v in verdicts
    ]
    here = (EngineVerdictKind.runs, EngineVerdictKind.may_run)
    if any(v.available and v.verdict in here for v in verdicts):
        return EligibilityLevel.works_here
    if any(v.available and v.verdict == EngineVerdictKind.after_preparation for v in verdicts):
        return EligibilityLevel.other_engine
    if any(
        not v.available and v.installable and v.verdict != EngineVerdictKind.no for v in verdicts
    ):
        return EligibilityLevel.other_engine
    return EligibilityLevel.not_here


def _order(verdict: EngineVerdict) -> tuple[bool, int, int, str]:
    return (
        not verdict.available,
        _RANK[verdict.verdict],
        verdict.preference if verdict.preference is not None else _LAST,
        _value(verdict.engine),
    )


def judge_facts(
    facts: Facts,
    engines: Sequence[EligibilityEngine],
    fit: tuple[engine_fit.Sizing, engine_fit.Question] | None = None,
) -> ModelEligibility:
    """Every engine against one model, best first, and the model's level.

    With `fit` (LS6), every verdict but `no` carries its engine's own fit,
    and the level counts an engine whose fit is `no` as one that cannot run
    the model."""
    judged = [_judge_engine(facts, e) for e in engines]
    if fit is not None:
        sizing, question = fit
        judged = [
            (
                verdict
                if verdict.verdict is EngineVerdictKind.no
                else verdict.model_copy(
                    update={"fit": engine_fit.engine_fit(sizing, engine, question)}
                ),
                guessed,
            )
            for (verdict, guessed), engine in zip(judged, engines, strict=True)
        ]
    verdicts = sorted((v for v, _ in judged), key=_order)
    return ModelEligibility(
        modelId=facts.id,
        level=level_of(verdicts),
        engines=verdicts,
        approximate=facts.approximate or any(guessed for _, guessed in judged),
    )


def judge(
    model: LibraryModel,
    engines: Sequence[EligibilityEngine],
    question: engine_fit.Question | None = None,
) -> ModelEligibility:
    fit = (engine_fit.sizing_of_model(model), question) if question is not None else None
    return judge_facts(facts_of_model(model), engines, fit)


def judge_candidate(
    candidate: EligibilityCandidate,
    engines: Sequence[EligibilityEngine],
    question: engine_fit.Question | None = None,
) -> ModelEligibility:
    fit = (engine_fit.sizing_of_candidate(candidate), question) if question is not None else None
    return judge_facts(facts_of_candidate(candidate), engines, fit)
