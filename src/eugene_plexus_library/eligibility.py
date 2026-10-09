"""The one judge of model against engine (library-sources-and-engines.md, LS1).

Each engine's adapter declares what it loads as data (`ModelRequirement`);
this matches a model's own facts against it. The console's Library page
and the agent's Run both ask here, so no rule is written twice: the MLX
rule used to be, once in the console and once in the agent.

Pure: the facts are the library's own, the engines are the caller's, and
nothing is fetched. Engines are named by kind only; how an engine is
called in words is the console's business, not the library's.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._generated.models import (
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


def _value(item: object) -> str:
    return str(getattr(item, "value", item))


def _either(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _quantization(model: LibraryModel) -> str | None:
    return model.gguf.quantization if model.gguf else None


def _mlx_marked(model: LibraryModel) -> bool:
    return bool(model.safetensors and model.safetensors.mlxQuantization is not None)


def _failed_term(model: LibraryModel, need: ModelRequirement) -> str | None:
    """Why `model` does not meet `need`, naming the term; None when it does."""
    if _value(model.format) != _value(need.format):
        return f"loads {_value(need.format)} models, and this one is {_value(model.format)}"
    if need.architectures and model.architecture not in need.architectures:
        it = model.architecture or "of an architecture this library could not read"
        return f"loads only the {_either(need.architectures)} architecture, and this one is {it}"
    if need.quantizations and _quantization(model) not in need.quantizations:
        it = _quantization(model) or "of a quantization this library could not read"
        return f"loads only {_either(need.quantizations)}, and this one is {it}"
    if need.mlxQuantization == MlxQuantizationRule.required and not _mlx_marked(model):
        return "loads only MLX-quantized folders, and this one is not"
    if need.mlxQuantization == MlxQuantizationRule.forbidden and _mlx_marked(model):
        return "cannot load MLX-quantized weights; only MLX reads them"
    return None


def _verdict_of(need: ModelRequirement) -> EngineVerdictKind:
    if need.preparation is not None:
        return EngineVerdictKind.after_preparation
    if need.authority == ModelRequirementAuthority.engine:
        return EngineVerdictKind.may_run
    return EngineVerdictKind.runs


def _words(verdict: EngineVerdictKind, need: ModelRequirement) -> str:
    if verdict == EngineVerdictKind.after_preparation:
        made = need.preparation.note if need.preparation and need.preparation.note else None
        return "runs it after preparing it" + (f" ({made})" if made else "")
    if verdict == EngineVerdictKind.may_run:
        return need.note or "may run it; only the engine can tell, when it loads"
    return "runs it as it is" + (f"; {need.note}" if need.note else "")


def judge_engine(model: LibraryModel, engine: EligibilityEngine) -> EngineVerdict:
    """One model against one engine: the best requirement it meets, or why none."""
    met = [need for need in engine.accepts if _failed_term(model, need) is None]
    best = min(met, key=lambda n: (_RANK[_verdict_of(n)], _preference(n))) if met else None
    verdict = _verdict_of(best) if best else EngineVerdictKind.no
    return EngineVerdict(
        engine=engine.engine,
        available=engine.available,
        installable=bool(engine.installable),
        experimental=bool(engine.experimental),
        verdict=verdict,
        reason=_words(verdict, best) if best else _why_not(model, engine),
        preparation=best.preparation if best else None,
        preference=_preference(best) if best else None,
    )


def _preference(need: ModelRequirement) -> int:
    return need.preference if need.preference is not None else 100


def _why_not(model: LibraryModel, engine: EligibilityEngine) -> str:
    """The most telling reason: a requirement of this model's own format
    fails on a finer term, which says more than "wrong format" does."""
    if not engine.accepts:
        return "loads no model from the library"
    same_format = [n for n in engine.accepts if _value(n.format) == _value(model.format)]
    if same_format:
        return _failed_term(model, same_format[0]) or "does not load this model"
    formats = sorted({_value(n.format) for n in engine.accepts})
    return f"loads {_either(formats)} models, and this one is {_value(model.format)}"


def level_of(verdicts: Iterable[EngineVerdict]) -> EligibilityLevel:
    """Troy's three-level dot (L5)."""
    verdicts = list(verdicts)
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


def judge(model: LibraryModel, engines: Sequence[EligibilityEngine]) -> ModelEligibility:
    """Every engine against one model, best first, and the model's level."""
    verdicts = sorted((judge_engine(model, e) for e in engines), key=_order)
    return ModelEligibility(modelId=model.id, level=level_of(verdicts), engines=verdicts)
