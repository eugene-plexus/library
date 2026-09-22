"""The starter set: a handful of models, and which one this machine takes.

A first install stalls at "which of the four hundred thousand models on
that site do I want". Discovery, fit scoring and the quant table all
answer *once you have a candidate*; nothing answered the question before
that, and a person with no candidate cannot use a search box.

## Why the list is data and never code

No model name appears in this component's source. The list ships as
`starter_models.yaml` inside the wheel, `starterModelsFile` points at
another one, and an empty list is a valid answer meaning "we have no
recommendation" -- which is what a client must render as *find a model*
rather than as a bug. That is the shape decision #4 was taken on: a
default recommendation in a field that moves monthly is only acceptable
if replacing it is a data edit and if something keeps checking it.
`starter_review.py` is the something.

## Why it makes no upstream call

Every number a fit needs -- the recommended file's size, the parameter
count, the layer and attention counts, the trained context -- was
measured once by the review that produced the list and is carried in it.
So this answers with the catalogue disabled, the hub down, or no
internet at all. Downloading the model still needs the hub; *choosing*
it does not, and the screen where a person decides should not be the
screen that fails first.

It is also why the list carries `reviewed`. The failure mode here is not
an error, it is silence: a list nobody has looked at in a year still
answers, confidently, with last year's models. So the date travels with
the recommendation, `reviewedDaysAgo` is computed here so no client has
to, and the release checklist refuses a review older than thirty days.

## The recommendation

The largest entry that fits **entirely** in GPU memory at the scored
context -- `catalogue.recommend()`'s rule, one level up: partial offload
is a decision someone should make knowingly rather than inherit from a
default, and the first model a person ever runs is the worst possible
place to hand them one that pages to host memory at three tokens a
second.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import fit as fit_mod
from ._generated.models import (
    AlreadyOwned,
    KvCacheType,
    MatchedOn,
    MemoryBudget,
    ModelStatus,
    StarterModel,
    StarterRecommendation,
    StarterSet,
    StarterSetSource,
)
from .store import StateStore

log = logging.getLogger(__name__)

PACKAGED_FILE = Path(__file__).parent / "starter_models.yaml"

STALE_AFTER_DAYS = 30
"""What the release checklist calls stale. Repeated here rather than
imported from the review, because this module is the one a person's
browser reaches and the number has to be visible from the answer."""


@dataclass(frozen=True)
class StarterEntry:
    """One row of the file, before any machine is involved.

    A typed carrier so a renamed key in the YAML fails where it is read
    rather than becoming a silently absent field three layers later.
    """

    size_class: str
    base_model: str
    repo: str
    file: str
    label: str
    size_bytes: int
    why: str
    parameters: int | None = None
    architecture: str | None = None
    context_length: int | None = None
    license: str | None = None
    downloads_30d: int | None = None
    shape: fit_mod.ModelShape = field(default_factory=fit_mod.ModelShape)


@dataclass(frozen=True)
class StarterList:
    reviewed: date
    engine: str | None
    entries: list[StarterEntry]
    source: StarterSetSource
    notes: list[str]


def load(path: str | None = None) -> StarterList:
    """Read the list, from `starterModelsFile` or from the wheel.

    Never raises. A file an operator pointed at and then broke must
    degrade to "no recommendation, and here is why" -- `degraded-mode-
    required` applied to a data file: the Discover screen still has a
    search box, and a traceback on the way to it would take that away
    too.
    """
    notes: list[str] = []
    source = StarterSetSource.shipped
    target = PACKAGED_FILE
    if path:
        source = StarterSetSource.configured
        target = Path(path)

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        notes.append(f"No starter list at {target}. Nothing is recommended; search is the way in.")
        return StarterList(_epoch(), None, [], source, notes)
    except Exception as exc:
        notes.append(f"The starter list at {target} could not be read ({exc}).")
        return StarterList(_epoch(), None, [], source, notes)

    if not isinstance(raw, dict):
        notes.append(f"The starter list at {target} is not a mapping.")
        return StarterList(_epoch(), None, [], source, notes)

    reviewed = _as_date(raw.get("reviewed"))
    if reviewed is None:
        notes.append(
            f"The starter list at {target} carries no `reviewed:` date, so how old it is "
            "cannot be said. Treating it as unreviewed."
        )
        reviewed = _epoch()

    engine = raw.get("engine") if isinstance(raw.get("engine"), str) else None
    entries: list[StarterEntry] = []
    for index, item in enumerate(raw.get("classes") or []):
        entry = _entry(item, index=index, notes=notes)
        if entry is not None:
            entries.append(entry)

    return StarterList(reviewed, engine, entries, source, notes)


def _entry(item: Any, *, index: int, notes: list[str]) -> StarterEntry | None:
    if not isinstance(item, dict):
        notes.append(f"Entry {index} of the starter list is not a mapping; skipped.")
        return None
    rec = item.get("recommended")
    rec = rec if isinstance(rec, dict) else {}
    missing = [
        key
        for key, value in (
            ("class", item.get("class")),
            ("baseModel", item.get("baseModel")),
            ("repo", item.get("repo")),
            ("recommended.file", rec.get("file")),
            ("recommended.sizeBytes", rec.get("sizeBytes")),
        )
        if not value
    ]
    if missing:
        notes.append(f"Entry {index} of the starter list is missing {', '.join(missing)}; skipped.")
        return None

    raw_evidence = item.get("evidence")
    evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
    raw_shape = item.get("shape")
    shape_raw: dict[str, Any] = raw_shape if isinstance(raw_shape, dict) else {}
    return StarterEntry(
        size_class=str(item["class"]),
        base_model=str(item["baseModel"]),
        repo=str(item["repo"]),
        file=str(rec["file"]),
        label=str(rec.get("label") or PurePosixPath(str(rec["file"])).stem),
        size_bytes=int(rec["sizeBytes"]),
        why=str(item.get("why") or ""),
        parameters=_as_int(item.get("parameters")),
        architecture=_str_or_none(item.get("architecture")),
        context_length=_as_int(item.get("contextLength")),
        license=_str_or_none(item.get("license")),
        downloads_30d=_as_int(evidence.get("downloads30d")),
        shape=fit_mod.ModelShape(
            block_count=_as_int(shape_raw.get("blockCount")),
            attention_layers=_as_int(shape_raw.get("attentionLayers")),
            head_count_kv=_as_int(shape_raw.get("headCountKv")),
            key_length=_as_int(shape_raw.get("keyLength")),
            value_length=_as_int(shape_raw.get("valueLength")),
            embedding_length=_as_int(shape_raw.get("embeddingLength")),
            head_count=_as_int(shape_raw.get("headCount")),
            context_length=_as_int(item.get("contextLength")),
            parameters=_as_int(item.get("parameters")),
            layers=fit_mod.decode_layers(shape_raw.get("layerRuns")),
        ),
    )


def build(
    starter: StarterList,
    *,
    budget: MemoryBudget,
    context_length: int,
    kv_cache_type: KvCacheType,
    store: StateStore | None = None,
    today: date | None = None,
) -> StarterSet:
    """Score every entry against one machine, and pick one.

    Pure but for the store lookup, so the states a 32 GB card, a laptop
    and an empty list each produce are assertable without a process.
    """
    models = [
        _score(entry, budget=budget, context_length=context_length, kv=kv_cache_type, store=store)
        for entry in starter.entries
    ]
    recommended = recommend(models, context_length=context_length, budget=budget)

    days = ((today or datetime.now(tz=UTC).date()) - starter.reviewed).days
    notes = list(starter.notes)
    if starter.entries and days > STALE_AFTER_DAYS:
        # Said out loud rather than logged. The person reading a
        # recommendation is the one who can judge whether a
        # three-month-old list still describes the field.
        notes.append(
            f"This list was last reviewed {days} days ago, which is past the "
            f"{STALE_AFTER_DAYS}-day window a release is allowed. It may name models that "
            "have been superseded; Discover has whatever is current."
        )

    return StarterSet(
        reviewed=starter.reviewed,
        reviewedDaysAgo=days,
        engine=starter.engine,
        source=starter.source,
        models=models,
        recommended=recommended,
        notes=notes,
    )


def _score(
    entry: StarterEntry,
    *,
    budget: MemoryBudget,
    context_length: int,
    kv: KvCacheType,
    store: StateStore | None,
) -> StarterModel:
    fit = fit_mod.compute(
        weights_bytes=entry.size_bytes,
        budget=budget,
        context_length=context_length,
        shape=entry.shape,
        kv_cache_type=kv,
    )
    return StarterModel(
        sizeClass=entry.size_class,
        baseModel=entry.base_model,
        repo=entry.repo,
        file=entry.file,
        label=entry.label,
        sizeBytes=entry.size_bytes,
        parameters=entry.parameters,
        architecture=entry.architecture,
        contextLength=entry.context_length,
        license=entry.license,
        why=entry.why,
        downloads30d=entry.downloads_30d,
        fit=fit,
        maxContextLength=fit_mod.max_context_that_fits(
            weights_bytes=entry.size_bytes,
            budget=budget,
            shape=entry.shape,
            kv_cache_type=kv,
        ),
        alreadyOwned=_owned(entry, store) if store is not None else None,
    )


def has_no_accelerator(budget: fit_mod.MemoryBudget) -> bool:
    """Is there genuinely no GPU on this machine?

    **Not `vramTotalBytes == 0`** (review §6.2 #28). That is also true of
    a machine whose card exists and whose size no vendor tool would
    state, and this function decides whether to *invert the whole
    recommendation* — so an Intel Arc owner opening Discover for the
    first time was handed the smallest model in the set, with a sentence
    explaining that their machine has no graphics card, on a machine
    built around one.

    `gpuCount` is the field that answers the question asked.
    """
    return (budget.gpuCount or 0) == 0 and not budget.unifiedMemory


def recommend(
    models: list[StarterModel], *, context_length: int, budget: MemoryBudget
) -> StarterRecommendation | None:
    """The largest that fits entirely, with the numbers said out loud.

    `None` only when the list is empty -- when nothing fits, the answer
    is still a recommendation object, carrying the sentence that says so
    and what to do instead. A silent absence would read as "we have no
    opinion", which is the opposite of what a machine too small for
    every entry needs to hear.
    """
    if not models:
        return None

    # **A machine we could not measure gets a sentence, not silence**
    # (review §6.2 #28, found by the live run). Every verdict here is
    # `unknown`, so nothing "fits" and the branch below would have said
    # *none of these fits entirely in 0 B* -- a number that is not a
    # number, about a card we can see and cannot size.
    if fit_mod.card_of_unknown_size(budget) and not budget.unifiedMemory:
        cards = budget.gpuCount or 1
        return StarterRecommendation(
            reason=(
                f"This machine has {cards} graphics card"
                f"{'' if cards == 1 else 's'} and nothing here would say how much memory "
                "it has, so none of these can be scored against it. Install the card's "
                "own tool (nvidia-smi, rocm-smi or xpu-smi) and reload, or pick from "
                "Discover using the size you know your card to be."
            )
        )

    fitting = [m for m in models if m.fit and m.fit.verdict.value == "fits"]
    if not fitting:
        smallest = min(models, key=lambda m: m.sizeBytes)
        free = fit_mod.format_bytes(budget.vramFreeBytes or budget.ramAvailableBytes)
        return StarterRecommendation(
            reason=(
                f"None of these fits entirely in {free} at {context_length:,} tokens of "
                f"context. The smallest is {smallest.baseModel} at "
                f"{fit_mod.format_bytes(smallest.sizeBytes)}, which needs "
                f"{fit_mod.format_bytes(smallest.fit.requiredBytes if smallest.fit else None)}. "
                "A shorter context, or a smaller model from Discover, is the way in."
            )
        )

    # **On a machine with no accelerator the rule inverts, and it has to.**
    # "Fits" is a question about memory; on a CPU the question a person
    # actually has is about speed, and the two have opposite answers.
    # 16 GB of weights fits comfortably in 32 GB of host memory and
    # generates at a couple of tokens a second -- which, as somebody's
    # first sentence out of this software, is indistinguishable from
    # broken. The guidance record already has this mistake once, in the
    # other direction: a library measuring a NAS recommended a 57 GB
    # BF16 download for CPU inference to an operator holding a 5090.
    if has_no_accelerator(budget):
        best = min(fitting, key=lambda m: (m.sizeBytes, m.parameters or 0))
        assert best.fit is not None
        return StarterRecommendation(
            sizeClass=best.sizeClass,
            reason=(
                f"{best.baseModel} is the smallest of these, and on a machine with no "
                f"graphics card that is the one to start with: {best.label}, "
                f"{fit_mod.format_bytes(best.sizeBytes)}. Bigger ones fit in memory here — "
                "they just generate a few words a second, because every token is computed "
                "on the processor. Discover has the larger ones if you want to trade speed "
                "for quality."
            ),
        )

    best = max(fitting, key=lambda m: (m.sizeBytes, m.parameters or 0))
    assert best.fit is not None
    where = "GPU memory" if (budget.vramTotalBytes or 0) > 0 else "memory"
    reason = (
        f"{best.baseModel} is the largest of these that runs entirely in {where} with room "
        f"for {context_length:,} tokens of context: {best.label}, "
        f"{fit_mod.format_bytes(best.sizeBytes)} of weights plus "
        f"{fit_mod.format_bytes(best.fit.kvCacheBytes)} of KV cache, "
        f"{fit_mod.format_bytes(best.fit.requiredBytes)} in total against "
        f"{fit_mod.format_bytes(budget.vramFreeBytes or budget.ramAvailableBytes)} free."
    )
    if best.maxContextLength:
        reason += f" It fits up to {best.maxContextLength:,} tokens here."
    return StarterRecommendation(sizeClass=best.sizeClass, reason=reason)


def _owned(entry: StarterEntry, store: StateStore) -> AlreadyOwned | None:
    """Already on disk? Same rule as a catalogue candidate.

    Name and size, never a hash -- the library does not read 40 GB to
    answer a question about a download it is trying to avoid. Duplicated
    from `catalogue._owned` rather than shared because that one takes a
    `_Group` built from an upstream file list, and this has a row of
    YAML; the two-line join is cheaper than the shared abstraction.
    """
    wanted = PurePosixPath(entry.file).name.lower()
    for model in store.list_models():
        if model.status is not ModelStatus.present:
            continue
        if PurePosixPath(model.path.replace("\\", "/")).name.lower() != wanted:
            continue
        if model.sizeBytes and abs(model.sizeBytes - entry.size_bytes) > 1024 * 1024:
            continue
        return AlreadyOwned(modelId=model.id, path=model.path, matchedOn=MatchedOn.name_and_size)
    return None


def _epoch() -> date:
    """The date an unreviewed list carries.

    Deliberately ancient rather than today: an unreadable or undated
    file must score as stale, so the staleness note fires instead of a
    missing date reading as a fresh one.
    """
    return date(1970, 1, 1)


def _as_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
