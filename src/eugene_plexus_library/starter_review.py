"""The monthly review that keeps the starter set honest.

Decision #4 -- ship a recommended first model -- was taken on one
condition, in Troy's words:

> This is something that will quickly grow stale as models continue to
> improve. If we make a default, this project needs an automated process
> before new releases to review the state of local inference and make a
> recommendation about whether we keep or replace our default selection.

This module is that process. It recommends; a person decides; nothing is
ever applied automatically. The release checklist refuses a review older
than thirty days or one carrying an unresolved verdict, so the gate is a
failing check rather than a note somebody is supposed to read.

## What it ranks on, and what it refuses to rank on

**30-day downloads, and nothing else.** That is the community's
judgement rather than ours, it is a number anyone can check, and it is
the same rule `quants.py` follows in refusing to say which of `IQ2_S`
and `Q2_K` is *better*. There is no benchmark column, no leaderboard,
no invented score. The report's header says so, because a reader who
assumes a quality ranking is reading a different document from the one
written.

## The five steps

1. **Rank.** One listing call per page, ranked by 30-day downloads,
   with `expand[]` carrying `gguf` and `cardData`. Aggregate the dozen
   quant mirrors of one model by `base_model` -- official, `unsloth`,
   `bartowski`, `ggml-org` and `lmstudio-community` are one candidate,
   not five -- and bucket by total parameter count.
2. **Compare** each class's current entry with the ranking.
3. **Prove it runs.** The pinned llama.cpp build's own architecture
   list, fetched from its source at that tag, must contain the
   candidate's `general.architecture`. A model the engine we install
   cannot load is the one recommendation worse than none.
4. **Verdict**, with two-month hysteresis so a launch-week spike is not
   a replacement.
5. **Report**, plus a proposed file. Never applied.

## What measuring the live hub corrected, against the design's §6.5

Four things, each of which would have produced a wrong list:

- **`pipeline_tag` cannot be a filter.** Adding `filter=text-generation`
  dropped the *second* most-downloaded GGUF repo on the hub, because
  that field is simply absent on many repos. The discriminator that
  works is the presence of a **chat template** in the repo's own `gguf`
  block, which also excludes the ASR, TTS, embedding and projector repos
  that rank in the top twenty.
- **`base_model` is sometimes a list of sixty.** One repo bundles
  every model it supports. A multi-valued `base_model` is not an
  aggregation key, so those entries are counted as their own candidate
  and flagged rather than merged into sixty families.
- **`gguf.total` is read off one file in the repo and is sometimes the
  wrong file.** A 27B repo reported 0.5B because the hub read its
  vision projector. So a class's parameter count is the **mode** of its
  contributors, never any single reading.
- **Capitalisation splits a family.** `google/gemma-4-E4B-it` and
  `google/gemma-4-e4b-it` are the same model named two ways by two
  publishers, and unnormalised they are two candidates with half the
  downloads each. Keys are case-folded; the display name is the most
  common spelling.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import catalogue as catalogue_mod
from . import fit as fit_mod
from . import preflight as preflight_mod
from . import starter as starter_mod
from .hub import HubClient, HubError

log = logging.getLogger(__name__)

# --- policy, all of it in one place ------------------------------------

SIZE_CLASSES: list[tuple[str, int, int]] = [
    # (name, min parameters inclusive, max exclusive). Total parameters,
    # not active: every expert of a mixture-of-experts model is resident,
    # so 30B-A3B occupies a 30B's worth of memory, which is the only
    # thing this bucket is used for.
    ("4B", 2_500_000_000, 6_000_000_000),
    ("8B", 6_000_000_000, 11_000_000_000),
    ("14B", 11_000_000_000, 20_000_000_000),
    ("30B", 20_000_000_000, 40_000_000_000),
    ("70B", 40_000_000_000, 90_000_000_000),
]

KNOWN_PUBLISHERS = frozenset(
    {
        # Whose quant repo we are willing to hand a stranger. The list
        # grows by a human adding a line and never by the tool: it is
        # the one curation here, and an unknown publisher at the top of
        # a class surfaces as REVIEW rather than being silently dropped.
        "unsloth",
        "bartowski",
        "ggml-org",
        "lmstudio-community",
        "google",
        "qwen",
        "meta-llama",
        "mistralai",
        "microsoft",
        "nvidia",
        "ibm-granite",
        "allenai",
        "huggingfacetb",
        "liquidai",
        "deepseek-ai",
    }
)

LICENCE_ALLOWLIST = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "gemma",
        "llama3.3",
        "llama4",
        "qwen",
    }
)
"""Permissive, or a published model licence a person can read. Anything
outside it is REVIEW rather than a rejection -- a licence we have not
seen before is a thing for a human to look at once, not a verdict."""

SPECIALISED = (
    # A first model is a general-purpose chat model. These are all
    # perfectly good models and none of them is the right answer to
    # "I have never run one of these before".
    "coder",
    "code-",
    "-ocr",
    "ocr-",
    "asr",
    "tts",
    "embed",
    "rerank",
    "guard",
    "math",
    "-vl-",
    "medical",
    "distill",
)

DE_ALIGNED = (
    # Not a judgement about whether these should exist. A *default*
    # recommendation is the one place this project's own choice shows,
    # and shipping an abliterated model to someone who did not ask for
    # one is a choice we would be making for them.
    "abliterated",
    "uncensored",
    "heretic",
    "defiant",
    "unaligned",
    "nsfw",
)

TARGET_BITS_PER_WEIGHT = 4.8
"""Which quant ships per class -- roughly `Q4_K_M`, the community's
default trade. **One axis, not two:** the size class is what adapts to
the machine, and offering a stranger a choice of eleven quants at the
same moment is exactly what the plan's §0.5 measured as the wall. An
expert changes it in Discover, one click away."""

HYSTERESIS_REVIEWS = 2
"""A different base model must hold first place for two consecutive
reviews before it is proposed. A launch-week spike is not a trend, and
this is the only defence against one -- which is also why the review
runs monthly rather than on release day: on release day there is no
second month of evidence to have."""

LLAMA_ARCH_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/src/llama-arch.cpp"


# --- the ranking -------------------------------------------------------


@dataclass
class Candidate:
    """One base model, with every repo that mirrors or derives from it."""

    key: str
    names: collections.Counter[str] = field(default_factory=collections.Counter)
    downloads: int = 0
    repos: list[tuple[str, int]] = field(default_factory=list)
    parameters: collections.Counter[int] = field(default_factory=collections.Counter)
    architectures: collections.Counter[str] = field(default_factory=collections.Counter)
    licences: collections.Counter[str] = field(default_factory=collections.Counter)
    contexts: collections.Counter[int] = field(default_factory=collections.Counter)
    flags: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return self.names.most_common(1)[0][0] if self.names else self.key

    @property
    def parameter_count(self) -> int | None:
        """The mode, never a single reading.

        The hub's repo-level `gguf.total` comes off one file in the repo
        and is sometimes the wrong file: a measured 27B repo reported
        0.5B because the projector was read. Contributors agreeing is
        the only signal available without downloading something.
        """
        return self.parameters.most_common(1)[0][0] if self.parameters else None

    @property
    def size_class(self) -> str | None:
        total = self.parameter_count
        if total is None:
            return None
        for name, low, high in SIZE_CLASSES:
            if low <= total < high:
                return name
        return None

    def known_repos(self) -> list[str]:
        """Repos from publishers on the list, most-downloaded first.

        A *list* and not a single best, because which repo to ship is
        not decided by downloads alone: publishers differ in what they
        publish. Measured on the first run -- `ggml-org`'s gemma repo
        holds `Q4_0`, `Q8_0` and `BF16` and no K-quant at all, so
        "the most-downloaded known publisher" chose a legacy layout for
        a model two other publishers ship a full K-quant ladder for.
        The repo and the quant have to be chosen together.
        """
        return [
            repo
            for repo, _ in sorted(self.repos, key=lambda r: -r[1])
            if repo.split("/")[0].lower() in KNOWN_PUBLISHERS
        ]

    def best_repo(self) -> str | None:
        return next(iter(self.known_repos()), None)


def _usable(entry: dict[str, Any]) -> tuple[bool, str | None]:
    """Is this repo a chat model anyone could launch? With the reason."""
    gguf = entry.get("gguf") or {}
    if not isinstance(gguf, dict) or not gguf:
        return False, "no GGUF metadata"
    if not gguf.get("chat_template"):
        # The discriminator that works. `pipeline_tag` does not: adding
        # `filter=text-generation` to the listing call dropped the
        # second most-downloaded GGUF repo on the hub, because many
        # repos simply do not carry the field.
        return False, "no chat template (a base model, or not a chat model at all)"
    gated = entry.get("gated")
    if gated not in (False, None, "false"):
        return False, f"gated ({gated})"
    if entry.get("private"):
        return False, "private"
    return True, None


def _base_model(entry: dict[str, Any]) -> tuple[str | None, str | None]:
    """`(base model, flag)`. The repo's own id when it declares none."""
    card = entry.get("cardData") or {}
    value = card.get("base_model") if isinstance(card, dict) else None
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        elif len(value) > 1:
            # One measured repo lists sixty. That is a bundle, not a
            # derivative, and merging it into sixty families would move
            # its downloads into every one of them.
            return entry.get("id"), "declares many base models; counted as itself"
        else:
            value = None
    if isinstance(value, str) and "/" in value:
        return value, None
    return entry.get("id"), "no base_model on the card; counted as itself"


def rank(entries: list[dict[str, Any]]) -> tuple[dict[str, Candidate], list[dict[str, str]]]:
    """Aggregate a listing into candidates, and say what was dropped."""
    candidates: dict[str, Candidate] = {}
    dropped: list[dict[str, str]] = []

    for entry in entries:
        repo = str(entry.get("id") or "")
        usable, why = _usable(entry)
        if not usable:
            dropped.append({"repo": repo, "why": why or "unusable"})
            continue

        base, flag = _base_model(entry)
        if not base:
            dropped.append({"repo": repo, "why": "no identifiable model"})
            continue

        # Case-folded, because `gemma-4-E4B-it` and `gemma-4-e4b-it` are
        # one model spelled two ways by two publishers and unnormalised
        # they are two candidates with half the downloads each.
        key = base.lower()
        candidate = candidates.setdefault(key, Candidate(key=key))
        candidate.names[base] += 1
        downloads = int(entry.get("downloads") or 0)
        candidate.downloads += downloads
        candidate.repos.append((repo, downloads))
        if flag:
            candidate.flags.add(flag)

        gguf = entry.get("gguf") or {}
        if isinstance(gguf.get("total"), int):
            candidate.parameters[gguf["total"]] += 1
        if isinstance(gguf.get("architecture"), str):
            candidate.architectures[gguf["architecture"]] += 1
        if isinstance(gguf.get("context_length"), int):
            candidate.contexts[gguf["context_length"]] += 1
        card = entry.get("cardData") or {}
        if isinstance(card, dict) and isinstance(card.get("license"), str):
            candidate.licences[card["license"]] += 1

    for candidate in candidates.values():
        lowered = candidate.name.lower()
        if any(token in lowered for token in SPECIALISED):
            candidate.flags.add("specialised, not a general-purpose chat model")
        if any(token in lowered for token in DE_ALIGNED):
            candidate.flags.add("a de-aligned derivative; never a default")
        if candidate.parameter_count is None:
            candidate.flags.add("no agreed parameter count")
        if len(candidate.repos) < 2:
            # One repo is a publisher, not a consensus. The ranking is a
            # popularity measure and a single mirror carries none of the
            # agreement that makes popularity worth reading -- so it is
            # flagged into the report for a human rather than proposed.
            candidate.flags.add("only one repo mirrors this model; thin evidence")
        if candidate.best_repo() is None:
            candidate.flags.add("no repo from a publisher on the known list")
        licence = candidate.licences.most_common(1)
        if not licence:
            candidate.flags.add("no licence on the card")
        elif licence[0][0].lower() not in LICENCE_ALLOWLIST:
            candidate.flags.add(f"licence `{licence[0][0]}` is not on the allowlist")

    return candidates, dropped


def eligible(candidates: dict[str, Candidate]) -> dict[str, list[Candidate]]:
    """Rankable candidates per size class, best first.

    A flag never removes a candidate from the *report* -- the point of a
    flag is that a human sees it -- but it does remove it from the
    ranking that produces a proposal.
    """
    by_class: dict[str, list[Candidate]] = {name: [] for name, _, _ in SIZE_CLASSES}
    for candidate in candidates.values():
        size_class = candidate.size_class
        if size_class is None or candidate.flags:
            continue
        by_class[size_class].append(candidate)
    for rows in by_class.values():
        rows.sort(key=lambda c: -c.downloads)
    return by_class


# --- step 3: prove the pinned engine can load it -----------------------


async def engine_architectures(tag: str) -> set[str] | None:
    """Every architecture the pinned llama.cpp build knows, from source.

    `llama-arch.cpp` maps each `LLM_ARCH_*` to the string a GGUF's
    `general.architecture` carries, which is the exact comparison that
    decides whether the engine we install can load a file. `None` when
    the fetch failed, which is REVIEW and never a silent pass.
    """
    url = LLAMA_ARCH_URL.format(tag=tag)
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code >= 400:
            log.warning("could not read %s (HTTP %s)", url, response.status_code)
            return None
    except httpx.HTTPError as exc:
        log.warning("could not read %s (%s)", url, exc)
        return None
    return parse_architectures(response.text)


def parse_architectures(source: str) -> set[str]:
    """`{ LLM_ARCH_QWEN3, "qwen3" },` -> `qwen3`."""
    return set(re.findall(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}', source))


# --- step 4: verdicts --------------------------------------------------


@dataclass
class ClassVerdict:
    size_class: str
    verdict: str
    detail: str
    current: dict[str, Any] | None
    leader: Candidate | None
    runners_up: list[Candidate]
    check: str


def verdict_for(
    size_class: str,
    *,
    current: dict[str, Any] | None,
    ranked: list[Candidate],
    architectures: set[str] | None,
    streak: dict[str, int],
) -> ClassVerdict:
    """KEEP, REPLACE or REVIEW for one class. A failed lookup is REVIEW."""
    leader = ranked[0] if ranked else None
    runners = ranked[1:4]
    check = "architecture" if architectures is not None else "none (engine list unavailable)"

    if architectures is None:
        return ClassVerdict(
            size_class,
            "REVIEW",
            "The pinned engine's architecture list could not be read, so no entry in this "
            "class was proved loadable. A failed lookup is never a silent KEEP.",
            current,
            leader,
            runners,
            check,
        )

    if leader is None:
        return ClassVerdict(
            size_class,
            "REVIEW" if current else "KEEP",
            (
                "No unflagged candidate ranked in this class. Either the class is genuinely "
                "empty at this download depth, or everything in it is flagged -- the table "
                "below says which."
                if current
                else "Nothing ranks here and nothing is shipped here. No action."
            ),
            current,
            None,
            runners,
            check,
        )

    leader_arch = leader.architectures.most_common(1)
    if leader_arch and leader_arch[0][0] not in architectures:
        return ClassVerdict(
            size_class,
            "REVIEW",
            f"The leader `{leader.name}` is architecture `{leader_arch[0][0]}`, which the "
            "pinned engine does not list. Either the engine pin moves or this class keeps "
            "what it has.",
            current,
            leader,
            runners,
            check,
        )

    if current is None:
        return ClassVerdict(
            size_class,
            "REPLACE",
            f"Nothing is shipped in this class and `{leader.name}` leads it with "
            f"{leader.downloads:,} downloads in 30 days.",
            None,
            leader,
            runners,
            check,
        )

    current_base = str(current.get("baseModel") or "").lower()
    top_three = [c.key for c in ranked[:3]]

    if current_base == leader.key:
        return ClassVerdict(
            size_class,
            "KEEP",
            f"`{current.get('baseModel')}` still leads its class with {leader.downloads:,} "
            "downloads in 30 days.",
            current,
            leader,
            runners,
            check,
        )

    if current_base not in top_three:
        return ClassVerdict(
            size_class,
            "REPLACE",
            f"`{current.get('baseModel')}` has fallen out of the top three of its class. "
            f"`{leader.name}` leads with {leader.downloads:,} downloads in 30 days.",
            current,
            leader,
            runners,
            check,
        )

    held = streak.get(leader.key, 0) + 1
    if held >= HYSTERESIS_REVIEWS:
        return ClassVerdict(
            size_class,
            "REPLACE",
            f"`{leader.name}` has led this class for {held} consecutive reviews "
            f"({leader.downloads:,} downloads in 30 days) while "
            f"`{current.get('baseModel')}` is still in the top three.",
            current,
            leader,
            runners,
            check,
        )

    second = ranked[1] if len(ranked) > 1 else None
    close = (
        second is not None
        and leader.downloads > 0
        and (leader.downloads - second.downloads) / leader.downloads < 0.20
    )
    return ClassVerdict(
        size_class,
        "REVIEW",
        f"`{leader.name}` leads for the first time ({held} of {HYSTERESIS_REVIEWS} reviews "
        f"needed)."
        + (" The top two are within 20% of each other." if close else "")
        + f" `{current.get('baseModel')}` is still in the top three, so nothing is proposed "
        "yet.",
        current,
        leader,
        runners,
        check,
    )


# --- building a proposed entry ----------------------------------------


async def build_entry(
    client: HubClient, candidate: Candidate, *, size_class: str
) -> dict[str, Any] | None:
    """The YAML row for one candidate: repo, file, size, and the shape.

    The shape costs one ranged read of about 11 MB, and it is what makes
    `GET /v1/catalogue/starter` answer with `basis: metadata` and with no
    upstream call at all. Paying it once a month here is the whole
    reason that endpoint works offline.

    It groups the repo's files with `catalogue`'s own grouping rather
    than through `build_model`: that one wants a `StateStore` to answer
    "already on disk", which is a question about somebody's machine and
    this is a question about the file.
    """
    picked = await _pick_repo_and_quant(client, candidate)
    if picked is None:
        return None
    repo, chosen = picked

    entry: dict[str, Any] = {
        "class": size_class,
        "baseModel": candidate.name,
        "repo": repo,
        "why": (
            f"most-downloaded well-known instruct GGUF in its size class over the last 30 "
            f"days ({candidate.downloads:,} downloads across {len(candidate.repos)} "
            f"{'repo' if len(candidate.repos) == 1 else 'repos'})"
        ),
        "license": _mode(candidate.licences),
        "parameters": candidate.parameter_count,
        "architecture": _mode(candidate.architectures),
        "contextLength": _mode(candidate.contexts),
        "evidence": {
            "downloads30d": candidate.downloads,
            "repos": [r for r, _ in sorted(candidate.repos, key=lambda x: -x[1])[:6]],
        },
        "recommended": {
            "file": chosen.files[0].path,
            "label": chosen.label,
            "sizeBytes": chosen.size,
            "bitsPerWeight": fit_mod.bits_per_weight(chosen.size, candidate.parameter_count),
        },
    }

    # The shape, read off the file itself. Failing this is not fatal --
    # the entry still works, with `basis: estimate` -- but it is worth a
    # line in the report, because a starter fit is the one fit nobody
    # will ever preflight by hand.
    try:
        meta, _read = await preflight_mod.read_gguf_header(
            client, repo=repo, revision="main", path=chosen.files[0].path
        )
    except HubError as exc:
        log.warning("%s: could not read %s (%s)", repo, chosen.files[0].path, exc)
        entry["shapeUnavailable"] = str(exc)
        return entry

    if meta.is_embedding:
        log.warning("%s: %s is an embedding model; not shipping it", repo, chosen.label)
        return None

    shape = preflight_mod.shape_from_gguf(meta)
    entry["architecture"] = meta.architecture or entry["architecture"]
    entry["contextLength"] = meta.context_length or entry["contextLength"]
    entry["shape"] = {
        k: v
        for k, v in {
            "blockCount": shape.block_count,
            "attentionLayers": shape.attention_layers,
            "headCountKv": shape.head_count_kv,
            "keyLength": shape.key_length,
            "valueLength": shape.value_length,
            "embeddingLength": shape.embedding_length,
            "headCount": shape.head_count,
            # Run-length `[count, heads, key, value, window]`. Present
            # only for the models whose layers differ from each other,
            # and load-bearing for exactly those: without it a current
            # 12B's KV cache reads 43x too large.
            "layerRuns": fit_mod.encode_layers(shape.layers),
        }.items()
        if v is not None
    }
    return entry


def _mode(counter: collections.Counter[Any]) -> Any:
    """The most common value, or None. A helper because `most_common(1)`
    on an empty counter is an empty list and indexing it raises."""
    top = counter.most_common(1)
    return top[0][0] if top else None


REPOS_TRIED = 3
"""How many known-publisher repos to open before settling. Bounded
because each one is two upstream calls and the answer converges fast:
the first repo that ships a preferred K-quant wins outright."""


async def _pick_repo_and_quant(client: HubClient, candidate: Candidate) -> tuple[str, Any] | None:
    """The repo and the file together, best preferred-family first.

    Stops at the first repo offering a rank-0 quant; otherwise takes the
    best of whatever the tried repos had. A repo that cannot be read is
    skipped rather than fatal -- there are others.
    """
    best: tuple[int, str, Any] | None = None
    for repo in candidate.known_repos()[:REPOS_TRIED]:
        try:
            info, files = await asyncio.gather(client.repo_info(repo), client.tree(repo))
        except HubError as exc:
            log.warning("%s: could not read the repo (%s)", repo, exc)
            continue
        groups, _projectors, _other = catalogue_mod._gguf_groups(files)
        chosen = _choose_quant(groups, parameters=info.parameters or candidate.parameter_count)
        if chosen is None:
            continue
        rank = quant_rank(chosen.label)
        if best is None or rank < best[0]:
            best = (rank, repo, chosen)
        if rank == 0:
            break
    if best is None:
        log.warning("%s: no known-publisher repo yielded a GGUF candidate", candidate.name)
        return None
    return best[1], best[2]


def quant_rank(label: str) -> int:
    """Which quant *family* a starter should prefer, lower first.

    Width alone is not enough, and the first run of this review proved
    it: at a 4.8-bit target it chose `IQ4_XS` for one class and `Q4_0`
    for another, both within 0.1 bits of `Q4_K_M` and neither the file a
    person should be handed first. An I-quant needs an importance matrix
    and is materially slower on CPU and on several GPU backends; `Q4_0`
    is a legacy layout kept for compatibility. The K-quants are what
    every current guide means by "the 4-bit one".

    So: family first, then width. The starter is the one file someone
    runs before they know that any of this is a decision.
    """
    upper = label.upper()
    if any(token in upper for token in ("Q4_K_M", "Q4_K_L", "Q4_K_XL")):
        return 0
    if "_K_" in upper or upper.endswith("_K"):
        return 1
    return 2


def _choose_quant(groups: list[Any], *, parameters: int | None) -> Any | None:
    """One quant per class: the preferred family nearest the target width.

    Bits per weight rather than a filename match for the width, because
    the naming schemes disagree (`Q4_K_M`, `UD-Q4_K_XL`, `IQ4_NL`) and
    the measured width is the one comparable number -- which is
    `fit.bits_per_weight`'s whole reason for existing. A tie goes to the
    smaller file.
    """
    if not groups:
        return None
    scored = [(g, fit_mod.bits_per_weight(g.size, parameters)) for g in groups]
    usable = [(g, bpw) for g, bpw in scored if bpw]
    if usable:
        return min(
            usable,
            key=lambda p: (quant_rank(p[0].label), abs(p[1] - TARGET_BITS_PER_WEIGHT), p[0].size),
        )[0]
    return min(groups, key=lambda g: (quant_rank(g.label), g.size))


# --- the report --------------------------------------------------------


def report(
    *,
    verdicts: list[ClassVerdict],
    dropped: list[dict[str, str]],
    new_this_month: list[Candidate],
    engine: str,
    architectures: set[str] | None,
    listed: int,
    when: date,
) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Starter model review — {when.isoformat()}")
    a("")
    a(
        "Ranking is **upstream's 30-day download count and nothing else**. There is no "
        "quality score here, invented or borrowed: which model is *better* is not a "
        "question this project answers, and downloads are the community's judgement "
        "rather than ours. Nothing in this report is applied automatically."
    )
    a("")
    a(f"- Engine checked against: **{engine}**")
    a(
        f"- Architectures the engine lists: "
        f"**{len(architectures) if architectures is not None else 'unavailable'}**"
    )
    a(f"- Repos read: **{listed}**")
    a("")

    verdict_counts = collections.Counter(v.verdict for v in verdicts)
    a(
        "## Verdicts — "
        + ", ".join(f"{count} {name}" for name, count in sorted(verdict_counts.items()))
    )
    a("")
    a("| class | verdict | current | leader | 30-day downloads | proof |")
    a("| --- | --- | --- | --- | --- | --- |")
    for v in verdicts:
        current = str((v.current or {}).get("baseModel") or "—")
        leader = v.leader.name if v.leader else "—"
        downloads = f"{v.leader.downloads:,}" if v.leader else "—"
        a(f"| {v.size_class} | **{v.verdict}** | {current} | {leader} | {downloads} | {v.check} |")
    a("")

    for v in verdicts:
        a(f"### {v.size_class} — {v.verdict}")
        a("")
        a(v.detail)
        a("")
        if v.runners_up:
            a("Runners-up:")
            a("")
            for c in v.runners_up:
                a(f"- `{c.name}` — {c.downloads:,} downloads across {len(c.repos)} repos")
            a("")

    a("## New this month")
    a("")
    a(
        "Models in the ranking that are not the leader of any class, with what stands "
        "between them and being recommendable. **This section is information for the "
        'reader, not an input to any verdict** — it is the "state of local inference" '
        "half of what this review is for."
    )
    a("")
    if not new_this_month:
        a("Nothing new above the download threshold.")
    else:
        a("| model | 30-day downloads | parameters | architecture | why it is not ranked |")
        a("| --- | --- | --- | --- | --- |")
        for c in new_this_month[:20]:
            params = f"{c.parameter_count / 1e9:.1f}B" if c.parameter_count else "—"
            arch = (c.architectures.most_common(1) or [("—", 0)])[0][0]
            flags = "; ".join(sorted(c.flags)) or "—"
            a(f"| `{c.name}` | {c.downloads:,} | {params} | {arch} | {flags} |")
    a("")

    a("## Repos the ranking dropped")
    a("")
    a(
        f"{len(dropped)} of {listed}. Listed rather than hidden, for the same reason the "
        "scanner reports what it skipped: a filter nobody can see is indistinguishable "
        "from a broken one."
    )
    a("")
    reasons = collections.Counter(d["why"].split(" (")[0] for d in dropped)
    for why, count in reasons.most_common():
        a(f"- {count} x {why}")
    a("")
    return "\n".join(lines)


# --- the run -----------------------------------------------------------


async def review(
    *,
    starter_file: Path,
    engine_tag: str,
    pages: int,
    limit: int,
    base_url: str,
    token: str | None,
) -> tuple[str, dict[str, Any], list[ClassVerdict]]:
    client = HubClient()
    client.configure(base_url=base_url, token=token, enabled=True)
    try:
        entries = await client.list_models(limit=limit, pages=pages)
        candidates, dropped = rank(entries)
        by_class = eligible(candidates)
        architectures = await engine_architectures(engine_tag)

        current_raw = _read_current(starter_file)
        streak = {
            str(item.get("baseModel", "")).lower(): int(
                (item.get("evidence") or {}).get("consecutiveReviewsAtTop") or 0
            )
            for item in current_raw.get("classes") or []
        }
        by_current = {str(item.get("class")): item for item in current_raw.get("classes") or []}

        verdicts = [
            verdict_for(
                name,
                current=by_current.get(name),
                ranked=by_class.get(name, []),
                architectures=architectures,
                streak=streak,
            )
            for name, _, _ in SIZE_CLASSES
        ]

        proposed: dict[str, Any] = {
            "reviewed": date.today().isoformat(),
            "engine": f"llama_cpp {engine_tag}",
            "classes": [],
        }
        for v in verdicts:
            if v.verdict == "REPLACE" and v.leader is not None:
                entry = await build_entry(client, v.leader, size_class=v.size_class)
                if entry is not None:
                    entry.setdefault("evidence", {})["consecutiveReviewsAtTop"] = 1
                    proposed["classes"].append(entry)
                    continue
                v.verdict = "REVIEW"
                v.detail += (
                    " The proposed replacement could not be turned into an entry — no repo "
                    "from a known publisher, or its files could not be read."
                )
            if v.current is not None:
                kept = dict(v.current)
                if v.verdict == "KEEP":
                    evidence = dict(kept.get("evidence") or {})
                    evidence["consecutiveReviewsAtTop"] = (
                        int(evidence.get("consecutiveReviewsAtTop") or 0) + 1
                    )
                    kept["evidence"] = evidence
                proposed["classes"].append(kept)

        leaders = {v.leader.key for v in verdicts if v.leader}
        new_this_month = sorted(
            (c for c in candidates.values() if c.key not in leaders and c.downloads > 0),
            key=lambda c: -c.downloads,
        )
        text = report(
            verdicts=verdicts,
            dropped=dropped,
            new_this_month=new_this_month,
            engine=f"llama_cpp {engine_tag}",
            architectures=architectures,
            listed=len(entries),
            when=datetime.now(tz=UTC).date(),
        )
        return text, proposed, verdicts
    finally:
        await client.aclose()


def _read_current(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("could not read %s (%s); treating it as empty", path, exc)
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eugene-plexus-library starter-review",
        description=(
            "Review the starter model list against the catalogue. Recommends; never "
            "applies. Exits non-zero when any class is REPLACE or REVIEW, which is what "
            "the release gate reads."
        ),
    )
    parser.add_argument(
        "--starter-file",
        type=Path,
        default=starter_mod.PACKAGED_FILE,
        help="The list to review. Defaults to the one inside the wheel.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help=(
            "The llama.cpp build tag to check architectures against, e.g. `b10948`. "
            "Defaults to the tag in the starter file's own `engine:` line."
        ),
    )
    parser.add_argument("--pages", type=int, default=4, help="Listing pages to read.")
    parser.add_argument("--limit", type=int, default=100, help="Repos per page.")
    parser.add_argument("--base-url", default="https://huggingface.co")
    parser.add_argument("--token", default=None)
    parser.add_argument("--report", type=Path, default=None, help="Write the report here.")
    parser.add_argument(
        "--proposed",
        type=Path,
        default=None,
        help=(
            "Write the proposed starter file here. Never the file under review: a review "
            "that edits what it reviews has nothing to compare against next month."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Also print the verdicts as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = args.engine
    if not engine:
        current = _read_current(args.starter_file)
        engine = str(current.get("engine") or "").split()[-1] or None
    if not engine:
        parser.error("no --engine given and the starter file names none")

    text, proposed, verdicts = asyncio.run(
        review(
            starter_file=args.starter_file,
            engine_tag=engine,
            pages=args.pages,
            limit=args.limit,
            base_url=args.base_url,
            token=args.token,
        )
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"report: {args.report}")
    else:
        print(text)

    if args.proposed:
        if args.proposed.resolve() == Path(args.starter_file).resolve():
            parser.error("--proposed must not be the file under review")
        args.proposed.parent.mkdir(parents=True, exist_ok=True)
        args.proposed.write_text(
            # `default_flow_style=None` puts the leaf sequences inline, so
            # a 48-layer run table is sixteen short rows rather than
            # eighty lines of one integer each. This file is read by a
            # person before they accept it.
            yaml.safe_dump(
                proposed, sort_keys=False, allow_unicode=True, default_flow_style=None, width=100
            ),
            encoding="utf-8",
        )
        print(f"proposed: {args.proposed}")

    if args.json:
        print(
            json.dumps(
                [
                    {"class": v.size_class, "verdict": v.verdict, "detail": v.detail}
                    for v in verdicts
                ],
                indent=2,
            )
        )

    unresolved = [v for v in verdicts if v.verdict != "KEEP"]
    for v in unresolved:
        print(f"{v.verdict}: {v.size_class} — {v.detail}", file=sys.stderr)
    return 1 if unresolved else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
