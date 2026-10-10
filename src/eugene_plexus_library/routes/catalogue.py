"""Catalogue routes: search, detail with guidance, the card, and preflight.

The repo id is a **query parameter**, not a path segment. It contains a
slash and may be one or two segments (`unsloth/Qwen3.8-27B-GGUF`, but
also `gpt2`), so `%2F` in a path would be mangled by intermediaries —
and every UI call to this component goes through the agent's proxy —
while `{owner}/{name}` as two parameters cannot address the
single-segment canonical repos that also exist.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from .. import catalogue as catalogue_mod
from .. import catalogue_sources as sources_mod
from .. import fit as fit_mod
from .. import hardware, repo_ref
from .. import preflight as preflight_mod
from .. import starter as starter_mod
from .._generated.models import (
    CatalogueCard,
    CatalogueModel,
    CataloguePreflight,
    CatalogueSearchPage,
    CatalogueSearchRequest,
    CatalogueSearchResult,
    CatalogueSort,
    CatalogueSortDirection,
    CatalogueSource,
    CatalogueSourceKind,
    CatalogueSourceStatus,
    InterpretedAs,
    KvCacheType,
    MemoryBudget,
    ModelFormat,
    Problem,
    StarterSet,
)
from ..catalogue_sources import HubClients, SourceProblem
from ..config import ConfigStore
from ..formats import safetensors
from ..hub import HubClient, HubError, gather_limited
from ..store import StateStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["catalogue"])

# Repeated across the endpoints that score a fit, and long enough that
# inlining them three times would be unreadable.
CONTEXT_DESCRIPTION = (
    "Context length to score against. Defaults to the configured "
    "`guidanceContextLength`; the KV cache grows linearly with it, so this is the "
    "number that decides which quant is recommended."
)
KV_CACHE_DESCRIPTION = (
    "KV cache element type to assume. `f16` is llama.cpp's own default; the "
    "quantized types shrink the cache term and are a launch flag, which is why "
    '"does this fit" has no answer independent of how it will be launched.'
)
VRAM_DESCRIPTION = (
    "Override the detected VRAM budget, in bytes. How a model gets scored against "
    "*another* host's memory, which is the only honest answer when the GPU is in a "
    "different building."
)
RAM_DESCRIPTION = "Override the detected host-memory budget, in bytes."
GPU_COUNT_DESCRIPTION = (
    "How many cards `vramBytes` is the combined free memory of, for a launch that "
    "spreads one model across them. Each card holds its own compute buffers, so the "
    "overhead allowance is counted once per card. Omitted, `vramBytes` is one card."
)
UNIFIED_DESCRIPTION = (
    "Score against one pool shared with host memory: an integrated GPU, Apple silicon "
    "or a GB10 on a host this library did not measure. Pass the device's own "
    "`sharedMemory`. Without it a `vramBytes` override reads as a card with memory of "
    "its own, and a partial offload is scored as spilling into the same RAM."
)


def _problem(
    status_code: int, title: str, detail: str, *, code: str | None = None
) -> HTTPException:
    slug = (code or title).replace(" ", "-").lower()
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/library#{slug}",
            title=title,
            status=status_code,
            detail=detail,
            component="library",
        ).model_dump(exclude_none=True),
    )


def _from_hub_error(exc: HubError) -> HTTPException:
    """Upstream's own failure, translated once.

    `code` is carried through because `GatedRepo` means "accept the
    licence on the model's page", which is something the operator can
    act on, and a bare 403 is not.
    """
    titles = {
        400: "Bad catalogue request",
        403: "Access restricted upstream",
        404: "Not found upstream",
        409: "Catalogue unavailable",
        429: "Upstream rate limit",
        502: "Catalogue error",
        503: "Catalogue unreachable",
    }
    return _problem(exc.status, titles.get(exc.status, "Catalogue error"), str(exc), code=exc.code)


SOURCE_DESCRIPTION = (
    "The `hf_hub` source the repo is on (`CatalogueSource.id`, a search result's "
    "`hubSource`, LS4). Absent: the first enabled hub, which is what a console older "
    "than the sources list meant."
)


def _hub(request: Request, source: str | None) -> tuple[CatalogueSource, HubClient]:
    """The hub a call names, or the default, with its client.

    Address, token and the enabled flags are read from live config on every
    call rather than captured at construction, so a `PATCH /v1/config` takes
    effect without a restart — which is the whole point of the config trio.
    """
    hubs: HubClients = request.app.state.hub_clients
    try:
        return hubs.resolve(source)
    except SourceProblem as exc:
        raise _problem(exc.status, exc.title, exc.detail, code=exc.code) from exc


def _client(request: Request, source: str | None = None) -> HubClient:
    return _hub(request, source)[1]


async def _budget(
    request: Request,
    *,
    vram: int | None = None,
    ram: int | None = None,
    unified: bool | None = None,
    gpu_count: int | None = None,
) -> MemoryBudget:
    detected = await asyncio.to_thread(hardware.detect)
    return fit_mod.budget_from_hardware(
        detected,
        vram_override=vram,
        ram_override=ram,
        unified_override=unified,
        gpu_count_override=gpu_count,
    )


@router.get("/v1/catalogue/search", response_model=CatalogueSearchPage)
async def search_catalogue(
    request: Request,
    q: str | None = Query(default=None, description="Free-text query."),
    format: ModelFormat | None = Query(
        default=None,
        description=(
            "Restrict to repos serving this format. `gguf` maps to upstream's own "
            "filter; `safetensors` is inferred from tags and is approximate, since a "
            "repo can hold both."
        ),
    ),
    author: str | None = Query(
        default=None,
        description=(
            'Publisher, e.g. `unsloth`. A first-class parameter because "the quant '
            'publisher I trust" is how people actually navigate this catalogue.'
        ),
    ),
    sort: CatalogueSort = Query(
        default=CatalogueSort.downloads,
        description="Ordering. `trending` is upstream's own trending score.",
    ),
    direction: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort direction.",
    ),
    limit: int = Query(default=25, ge=1, le=100, description="Rows per page."),
    cursor: str | None = Query(
        default=None,
        description=(
            "Opaque continuation token from a previous response's `nextCursor`. "
            "Cursor-based because upstream paginates with an opaque cursor and there "
            "is no page number to pass through."
        ),
    ),
) -> CatalogueSearchPage:
    """Search upstream. Repos, not candidates.

    No sizes and no fit verdicts here: the upstream search response
    carries filenames without sizes, so scoring a row would cost one
    extra call per row — fifty requests for one keystroke, against a
    budget of 500 per five minutes. Sizes, quant options and guidance
    are on `GET /v1/catalogue/model`.

    Debounce this client-side. Responses are cached here for a short
    window, which is not a substitute.
    """
    client = _client(request)

    # A pasted repo reference is a lookup, not a query. Upstream's
    # full-text index has never matched a URL, so before this the
    # commonest way a person arrives with a model in mind -- someone
    # linked them one -- produced an empty list and no clue.
    reference = repo_ref.parse(q)
    if reference is not None:
        resolved = await _resolve_reference(client, reference)
        if resolved is not None:
            return resolved

    try:
        results, next_cursor, _cached_at = await client.search(
            query=q,
            author=author,
            gguf_only=format is ModelFormat.gguf,
            sort=sort,
            descending=direction == "desc",
            limit=limit,
            cursor=cursor,
        )
    except HubError as exc:
        raise _from_hub_error(exc) from exc

    rows = [
        catalogue_mod.build_search_result(entry) for entry in results if isinstance(entry, dict)
    ]
    if format is ModelFormat.safetensors:
        # Upstream has no reliable safetensors filter, so this is a
        # post-filter on tags and is approximate by nature — a repo can
        # hold both formats. Named as approximate in the contract rather
        # than presented as authoritative.
        rows = [r for r in rows if ModelFormat.safetensors in (r.formats or [])]

    return CatalogueSearchPage(
        results=rows,
        nextCursor=next_cursor,
        cachedAt=None,
        interpretedAs=InterpretedAs.search,
    )


_CURSOR_VERSION = 1


def _encode_cursor(hubs: dict[str, str]) -> str:
    """This library's continuation: each hub's own cursor, by source id."""
    raw = json.dumps({"v": _CURSOR_VERSION, "hubs": hubs}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> dict[str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data: Any = json.loads(base64.urlsafe_b64decode(padded.encode()))
        hubs = data["hubs"]
        if data.get("v") != _CURSOR_VERSION or not isinstance(hubs, dict):
            raise ValueError("not this library's cursor")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in hubs.items()):
            raise ValueError("not this library's cursor")
        return hubs
    except (ValueError, KeyError, TypeError, binascii.Error, UnicodeDecodeError) as exc:
        raise _problem(
            400,
            "Not a cursor this library made",
            "`cursor` must be the `nextCursor` of an earlier answer from this endpoint. A "
            "hub's own cursor belongs on `GET /v1/catalogue/search`.",
            code="bad-cursor",
        ) from exc


def _status(source: CatalogueSource) -> CatalogueSourceStatus:
    return CatalogueSourceStatus(
        id=source.id, kind=source.kind, label=sources_mod.label_of(source), searched=False
    )


@router.post("/v1/catalogue/search", response_model=CatalogueSearchPage)
async def search_catalogue_sources(
    request: Request, body: CatalogueSearchRequest
) -> CatalogueSearchPage:
    """Search the chosen sources together (LS4, design §4.4).

    Every enabled source in `catalogueSources`, or the ones named. A hub is
    searched as `GET` searches one; an engine's list from the lists the
    caller sent (the picked node's, as it sends that node's `accepts` to the
    judge, so this calls no agent). Every result names its source, and
    `sources` says what each answered: one hub down is not an empty search.
    """
    hubs: HubClients = request.app.state.hub_clients
    sources = hubs.sources()
    if body.sources is not None:
        unknown = sorted(set(body.sources) - {s.id for s in sources})
        if unknown:
            raise _problem(
                400,
                "No such source",
                f"This library has no catalogue source {', '.join(map(repr, unknown))}. It may "
                "have been removed in the Library's settings since the page was loaded.",
                code="source-not-found",
            )
    named = set(body.sources) if body.sources is not None else None
    chosen = [s for s in sources if named is None or s.id in named]
    default = sources_mod.default_hub(sources)
    cursors = _decode_cursor(body.cursor) if body.cursor else None

    # A pasted link names one repo, on the hub at its host (or the default).
    reference = repo_ref.parse(body.q) if cursors is None else None
    if reference is not None:
        hub = sources_mod.hub_for_host(
            [s for s in chosen if sources_mod.enabled(s)], reference.host
        )
        if hub is not None:
            resolved = await _resolve_reference(hubs.client_for(hub), reference, source=hub.id)
            if resolved is not None:
                looked = _status(hub)
                looked.searched, looked.results = True, len(resolved.results)
                resolved.sources = [looked]
                return resolved

    statuses: list[CatalogueSourceStatus] = []
    # Each source's rows, answered in the list's order whatever its kind
    # (LS7, Troy: the order is the person's; the default list puts the
    # engines' lists first).
    by_source: dict[str, list[CatalogueSearchResult]] = {}
    asked: list[tuple[CatalogueSource, CatalogueSourceStatus]] = []
    for source in sources:
        status = _status(source)
        statuses.append(status)
        if named is not None and source.id not in named:
            continue
        if not sources_mod.enabled(source):
            status.problem = "switched off in the Library's settings (Where to find models)"
            continue
        if cursors is not None and source.id not in cursors:
            continue  # a later page: only the hubs that had more
        status.searched = True
        if source.kind is CatalogueSourceKind.engine_list:
            if body.engines is None:
                status.results = 0
                status.problem = (
                    "no engine's list came with the search: the node's agent is older than "
                    "the sources list, or the console did not send it"
                )
                continue
            if not any(
                source.engine is None or listed_by.engine == source.engine
                for listed_by in body.engines
            ):
                status.results = 0
                who = (
                    f"{source.engine.value} on the node publishes"
                    if source.engine
                    else "none of the node's engines publishes"
                )
                status.problem = (
                    f"{who} a list of its own (an agent older than the sources list publishes none)"
                )
                continue
            rows = catalogue_mod.supported_rows(
                body.engines,
                source=source.id,
                hub_source=default.id if default else None,
                engine=source.engine,
                query=body.q,
                fmt=body.format,
                author=body.author,
            )
            status.results = len(rows)
            by_source[source.id] = rows
            continue
        asked.append((source, status))

    # By value: the generated default is the plain string "desc", which is
    # never the enum member, so `is` asked every hub for ascending order
    # (the least-downloaded first; Hugging Face now refuses it with a 400).
    descending = CatalogueSortDirection(body.direction or "desc") is CatalogueSortDirection.desc

    async def one_hub(source: CatalogueSource) -> tuple[list[Any], str | None, Any]:
        return await hubs.client_for(source).search(
            query=body.q,
            author=body.author,
            gguf_only=body.format is ModelFormat.gguf,
            sort=body.sort or CatalogueSort.downloads,
            descending=descending,
            limit=body.limit or 25,
            cursor=cursors.get(source.id) if cursors else None,
        )

    answers = await asyncio.gather(*(one_hub(s) for s, _ in asked), return_exceptions=True)
    more: dict[str, str] = {}
    for (source, status), answer in zip(asked, answers, strict=True):
        if isinstance(answer, HubError):
            status.results = 0
            status.problem = str(answer)
            continue
        if isinstance(answer, BaseException):
            raise answer
        results, next_cursor, _cached_at = answer
        rows = [
            catalogue_mod.build_search_result(entry, source=source.id, hub_source=source.id)
            for entry in results
            if isinstance(entry, dict)
        ]
        if body.format is ModelFormat.safetensors:
            rows = [r for r in rows if ModelFormat.safetensors in (r.formats or [])]
        status.results = len(rows)
        by_source[source.id] = rows
        if next_cursor:
            more[source.id] = next_cursor

    return CatalogueSearchPage(
        results=[row for s in sources for row in by_source.get(s.id, [])],
        nextCursor=_encode_cursor(more) if more else None,
        cachedAt=None,
        interpretedAs=InterpretedAs.search,
        sources=statuses,
    )


async def _resolve_reference(
    client: HubClient, reference: repo_ref.RepoReference, *, source: str | None = None
) -> CatalogueSearchPage | None:
    """One repo lookup for a pasted reference.

    `None` means "fall through to an ordinary search", which is what a
    bare `owner/name` that upstream does not have deserves -- it may be
    what the person meant to type. A URL is different: it names one repo
    and nothing else, so a miss answers with a Problem about that repo
    rather than quietly searching for the URL's own text.

    **A repo that does not exist does not come back as 404.** Measured
    against the live hub: `GET /api/models/nobody/nothing` answers **401
    `Invalid username or password`**, because to an unauthenticated
    caller "gone" and "private" are deliberately the same answer -- an
    enumeration defence. So "could not resolve" is 401, 403 and 404
    together, and the message has to name both possibilities instead of
    asserting the one we cannot tell from the other. A genuinely gated
    repo is the exception and keeps its own advice, because accepting a
    licence is a thing the operator can go and do.
    """
    try:
        info = await client.repo_info(reference.repo, revision=reference.revision or "main")
    except HubError as exc:
        if exc.code == "GatedRepo":
            raise _from_hub_error(exc) from exc
        if exc.status not in (401, 403, 404):
            raise _from_hub_error(exc) from exc
        if not reference.certain:
            return None
        raise _problem(
            404,
            "No such model",
            f"{client.base_url} has no repository {reference.repo!r} that this install can "
            "see. It may not exist, it may have been renamed, or it may be private -- "
            "upstream answers the same way for all three. If it is private, give this hub "
            "a token with access in the Library's settings (Where to find models). A link "
            "to a dataset or a space will land here too; only model repos can be opened "
            "from this screen.",
            code="repo-not-found",
        ) from exc

    return CatalogueSearchPage(
        results=[
            catalogue_mod.build_search_result(
                {"id": reference.repo, **info.raw}, source=source, hub_source=source
            )
        ],
        cachedAt=None,
        interpretedAs=InterpretedAs.repo,
        interpretedFrom=reference.repo,
    )


@router.get("/v1/catalogue/model", response_model=CatalogueModel)
async def get_catalogue_model(
    request: Request,
    repo: str = Query(
        default=...,
        description="Upstream repo id, e.g. `unsloth/Qwen3.8-27B-GGUF`.",
    ),
    source: str | None = Query(default=None, description=SOURCE_DESCRIPTION),
    revision: str = Query(
        default="main",
        description=(
            "Branch, tag or commit. The response reports the `resolvedCommit` it landed "
            "on, because `main` moves: repos are requantized and re-uploaded under the "
            "same filenames."
        ),
    ),
    contextLength: int | None = Query(
        default=None,
        ge=1,
        description=CONTEXT_DESCRIPTION,
    ),
    kvCacheType: KvCacheType = Query(
        default=KvCacheType.f16,
        description=KV_CACHE_DESCRIPTION,
    ),
    vramBytes: int | None = Query(
        default=None,
        ge=0,
        description=VRAM_DESCRIPTION,
    ),
    ramBytes: int | None = Query(
        default=None, ge=0, description="Override the detected host-memory budget, in bytes."
    ),
    unifiedMemory: bool | None = Query(default=None, description=UNIFIED_DESCRIPTION),
    gpuCount: int | None = Query(default=None, ge=1, description=GPU_COUNT_DESCRIPTION),
) -> CatalogueModel:
    """One repo, its download candidates, and which of them will fit.

    Guidance and discovery are the same response deliberately: the
    moment someone is choosing between `Q3_K_S` and `Q2_K_M` is the
    moment they need to be told which one their machine can hold, and a
    UI that has to join two calls to say so will ship without the second
    one.

    Two upstream calls: the repo's metadata and its file list. The file
    list is where sizes live, which is why this is a per-repo call and
    not something the search screen can do for every row. Plus, since
    LS2, one small ranged read of each safetensors folder's `config.json`
    (cached like the rest), so its facts carry the MLX marker and the
    architecture before anything is downloaded.
    """
    client = _client(request, source)
    config: ConfigStore = request.app.state.config_store
    store: StateStore = request.app.state.state_store

    try:
        info, files = await asyncio.gather(
            client.repo_info(repo, revision=revision),
            client.tree(repo, revision=revision),
        )
    except HubError as exc:
        raise _from_hub_error(exc) from exc

    budget = await _budget(
        request, vram=vramBytes, ram=ramBytes, unified=unifiedMemory, gpu_count=gpuCount
    )
    paths = catalogue_mod.config_paths(files, repo=repo)
    read = await gather_limited(
        [_remote_config(client, repo, revision, path) for path in paths], limit=4
    )
    return catalogue_mod.build_model(
        info=info,
        files=files,
        revision=revision,
        store=store,
        budget=budget,
        context_length=contextLength or config.guidance_context_length(),
        kv_cache_type=kvCacheType,
        configs=dict(zip(paths, read, strict=True)),
    )


async def _remote_config(
    client: HubClient, repo: str, revision: str, path: str
) -> safetensors.ModelConfig | None:
    """One folder's `config.json`, read before download (LS2), or None.

    A failure costs one fact, not the answer: the folder is judged with its
    MLX marker and architecture not known, which the verdict says."""
    try:
        return safetensors.parse_config(
            await client.read_small(repo, revision=revision, path=path), name=path
        )
    except (HubError, safetensors.SafetensorsError) as exc:
        log.info("catalogue %s: could not read %s (%s)", repo, path, exc)
        return None


@router.get("/v1/catalogue/starter", response_model=StarterSet)
async def get_starter_models(
    request: Request,
    contextLength: int | None = Query(
        default=None,
        ge=1,
        description=CONTEXT_DESCRIPTION,
    ),
    kvCacheType: KvCacheType = Query(
        default=KvCacheType.f16,
        description=KV_CACHE_DESCRIPTION,
    ),
    vramBytes: int | None = Query(default=None, ge=0, description=VRAM_DESCRIPTION),
    ramBytes: int | None = Query(default=None, ge=0, description=RAM_DESCRIPTION),
    unifiedMemory: bool | None = Query(default=None, description=UNIFIED_DESCRIPTION),
    gpuCount: int | None = Query(default=None, ge=1, description=GPU_COUNT_DESCRIPTION),
) -> StarterSet:
    """The shipped starter set, scored against this machine.

    **No upstream call**, which is the property worth protecting: every
    number a fit needs was measured once by the review that produced the
    list, so the screen where a person picks their first model still
    works with the catalogue disabled or the hub down. It is also the
    only catalogue endpoint that answers in a lab with no internet.

    Not cached, because there is nothing to cache: this is a YAML read
    and some arithmetic. The list is re-read per request so an operator
    editing `starterModelsFile` sees the change without a restart, the
    same rule the config trio follows everywhere else.
    """
    config: ConfigStore = request.app.state.config_store
    store: StateStore = request.app.state.state_store
    starter = starter_mod.load(config.starter_models_file())
    budget = await _budget(
        request, vram=vramBytes, ram=ramBytes, unified=unifiedMemory, gpu_count=gpuCount
    )
    return starter_mod.build(
        starter,
        budget=budget,
        context_length=contextLength or config.guidance_context_length(),
        kv_cache_type=kvCacheType,
        store=store,
    )


@router.get("/v1/catalogue/model/card", response_model=CatalogueCard)
async def get_catalogue_card(
    request: Request,
    repo: str = Query(default=..., description="Upstream repo id."),
    source: str | None = Query(default=None, description=SOURCE_DESCRIPTION),
    revision: str = Query(default="main", description="Branch, tag or commit."),
) -> CatalogueCard:
    """The model card prose, as Markdown.

    This is the "with summaries" half of the search complaint. A
    separate call from the detail response because it is a separate
    upstream fetch and a collapsed "about this model" panel should not
    pay for it until it opens.

    **Untrusted content.** It is written by whoever uploaded the model,
    so a renderer must not execute anything in it.
    """
    client = _client(request, source)
    try:
        markdown = await client.card(repo, revision=revision)
        info = await client.repo_info(repo, revision=revision)
    except HubError as exc:
        raise _from_hub_error(exc) from exc

    from datetime import UTC, datetime

    return CatalogueCard(
        repo=repo,
        revision=revision,
        markdown=markdown,
        frontMatter=info.card_data or None,
        fetchedAt=datetime.now(tz=UTC),
    )


@router.get("/v1/catalogue/model/preflight", response_model=CataloguePreflight)
async def preflight_catalogue_file(
    request: Request,
    repo: str = Query(default=..., description="Upstream repo id."),
    source: str | None = Query(default=None, description=SOURCE_DESCRIPTION),
    file: str = Query(
        default=...,
        description=(
            "Repo-relative path of the file to read. For a split candidate, the "
            "**first** shard: it carries the whole KV block."
        ),
    ),
    revision: str = Query(default="main", description="Branch, tag or commit."),
    contextLength: int | None = Query(
        default=None,
        ge=1,
        description=CONTEXT_DESCRIPTION,
    ),
    kvCacheType: KvCacheType = Query(
        default=KvCacheType.f16,
        description=KV_CACHE_DESCRIPTION,
    ),
    vramBytes: int | None = Query(
        default=None,
        ge=0,
        description=VRAM_DESCRIPTION,
    ),
    ramBytes: int | None = Query(
        default=None, ge=0, description="Override the detected host-memory budget, in bytes."
    ),
    unifiedMemory: bool | None = Query(default=None, description=UNIFIED_DESCRIPTION),
    gpuCount: int | None = Query(default=None, ge=1, description=GPU_COUNT_DESCRIPTION),
) -> CataloguePreflight:
    """Read one remote file's real metadata before downloading it.

    Costs about 11 MB of ranged read on a large-vocab GGUF — 0.07% of a
    16.5 GB file — and returns the machine-readable quant, the layer
    count, the attention-layer count and the trained context. That last
    one is the difference between a KV-cache estimate that is right and
    one that is 4.1x too large on a hybrid model.

    **Explicit, never automatic.** It spends upstream bandwidth per
    call, so it is for the one candidate an operator has settled on; a
    UI that fires it on hover would pull ~270 MB across one repo's
    candidates.
    """
    client = _client(request, source)
    config: ConfigStore = request.app.state.config_store

    # The candidate's whole size, so a split model is not scored as its
    # first shard alone — which understates it by every other shard.
    candidate_size: int | None = None
    try:
        files = await client.tree(repo, revision=revision)
        candidate_size = catalogue_mod.candidate_size(files, file)
    except HubError as exc:
        # A failure here costs precision, not the answer: fall back to
        # the probed file's own size from its HEAD.
        log.info("preflight %s: could not read the file list (%s)", repo, exc)

    budget = await _budget(
        request, vram=vramBytes, ram=ramBytes, unified=unifiedMemory, gpu_count=gpuCount
    )
    try:
        return await preflight_mod.preflight(
            client,
            repo=repo,
            revision=revision,
            path=file,
            budget=budget,
            context_length=contextLength or config.guidance_context_length(),
            kv_cache_type=kvCacheType,
            candidate_size=candidate_size,
        )
    except HubError as exc:
        raise _from_hub_error(exc) from exc
