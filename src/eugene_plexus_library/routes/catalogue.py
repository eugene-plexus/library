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
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from .. import catalogue as catalogue_mod
from .. import fit as fit_mod
from .. import hardware
from .. import preflight as preflight_mod
from .._generated.models import (
    CatalogueCard,
    CatalogueModel,
    CataloguePreflight,
    CatalogueSearchPage,
    CatalogueSort,
    KvCacheType,
    MemoryBudget,
    ModelFormat,
    Problem,
)
from ..config import ConfigStore
from ..hub import HubClient, HubError
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


def _client(request: Request) -> HubClient:
    """The shared client, reconfigured from live config on every call.

    Base URL, token and the enabled flag are read here rather than
    captured at construction so a `PATCH /v1/config` takes effect
    without a restart — which is the whole point of the config trio.
    """
    client: HubClient = request.app.state.hub_client
    config: ConfigStore = request.app.state.config_store
    client.configure(
        base_url=config.catalogue_base_url(),
        token=config.hf_token(),
        enabled=config.catalogue_enabled(),
    )
    return client


async def _budget(
    request: Request, *, vram: int | None = None, ram: int | None = None
) -> MemoryBudget:
    detected = await asyncio.to_thread(hardware.detect)
    return fit_mod.budget_from_hardware(detected, vram_override=vram, ram_override=ram)


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

    return CatalogueSearchPage(results=rows, nextCursor=next_cursor, cachedAt=None)


@router.get("/v1/catalogue/model", response_model=CatalogueModel)
async def get_catalogue_model(
    request: Request,
    repo: str = Query(
        default=...,
        description="Upstream repo id, e.g. `unsloth/Qwen3.8-27B-GGUF`.",
    ),
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
) -> CatalogueModel:
    """One repo, its download candidates, and which of them will fit.

    Guidance and discovery are the same response deliberately: the
    moment someone is choosing between `Q3_K_S` and `Q2_K_M` is the
    moment they need to be told which one their machine can hold, and a
    UI that has to join two calls to say so will ship without the second
    one.

    Two upstream calls: the repo's metadata and its file list. The file
    list is where sizes live, which is why this is a per-repo call and
    not something the search screen can do for every row.
    """
    client = _client(request)
    config: ConfigStore = request.app.state.config_store
    store: StateStore = request.app.state.state_store

    try:
        info, files = await asyncio.gather(
            client.repo_info(repo, revision=revision),
            client.tree(repo, revision=revision),
        )
    except HubError as exc:
        raise _from_hub_error(exc) from exc

    budget = await _budget(request, vram=vramBytes, ram=ramBytes)
    return catalogue_mod.build_model(
        info=info,
        files=files,
        revision=revision,
        store=store,
        budget=budget,
        context_length=contextLength or config.guidance_context_length(),
        kv_cache_type=kvCacheType,
    )


@router.get("/v1/catalogue/model/card", response_model=CatalogueCard)
async def get_catalogue_card(
    request: Request,
    repo: str = Query(default=..., description="Upstream repo id."),
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
    client = _client(request)
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
    client = _client(request)
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

    budget = await _budget(request, vram=vramBytes, ram=ramBytes)
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
