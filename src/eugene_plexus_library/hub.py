"""The upstream catalogue client. The only outbound network in this component.

Everything here was verified against the live hub on 2026-09-08, because
five of its behaviours are traps that a plausible-looking client gets
wrong silently.

## Trap 1 — the digest is `lfs.oid`, and there is a decoy beside it

Each tree entry carries three hashes:

    oid       40 hex   git sha1 of the 136-byte LFS *pointer file*
    lfs.oid   64 hex   sha256 of the content        <- the one to verify against
    xetHash   64 hex   Xet chunk-tree hash

A verifier that uses `oid` compares a 16 GB download against a hash of
the stub that stands in for it, and passes on anything. Non-LFS files
have no `lfs` object at all and are verified by git blob sha1 instead —
`sha1("blob <len>\\0" + bytes)`, checked exactly against a real
`config.json`. Between the two, every file has something to check.

## Trap 2 — the metadata is on the redirect, not on the response

`GET /{repo}/resolve/{rev}/{file}` answers **302** to a signed CDN URL,
and `X-Linked-Size`, `X-Linked-ETag` and `X-Repo-Commit` are all on that
302. Follow the redirect — which every HTTP client does by default — and
all three are gone; the CDN's own `etag` is the *Xet* hash, which
matches nothing in the tree API. So `follow_redirects` is off and the
downloader reads the first response's headers before following.

The mirror-image trap: a client that does not follow the redirect writes
the redirect *body* to disk. Non-LFS files answer **307** to an internal
`/api/resolve-cache/…` path, and a 230-byte file reading
`Temporary Redirect. Redirecting to …` is what lands where `config.json`
was meant to be.

## Trap 3 — the CDN URL expires

`…?Expires=1788926395&Policy=…&Signature=…`. A downloader that stores
the resolved URL beside its `.part` and reuses it an hour later gets a
403 that looks like a network failure. **Resume always re-resolves.**

## Trap 4 — a gated repo browses perfectly and only the download 401s

Metadata, file list, sizes and digests are public; the bytes are not.
`gated` is `false | "auto" | "manual"` on the model API, so the warning
is available before anything is attempted, and the 401 carries
`X-Error-Code: GatedRepo` so the failure is diagnosable rather than
generic.

## Trap 5 — two rate-limit buckets, and search has the small one

    RateLimit-Policy: "fixed window";"api";q=500;w=300
    RateLimit-Policy: "fixed window";"resolvers";q=3000;w=300

500 API requests per 300 seconds. Search-as-you-type at 100 keystrokes a
minute exceeds that inside two minutes, so responses are cached here for
a short window and the UI is expected to debounce as well.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._generated.models import CatalogueSort, GateKind

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://huggingface.co"

USER_AGENT = "eugene-plexus-library"

CACHE_TTL_SECONDS = 60.0
"""How long a search or repo lookup is reused. Short enough that a
download count is not stale in any way that matters, long enough that a
UI re-rendering a screen does not re-spend the API budget."""

CACHE_MAX_ENTRIES = 256

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

TRANSFER_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
"""Separate from `REQUEST_TIMEOUT`: a stalled *transfer* needs a longer
read timeout than a stalled API call, because a large chunk over a slow
link is not a hung connection."""


class HubError(Exception):
    """An upstream call did not produce a usable answer.

    Carries the status a route should return and, where upstream gave
    one, its own error code — `GatedRepo` being the one that will
    actually happen, and the one where the operator has something
    specific to do about it.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int = 502,
        code: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after


class CatalogueDisabled(HubError):
    """`catalogueEnabled` is off.

    A real configuration, not a fault: an air-gapped install turns the
    catalogue off and should get a legible refusal rather than a
    sequence of connection timeouts.
    """

    def __init__(self) -> None:
        super().__init__(
            "The model catalogue is switched off. Set `catalogueEnabled` in this "
            "component's config to search, browse or download models from upstream. "
            "Scanning your own directories is unaffected.",
            status=409,
            code="CatalogueDisabled",
        )


@dataclass
class FileMetadata:
    """What the hub says about one file, before a byte is transferred."""

    path: str
    size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None
    lfs: bool = False

    @property
    def digest(self) -> tuple[str, str] | None:
        """`(algorithm, hex)` — whichever this file actually has.

        sha256 for LFS-backed weights, git blob sha1 for the small
        sidecars that are stored as plain git objects. `None` only if
        upstream gave neither, in which case nothing is verified and the
        caller has to say so.
        """
        if self.sha256:
            return ("sha256", self.sha256)
        if self.git_blob_sha1:
            return ("git-blob-sha1", self.git_blob_sha1)
        return None


@dataclass
class ResolvedFile:
    """The answer to a HEAD on a resolve URL — read off the 302."""

    size: int | None
    sha256: str | None
    commit: str | None
    accept_ranges: bool


@dataclass
class RepoInfo:
    """One repo's metadata, as the model API reports it."""

    repo: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def owner(self) -> str | None:
        author = self.raw.get("author")
        if isinstance(author, str) and author:
            return author
        return self.repo.split("/")[0] if "/" in self.repo else None

    @property
    def name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def sha(self) -> str | None:
        value = self.raw.get("sha")
        return value if isinstance(value, str) else None

    @property
    def gated(self) -> GateKind:
        """`false | "auto" | "manual"` upstream, normalized to three names.

        `open` rather than a boolean because a three-valued enum beats a
        boolean-or-string union in every generated client, and because
        the two gated kinds need different advice: `auto` is a
        click-through, `manual` is an approval that can take days.
        """
        value = self.raw.get("gated")
        if value == "auto":
            return GateKind.auto
        if value == "manual":
            return GateKind.manual
        return GateKind.open

    @property
    def gguf(self) -> dict[str, Any]:
        """The hub's repo-level GGUF block.

        `total` here is a **parameter count** (27,320,697,856 on a
        27B), not a byte count — which is what makes bits-per-weight
        computable for a remote candidate with no extra request, and is
        the exact inverse of the local case where a GGUF file carries no
        parameter count at all.
        """
        value = self.raw.get("gguf")
        return value if isinstance(value, dict) else {}

    @property
    def parameters(self) -> int | None:
        value = self.gguf.get("total")
        return int(value) if isinstance(value, int) else None

    @property
    def architecture(self) -> str | None:
        value = self.gguf.get("architecture")
        if isinstance(value, str):
            return value
        config = self.raw.get("config")
        if isinstance(config, dict):
            model_type = config.get("model_type")
            if isinstance(model_type, str):
                return model_type
        return None

    @property
    def context_length(self) -> int | None:
        value = self.gguf.get("context_length")
        return int(value) if isinstance(value, int) else None

    @property
    def chat_template(self) -> bool | None:
        if "chat_template" in self.gguf:
            return bool(self.gguf.get("chat_template"))
        return None

    @property
    def card_data(self) -> dict[str, Any]:
        value = self.raw.get("cardData")
        return value if isinstance(value, dict) else {}

    @property
    def license(self) -> str | None:
        value = self.card_data.get("license")
        if isinstance(value, str):
            return value
        for tag in self.tags:
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
        return None

    @property
    def tags(self) -> list[str]:
        value = self.raw.get("tags")
        return [t for t in value if isinstance(t, str)] if isinstance(value, list) else []


def git_blob_sha1(data: bytes) -> str:
    """Git's own hash of a blob: `sha1("blob <len>\\0" + bytes)`.

    The only digest available for the small non-LFS files in a repo —
    `config.json`, the tokenizer files, the safetensors index — none of
    which are optional and all of which would otherwise be written
    unverified. Verified exactly against a real 728-byte `config.json`
    whose reported `oid` this reproduces.
    """
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


class _TtlCache:
    """A tiny TTL cache, sized for the rate-limit budget rather than for
    memory. Not an LRU: entries expire on time and the map is cleared
    wholesale when it grows past the cap, which is the right trade for a
    few hundred small JSON bodies."""

    def __init__(self, ttl: float = CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> tuple[Any, float] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stamped, value = entry
        if time.monotonic() - stamped > self._ttl:
            self._entries.pop(key, None)
            return None
        return value, stamped

    def put(self, key: str, value: Any) -> None:
        if len(self._entries) >= CACHE_MAX_ENTRIES:
            self._entries.clear()
        self._entries[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._entries.clear()


_SORT_FIELDS: dict[CatalogueSort, str] = {
    CatalogueSort.downloads: "downloads",
    CatalogueSort.likes: "likes",
    CatalogueSort.trending: "trendingScore",
    CatalogueSort.modified: "lastModified",
    CatalogueSort.created: "createdAt",
}


class HubClient:
    """Async client for one upstream hub.

    Holds the connection pool; base URL, token and the enabled flag are
    read from config on every call rather than captured at construction,
    so a `PATCH /v1/config` takes effect without a restart.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache: _TtlCache | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,  # Trap 2. Never turn this on.
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        self._cache = cache or _TtlCache()
        self._base_url = DEFAULT_BASE_URL
        self._token: str | None = None
        self._enabled = True

    def configure(self, *, base_url: str | None, token: str | None, enabled: bool) -> None:
        base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if base != self._base_url:
            # A different hub is a different catalogue; keeping the old
            # answers would attribute one hub's models to another.
            self._cache.clear()
        self._base_url = base
        self._token = token or None
        self._enabled = enabled

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- plumbing ---------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if self._token:
            # Sent for public repos too, not just gated ones: upstream's
            # own warning header says a token raises the rate limit and
            # speeds up transfers.
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise CatalogueDisabled()

    def _raise_for(self, response: httpx.Response, *, what: str) -> None:
        """Map an upstream failure onto something an operator can act on."""
        code = response.headers.get("X-Error-Code")
        message = response.headers.get("X-Error-Message") or ""

        if response.status_code in (401, 403):
            if code == "GatedRepo" or "restricted" in message.lower():
                raise HubError(
                    f"{what} is gated upstream. {message or 'Access is restricted.'} "
                    "Accept the licence on the model's own page, then set `hfToken` in "
                    "this component's config to a token from an account that has access.",
                    status=403,
                    code=code or "GatedRepo",
                )
            raise HubError(
                f"{what} requires authentication upstream"
                f"{f': {message}' if message else ''}. Set `hfToken` in this component's "
                "config.",
                status=403,
                code=code or "Unauthorized",
            )
        if response.status_code == 404:
            raise HubError(
                f"{what} was not found upstream. Check the spelling, and note that a "
                "private repo reads as missing until a token with access is configured.",
                status=404,
                code=code or "NotFound",
            )
        if response.status_code == 429:
            retry = response.headers.get("Retry-After")
            raise HubError(
                "Upstream rate limit reached. The hub allows 500 API requests per 300 "
                "seconds unauthenticated; setting `hfToken` raises that. Try again "
                "shortly.",
                status=429,
                code="RateLimited",
                retry_after=int(retry) if retry and retry.isdigit() else None,
            )
        raise HubError(
            f"Upstream returned {response.status_code} for {what}"
            f"{f': {message}' if message else ''}.",
            status=502,
            code=code,
        )

    async def _get_json(self, url: str, *, what: str, cache_key: str | None = None) -> Any:
        self._require_enabled()
        if cache_key:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit[0]
        try:
            response = await self._client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise HubError(
                f"Could not reach {self._base_url} for {what} ({exc}). The catalogue needs "
                "outbound HTTPS; set `catalogueEnabled` off if this install has none.",
                status=503,
            ) from exc

        if response.status_code >= 400:
            self._raise_for(response, what=what)

        try:
            payload = response.json()
        except ValueError as exc:
            raise HubError(f"Upstream sent a non-JSON answer for {what} ({exc}).") from exc

        if cache_key:
            self._cache.put(cache_key, payload)
        return payload

    # -- the catalogue -----------------------------------------------------

    async def search(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        gguf_only: bool = False,
        sort: CatalogueSort = CatalogueSort.downloads,
        descending: bool = True,
        limit: int = 25,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, float | None]:
        """`(results, next_cursor, cached_at_monotonic)`.

        Pagination is by opaque cursor because upstream's is: the next
        page arrives as a whole URL in a `Link: rel="next"` header with a
        base64 blob in it, and there is no page number to pass through.
        The cursor handed back to callers is that blob.
        """
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("limit", str(limit)),
            ("full", "true"),
            ("sort", _SORT_FIELDS[sort]),
            ("direction", "-1" if descending else "1"),
        ]
        if query:
            params.append(("search", query))
        if author:
            params.append(("author", author))
        if gguf_only:
            params.append(("filter", "gguf"))
        if cursor:
            params.append(("cursor", cursor))

        url = f"{self._base_url}/api/models"
        key = f"search:{params}"

        self._require_enabled()
        hit = self._cache.get(key)
        if hit is not None:
            payload, stamped = hit
            return payload[0], payload[1], stamped

        try:
            response = await self._client.get(url, params=params, headers=self._headers())
        except httpx.HTTPError as exc:
            raise HubError(
                f"Could not reach {self._base_url} to search ({exc}).", status=503
            ) from exc
        if response.status_code >= 400:
            self._raise_for(response, what="the model search")

        try:
            results = response.json()
        except ValueError as exc:
            raise HubError(f"Upstream sent a non-JSON search answer ({exc}).") from exc
        if not isinstance(results, list):
            raise HubError("Upstream search did not return a list of models.")

        next_cursor = _next_cursor(response.headers.get("Link"))
        self._cache.put(key, (results, next_cursor))
        return results, next_cursor, None

    async def repo_info(self, repo: str, *, revision: str = "main") -> RepoInfo:
        url = f"{self._base_url}/api/models/{repo}"
        if revision != "main":
            url = f"{url}/revision/{revision}"
        payload = await self._get_json(
            url, what=f"repo {repo!r}", cache_key=f"info:{repo}@{revision}"
        )
        if not isinstance(payload, dict):
            raise HubError(f"Upstream sent an unexpected shape for repo {repo!r}.")
        return RepoInfo(repo=repo, raw=payload)

    async def tree(self, repo: str, *, revision: str = "main") -> list[FileMetadata]:
        """Every file in the repo, with the digest each one actually has.

        Sizes live here and nowhere else: the search response carries
        filenames only, which is why fit verdicts belong on the detail
        screen and not on a search row.
        """
        url = f"{self._base_url}/api/models/{repo}/tree/{revision}?recursive=1&expand=true"
        payload = await self._get_json(
            url, what=f"the file list for {repo!r}", cache_key=f"tree:{repo}@{revision}"
        )
        if not isinstance(payload, list):
            raise HubError(f"Upstream sent an unexpected file list for {repo!r}.")

        files: list[FileMetadata] = []
        for entry in payload:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            size = entry.get("size")
            lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else None
            sha256 = None
            if lfs is not None:
                oid = lfs.get("oid")
                # Trap 1: `lfs.oid` is the content sha256. The sibling
                # `oid` is the sha1 of the pointer file and is not a
                # digest of anything the operator is downloading.
                if isinstance(oid, str) and len(oid) == 64:
                    sha256 = oid
                if isinstance(lfs.get("size"), int):
                    size = lfs["size"]
            blob_sha1 = None
            if sha256 is None:
                oid = entry.get("oid")
                if isinstance(oid, str) and len(oid) == 40:
                    blob_sha1 = oid
            files.append(
                FileMetadata(
                    path=path,
                    size=int(size) if isinstance(size, int) else 0,
                    sha256=sha256,
                    git_blob_sha1=blob_sha1,
                    lfs=lfs is not None,
                )
            )
        return files

    async def card(self, repo: str, *, revision: str = "main") -> str:
        """The README, verbatim. 7-10 KB for a typical repo.

        Untrusted content — written by whoever uploaded the model — and
        the only thing here that answers the "no search page with
        summaries" half of the complaint.
        """
        self._require_enabled()
        url = f"{self._base_url}/{repo}/raw/{revision}/README.md"
        try:
            response = await self._client.get(url, headers=self._headers(), follow_redirects=True)
        except httpx.HTTPError as exc:
            raise HubError(
                f"Could not fetch the model card for {repo!r} ({exc}).", status=503
            ) from exc
        if response.status_code == 404:
            return ""
        if response.status_code >= 400:
            self._raise_for(response, what=f"the model card for {repo!r}")
        return response.text

    # -- files -------------------------------------------------------------

    def resolve_url(self, repo: str, *, revision: str, path: str) -> str:
        return f"{self._base_url}/{repo}/resolve/{revision}/{path}"

    async def head_file(self, repo: str, *, revision: str, path: str) -> ResolvedFile:
        """Size, digest and commit, read off the **302** — see Trap 2.

        This is the resume validator: comparing the digest here against
        what a download recorded is what distinguishes "the transfer
        broke" from "upstream requantized under the same filename", and
        only the second one means the partial file has to go.
        """
        self._require_enabled()
        url = self.resolve_url(repo, revision=revision, path=path)
        try:
            response = await self._client.head(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise HubError(f"Could not resolve {path!r} in {repo!r} ({exc}).", status=503) from exc

        if response.status_code >= 400:
            self._raise_for(response, what=f"{path!r} in {repo!r}")

        size = response.headers.get("X-Linked-Size") or response.headers.get("Content-Length")
        etag = response.headers.get("X-Linked-ETag") or ""
        digest = etag.strip('"') or None
        # The CDN's own etag is the Xet hash; only a 64-hex value from
        # X-Linked-ETag is the sha256 that matches the tree API.
        if digest is not None and len(digest) != 64:
            digest = None
        return ResolvedFile(
            size=int(size) if size and size.isdigit() else None,
            sha256=digest,
            commit=response.headers.get("X-Repo-Commit"),
            accept_ranges=response.headers.get("Accept-Ranges", "").lower() == "bytes",
        )

    async def read_range(
        self, repo: str, *, revision: str, path: str, first: int, last: int
    ) -> bytes:
        """Fetch `[first, last]` inclusive.

        The mechanism behind preflight: the KV block sits at the front
        of a GGUF, so ~11 MB of a 16.5 GB file — 0.07% of it — yields
        the machine-readable quant and the layer shape. Redirects are
        followed here (unlike the HEAD) because the *body* is the point.
        """
        self._require_enabled()
        url = self.resolve_url(repo, revision=revision, path=path)
        headers = self._headers({"Range": f"bytes={first}-{last}"})
        try:
            response = await self._client.get(url, headers=headers, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise HubError(f"Could not read {path!r} from {repo!r} ({exc}).", status=503) from exc
        if response.status_code >= 400:
            self._raise_for(response, what=f"{path!r} in {repo!r}")
        if response.status_code != 206:
            # 200 to a Range request means the whole file came back. For
            # a 16 GB model that is not a fallback, it is a disaster, so
            # it is refused rather than accommodated.
            raise HubError(
                f"Upstream ignored a Range request for {path!r} (status "
                f"{response.status_code}); refusing to read the whole file.",
            )
        return response.content

    def stream(
        self, repo: str, *, revision: str, path: str, first: int | None = None
    ) -> AbstractAsyncContextManager[httpx.Response]:
        """An open streaming response for a transfer, resuming at `first`.

        Returns the context manager rather than the bytes: the caller
        writes to disk in chunks and must not hold a 40 GB body in
        memory. Redirects are followed, so the caller reads the digest
        from `head_file` first (Trap 2) rather than from this response.
        """
        headers = self._headers()
        if first:
            headers["Range"] = f"bytes={first}-"
        url = self.resolve_url(repo, revision=revision, path=path)
        return self._client.stream(
            "GET", url, headers=headers, follow_redirects=True, timeout=TRANSFER_TIMEOUT
        )


def _next_cursor(link_header: str | None) -> str | None:
    """Pull the opaque `cursor` out of a `Link: rel="next"` header.

    Upstream hands back a whole URL; only the cursor parameter is worth
    passing through a contract, because a full upstream URL in a
    client's hands is an invitation to bypass this component.
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        start = section.find("<")
        end = section.find(">", start + 1)
        if start < 0 or end < 0:
            continue
        url = section[start + 1 : end]
        query = httpx.URL(url).params
        cursor = query.get("cursor")
        return str(cursor) if cursor else None
    return None


async def gather_limited(coros: list[Any], *, limit: int) -> list[Any]:
    """Run coroutines with a concurrency cap, preserving order.

    Used where a screen needs several upstream lookups at once: the
    point of the cap is the rate-limit budget, not local resources.
    """
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros), return_exceptions=True)
