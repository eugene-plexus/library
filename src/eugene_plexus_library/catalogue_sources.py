"""Where Discover finds models: the `catalogueSources` list (LS4, design §4.4).

Each source has an id, a kind and its own settings. Two kinds:

* `hf_hub` -- a Hugging Face-compatible hub (the public one, a mirror, an
  enterprise instance) at its own `address`, with its own `token`;
* `engine_list` -- the models engines publish as supported
  (`SupportedModel`). The lists ship with the adapters in the agent and the
  caller of a search sends the picked node's, so a source of this kind holds
  only which engine's list (`engine`, absent for every engine) and on/off.

The list replaced `catalogueBaseUrl` and `hfToken`, the single hub's address
and token. A config file holding only those becomes one `hf_hub` source with
both (`migrate`), and the file keeps them beside the list, mirroring the
first `hf_hub` source (`mirror`), so a library older than LS4 reading the
same file still has its hub and its token.

The token is the one secret, per entry, with `share_credentials`' rules
(common.yaml `catalogue_sources`): `GET` answers `null` and `hasToken`; a
`PATCH` entry without it keeps the token stored under the same id; `""`
clears it. In memory tokens are plain; `config.py` seals them on disk.

Pure functions over plain dicts, so the config store, the routes and the
download manager share one reading of the list.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import hub as hub_mod
from ._generated.models import CatalogueSource, CatalogueSourceKind, EngineKind
from .hub import HubClient, HubError

DEFAULT_ADDRESS = "https://huggingface.co"
DEFAULT_HUB_ID = "huggingface"
DEFAULT_LIST_ID = "engines"

SOURCES_KEY = "catalogueSources"
#: The keys LS4 replaced, still mirrored in the file and accepted in PATCH.
OLD_ADDRESS_KEY = "catalogueBaseUrl"
OLD_TOKEN_KEY = "hfToken"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_KEYS = frozenset({"id", "kind", "label", "enabled", "address", "token", "hasToken", "engine"})
_ENGINES = frozenset(k.value for k in EngineKind)


def default_sources() -> list[dict[str, Any]]:
    """Every engine's list, then the public hub: a search answers in the
    list's order (LS7), and what an engine says it runs best comes first."""
    return [
        {
            "id": DEFAULT_LIST_ID,
            "kind": "engine_list",
            "label": "Engines' own lists",
            "enabled": True,
        },
        {
            "id": DEFAULT_HUB_ID,
            "kind": "hf_hub",
            "label": "Hugging Face",
            "enabled": True,
            "address": DEFAULT_ADDRESS,
        },
    ]


def _hub_label(address: str) -> str:
    """What to call a migrated hub: its own name only when it is that hub."""
    if address.rstrip("/") == DEFAULT_ADDRESS:
        return "Hugging Face"
    host = urlsplit(address).netloc or address
    return f"Hub at {host}"


def migrate(address: Any, token: Any) -> list[dict[str, Any]]:
    """The list a pre-LS4 config file means: its hub, at its address with its
    token (already decrypted by the caller, or None), then every engine's
    list. A blank address was always the public hub."""
    where = address.strip().rstrip("/") if isinstance(address, str) and address.strip() else ""
    where = where or DEFAULT_ADDRESS
    sources = default_sources()
    hub = next(s for s in sources if s["kind"] == "hf_hub")
    hub["address"] = where
    hub["label"] = _hub_label(where)
    if isinstance(token, str) and token.strip():
        hub["token"] = token.strip()
    return sources


def validate(value: Any) -> str | None:
    """A sentence naming the first thing wrong with a `catalogueSources`
    value, or None. Shape only: whether a hub answers is a fact about a
    network, and `POST /v1/catalogue/search` reports it per source."""
    if not isinstance(value, list):
        return f"expected a list of catalogue sources, got {type(value).__name__}"
    seen: set[str] = set()
    for index, entry in enumerate(value):
        where = f"entry {index + 1}"
        if not isinstance(entry, dict):
            return f"{where}: expected an object, got {type(entry).__name__}"
        unknown = sorted(set(entry) - _KEYS)
        if unknown:
            return f"{where}: unknown field(s) {', '.join(unknown)}"
        ident = entry.get("id")
        if not isinstance(ident, str) or not _ID.match(ident):
            return (
                f"{where}: `id` must be 1-40 lowercase letters, digits or dashes, starting "
                f"with a letter or digit (got {ident!r})"
            )
        if ident in seen:
            return f"{where}: the id {ident!r} is used twice"
        seen.add(ident)
        kind = entry.get("kind")
        if kind not in ("hf_hub", "engine_list"):
            return f"{where} ({ident}): `kind` must be hf_hub or engine_list (got {kind!r})"
        for key in ("label",):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                return f"{where} ({ident}): `{key}` must be text"
        if entry.get("enabled") is not None and not isinstance(entry["enabled"], bool):
            return f"{where} ({ident}): `enabled` must be true or false"
        address = entry.get("address")
        token = entry.get("token")
        engine = entry.get("engine")
        if kind == "hf_hub":
            if engine is not None:
                return f"{where} ({ident}): a hub has no `engine`; that is for an engine_list"
            if address is not None and (
                not isinstance(address, str) or not re.match(r"^https?://\S+$", address)
            ):
                return (
                    f"{where} ({ident}): `address` must be an http:// or https:// address "
                    f"(got {address!r})"
                )
            if token is not None and not isinstance(token, str):
                return f"{where} ({ident}): `token` must be text"
        else:
            if address is not None or token not in (None, ""):
                return f"{where} ({ident}): an engine_list has no address or token"
            if engine is not None and engine not in _ENGINES:
                return (
                    f"{where} ({ident}): `engine` must be one of {', '.join(sorted(_ENGINES))} "
                    f"(got {engine!r})"
                )
    return None


def _clean(entry: dict[str, Any]) -> dict[str, Any]:
    """One entry as it is held: known keys, no `hasToken` (it is derived)."""
    out: dict[str, Any] = {"id": entry["id"], "kind": entry["kind"]}
    for key in ("label", "enabled", "address", "engine"):
        if entry.get(key) is not None:
            out[key] = entry[key]
    if isinstance(out.get("address"), str):
        out["address"] = out["address"].rstrip("/")
    return out


def merge(incoming: list[dict[str, Any]], stored: Any) -> list[dict[str, Any]]:
    """A validated PATCH value, with tokens resolved against what is stored.

    An entry without `token` (or with `null`) keeps the token stored under
    its id: `GET` redacts, so a round trip that renames a source would
    otherwise clear it. `""` clears it; any other text replaces it."""
    previous = {
        e.get("id"): e.get("token")
        for e in (stored if isinstance(stored, list) else [])
        if isinstance(e, dict)
    }
    out: list[dict[str, Any]] = []
    for entry in incoming:
        kept = _clean(entry)
        token = entry.get("token")
        if entry["kind"] == "hf_hub":
            if token is None:
                old = previous.get(entry["id"])
                if old:
                    kept["token"] = old
            elif token.strip():
                kept["token"] = token.strip()
        out.append(kept)
    return out


def redact(sources: Any) -> list[dict[str, Any]]:
    """What `GET /v1/config` answers: every entry, no token, and whether one
    is stored (`hasToken`), so a UI can tell a stored token from none."""
    out: list[dict[str, Any]] = []
    for entry in sources if isinstance(sources, list) else []:
        if not isinstance(entry, dict):
            continue
        shown = _clean(entry)
        if entry.get("kind") == "hf_hub":
            shown["token"] = None
            shown["hasToken"] = bool(entry.get("token"))
        out.append(shown)
    return out


def first_hub_index(sources: Sequence[dict[str, Any]], *, enabled_only: bool = False) -> int | None:
    for index, entry in enumerate(sources):
        if entry.get("kind") == "hf_hub" and (not enabled_only or entry.get("enabled", True)):
            return index
    return None


def mirror(sources: Any) -> tuple[str, str | None]:
    """`(catalogueBaseUrl, hfToken)` for the file, from the first hub: what a
    library older than LS4 reads there. With no hub, the public one, no token."""
    listed = sources if isinstance(sources, list) else []
    index = first_hub_index(listed)
    if index is None:
        return DEFAULT_ADDRESS, None
    entry = listed[index]
    return entry.get("address") or DEFAULT_ADDRESS, entry.get("token") or None


def apply_old_key(sources: Any, key: str, value: Any) -> tuple[list[dict[str, Any]], str | None]:
    """A PATCH of `catalogueBaseUrl` or `hfToken`, which LS4 replaced: set
    on the first hub. `(sources, error)`; null clears to the default."""
    listed = [dict(e) for e in sources] if isinstance(sources, list) else []
    index = first_hub_index(listed)
    if index is None:
        return listed, (
            f"`{key}` sets the first hub in `catalogueSources`, and there is no hub there; "
            "add one to `catalogueSources` instead"
        )
    if value is not None and not isinstance(value, str):
        return listed, f"expected string, got {type(value).__name__}"
    entry = listed[index]
    if key == OLD_ADDRESS_KEY:
        address = (value or "").strip() or DEFAULT_ADDRESS
        if not re.match(r"^https?://\S+$", address):
            return listed, f"must be an http:// or https:// address (got {address!r})"
        entry["address"] = address.rstrip("/")
    else:
        token = (value or "").strip()
        if token:
            entry["token"] = token
        else:
            entry.pop("token", None)
    return listed, None


# -- resolving a source ---------------------------------------------------------


class SourceProblem(HubError):
    """Why a call naming a source cannot use it, as an HTTP answer. A
    `HubError`, so a download resumed against a source since removed or
    switched off fails with this sentence like any other upstream refusal."""

    def __init__(self, status: int, title: str, detail: str, code: str) -> None:
        super().__init__(detail, status=status, code=code)
        self.title = title
        self.detail = detail


def as_models(sources: Iterable[Any]) -> list[CatalogueSource]:
    """The held entries as the generated model (tokens plain), skipping any a
    hand-edited file broke: the config endpoints are how it is repaired."""
    out: list[CatalogueSource] = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                CatalogueSource.model_validate({k: v for k, v in entry.items() if k in _KEYS})
            )
        except ValueError:
            continue
    return out


def label_of(source: CatalogueSource) -> str:
    return source.label or source.id


def address_of(source: CatalogueSource) -> str:
    return (source.address or DEFAULT_ADDRESS).rstrip("/")


def enabled(source: CatalogueSource) -> bool:
    return source.enabled is not False


def default_hub(sources: Sequence[CatalogueSource]) -> CatalogueSource | None:
    """The first enabled hub: what a call naming no source means."""
    return next((s for s in sources if s.kind is CatalogueSourceKind.hf_hub and enabled(s)), None)


def hub_for(sources: Sequence[CatalogueSource], ident: str | None) -> CatalogueSource:
    """The hub a repo call names (`source`), or the default; else why not."""
    if ident is None or ident == "":
        hub = default_hub(sources)
        if hub is None:
            raise SourceProblem(
                409,
                "No hub switched on",
                "Every hub in `catalogueSources` is switched off or removed, so there is "
                "nowhere to look this up. Switch one on in the Library's settings, under "
                "Where to find models.",
                "no-hub",
            )
        return hub
    found = next((s for s in sources if s.id == ident), None)
    if found is None:
        raise SourceProblem(
            404,
            "No such source",
            f"This library has no catalogue source {ident!r}. It may have been removed in "
            "the Library's settings since the page was loaded.",
            "source-not-found",
        )
    if found.kind is not CatalogueSourceKind.hf_hub:
        raise SourceProblem(
            400,
            "Not a hub",
            f"{label_of(found)!r} is an engine's list, not a hub: its models' files are on "
            "a hub, which is what to name here.",
            "source-not-a-hub",
        )
    if not enabled(found):
        raise SourceProblem(
            409,
            "Source switched off",
            f"{label_of(found)!r} is switched off in the Library's settings (Where to find "
            "models). Switch it on to search, open or download from it.",
            "source-disabled",
        )
    return found


def hub_for_host(sources: Sequence[CatalogueSource], host: str | None) -> CatalogueSource | None:
    """The enabled hub at this host (a pasted link's), or the default."""
    if host:
        for source in sources:
            if (
                source.kind is CatalogueSourceKind.hf_hub
                and enabled(source)
                and urlsplit(address_of(source)).netloc.lower() == host.lower()
            ):
                return source
    return default_hub(sources)


class HubClients:
    """One `HubClient` per hub source, over one shared HTTP client.

    One connection pool for the process, as before LS4 (one client per
    instance, never per call); each hub keeps its own address, token and
    answer cache, so one hub's results are never attributed to another.
    Address, token and on/off are read from config on every `resolve`, so a
    `PATCH /v1/config` takes effect without a restart.
    """

    def __init__(
        self,
        sources: Callable[[], list[CatalogueSource]],
        catalogue_enabled: Callable[[], bool],
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._sources = sources
        self._enabled = catalogue_enabled
        self._http = http or hub_mod.egress_client(
            follow_redirects=False,  # hub.py's trap 2: never follow here.
            timeout=hub_mod.REQUEST_TIMEOUT,
            headers={"User-Agent": hub_mod.USER_AGENT},
        )
        self._clients: dict[str, HubClient] = {}

    def sources(self) -> list[CatalogueSource]:
        return self._sources()

    def client_for(self, source: CatalogueSource) -> HubClient:
        client = self._clients.get(source.id)
        if client is None:
            client = self._clients[source.id] = HubClient(client=self._http)
        client.configure(
            base_url=address_of(source),
            token=source.token or None,
            enabled=self._enabled(),
        )
        return client

    def resolve(self, ident: str | None) -> tuple[CatalogueSource, HubClient]:
        """The hub a repo call names, or the default, with its client; raises
        `SourceProblem` naming why it cannot be used."""
        source = hub_for(self._sources(), ident)
        return source, self.client_for(source)

    async def aclose(self) -> None:
        await self._http.aclose()
