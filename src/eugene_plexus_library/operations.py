"""Operations rights for people who are not the owner (row 39), on the library.

`specs/docs/design/rows/row-39.md`. A person the owner gave the **Models**
power (`operate-models`) on this machine holds an `ep-operations+jwt`
addressed to it. That token opens exactly the routes the table below
names; every other route keeps the owner's gate (`require_operator`):
the config and its folders, the directory picker, restart, Keep and run
operations, adopting and moving models.

R39-DELETE: Models includes deleting model files, each delete in the
machine's record.

**Every change says who made it.** A change on a table route -- the
owner's included -- is reported to this machine's agent
(`POST /v1/node/operations-log`) with the library's own service token. The
person comes from the token verified here, never from a header.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import tokens
from ._http import shared_internal_client
from .auth_state import AuthState
from .dependencies import _bearer_scheme, _problem, require_operator

log = logging.getLogger(__name__)

MODELS = tokens.POWER_OPERATE_MODELS
POWER_NAMES = {MODELS: "Models", tokens.POWER_OPERATE_MACHINES: "Updates and restarts"}
CLIENT_KEY = "operations-log"

#: Reads any operations power may make.
OPERATIONS_READS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/v1/folders"),
        ("GET", "/v1/models"),
        ("GET", "/v1/models/{model_id}"),
        ("GET", "/v1/models/{model_id}/profiles"),
        ("GET", "/v1/models/{model_id}/profiles/{profile_id}"),
        ("GET", "/v1/downloads"),
        ("GET", "/v1/downloads/{download_id}"),
        ("GET", "/v1/scan"),
    }
)

#: Routes a power opens, with the action its record names.
OPERATIONS_ROUTES: dict[tuple[str, str], tuple[str, str]] = {
    ("POST", "/v1/downloads"): (MODELS, "download model"),
    ("DELETE", "/v1/downloads/{download_id}"): (MODELS, "cancel download"),
    ("POST", "/v1/downloads/{download_id}/pause"): (MODELS, "pause download"),
    ("POST", "/v1/downloads/{download_id}/resume"): (MODELS, "resume download"),
    ("GET", "/v1/models/{model_id}/deletion"): (MODELS, "plan model delete"),
    ("POST", "/v1/models/{model_id}/delete"): (MODELS, "delete model"),
    ("DELETE", "/v1/models/{model_id}"): (MODELS, "forget model"),
    ("POST", "/v1/models/{model_id}/profiles"): (MODELS, "create profile"),
    ("PUT", "/v1/models/{model_id}/profiles/{profile_id}"): (MODELS, "change profile"),
    ("DELETE", "/v1/models/{model_id}/profiles/{profile_id}"): (MODELS, "delete profile"),
    ("POST", "/v1/scan"): (MODELS, "re-scan models"),
    ("DELETE", "/v1/scan"): (MODELS, "stop scan"),
    ("POST", "/v1/catalogue/search"): (MODELS, "search catalogue"),
    ("GET", "/v1/catalogue/model"): (MODELS, "read catalogue"),
    ("GET", "/v1/catalogue/starter"): (MODELS, "read catalogue"),
    ("GET", "/v1/catalogue/model/card"): (MODELS, "read catalogue"),
    ("GET", "/v1/catalogue/model/preflight"): (MODELS, "read catalogue"),
}

READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STATE_KEY = "ep_operation"
CLAIMS_KEY = "ep_operations_claims"
#: Review S3: what a power may do that is a question, not a change.
NOT_RECORDED = frozenset({("POST", "/v1/catalogue/search")})
_TARGET_PARAMS = ("model_id", "download_id", "profile_id")


def _route_of(request: Request) -> tuple[str, str | None]:
    method = str(request.scope.get("method") or "").upper()
    return method, getattr(request.scope.get("route"), "path", None)


def is_operations(token: str) -> bool:
    try:
        return bool(jwt.get_unverified_header(token).get("typ") == tokens.TYP_OPERATIONS)
    except jwt.InvalidTokenError:
        return False


def only_the_owner() -> HTTPException:
    return _problem(
        status.HTTP_403_FORBIDDEN,
        "Owner only",
        "Only the owner can do this. Your operations powers do not include it.",
    )


def _invalid(exc: Exception) -> HTTPException:
    return _problem(status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}")


def admit(request: Request, token: str) -> tokens.Claims:
    """An operations token on this request's route, or the refusal.

    A read in `OPERATIONS_READS` takes any power; a route in
    `OPERATIONS_ROUTES` takes its own; anything else, none."""
    auth: AuthState = request.app.state.auth_state
    try:
        claims = auth.verify(token, classes=(tokens.TYP_OPERATIONS,))
    except tokens.TokenError as exc:
        raise _invalid(exc) from exc
    route = _route_of(request)
    if route in OPERATIONS_READS:
        setattr(request.state, CLAIMS_KEY, claims)
        return claims
    listed = OPERATIONS_ROUTES.get(route)  # type: ignore[arg-type]
    if listed is None:
        raise only_the_owner()
    if not claims.holds(listed[0]):
        raise _problem(
            status.HTTP_403_FORBIDDEN,
            "Power missing",
            f"This needs the {POWER_NAMES[listed[0]]} power on this machine. "
            "The owner gives it on Eugene's People page.",
        )
    return claims


def _mark(request: Request, claims: tokens.Claims, power: str, action: str) -> None:
    holder = getattr(request.state, STATE_KEY, None)
    if not isinstance(holder, dict) or request.method.upper() in READ_METHODS:
        return
    if _route_of(request) in NOT_RECORDED:
        return
    owner = not claims.is_operations
    params = request.path_params
    target = next((str(params[p]) for p in _TARGET_PARAMS if params.get(p)), "")
    holder.update(
        personId=tokens.SUB_OPERATOR if owner else claims.sub,
        name="Owner" if owner else (claims.name or claims.sub),
        owner=owner,
        power=power,
        action=action,
        target=target[:256],
    )


def operation(power: str) -> Callable[..., tokens.Claims | None]:
    """`require_operation(power)`: an operations token holding `power` on a
    route the table names with it, or the owner's gate as before."""

    def require_operation(
        request: Request,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> tokens.Claims | None:
        listed = OPERATIONS_ROUTES.get(_route_of(request))  # type: ignore[arg-type]
        if creds is not None and creds.credentials and is_operations(creds.credentials):
            claims = admit(request, creds.credentials)
            if listed is None or listed[0] != power:
                raise only_the_owner()
            _mark(request, claims, power, listed[1])
            setattr(request.state, CLAIMS_KEY, claims)
            return claims
        owner = require_operator(request, creds)
        if owner is not None and listed is not None:
            _mark(request, owner, listed[0], listed[1])
        return owner

    return require_operation


def note_target(request: Request, target: str) -> None:
    """Name what a change with no target in its path is about (a download's
    repository), for its row in the machine's operations log."""
    holder = getattr(request.state, STATE_KEY, None)
    if isinstance(holder, dict) and holder.get("action"):
        holder["target"] = target[:256]


def operations_claims(request: Any) -> tokens.Claims | None:
    found = getattr(getattr(request, "state", None), CLAIMS_KEY, None)
    return found if isinstance(found, tokens.Claims) else None


OWNER_PROFILE_FIELDS = {"env": "environment variables", "extraArgs": "extra command-line arguments"}

# library#22: a profile's engine flags become a runtime's arguments when the
# owner next runs it, so with an operations token each flag is judged by three
# explicit lists, keyed by (engine, flag), never by the key's spelling. They
# are the agent's lists (agent#53), duplicated: components share schemas,
# never code. The Library holds no engine adapters; `tests/fixtures/
# engine_flags.json` records every flag the agent's adapters declare, and
# `tests/test_operations_flags.py` fails on a flag on no list, so a new flag
# is classified here before an operations token can write it.

#: Flags only the owner turns on, with what each is, for the refusal.
OWNER_FLAGS: dict[tuple[str, str], str] = {
    # MLX and vLLM `--trust-remote-code`: a model folder's own Python runs in
    # the agent's account.
    ("mlx", "trustRemoteCode"): "running code that comes with a model",
    ("vllm", "trustRemoteCode"): "running code that comes with a model",
}
#: Flags that name a file, a folder or a model: for an operations token each
#: must lie inside a Library folder, so a Hugging Face Hub id is refused too.
PATH_FLAGS: frozenset[tuple[str, str]] = frozenset(
    {
        ("llama_cpp", "chatTemplateFile"),  # --chat-template-file
        ("llama_cpp", "draftModelPath"),  # --model-draft
        ("llama_cpp", "projectorPath"),  # --mmproj
        ("vllm", "draftModelPath"),  # the speculative model
        ("vllm", "tokenizer"),  # --tokenizer: a folder or a Hub id
        ("mlx", "draftModel"),  # --draft-model: a folder or a Hub id
        ("mlx", "adapterPath"),  # --adapter-path: LoRA weights
        ("kev", "fallback"),  # --fallback: a checkpoint folder
    }
)
_RUN = (
    "sameAnswerEveryTime",
    "reuseRepeatedText",
    "draftWith",
    "draftTokens",
    "missingCard",
)
#: The reviewed flags an operations token may set: sizes, counts, choices and
#: switches that change how a model runs on this machine, and nothing that
#: names a file, a program, an address or a secret.
OPERATIONS_FLAGS: frozenset[tuple[str, str]] = frozenset(
    [("llama_cpp", key) for key in _RUN]
    + [("vllm", key) for key in _RUN]
    + [
        ("llama_cpp", key)
        for key in (
            "projectorOnCpu",
            "contextSize",
            "gpuLayers",
            "parallelSlots",
            "batchSize",
            "ubatchSize",
            "threads",
            "flashAttention",
            "cacheType",
            "memoryMargin",
            "cacheRam",
            "cacheReuse",
            "ctxCheckpoints",
            "continuousBatching",
            "slotPinning",
            "mainGpu",
            "devices",  # GPU names, checked against this machine's cards
            "splitMode",
            "tensorSplit",
            "mlock",
            "noMmap",
        )
    ]
    + [
        ("vllm", key)
        for key in (
            "source",  # this release's pinned image or the owner's own vLLM
            "shmSizeGiB",
            "maxModelLen",
            "gpuMemoryUtilization",
            "kvCacheDtype",
            "tensorParallelSize",
            "pipelineParallelSize",
            "maxNumSeqs",
            "maxNumBatchedTokens",
            "dtype",
            "enforceEager",
            "quantization",  # a method name from vLLM's own closed list
            "kvOffloadGb",
        )
    ]
    + [
        ("mlx", key)
        for key in (
            "maxTokens",
            "promptCacheSize",
            "promptCacheBytes",
            "decodeConcurrency",
            "promptConcurrency",
            "prefillStepSize",
            "numDraftTokens",
            # mlx-lm passes --chat-template to the tokenizer as the
            # template's own text, never as a file name.
            "chatTemplate",
        )
    ]
    + [("strata", key) for key in ("speedProjection", "gpus", "layerSplit")]
)
ENGINE_NAMES = {
    "llama_cpp": "llama.cpp",
    "vllm": "vLLM",
    "mlx": "MLX",
    "kev": "Kev",
    "strata": "Strata",
}
#: A negative number written as text (`tensorSplit: "-1"`), as the agent
#: reads one: the whole value, after spaces.
_NEGATIVE_NUMBER = re.compile(r"-(\d+(\.\d*)?|\.\d+)")
#: Flags that choose which program a kept owner flag applies to: vLLM's
#: `source` (this release's container or the owner's own vLLM on the host).
#: Like a path flag, changing it needs the owner again while an owner flag
#: stays on; removing it is a change too.
SOURCE_FLAGS: frozenset[tuple[str, str]] = frozenset({("vllm", "source")})


def _engine(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def flag_class(engine: Any, key: str) -> str | None:
    """What `engine`'s flag `key` is for an operations token: `owner`,
    `path`, `operations` (reviewed as safe), or None when nobody has
    classified it for that engine."""
    pair = (_engine(engine), key)
    if pair in OWNER_FLAGS:
        return "owner"
    if pair in PATH_FLAGS:
        return "path"
    if pair in OPERATIONS_FLAGS:
        return "operations"
    return None


def _is_on(value: Any) -> bool:
    return value not in (None, False, "")


def _dash_value(value: Any) -> str | None:
    """A text value an engine could read as another option: one that starts
    with "-" (after spaces) and is not a negative number."""
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str):
            text = item.strip()
            if text.startswith("-") and not _NEGATIVE_NUMBER.fullmatch(text):
                return item
    return None


def refuse_owner_fields(request: Request, spec: Any, previous: Any = None) -> None:
    """Review B1 and library#22: a profile is what a runtime is later
    declared from, under the owner's authority. With an operations token,
    its environment, its extra arguments, an engine flag only the owner
    sets, a file outside the Library's folders, and a flag on no list stay
    the owner's. A value the profile already had is left alone, for the
    same engine only; a kept owner flag does not follow a changed file or a
    changed or removed `source`. The owner is never checked here."""
    if operations_claims(request) is None:
        return
    engine = _engine(getattr(spec, "engine", None))
    moved = previous is not None and _engine(getattr(previous, "engine", None)) != engine
    if moved:
        # What the owner set for one engine does not carry over to another.
        previous = None

    def kept(field: str) -> bool:
        return previous is not None and getattr(previous, field, None) == getattr(spec, field)

    for field, what in OWNER_PROFILE_FIELDS.items():
        if getattr(spec, field, None) and not kept(field):
            raise _owner_field(field, what, moved=moved)
    roots = request.app.state.config_store.model_roots()
    flags = getattr(spec, "flags", None) or {}
    before = (getattr(previous, "flags", None) or {}) if previous is not None else {}
    changed_path: str | None = None
    for key, value in flags.items():
        field = f"flags.{key}"
        kind = flag_class(engine, key)
        if kind == "path" and value not in (None, "") and before.get(key) != value:
            changed_path = field
        if (engine, key) in SOURCE_FLAGS and before.get(key) != value:
            changed_path = field
        if value in (None, "") or before.get(key) == value:
            continue
        if kind == "owner":
            if _is_on(value):
                raise _owner_field(field, OWNER_FLAGS[(engine, key)], moved=moved)
        elif kind == "path":
            _refuse_outside(field, value, roots)
        elif kind == "operations":
            dash = _dash_value(value)
            if dash is not None:
                raise _problem(
                    status.HTTP_403_FORBIDDEN,
                    "Owner only",
                    f'Only the owner can set a value that starts with "-" ({field}): '
                    f"{dash!r} could reach the engine as another setting. Leave it as it "
                    "was, or ask the owner.",
                )
        else:
            name = ENGINE_NAMES.get(engine, engine)
            raise _problem(
                status.HTTP_403_FORBIDDEN,
                "Owner only",
                f"Only the owner can set {key} ({field}): it is not a setting the Models "
                f"power may change for {name}. Leave it out, or ask the owner.",
            )
    for source_engine, key in SOURCE_FLAGS:
        if source_engine == engine and before.get(key) not in (None, "") and key not in flags:
            changed_path = f"flags.{key}"  # removed
    if changed_path is not None:
        # A kept owner flag was the owner's choice for these files and this
        # program.
        for key, value in flags.items():
            owned = OWNER_FLAGS.get((engine, key))
            if owned is not None and _is_on(value):
                raise _problem(
                    status.HTTP_403_FORBIDDEN,
                    "Owner only",
                    f"Only the owner can keep {owned} (flags.{key}) with another file or "
                    f"program ({changed_path}). Turn it off, leave {changed_path} as it "
                    "was, or ask the owner.",
                )


def _refuse_outside(field: str, value: Any, roots: list[Any]) -> None:
    """A path flag's value, refused unless every path in it lies inside a
    Library folder; the refusal says what was observed."""
    values = value if isinstance(value, list) else [value]
    for item in values or [value]:
        if inside_a_folder(item, roots):
            continue
        if not _full_path(item):
            raise _problem(
                status.HTTP_403_FORBIDDEN,
                "Owner only",
                f"Only the owner can set {item!r} ({field}): it is not a full path to a "
                "file or folder on this machine, and a Hugging Face model id is not one. "
                "Use a file inside a Library folder, or ask the owner.",
            )
        raise _owner_field(field, f"a file outside the Library's folders ({item})")


_SEPARATORS = re.compile(r"[\\/]")


def _full_path(value: Any) -> bool:
    """An absolute path; on Windows also with a drive or a UNC share, since
    Python 3.12 takes `\\models\\x.gguf` as absolute there, on whichever
    drive is current."""
    if not isinstance(value, str) or not os.path.isabs(value):
        return False
    return os.name != "nt" or bool(os.path.splitdrive(value)[0])


def inside_a_folder(value: Any, roots: list[Any]) -> bool:
    """Review B1, second pass: a full path (`_full_path`) with no `..` under
    either separator that lies in one of `roots` both as written and after
    this machine resolves its links, so a link inside a folder that points
    out is outside."""
    from .paths import is_within, real_within

    if not _full_path(value):
        return False
    if ".." in _SEPARATORS.split(value):
        return False
    return any(is_within(value, root) and real_within(value, root) for root in roots)


def _owner_field(field: str, what: str, *, moved: bool = False) -> HTTPException:
    if moved:
        return _problem(
            status.HTTP_403_FORBIDDEN,
            "Owner only",
            f"Only the owner can set {what} ({field}) for another engine. "
            "Remove it, or ask the owner.",
        )
    return _problem(
        status.HTTP_403_FORBIDDEN,
        "Owner only",
        f"Only the owner can set {what} ({field}). Leave it as it was, or ask the owner.",
    )


def carry_owner_fields[M: BaseModel](request: Request, spec: M, previous: Any) -> M:
    """Review of library#22: an operations token reads a profile without its
    environment and extra arguments (`scrub_profile`), so the console's save
    sends them empty. On a replace with an operations token, the stored ones
    are kept when the body omits or empties them; `refuse_owner_fields`
    still refuses any other value. A move to another engine with them set
    needs the owner, since they were set for this engine."""
    if operations_claims(request) is None or previous is None:
        return spec
    missing = {
        field: getattr(previous, field)
        for field in OWNER_PROFILE_FIELDS
        if getattr(previous, field, None) and not getattr(spec, field, None)
    }
    if not missing:
        return spec
    if _engine(getattr(spec, "engine", None)) != _engine(getattr(previous, "engine", None)):
        what = " and ".join(OWNER_PROFILE_FIELDS[field] for field in missing)
        raise _problem(
            status.HTTP_403_FORBIDDEN,
            "Owner only",
            f"Only the owner can move this profile to another engine: the owner set {what} "
            f"on it ({', '.join(missing)}). Keep the engine, or ask the owner.",
        )
    return spec.model_copy(update=missing)


def scrub_profile[M: BaseModel](request: Request, profile: M) -> M:
    """Review S6: a profile read with an operations token carries no
    environment values and no extra arguments; either can hold a key."""
    if operations_claims(request) is None:
        return profile
    return profile.model_copy(update={"env": None, "extraArgs": None})


class ReportOperations:
    """Pure ASGI: after a request `operation(...)` marked, tell the agent."""

    def __init__(self, app: ASGIApp, owner: Any) -> None:
        self.app = app
        self.owner = owner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "GET").upper() in READ_METHODS:
            await self.app(scope, receive, send)
            return
        holder: dict[str, Any] = {}
        scope.setdefault("state", {})[STATE_KEY] = holder
        status_code = 500

        async def capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            if holder.get("action"):
                await report(self.owner, {**holder, "status": status_code})


async def report(app: Any, entry: dict[str, Any]) -> None:
    """One row in this machine's operations log, at its agent. A failure
    is logged with the row, so the change is still on record here."""
    agent = getattr(app.state.settings, "agent_url", None)
    token = getattr(app.state.auth_state, "service_token", None)
    if not agent or not token:
        log.info("operations (no agent to record it): %s", entry)
        return
    try:
        client = shared_internal_client(CLIENT_KEY, timeout=httpx.Timeout(5.0))
        answer = await client.post(
            agent.rstrip("/") + "/v1/node/operations-log",
            json=entry,
            headers={"Authorization": "Bearer " + token},
        )
        answer.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("the agent did not record this change (%s): %s", exc, entry)


def install(app: Any) -> None:
    app.add_middleware(ReportOperations, owner=app)
