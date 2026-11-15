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
from collections.abc import Callable
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
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
            return claims
        owner = require_operator(request, creds)
        if owner is not None and listed is not None:
            _mark(request, owner, listed[0], listed[1])
        return owner

    return require_operation


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
