"""FastAPI dependencies for bearer auth, against this machine's trust bundle.

Same shape as every other component: `require_authorized` for reads,
`require_operator` for everything that writes. Both pass through when
`AuthState.auth_disabled` is true (the dev path). All rejection paths
produce RFC 7807 Problem JSON so the UI renders one error template.

Reads take an operator session addressed to this machine, a service
token from this machine's own components, or an `agent` or `gateway`
token from any machine of the install (per-node token keys, D5): the
agent on a GPU node resolving a model it is about to launch, the
gateway reading a model's profile. **No `service:*` wildcard**: a
driver's token from another machine reads nothing here.

Everything that writes is operator-only, including profile edits: a
profile is the operator's tuning work, and a compromised peer should not
be able to rewrite the flags a model launches with.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import Problem
from .auth_state import AuthState

_bearer_scheme = HTTPBearer(auto_error=False)


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
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


_REMOTE_READERS = frozenset({tokens.SUB_AGENT, tokens.SUB_GATEWAY})


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    reads: bool,
) -> tokens.Claims | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    classes = (tokens.TYP_SESSION, tokens.TYP_SERVICE) if reads else (tokens.TYP_SESSION,)
    try:
        claims = auth.verify(creds.credentials, classes=classes)
    except tokens.TokenError as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e
    if claims.is_service and not (
        claims.is_local_service(str(auth.recipient)) or claims.sub in _REMOTE_READERS
    ):
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong audience",
            f"A {claims.sub!r} service token from {claims.iss!r} may not read the library; "
            "only this machine's components and any machine's agent or gateway may.",
        )
    return claims


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """Reads: a session, this machine's services, or an agent or gateway."""
    return _validate(request, creds, reads=True)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """An operator session only -- for everything that writes."""
    return _validate(request, creds, reads=False)
