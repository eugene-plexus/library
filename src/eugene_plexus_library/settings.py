"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable
via `PATCH /v1/config` at runtime. These only control bootstrap: where
the config and state files live, which interface to bind. Once the
config file is loaded, runtime config takes precedence for everything it
covers — model roots are a config field, never an env var, because if
it's tunable the UI exposes it.

`default_model_roots` is not an exception to that rule; it is the
*default* of that field, for a host whose layout is fixed before any
config exists. A container image knows its models volume is mounted at
`/models` before anyone has opened the UI, and a library that boots with
nowhere to look is useless until someone types that path in — which, in
a container, is the one path the operator cannot see from their own
desk. The UI still shows the value, replaces it like any other, and
clearing the field returns to it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_LIBRARY_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where runtime config is persisted. PATCH /v1/config writes here."""

    state_file: Path = Path("library-state.json")
    """Scanned model records and saved profiles. Machine-owned, hence
    JSON rather than the YAML the operator-facing config uses."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. The gateway is the thing that gets
    exposed; this stays on loopback unless an operator has a reason."""

    safe_mode: bool = False
    """Skip the persisted config at startup and run on built-in
    defaults. Set by the agent as EUGENE_PLEXUS_LIBRARY_SAFE_MODE=1
    after a boot failed on bad config, so the operator can reach
    /v1/config to fix it. PATCH still writes through to `config_file`,
    so the repair survives the next boot."""

    auth_signing_key: str | None = None
    """Base64 32-byte HMAC key from the agent at spawn time. Absent
    means unauthenticated — the dev/standalone path only."""

    service_token: str | None = None
    """Long-lived service JWT. Not consumed: the library makes no
    outbound calls to peers, deliberately."""

    master_key: str | None = None
    """Base64 32-byte secretbox key for at-rest config decryption."""

    default_model_roots: str = ""
    """Model directories to use while the operator has configured none,
    separated by `os.pathsep` (`:` on POSIX, `;` on Windows — exactly as
    PATH is, and for the same reason: a Windows path has a colon in it).

    The default of the `modelRoots` config field, not an override: the
    schema reports it as the field's default, `GET /v1/config` shows it
    as the effective value, an explicit list replaces it, and clearing
    the list (`null` or `[]`) returns to it. It is never written to the
    config file, so the file records only what the operator set and a
    later change to this variable still applies.

    Set by the container image (`/models`, where its models volume is
    mounted) so a fresh container has somewhere to scan and Discover has
    somewhere to put a download with nobody having opened Config. Unset
    everywhere else: on bare metal nothing but the operator knows where
    the models are, and inventing a directory would be exactly the
    managed store this project refuses to be."""

    def default_roots(self) -> list[str]:
        """The variable split and cleaned: blanks dropped, order kept."""
        return [part.strip() for part in self.default_model_roots.split(os.pathsep) if part.strip()]


def load_settings() -> Settings:
    return Settings()
