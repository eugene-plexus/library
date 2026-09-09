"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable
via `PATCH /v1/config` at runtime. These only control bootstrap: where
the config and state files live, which interface to bind. Once the
config file is loaded, runtime config takes precedence for everything it
covers — model roots are a config field, never an env var, because if
it's tunable the UI exposes it.
"""

from __future__ import annotations

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


def load_settings() -> Settings:
    return Settings()
