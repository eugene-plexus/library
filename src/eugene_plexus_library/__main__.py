"""Entrypoint: `python -m eugene_plexus_library`."""

from __future__ import annotations

import contextlib
import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

# Bind port when the watchdog doesn't supply one — i.e. standalone runs.
# Under supervision the topology owns it, which is the only source of
# truth: a port in this component's own config would be a second one,
# and the watchdog spawning at 8082 while the config claims 9000 is a
# failure nobody can see from either side.
_DEFAULT_PORT = 8082


def _resolve_port() -> int:
    env_port = os.environ.get("EUGENE_PLEXUS_LIBRARY_BIND_PORT")
    return int(env_port) if env_port else _DEFAULT_PORT


def main() -> None:
    settings = load_settings()

    # Bootstrap the config store only to discover the log level. The
    # real one is built in the lifespan, where the master key exists.
    bootstrap = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        # A broken config must not stop logging from being configured;
        # the lifespan reports it properly a moment later.
        with contextlib.suppress(Exception):
            bootstrap.load()

    log_level = str(bootstrap.get("logLevel") or "INFO").upper()
    # uvicorn configures only its own loggers, so without this our
    # warnings arrive bare and untimestamped. force=True overrides any
    # basicConfig uvicorn already applied.
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=_resolve_port(),
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
