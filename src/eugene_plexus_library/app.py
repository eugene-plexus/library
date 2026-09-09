"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .downloads import DownloadManager
from .hub import HubClient
from .routes import admin as admin_routes
from .routes import catalogue as catalogue_routes
from .routes import config as config_routes
from .routes import downloads as download_routes
from .routes import guidance as guidance_routes
from .routes import health as health_routes
from .routes import models as model_routes
from .routes import profiles as profile_routes
from .routes import scan as scan_routes
from .scan_manager import ScanManager
from .settings import Settings, load_settings
from .store import StateStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # Auth state is built before the config store so its master key can
    # be threaded in for at-rest decryption. Tests pre-populate
    # `app.state.auth_state`; production reads it from env.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    config_store = ConfigStore(settings.config_file, master_key=app.state.auth_state.master_key)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_LIBRARY_SAFE_MODE=1); ignoring %s and "
            "running on defaults — no model roots, so no scan. Fix config via /v1/config, "
            "then restart without the env var.",
            settings.config_file,
        )
    else:
        try:
            config_store.load()
        # Bad config must never stop startup — see below.
        except Exception as exc:
            # Degraded, not dead. The config endpoints are how an
            # operator repairs a broken config file, so refusing to
            # start over one locks them out of the fix — the exact
            # failure mode this project exists to avoid.
            log.error(
                "config file %s could not be loaded (%s); running on defaults. "
                "Fix it via /v1/config and restart.",
                settings.config_file,
                exc,
            )
    app.state.config_store = config_store
    app.state.safe_mode = settings.safe_mode

    state_store = StateStore(settings.state_file)
    state_store.load()
    app.state.state_store = state_store

    manager = ScanManager(state_store, roots=config_store.model_roots)
    app.state.scan_manager = manager

    # One client for the process, so the connection pool is reused;
    # base URL, token and the enabled flag are re-read from config on
    # every call, so a PATCH takes effect without a restart.
    hub_client = HubClient()
    hub_client.configure(
        base_url=config_store.catalogue_base_url(),
        token=config_store.hf_token(),
        enabled=config_store.catalogue_enabled(),
    )
    app.state.hub_client = hub_client

    app.state.download_manager = DownloadManager(
        client=hub_client,
        store=state_store,
        scan_manager=manager,
        roots=config_store.model_roots,
        layout=config_store.download_layout,
        concurrency=config_store.max_concurrent_downloads,
    )

    # A startup scan is fire-and-forget: the API is up and answering
    # while it runs, which is the whole reason scanning is a background
    # task with its own status resource.
    if not settings.safe_mode and config_store.get("scanOnStartup"):
        roots = config_store.model_roots()
        if roots:
            log.info("scanning %d model %s", len(roots), "root" if len(roots) == 1 else "roots")
            await manager.start()
        else:
            log.info("no model directories configured; skipping the startup scan")

    try:
        yield
    finally:
        # Downloads first: a transfer in flight is parked as `paused`
        # with its partial file intact, and that has to be written to
        # the state file before anything else stops.
        await app.state.download_manager.shutdown()
        await manager.shutdown()
        await hub_client.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with every router mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — library",
        description="The operator's own model directories, scanned and profiled.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors probe it without
    # holding credentials.
    app.include_router(health_routes.router)

    # Baseline for everything else: operator OR any service token. The
    # runtime dashboard resolving a `modelPath` back to a library entry
    # arrives with a service token, so reads have to accept one.
    #
    # Mutations raise the bar to operator-only individually, declared on
    # the route itself so the level is visible next to the handler
    # rather than inferred from which router something was mounted in.
    # Models and scan each mix both kinds, so a per-router level would
    # have let a service token delete an entry or kick off a scan.
    authorized = [Depends(require_authorized)]
    app.include_router(model_routes.router, dependencies=authorized)
    app.include_router(scan_routes.router, dependencies=authorized)
    app.include_router(profile_routes.router, dependencies=authorized)

    # Guidance is read-only and answers questions about this host, so it
    # sits at the same level as the model reads. The catalogue is a read
    # too, but it spends *upstream* rate-limit budget and a token that
    # belongs to the operator, so it is operator-only: a service token
    # is for a component doing its job, and no component's job involves
    # searching a model catalogue.
    app.include_router(guidance_routes.router, dependencies=authorized)

    operator_catalogue = [Depends(require_operator)]
    app.include_router(catalogue_routes.router, dependencies=operator_catalogue)

    # Downloads mix reads and mutations; the mutations are declared on
    # their own routes, so the router keeps the baseline level and the
    # bar is visible next to each handler.
    app.include_router(download_routes.router, dependencies=authorized)

    # Wholly operator-only: config carries the roots, and restart is a
    # process-lifecycle action.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)

    return app
