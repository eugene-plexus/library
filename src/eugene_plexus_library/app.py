"""FastAPI app factory."""

from __future__ import annotations

import asyncio
import logging
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from ._causes import plain_cause
from .auth_state import load_auth_state
from .catalogue_sources import HubClients
from .config import DEFAULT_ROOTS_VARIABLE, ConfigStore
from .dependencies import require_authorized, require_operator
from .downloads import DownloadManager
from .keep import KeepError, Keeps
from .node_ollama import NodeStores
from .profile_storage import StorageUnavailable
from .routes import admin as admin_routes
from .routes import catalogue as catalogue_routes
from .routes import config as config_routes
from .routes import directories as directory_routes
from .routes import downloads as download_routes
from .routes import guidance as guidance_routes
from .routes import health as health_routes
from .routes import models as model_routes
from .routes import ollama as ollama_routes
from .routes import profiles as profile_routes
from .routes import run_operations as run_routes
from .routes import scan as scan_routes
from .run_operations import Journal
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
            trust_bundle_file=settings.trust_bundle_file,
            trust_authority=settings.trust_authority,
            auth_recipient=settings.auth_recipient,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    config_store = ConfigStore(
        settings.config_file,
        master_key=app.state.auth_state.master_key,
        default_roots=settings.default_roots(),
    )
    # Why the config was not read, when it was not. `/healthz` says it
    # (`configError`) in a sentence that names the file by its name, not its path.
    app.state.config_error = None
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
            app.state.config_error = (
                f"{settings.config_file.name} could not be loaded "
                f"({plain_cause(exc, settings.config_file)}). Running on defaults. "
                "Repair the config and restart."
            )
            log.error(
                "config file %s could not be loaded (%s); running on defaults. "
                "Fix it via /v1/config and restart.",
                settings.config_file,
                exc,
            )
    app.state.config_store = config_store
    app.state.safe_mode = settings.safe_mode
    _announce_default_roots(config_store)

    # A state the store cannot read (v0.1's included) leaves it refusing writes
    # and naming why on /healthz (storageError): never a crash.
    state_store = StateStore(settings.state_file)
    state_store.load()
    app.state.state_store = state_store
    app.state.keeps = Keeps(config_store, state_store)
    app.state.node_ollama = NodeStores(settings.state_file.with_suffix(".ollama.json"))

    manager = ScanManager(
        state_store,
        roots=config_store.model_roots,
        follow_symlinks=lambda: bool(config_store.get("followSymlinks")),
        ollama_roots=config_store.ollama_roots,
    )
    app.state.scan_manager = manager

    # One HTTP client for the process, so the connection pool is reused,
    # and one hub client per hub source over it (LS4); address, token and
    # the enabled flags are re-read from config on every call, so a PATCH
    # takes effect without a restart.
    hub_clients = HubClients(config_store.catalogue_sources, config_store.catalogue_enabled)
    app.state.hub_clients = hub_clients

    app.state.download_manager = DownloadManager(
        clients=hub_clients,
        store=state_store,
        scan_manager=manager,
        roots=config_store.writable_roots,
        read_only_roots=config_store.ollama_roots,
        layout=config_store.download_layout,
        concurrency=config_store.max_concurrent_downloads,
    )
    app.state.run_operations = Journal(settings.state_file.with_suffix(".runs.sqlite3"))
    app.state.run_download_lock = asyncio.Lock()
    run_task = (
        None if settings.safe_mode else asyncio.create_task(run_routes.advance_downloads(app))
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
        if run_task is not None:
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        # Downloads first: a transfer in flight is parked as `paused`
        # with its partial file intact, and that has to be written to
        # the state file before anything else stops.
        await app.state.download_manager.shutdown()
        await app.state.keeps.shutdown()
        await manager.shutdown()
        await hub_clients.aclose()


def _announce_default_roots(config_store: ConfigStore) -> None:
    """Say where the model directories came from when it was not the
    operator, and say it plainly when one of them is not there.

    In a container the missing case has exactly one meaning -- nothing
    is mounted at that path -- and the scan will go on to report the
    root as missing and health as degraded. This line is the sentence
    that explains those, in the log an operator reads when the library
    is empty.
    """
    if not config_store.default_roots:
        return
    roots = config_store.default_roots
    if not config_store.roots_are_defaulted():
        log.info(
            "model directories are configured (%s); the environment's default (%s) is not in use",
            ", ".join(str(r) for r in config_store.configured_roots()),
            ", ".join(roots),
        )
        return
    log.info(
        "model directories default to %s (%s); set them under Config to use others",
        ", ".join(roots),
        DEFAULT_ROOTS_VARIABLE,
    )
    for root in roots:
        # A system install runs as Eugene's account, which may not be able
        # to traverse the person's home. This diagnostic must not kill the
        # service that provides the folder picker and configuration repair.
        try:
            mode = Path(root).expanduser().stat().st_mode
        except FileNotFoundError:
            log.warning(
                "%s does not exist -- in a container that means nothing is mounted there. "
                "The scan will report it missing until the directory is mounted.",
                root,
            )
        except OSError as exc:
            log.warning(
                "model directory %s could not be accessed: %s. "
                "The Library will stay available. Check access for the account running "
                "Eugene, including traversal of parent directories, or choose another "
                "folder under Library -> Folders.",
                root,
                exc,
            )
        else:
            if not stat.S_ISDIR(mode):
                log.warning(
                    "%s is not a directory. Choose a model folder under Library -> Folders.",
                    root,
                )


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

    @app.exception_handler(KeepError)
    async def keep_error(request: Request, exc: KeepError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={
                "detail": {
                    "title": exc.code,
                    "status": exc.status,
                    "detail": exc.detail,
                    "component": "library",
                }
            },
        )

    @app.exception_handler(StorageUnavailable)
    async def storage_unavailable(request: Request, exc: StorageUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

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
    app.include_router(ollama_routes.router)
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
    app.include_router(run_routes.router)

    # Wholly operator-only: config carries the roots, and restart is a
    # process-lifecycle action. The directory listing (M11) sits with
    # them: it is the picker behind the roots field, and it walks the
    # operator's disk on request, which no component's job involves.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(directory_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)

    return app
