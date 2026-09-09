"""The scan singleton: one walk at a time, with phases and progress.

Follows M1's engine-install precedent — a singleton sub-resource rather
than a job collection. Nobody scans two libraries at once, and inventing
job ids for something that has exactly one is worse than not.

## Why it is asynchronous at all

Reading a large-vocab GGUF header costs ~50-60 ms warm and ~10 MB of
disk cold (see `formats.gguf`). A hundred models is minutes on a cold
spinning disk. That does not go on a request the UI is waiting on, so
`POST /v1/scan` starts a walk and returns, and `GET /v1/scan` reports
where it is.

The walk itself is blocking file I/O, so it runs in a worker thread via
`asyncio.to_thread` and the event loop stays free to answer the polling
it exists to serve.

## Cancellation is cooperative and does not roll back

`DELETE /v1/scan` sets a flag the walk checks between files and between
directories. Whatever it already found is kept — a partial scan learned
true things, and discarding them to look tidy would mean an operator who
cancelled a long scan loses the models it had already found. `state`
reports `cancelled` so nobody mistakes a short list for a complete one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ._generated.models import Scan, ScanRoot, ScanState, SkippedPath
from ._generated.models import Status1 as ScanRootStatus
from .scanner import RootResult, ScanCounters, Scanner
from .store import StateStore

log = logging.getLogger(__name__)

# A pathological tree (a symlink loop, a network mount that lists
# forever) must not leave the library reporting `scanning` until it is
# restarted. Generous enough that a genuinely large cold library on a
# spinning disk finishes inside it.
DEFAULT_TIMEOUT_SECONDS = 60 * 60


class ScanManager:
    """Owns the one scan and the state it reports."""

    def __init__(
        self,
        store: StateStore,
        *,
        roots: Callable[[], list[Path]],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._store = store
        self._resolve_roots = roots
        self._timeout = timeout_seconds

        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._cancel = threading.Event()
        self._counters = ScanCounters()

        self._state = ScanState.idle
        self._full = False
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._roots: list[ScanRoot] = []
        self._skipped: list[SkippedPath] = []
        self._added = self._updated = self._missing = 0
        self._error: str | None = None

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> Scan:
        """Current state. Safe to call at any time, including mid-walk —
        the counters are read live so progress actually moves."""
        running = self._state == ScanState.scanning
        return Scan(
            state=self._state,
            full=self._full,
            startedAt=self._started_at,
            finishedAt=self._finished_at,
            roots=list(self._roots),
            currentPath=self._counters.current_path if running else None,
            filesScanned=self._counters.files_scanned,
            modelsFound=(
                self._counters.models_found if running else len(self._store.list_models())
            ),
            added=self._added,
            updated=self._updated,
            missing=self._missing,
            skipped=list(self._skipped),
            error=self._error,
        )

    @property
    def running(self) -> bool:
        return self._state == ScanState.scanning

    # -- lifecycle ----------------------------------------------------------

    async def start(self, *, full: bool = False) -> Scan:
        """Begin a walk. Raises `RuntimeError` if one is already running."""
        async with self._lock:
            if self.running:
                raise RuntimeError("a scan is already running")

            roots = self._resolve_roots()
            self._full = full
            self._started_at = datetime.now(tz=UTC)
            self._finished_at = None
            self._error = None
            self._skipped = []
            self._added = self._updated = self._missing = 0
            self._counters = ScanCounters()
            self._cancel = threading.Event()
            # Every root starts `pending` so the UI can show the list it
            # is about to work through rather than an empty table that
            # fills in from nowhere.
            self._roots = [ScanRoot(path=str(r), status=ScanRootStatus.pending) for r in roots]

            if not roots:
                # Not an error state to have nothing configured — that is
                # a fresh install. But it is a failed *scan*, because
                # there was nothing to scan, and saying so is more useful
                # than reporting success over zero directories.
                self._state = ScanState.failed
                self._finished_at = self._started_at
                self._error = "no model directories are configured"
                return self.snapshot()

            self._state = ScanState.scanning
            self._task = asyncio.create_task(self._run(roots, full=full))
            return self.snapshot()

    async def cancel(self) -> Scan:
        """Ask the walk to stop. Raises `RuntimeError` if none is running."""
        async with self._lock:
            if not self.running:
                raise RuntimeError("no scan is running")
            self._cancel.set()
        return self.snapshot()

    async def shutdown(self) -> None:
        """Stop a walk in flight at process exit, without waiting for it
        to finish a 10 MB header read."""
        self._cancel.set()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            # Shutdown swallows everything: the process is going away,
            # and a scan that failed on the way out is not news.
            with contextlib.suppress(BaseException):
                await task

    # -- the walk -----------------------------------------------------------

    async def _run(self, roots: list[Path], *, full: bool) -> None:
        try:
            scanner = Scanner(
                cache_lookup=(None if full else self._store.lookup),
                should_cancel=self._cancel.is_set,
                counters=self._counters,
            )
            result = await asyncio.wait_for(
                asyncio.to_thread(scanner.scan, roots), timeout=self._timeout
            )
        except TimeoutError:
            self._state = ScanState.failed
            self._error = (
                f"scan exceeded {self._timeout:.0f}s and was abandoned. A symlink loop or an "
                "unresponsive network mount is the usual cause; 'Follow symlinked directories' "
                "is off by default for the first of those."
            )
            self._finished_at = datetime.now(tz=UTC)
            log.error("%s", self._error)
            return
        except asyncio.CancelledError:
            self._state = ScanState.cancelled
            self._finished_at = datetime.now(tz=UTC)
            raise
        # A failed scan must never take the process down with it.
        except Exception as exc:
            self._state = ScanState.failed
            self._error = str(exc)
            self._finished_at = datetime.now(tz=UTC)
            log.exception("scan failed")
            return

        finished = datetime.now(tz=UTC)
        counts = self._store.replace_models(result.models, scanned_at=finished)

        self._roots = [_to_scan_root(r) for r in result.roots]
        self._skipped = result.skipped
        self._added = counts["added"]
        self._updated = counts["updated"]
        self._missing = counts["missing"]
        self._finished_at = finished
        self._state = ScanState.cancelled if self._cancel.is_set() else ScanState.done

        log.info(
            "scan %s: %d models (%d new, %d updated, %d missing), %d files, %d skipped",
            self._state.value,
            len(result.models),
            self._added,
            self._updated,
            self._missing,
            result.files_scanned,
            len(result.skipped),
        )


def _to_scan_root(result: RootResult) -> ScanRoot:
    return ScanRoot(
        path=result.path,
        status=result.status,
        modelsFound=result.models_found,
        filesScanned=result.files_scanned,
        error=result.error,
    )
