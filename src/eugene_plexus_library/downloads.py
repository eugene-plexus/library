"""Fetching a model into a directory the operator chose.

## One job, N files

A split GGUF is two to nine files, a multimodal model is a quant plus a
projector, and a safetensors model is a directory of which nothing is
optional — `config.json` and the tokenizer files are as necessary as the
weights. "Download this model" is one operator action, so it is one
record with per-file progress inside it.

## The loop, proven end to end before it was written

    1. resolve   HEAD the resolve URL, no redirect following
    2. compare   X-Linked-ETag and X-Linked-Size against what we stored
    3. continue  Range: bytes=<size of .part>-      expect 206
    4. verify    digest over the whole finished file
    5. rename    .part -> final, atomically

Measured against the live hub: 13.6 MB in two legs with the connection
abandoned mid-stream, `Range: bytes=5242880-` answering 206, sha256
matching upstream's own digest, `.part` renamed. Step 2 is the one that
matters — see `_resume_offset`.

## The `.part` is ours; the finished file is the operator's

That single line decides the delete semantics. Cancelling removes the
`.part`, because that file only ever existed as a side effect of a job
just abandoned. Nothing here ever deletes a completed model file — the
same guarantee `DELETE /v1/models/{id}` makes. Pausing keeps the
`.part`, which is why pause and cancel are different verbs.

A `.part` in a scanned root is reported by the scan as
`incomplete_download`, the reason M2 shipped one milestone early for
exactly this.

## Retries are automatic; resume is what a retry does

A 40 GB transfer over a domestic link will meet a transient failure, and
treating that as terminal makes the feature useless. So a failed
transfer re-resolves and continues with backoff, up to a budget, and
only then goes `failed` with the attempt count visible.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import socket
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx

from . import paths as path_utils
from ._generated.models import (
    Download,
    DownloadFile,
    DownloadSpec,
    DownloadState,
    ModelFileRole,
)
from .hub import FileMetadata, HubClient, HubError, git_blob_sha1
from .scan_manager import ScanManager
from .store import StateStore

log = logging.getLogger(__name__)

PART_SUFFIX = ".part"

CHUNK_BYTES = 1024 * 1024

MAX_ATTEMPTS = 5
"""Transfer attempts per download before it goes `failed`. Each one
re-resolves upstream and continues from the bytes on disk, so five
attempts on a large file is five chances to survive a blip, not five
restarts."""

MAX_DIGEST_ATTEMPTS = 2
"""Attempts allowed when the failure was a *digest mismatch*, which is
a different kind of failure from a dropped connection: the bytes on disk
are wrong, so the partial is discarded and the retry starts clean. One
retry covers corruption in flight; beyond that the download is
re-fetching 40 GB to compute the same wrong hash, and upstream
disagreeing with its own published digest is not something more attempts
will fix."""

BACKOFF_SECONDS = (2, 5, 15, 45)

RATE_WINDOW_SECONDS = 10.0
"""Progress rate is measured over a trailing window rather than the
whole transfer: a job that stalled for a minute and then recovered
should read as fast again, because the number is there to tell an
operator whether it is moving *now*."""

DISK_HEADROOM_BYTES = 512 * 1024 * 1024
"""Refuse a download that would leave less than this free. Filling the
operator's disk to zero while fetching a model they can then not load is
a worse outcome than a legible refusal."""


class DownloadError(Exception):
    """A download could not be set up. Carries the status a route returns."""

    def __init__(self, message: str, *, status: int = 400, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _role_for(path: str, *, index: int) -> ModelFileRole:
    name = PurePosixPath(path).name.lower()
    if "mmproj" in name:
        return ModelFileRole.projector
    if name.endswith(".gguf") or name.endswith(".safetensors"):
        return ModelFileRole.weights if index == 0 else ModelFileRole.shard
    if name.endswith(".index.json"):
        return ModelFileRole.index
    if "tokenizer" in name or name in ("vocab.json", "merges.txt"):
        return ModelFileRole.tokenizer
    if name.endswith(".json") or name.endswith(".jinja"):
        return ModelFileRole.config
    return ModelFileRole.other


def resolve_destination(
    *,
    repo: str,
    roots: list[Path],
    root: str | None,
    subdirectory: str | None,
    layout: str,
) -> tuple[Path, Path]:
    """`(root, directory)` for a download. Refuses anything outside a root.

    Root must be one of the configured model roots: this component will
    not write into a directory the operator did not nominate, which is
    the whole of the rule the milestone exists to honour. The default is
    the first configured root, which the `path_list` config type has
    promised since M2.
    """
    if not roots:
        # Where, precisely, and on which machine. The first operator to hit
        # this was browsing from a GPU worker while the library ran in a
        # container on a NAS, and could not tell which host's directory the
        # setting wanted. It wants this host's.
        raise DownloadError(
            "No model directories are configured, so there is nowhere to put a download. "
            "Add one under Config → Library → Model directories (`modelRoots`). Those are "
            f"paths on the machine the library runs on ({socket.gethostname()}) — inside its "
            "container, if it runs in one — and downloads land in the first of them.",
            status=409,
            code="NoRootsConfigured",
        )

    if root is None:
        chosen = roots[0]
    else:
        match = next((r for r in roots if path_utils.same_path(r, root)), None)
        if match is None:
            configured = ", ".join(str(r) for r in roots)
            raise DownloadError(
                f"{root!r} is not one of this library's configured model directories "
                f"({configured}). Downloads only ever land in a directory you nominated.",
                code="UnknownRoot",
            )
        chosen = match

    if subdirectory:
        candidate = (chosen / subdirectory).expanduser()
        if not path_utils.is_within(candidate, chosen):
            raise DownloadError(
                f"{subdirectory!r} resolves outside {chosen}. A download destination has "
                "to stay inside the root it names.",
                code="PathTraversal",
            )
        return chosen, candidate

    if layout == "flat":
        return chosen, chosen

    owner, _, name = repo.partition("/")
    if not name:
        return chosen, chosen / owner
    return chosen, chosen / owner / name


def _part_path(destination: Path) -> Path:
    return destination.with_name(destination.name + PART_SUFFIX)


async def _hash_file(path: Path, algorithm: str) -> str:
    """Digest a finished file off the request path.

    A 40 GB sha256 is tens of seconds of CPU, so it runs in a worker
    thread: the event loop is answering progress polls for this very
    download while this happens.
    """

    def run() -> str:
        if algorithm == "git-blob-sha1":
            # Git's blob hash needs the length in the prefix, so this is
            # a two-pass read for the small non-LFS sidecars — which is
            # free, because they are kilobytes.
            return git_blob_sha1(path.read_bytes())
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
                digest.update(block)
        return digest.hexdigest()

    return await asyncio.to_thread(run)


class DownloadJob:
    """One download's mutable state, and the transfer that advances it."""

    def __init__(
        self,
        *,
        record: Download,
        files: list[FileMetadata],
        client: HubClient,
        store: StateStore,
        scan_manager: ScanManager | None,
    ) -> None:
        self.record = record
        self.metadata = {f.path: f for f in files}
        self._client = client
        self._store = store
        self._scan_manager = scan_manager
        self.task: asyncio.Task[None] | None = None
        self._samples: list[tuple[float, int]] = []

    # -- progress ---------------------------------------------------------

    def _observe(self, total: int) -> None:
        """Record a progress sample and refresh rate and ETA."""
        now = time.monotonic()
        self._samples.append((now, total))
        while len(self._samples) > 2 and now - self._samples[0][0] > RATE_WINDOW_SECONDS:
            self._samples.pop(0)

        self.record.bytesDownloaded = total
        if len(self._samples) >= 2:
            (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
            elapsed = t1 - t0
            if elapsed > 0.5 and b1 > b0:
                rate = (b1 - b0) / elapsed
                self.record.bytesPerSecond = rate
                remaining = (self.record.bytesTotal or 0) - total
                self.record.etaSeconds = int(remaining / rate) if rate > 0 and remaining > 0 else 0

    def _sum_downloaded(self) -> int:
        return sum(f.bytesDownloaded or 0 for f in self.record.files)

    # -- the transfer -------------------------------------------------------

    async def run(self) -> None:
        """Transfer every file, verify each, then find the library entry.

        Cancellation propagates: `pause` and `cancel` both cancel this
        task and then decide what happens to the `.part` files.
        """
        attempt = 0
        digest_attempts = 0
        while True:
            attempt += 1
            self.record.attempts = attempt
            try:
                await self._transfer_all()
            except asyncio.CancelledError:
                raise
            except HubError as exc:
                # Upstream said no in a way that says why. A gated repo
                # or a deleted file will not succeed on a retry, so
                # those stop immediately with the reason intact.
                if exc.status in (400, 403, 404, 409):
                    self._fail(str(exc), code=exc.code)
                    return
                if exc.code == "DigestMismatch":
                    digest_attempts += 1
                    if digest_attempts >= MAX_DIGEST_ATTEMPTS:
                        self._fail(str(exc), code=exc.code)
                        return
                    # The bytes on disk are wrong, so keeping them and
                    # resuming would compute the same wrong hash again.
                    self._discard_partials()
                elif attempt >= MAX_ATTEMPTS:
                    self._fail(str(exc), code=exc.code)
                    return
                await self._backoff(attempt, str(exc))
                continue
            except (httpx.HTTPError, OSError) as exc:
                if attempt >= MAX_ATTEMPTS:
                    self._fail(
                        f"{exc}. Gave up after {attempt} attempts; the bytes already "
                        "fetched are kept, so resuming continues from there.",
                    )
                    return
                await self._backoff(attempt, str(exc))
                continue
            break

        self.record.state = DownloadState.done
        self.record.finishedAt = _now()
        self.record.message = "complete"
        self.record.etaSeconds = 0
        await self._locate_model()

    async def _backoff(self, attempt: int, reason: str) -> None:
        delay = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        self.record.message = (
            f"attempt {attempt} failed ({reason}); retrying in {delay}s from "
            f"{self._sum_downloaded():,} bytes"
        )
        log.warning("download %s: %s", self.record.id, self.record.message)
        await asyncio.sleep(delay)

    def _discard_partials(self) -> None:
        """Throw away every partial file and start the next attempt clean.

        Only for a verification failure: for a dropped connection the
        bytes on disk are good and discarding them would turn a blip
        into a full re-download.
        """
        for entry in self.record.files:
            if entry.state is DownloadState.done:
                continue
            part = _part_path(Path(entry.destinationPath))
            if part.exists():
                with contextlib.suppress(OSError):
                    part.unlink()
            entry.bytesDownloaded = 0
            entry.state = DownloadState.queued

    def _fail(self, reason: str, *, code: str | None = None) -> None:
        self.record.state = DownloadState.failed
        self.record.error = reason
        self.record.errorCode = code
        self.record.finishedAt = _now()
        log.error("download %s failed: %s", self.record.id, reason)

    async def _transfer_all(self) -> None:
        self.record.state = DownloadState.resolving
        self.record.message = "resolving upstream"
        self.record.error = None
        self.record.errorCode = None

        for entry in self.record.files:
            if entry.state is DownloadState.done:
                continue
            await self._transfer_one(entry)

        self.record.state = DownloadState.verifying

    async def _transfer_one(self, entry: DownloadFile) -> None:
        destination = Path(entry.destinationPath)
        part = _part_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if (
            destination.exists()
            and entry.sizeBytes
            and destination.stat().st_size == entry.sizeBytes
        ):
            # Already there at the right size. Verified below rather than
            # trusted, then left exactly as found: re-fetching a file the
            # operator already has is the one thing this whole component
            # is meant to avoid.
            entry.bytesDownloaded = entry.sizeBytes
            await self._verify(entry, destination)
            return

        resolved = await self._client.head_file(
            repo=self.record.repo,
            revision=self.record.resolvedCommit or self.record.revision or "main",
            path=entry.path,
        )
        offset = await self._resume_offset(entry, part, resolved.sha256)
        if resolved.sha256:
            # Authoritative at transfer time. The tree API's digest was
            # read when the download was created, and a file replaced
            # upstream since then would otherwise be verified against
            # the hash of the file that used to be there -- which can
            # never pass, however many times it is retried.
            entry.sha256 = resolved.sha256
        if resolved.size and entry.sizeBytes != resolved.size:
            entry.sizeBytes = resolved.size
            self.record.bytesTotal = sum(f.sizeBytes or 0 for f in self.record.files)

        self.record.state = DownloadState.downloading
        self.record.message = f"downloading {PurePosixPath(entry.path).name}"
        entry.state = DownloadState.downloading

        mode = "ab" if offset else "wb"
        base = self._sum_downloaded() - (entry.bytesDownloaded or 0) + offset
        async with self._client.stream(
            repo=self.record.repo,
            revision=self.record.resolvedCommit or self.record.revision or "main",
            path=entry.path,
            first=offset or None,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise HubError(
                    f"Upstream returned {response.status_code} for {entry.path}.",
                    status=502 if response.status_code >= 500 else response.status_code,
                    code=response.headers.get("X-Error-Code"),
                )
            if offset and response.status_code != 206:
                # The server ignored the Range header and is sending the
                # whole file. Appending it to what is on disk would
                # produce a plausible-sized corrupt file, so start over.
                log.warning(
                    "download %s: upstream ignored Range for %s; restarting the file",
                    self.record.id,
                    entry.path,
                )
                offset = 0
                mode = "wb"
                base = self._sum_downloaded() - (entry.bytesDownloaded or 0)

            written = offset
            with part.open(mode) as handle:
                async for chunk in response.aiter_bytes(CHUNK_BYTES):
                    handle.write(chunk)
                    written += len(chunk)
                    entry.bytesDownloaded = written
                    self._observe(base + written - offset)

        entry.bytesDownloaded = written
        await self._verify(entry, destination, part=part)

    async def _resume_offset(
        self, entry: DownloadFile, part: Path, remote_digest: str | None
    ) -> int:
        """How many bytes on disk can be kept.

        **The load-bearing step.** When the remote digest disagrees with
        what this download recorded, upstream has replaced the file —
        publishers requantize under the same filename — and appending the
        tail of the new file to the head of the old one yields exactly
        the right number of bytes and a corrupt model. The only correct
        move is to discard the partial and say so.
        """
        if not part.exists():
            return 0

        have = part.stat().st_size
        if have == 0:
            return 0

        expected = entry.sha256 or self.metadata.get(entry.path, FileMetadata(entry.path, 0)).sha256
        if expected and remote_digest and expected != remote_digest:
            log.warning(
                "download %s: %s changed upstream (%s -> %s); discarding %d partial bytes",
                self.record.id,
                entry.path,
                expected[:12],
                remote_digest[:12],
                have,
            )
            part.unlink(missing_ok=True)
            entry.sha256 = remote_digest
            entry.bytesDownloaded = 0
            self.record.restartedFromZero = True
            self.record.message = (
                f"{PurePosixPath(entry.path).name} was replaced upstream since this "
                "download started, so the partial file was discarded and it is being "
                "fetched again from the beginning."
            )
            return 0

        if entry.sizeBytes and have > entry.sizeBytes:
            # More on disk than the file has bytes: a previous run wrote
            # something else here. Not resumable in any safe sense.
            log.warning(
                "download %s: %s is larger than expected; starting over",
                self.record.id,
                part.name,
            )
            part.unlink(missing_ok=True)
            return 0

        entry.bytesDownloaded = have
        return have

    async def _verify(
        self, entry: DownloadFile, destination: Path, *, part: Path | None = None
    ) -> None:
        """Digest, then rename. Never the other way round.

        A file that fails stays a `.part` with the record in `failed`,
        so nothing that exists at `destinationPath` was written
        unverified — and a failed verification is never silently retried
        into the same bytes.
        """
        source = part if part is not None else destination
        # `entry.sha256` is kept current from the HEAD that preceded this
        # transfer; the tree metadata is the fallback and the only source
        # for the non-LFS sidecars, which have a git blob hash instead.
        digest: tuple[str, str] | None = None
        if entry.sha256:
            digest = ("sha256", entry.sha256)
        else:
            meta = self.metadata.get(entry.path)
            digest = meta.digest if meta else None

        if digest is None:
            # Upstream published no digest for this file. Size is the
            # only check available, and the record says as much rather
            # than claiming a verification that did not happen.
            entry.verified = False
            if entry.sizeBytes and source.stat().st_size != entry.sizeBytes:
                raise HubError(
                    f"{entry.path} finished at {source.stat().st_size} bytes, expected "
                    f"{entry.sizeBytes}.",
                )
            log.info(
                "download %s: %s has no upstream digest; size checked only",
                self.record.id,
                entry.path,
            )
        else:
            self.record.state = DownloadState.verifying
            self.record.message = f"verifying {PurePosixPath(entry.path).name}"
            algorithm, expected = digest
            actual = await self._hash(source, algorithm)
            if actual != expected:
                entry.error = (
                    f"{algorithm} mismatch: upstream says {expected}, the downloaded file "
                    f"is {actual}. The partial file is kept for inspection and was not "
                    "moved into place."
                )
                entry.state = DownloadState.failed
                raise HubError(entry.error, status=502, code="DigestMismatch")
            entry.verified = True

        if part is not None:
            os.replace(part, destination)
        entry.state = DownloadState.done

    async def _hash(self, path: Path, algorithm: str) -> str:
        return await _hash_file(path, algorithm)

    async def _locate_model(self) -> None:
        """Scan the library so the new model becomes a library entry.

        This is what closes discovery -> download -> library -> profile
        -> launch without the UI polling for a model to appear. The scan
        is incremental, so the cost is a stat per known file plus one
        header read for what just arrived.
        """
        weights = next(
            (f for f in self.record.files if f.role is ModelFileRole.weights),
            self.record.files[0] if self.record.files else None,
        )
        if weights is None or self._scan_manager is None:
            return

        try:
            if not self._scan_manager.running:
                await self._scan_manager.start()
            await self._scan_manager.wait()
        except Exception as exc:  # a scan failure must not fail the download
            log.warning("download %s: post-download scan did not run (%s)", self.record.id, exc)

        target = Path(weights.destinationPath)
        # A safetensors model is a directory; a GGUF is the file itself.
        for candidate in (target, target.parent):
            found = self._store.find_by_path(str(candidate))
            if found is not None:
                self.record.modelId = found.id
                return
        log.info(
            "download %s: %s is on disk but no library entry matched yet; the next scan "
            "will pick it up",
            self.record.id,
            target,
        )


class DownloadManager:
    """The collection, the queue, and the concurrency cap.

    A collection rather than M1's singleton install: several downloads
    can be queued, and `maxConcurrentDownloads` decides how many
    transfer at once. The default is 1 — two 40 GB fetches sharing one
    link both finish later than one after the other, and the progress
    bar people watch is the one they started first.
    """

    def __init__(
        self,
        *,
        client: HubClient,
        store: StateStore,
        scan_manager: ScanManager | None,
        roots: Callable[[], list[Path]],
        layout: Callable[[], str],
        concurrency: Callable[[], int],
    ) -> None:
        self._client = client
        self._store = store
        self._scan_manager = scan_manager
        self._roots = roots
        self._layout = layout
        self._concurrency = concurrency
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = asyncio.Lock()

        for record in store.list_downloads():
            self._jobs[record.id] = DownloadJob(
                record=record,
                files=[],
                client=client,
                store=store,
                scan_manager=scan_manager,
            )

    # -- reporting ----------------------------------------------------------

    def records(self) -> list[Download]:
        """Every record, newest first.

        Named `records` rather than `list`: a method called `list` on a
        class shadows the builtin for every annotation inside that
        class body, which silently turned `list[str]` into a reference
        to this method.
        """
        return sorted(
            (job.record for job in self._jobs.values()),
            key=lambda r: r.startedAt or _now(),
            reverse=True,
        )

    def get(self, download_id: str) -> Download | None:
        job = self._jobs.get(download_id)
        return job.record if job else None

    def _persist(self, record: Download) -> None:
        self._store.put_download(record)

    # -- starting -----------------------------------------------------------

    async def start(self, spec: DownloadSpec) -> Download:
        """Create a record, resolve its destination, and queue it.

        The destination is resolved and reported **before a byte
        moves** — the operator can see where their model is going while
        there is still time to change it.
        """
        if not spec.files:
            raise DownloadError("A download needs at least one file to fetch.")

        root, directory = resolve_destination(
            repo=spec.repo,
            roots=self._roots(),
            root=spec.root,
            subdirectory=spec.subdirectory,
            layout=self._layout(),
        )

        if spec.filename and len(spec.files) > 1:
            raise DownloadError(
                "`filename` renames a single file, and this download has "
                f"{len(spec.files)} of them. Renaming one shard of a set breaks the set, "
                "so the upstream names are kept.",
            )

        revision = spec.revision or "main"
        metadata = await self._file_metadata(spec.repo, revision, list(spec.files))
        commit = await self._pin_commit(spec.repo, revision, list(spec.files))

        entries: list[DownloadFile] = []
        for index, repo_path in enumerate(spec.files):
            meta = metadata[repo_path]
            name = spec.filename if spec.filename else PurePosixPath(repo_path).name
            entries.append(
                DownloadFile(
                    path=repo_path,
                    destinationPath=str(directory / name),
                    sizeBytes=meta.size,
                    bytesDownloaded=0,
                    state=DownloadState.queued,
                    sha256=meta.sha256,
                    role=_role_for(repo_path, index=index),
                )
            )

        total = sum(e.sizeBytes or 0 for e in entries)
        self._check_disk(directory, total)
        self._check_conflicts(entries)

        record = Download(
            id=uuid.uuid4().hex[:12],
            state=DownloadState.queued,
            repo=spec.repo,
            revision=revision,
            resolvedCommit=commit,
            root=str(root),
            destinationDirectory=str(directory),
            files=entries,
            bytesTotal=total,
            bytesDownloaded=0,
            attempts=0,
            startedAt=_now(),
            message=f"queued; {len(entries)} file(s) into {directory}",
        )

        job = DownloadJob(
            record=record,
            files=list(metadata.values()),
            client=self._client,
            store=self._store,
            scan_manager=self._scan_manager,
        )
        async with self._lock:
            self._jobs[record.id] = job
            self._persist(record)
            self._maybe_start_locked()
        return record

    async def _file_metadata(
        self, repo: str, revision: str, wanted: list[str]
    ) -> dict[str, FileMetadata]:
        """Sizes and digests for the requested files, from the tree API.

        One call for the whole repo rather than a HEAD per file: the
        tree is where sizes and `lfs.oid` live, and a 30-file repo would
        otherwise cost 30 requests against the resolver budget.
        """
        tree = await self._client.tree(repo, revision=revision)
        by_path = {f.path: f for f in tree}
        missing = [p for p in wanted if p not in by_path]
        if missing:
            raise DownloadError(
                f"{repo} at {revision} does not contain: {', '.join(missing)}. The file "
                "list comes from the catalogue detail response; a repo re-uploaded since "
                "then may have renamed things.",
                status=404,
                code="FileNotInRepo",
            )
        return {p: by_path[p] for p in wanted}

    async def _pin_commit(self, repo: str, revision: str, files: list[str]) -> str | None:
        """Resolve `main` to a commit and hold it for the whole download.

        `main` moves — repos are requantized and re-uploaded under the
        same filenames — so a transfer that starts on today's `main` and
        resumes next week against a moved branch would splice two
        different files together. Fetching by commit makes that
        impossible rather than merely detectable.
        """
        try:
            resolved = await self._client.head_file(repo=repo, revision=revision, path=files[0])
        except HubError as exc:
            if exc.status in (403, 404):
                raise
            log.warning("could not pin a commit for %s@%s (%s)", repo, revision, exc)
            return None
        return resolved.commit

    def _check_disk(self, directory: Path, needed: int) -> None:
        probe = directory
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            free = shutil.disk_usage(probe).free
        except OSError as exc:
            log.warning("could not read free space on %s (%s)", probe, exc)
            return
        if needed and free < needed + DISK_HEADROOM_BYTES:
            raise DownloadError(
                f"This download needs {needed / 1024**3:.1f} GiB and {probe} has "
                f"{free / 1024**3:.1f} GiB free. Refusing rather than filling the disk: "
                "a model that arrives on a full disk cannot be loaded anyway.",
                status=507,
                code="InsufficientStorage",
            )

    def _check_conflicts(self, entries: list[DownloadFile]) -> None:
        """Refuse a second download writing the same destination.

        Two jobs appending to one `.part` interleave their bytes and
        produce a file of the right length that fails verification, with
        nothing in either record explaining why.
        """
        active = {
            f.destinationPath
            for job in self._jobs.values()
            if job.record.state
            not in (DownloadState.done, DownloadState.failed, DownloadState.cancelled)
            for f in job.record.files
        }
        clash = [e.destinationPath for e in entries if e.destinationPath in active]
        if clash:
            raise DownloadError(
                f"Another download is already writing {clash[0]}. Wait for it, or cancel it first.",
                status=409,
                code="DestinationBusy",
            )

    def _maybe_start_locked(self) -> None:
        """Promote queued jobs up to the concurrency cap."""
        limit = max(1, self._concurrency())
        running = sum(
            1 for job in self._jobs.values() if job.task is not None and not job.task.done()
        )
        for job in sorted(self._jobs.values(), key=lambda j: j.record.startedAt or _now()):
            if running >= limit:
                return
            if job.record.state is not DownloadState.queued:
                continue
            if job.task is not None and not job.task.done():
                continue
            job.task = asyncio.create_task(self._supervise(job))
            running += 1

    async def _supervise(self, job: DownloadJob) -> None:
        try:
            await job.run()
        except asyncio.CancelledError:
            # pause/cancel set the state before cancelling, so there is
            # nothing to decide here.
            raise
        except Exception as exc:  # a job must never take the process down
            log.exception("download %s crashed", job.record.id)
            job.record.state = DownloadState.failed
            job.record.error = str(exc)
            job.record.finishedAt = _now()
        finally:
            self._persist(job.record)
            async with self._lock:
                self._maybe_start_locked()

    # -- control ------------------------------------------------------------

    async def pause(self, download_id: str) -> Download:
        job = self._require(download_id)
        if job.record.state in (DownloadState.done, DownloadState.cancelled):
            raise DownloadError(f"This download is already {job.record.state.value}.", status=409)
        await self._stop(job)
        job.record.state = DownloadState.paused
        job.record.message = (
            f"paused at {job.record.bytesDownloaded or 0:,} of "
            f"{job.record.bytesTotal or 0:,} bytes; the partial files are kept and will "
            "show up in the next scan as incomplete downloads"
        )
        self._persist(job.record)
        return job.record

    async def resume(self, download_id: str) -> Download:
        job = self._require(download_id)
        if job.record.state not in (
            DownloadState.paused,
            DownloadState.failed,
            DownloadState.queued,
        ):
            raise DownloadError(
                f"Only a paused or failed download can be resumed; this one is "
                f"{job.record.state.value}.",
                status=409,
            )

        # Re-read sizes and digests: the point of a resume is to check
        # whether upstream still holds the file this started on.
        if not job.metadata:
            with contextlib.suppress(HubError, DownloadError):
                job.metadata = await self._file_metadata(
                    job.record.repo,
                    job.record.resolvedCommit or job.record.revision or "main",
                    [f.path for f in job.record.files],
                )

        for entry in job.record.files:
            if entry.state is not DownloadState.done:
                entry.state = DownloadState.queued
                entry.error = None
        job.record.state = DownloadState.queued
        job.record.error = None
        job.record.errorCode = None
        job.record.finishedAt = None
        job.record.message = "resuming"
        async with self._lock:
            self._maybe_start_locked()
        self._persist(job.record)
        return job.record

    async def cancel(self, download_id: str) -> None:
        """Stop, remove the `.part` files, and forget the record.

        The `.part` goes because it is ours — created as a side effect
        of a job the operator just abandoned. Anything already finished
        and renamed into place stays: that is the operator's file now,
        and this component does not delete those.
        """
        job = self._require(download_id)
        await self._stop(job)

        removed = 0
        for entry in job.record.files:
            part = _part_path(Path(entry.destinationPath))
            if part.exists():
                with contextlib.suppress(OSError):
                    part.unlink()
                    removed += 1
        log.info(
            "download %s cancelled; removed %d partial file(s), kept every completed one",
            download_id,
            removed,
        )

        async with self._lock:
            self._jobs.pop(download_id, None)
            self._store.delete_download(download_id)
            self._maybe_start_locked()

    async def _stop(self, job: DownloadJob) -> None:
        task = job.task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        job.task = None

    def _require(self, download_id: str) -> DownloadJob:
        job = self._jobs.get(download_id)
        if job is None:
            raise DownloadError(f"No download with id {download_id!r}.", status=404)
        return job

    async def shutdown(self) -> None:
        """Stop every transfer at process exit, keeping the partials.

        A download in flight when the process goes away comes back as
        `paused` on the next start, which is what makes a 40 GB fetch
        survive a restart instead of beginning again.
        """
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.record.state = DownloadState.paused
                job.record.message = "paused by shutdown; resume to continue"
                self._persist(job.record)
                await self._stop(job)
