"""Keep immutable Ollama bytes in a nominated Library folder.

Jobs own every destination and expected digest. A sender supplies only a role
and an offset; neither remote paths nor remote URLs participate in placement.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import os
import re
import shutil
import threading
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from . import fingerprint
from ._generated.models import KeepJob, LibraryModel, Method, Method1, OllamaKept, State
from .config import ConfigStore
from .downloads import _in_store, _refuse_store
from .formats.ollama import blob_path
from .paths import is_within, model_id, real_within, same_path
from .scanner import Scanner
from .store import StateStore

CHUNK = 16 * 1024 * 1024
log = logging.getLogger(__name__)


class KeepError(Exception):
    def __init__(self, status: int, detail: str, code: str = "Keep refused") -> None:
        self.status, self.detail, self.code = status, detail, code
        super().__init__(detail)


@dataclass
class File:
    path: Path
    digest: str
    size: int
    source: Path | None = None
    verified: bool = False
    published: bool = False

    @property
    def partial(self) -> Path:
        return self.path.with_name(self.path.name + ".partial")


@dataclass
class Job:
    public: KeepJob
    root: Path
    provenance: OllamaKept
    files: dict[str, File]
    source_id: str | None = None
    lock: Any = field(default_factory=threading.RLock)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            value.update(block)
    return "sha256:" + value.hexdigest()


class Keeps:
    def __init__(self, config: ConfigStore, store: StateStore) -> None:
        self.config, self.store = config, store
        self.jobs: dict[str, Job] = {}
        self.tasks: set[asyncio.Task] = set()
        self.lock = threading.RLock()
        self.remove_orphan_partials()

    def remove_orphan_partials(self) -> None:
        """Discard interrupted Keep files; never traverse links or other layouts."""
        with self.lock:
            owned = {
                file.partial
                for job in self.jobs.values()
                if job.public.state == "running"
                for file in job.files.values()
            }
            for root in self.config.writable_roots():
                root = root.absolute()
                directory = root / "ollama"
                try:
                    self.guard_path(root, directory)
                    if not directory.is_dir():
                        continue
                    children = list(directory.iterdir())
                except (KeepError, OSError) as exc:
                    log.warning("Keep partials in %s were left in place: %s", directory, exc)
                    continue
                for child in children:
                    name = child.name
                    if not re.fullmatch(r"[A-Za-z0-9._-]{1,126}", name) or name != name.strip(".-"):
                        continue
                    for filename in (name + ".gguf.partial", "mmproj-" + name + ".gguf.partial"):
                        path = child / filename
                        if path in owned:
                            continue
                        try:
                            self.guard_path(root, path)
                            if path.is_file():
                                path.unlink()
                        except (KeepError, OSError) as exc:
                            log.warning("Keep partial %s was left in place: %s", path, exc)

    def get(self, id: str) -> Job:
        if id not in self.jobs:
            raise KeepError(404, "This Keep job is not listed.")
        return self.jobs[id]

    def guard(self, job: Job, path: Path) -> None:
        self.guard_path(job.root, path)

    def guard_path(self, root: Path, path: Path) -> None:
        protected = _in_store(path, self.config.ollama_roots())
        if protected is not None:
            exc = _refuse_store(path, protected)
            raise KeepError(409, str(exc), "ReadOnlyFolder")
        if not is_within(path, root) or not real_within(path, root):
            raise KeepError(409, f"{path} is outside the Library folder {root}.")
        # Refuse links and junctions below the nominated root, even ones that
        # stay inside it. This also protects an already kept hard link from
        # ever being opened for a retry's writes.
        current = path
        while not same_path(current, root):
            if current.is_symlink() or os.path.isjunction(current):
                raise KeepError(409, f"{current} is a link. Nothing was written.")
            current = current.parent

    def reserve(
        self,
        *,
        name: str,
        tags: list[str],
        digest: str,
        size: int,
        store: str,
        folder: str | None,
        node: str | None = None,
        projector_digest: str | None = None,
        projector_size: int = 0,
        source: Path | None = None,
        projector: Path | None = None,
        source_id: str | None = None,
    ) -> Job:
        roots = self.config.writable_roots()
        if folder is not None and (
            protected := _in_store(Path(folder), self.config.ollama_roots())
        ):
            exc = _refuse_store(Path(folder), protected)
            raise KeepError(409, str(exc), "ReadOnlyFolder")
        root = next((r for r in roots if folder is None or same_path(r, folder)), None)
        if root is None:
            raise KeepError(409, "Choose a writable Library folder before keeping this model.")
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-")[:120] or "model"
        # Prefix reserved Windows device names as well as dot components.
        if safe.split(".")[0].upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"COM{i}" for i in range(10)],
            *[f"LPT{i}" for i in range(10)],
        }:
            safe = "model-" + safe
        target = root.absolute() / "ollama" / safe / (safe + ".gguf")
        method = Method1.copy if node else Method1.link
        public = KeepJob(
            id=uuid.uuid4().hex,
            model=None,
            state=State.running,
            method=method,
            bytesDone=0,
            bytesTotal=size + projector_size,
            target=str(target),
            error=None,
            digest=digest,
            node=node,
        )
        provenance = OllamaKept(
            servedName=name,
            tags=tags,
            digest=digest,
            store=store,
            node=node,
            keptAt=datetime.now(UTC),
            method=Method(method),
        )
        files = {"weights": File(target, digest, size, source)}
        if projector_digest:
            files["projector"] = File(
                target.with_name("mmproj-" + safe + ".gguf"),
                projector_digest,
                projector_size,
                projector,
            )
        job = Job(public, root.absolute(), provenance, files, source_id)
        with self.lock:
            for file in files.values():
                self.guard(job, file.path)
                self.guard(job, file.partial)
                for path in (file.path, file.partial):
                    if path.exists():
                        raise KeepError(409, f"{path} already exists. Nothing was written.")
            if any(
                same_path(other.public.target, target) and other.public.state == "running"
                for other in self.jobs.values()
            ):
                raise KeepError(409, f"{target} is already being kept.")
            self.jobs[public.id] = job
        return job

    def local(self, model: LibraryModel, folder: str | None) -> Job:
        if model.ollama is None or model.status != "present":
            raise KeepError(409, "Keep needs a present model in Ollama.")
        source = Path(model.path)
        digest = "sha256:" + source.name.removeprefix("sha256-")
        root = cast(str, model.root)
        if blob_path(Path(root), digest) != source:
            raise KeepError(409, "The source is not an Ollama blob.")
        projector = next((Path(f.path) for f in model.files or [] if f.role == "projector"), None)
        return self.reserve(
            name=model.name,
            tags=model.ollama.tags,
            digest=digest,
            size=source.stat().st_size,
            store=root,
            folder=folder,
            source=source,
            projector=projector,
            projector_digest=(
                "sha256:" + projector.name.removeprefix("sha256-") if projector else None
            ),
            projector_size=projector.stat().st_size if projector else 0,
            source_id=model.id,
        )

    def launch(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def cleanup(self, job: Job) -> None:
        for file in job.files.values():
            for path in [file.partial, *([file.path] if file.published else [])]:
                # A changed junction is refused, including during cancellation.
                self.guard(job, path)
                path.unlink(missing_ok=True)

    def fail(self, job: Job, cause: str) -> None:
        job.public.state = type(job.public.state).failed
        job.public.error = cause
        try:
            self.cleanup(job)
        except (KeepError, OSError) as exc:
            job.public.error += f" Cleanup refused: {exc}"

    def cancel(self, job: Job) -> KeepJob:
        with job.lock:
            if job.public.state == "running":
                job.public.state = type(job.public.state).cancelled
                try:
                    self.cleanup(job)
                except (KeepError, OSError) as exc:
                    job.public.error = f"Keep cancelled. Cleanup refused: {exc}"
            return job.public

    def publish(self, job: Job, file: File) -> None:
        self.guard(job, file.path)
        self.guard(job, file.partial)
        # Exclusive creation fences an existing kept inode. It is never opened
        # for writing. Replace only this newly created empty reservation.
        fd = os.open(file.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        file.published = True
        os.replace(file.partial, file.path)
        file.verified = True

    async def run_local(self, job: Job) -> None:
        try:
            await asyncio.to_thread(self.transfer_local, job)
            if job.public.state == "running":
                await asyncio.to_thread(self.finish, job)
        except Exception as exc:
            with job.lock:
                self.fail(job, str(exc))

    def transfer_local(self, job: Job) -> None:
        for file in job.files.values():
            with job.lock:
                if job.public.state != "running":
                    return
                self.guard(job, file.path)
                self.guard(job, file.partial)
                file.path.parent.mkdir(parents=True, exist_ok=True)
            assert file.source is not None
            with job.lock:
                if job.public.state != "running":
                    return
                self.guard(job, file.partial)
                try:
                    os.link(file.source, file.partial)
                except OSError as exc:
                    job.public.method = type(job.public.method).copy
                    job.provenance.method = type(job.provenance.method).copy
                    job.public.note = f"Copied, not linked: {exc.strerror or exc}."
                    free = shutil.disk_usage(job.root).free
                    if free < file.size:
                        raise KeepError(
                            507,
                            f"Not kept: {job.provenance.servedName} "
                            f"needs {file.size / 1e9:.1f} GB, "
                            f"and {job.root} has {free / 1e9:.1f} GB free.",
                        ) from exc
            if not file.partial.exists():
                copied_digest = hashlib.sha256()
                with file.source.open("rb") as incoming:
                    while True:
                        block = incoming.read(CHUNK)
                        if not block:
                            break
                        with job.lock:
                            if job.public.state != "running":
                                return
                            self.guard(job, file.partial)
                            with file.partial.open(
                                "xb" if not file.partial.exists() else "ab"
                            ) as output:
                                output.write(block)
                            copied_digest.update(block)
                            job.public.bytesDone += len(block)
                if file.size == 0:
                    with file.partial.open("xb"):
                        pass
                if "sha256:" + copied_digest.hexdigest() != file.digest:
                    raise KeepError(
                        409,
                        "The copy did not match Ollama's file (SHA-256 differs). Nothing was kept.",
                    )
            with job.lock:
                if job.public.state != "running":
                    return
                self.verify(job, file)
                file.verified = True
                job.public.bytesDone = sum(f.size for f in job.files.values() if f.verified)

    def verify(self, job: Job, file: File) -> None:
        self.guard(job, file.partial)
        if file.partial.stat().st_size != file.size or sha256(file.partial) != file.digest:
            raise KeepError(
                409,
                "The copy did not match Ollama's file (SHA-256 differs or size changed). "
                "Nothing was kept.",
            )

    def finish(self, job: Job) -> None:
        with job.lock:
            if job.public.state != "running":
                return
            for file in job.files.values():
                self.guard(job, file.path)
                self.verify(job, file)
            for file in job.files.values():
                self.publish(job, file)
            result = Scanner().scan([Path(job.public.target).parent])
            model = next((m for m in result.models if m.id == model_id(job.public.target)), None)
            if model is None or model.status != "present":
                raise KeepError(409, "The kept GGUF could not be read by the Library.")
            model = fingerprint.stamp(model).model_copy(update={"root": str(job.root)})
            job.public.model = self.store.record_kept(model, job.provenance, job.source_id)
            job.public.bytesDone = job.public.bytesTotal
            job.public.state = type(job.public.state).done

    def file(self, job: Job, role: str) -> File:
        if role not in job.files:
            raise KeepError(409, "This file role is not part of this Keep job.")
        return job.files[role]

    def state(self, job: Job, file: File) -> dict:
        self.guard(job, file.path)
        self.guard(job, file.partial)
        return {
            "role": next(k for k, v in job.files.items() if v is file),
            "receivedBytes": file.partial.stat().st_size
            if file.partial.exists() and not file.verified
            else 0,
            "sizeBytes": file.size if file.verified else None,
            "sha256": file.digest[7:] if file.verified else None,
        }

    def write(self, job: Job, file: File, offset: int, data: bytes) -> dict:
        with job.lock:
            try:
                return self._write(job, file, offset, data)
            except OSError as exc:
                full = exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112
                cause = f"The disk holding {job.root} is full" if full else str(exc)
                self.fail(job, cause)
                raise KeepError(507 if full else 409, cause) from exc
            except KeepError as exc:
                if exc.status == 507:
                    self.fail(job, exc.detail)
                raise

    def _write(self, job: Job, file: File, offset: int, data: bytes) -> dict:
        with job.lock:
            if job.public.state != "running" or file.verified:
                raise KeepError(409, "This Keep file is no longer receiving bytes.")
            self.guard(job, file.path)
            self.guard(job, file.partial)
            have = file.partial.stat().st_size if file.partial.exists() else 0
            if file.partial.exists() and file.partial.stat().st_nlink > 1:
                raise KeepError(
                    409,
                    "This partial file shares its bytes with another file. Nothing was written.",
                )
            if offset not in (0, have) or offset + len(data) > file.size:
                raise KeepError(409, "The offset or chunk exceeds the expected file length.")
            file.path.parent.mkdir(parents=True, exist_ok=True)
            if not file.partial.exists() and shutil.disk_usage(job.root).free < file.size:
                raise KeepError(507, "There is not enough free space in the Library folder.")
            with file.partial.open("wb" if offset == 0 else "r+b") as output:
                output.seek(offset)
                output.write(data)
            job.public.bytesDone = sum(
                f.size if f.verified else f.partial.stat().st_size if f.partial.exists() else 0
                for f in job.files.values()
            )
            return self.state(job, file)

    def complete(self, job: Job, file: File) -> dict:
        with job.lock:
            if file.verified and job.public.state in ("running", "done"):
                return self.state(job, file)
            if job.public.state != "running":
                raise KeepError(409, "This Keep is no longer running.")
            try:
                self.verify(job, file)
                file.verified = True
                if all(f.verified for f in job.files.values()):
                    self.finish(job)
                return self.state(job, file)
            except Exception as exc:
                self.fail(job, str(exc))
                if isinstance(exc, KeepError):
                    raise
                raise KeepError(409, str(exc)) from exc

    async def shutdown(self) -> None:
        for job in self.jobs.values():
            self.cancel(job)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
