"""The downloader: destinations, the resume decision, and what gets deleted.

The transfer loop is exercised end to end against a mock transport,
including the case that matters most — a remote file replaced under a
partial download, where appending would produce a file of exactly the
right length and a corrupt model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from eugene_plexus_library import downloads, hub
from eugene_plexus_library._generated.models import (
    DownloadSpec,
    DownloadState,
    ModelFileRole,
)
from eugene_plexus_library.store import StateStore


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry delays are real seconds in production and pure waiting here."""
    monkeypatch.setattr(downloads, "BACKOFF_SECONDS", (0, 0, 0, 0))


BODY = b"GGUF" + b"payload" * 5000
DIGEST = hashlib.sha256(BODY).hexdigest()
SIDECAR = b'{"model_type": "qwen"}'
SIDECAR_OID = hub.git_blob_sha1(SIDECAR)


def tree_payload(body: bytes = BODY, digest: str = DIGEST) -> list[dict]:
    return [
        {
            "type": "file",
            "path": "model.gguf",
            "size": len(body),
            "oid": "0" * 40,
            "lfs": {"oid": digest, "size": len(body)},
        },
        {"type": "file", "path": "config.json", "size": len(SIDECAR), "oid": SIDECAR_OID},
    ]


class FakeHub:
    """A mock transport shaped like the real hub's two redirect forms."""

    def __init__(self) -> None:
        self.body = BODY
        self.digest = DIGEST
        self.range_requests: list[str] = []
        self.ignore_range = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tree/main") or "/tree/" in path:
            # The tree and the resolve endpoint agree, as upstream's do:
            # a replaced file changes both.
            return httpx.Response(200, json=tree_payload(self.body, self.digest))
        if path.startswith("/api/models/"):
            return httpx.Response(200, json={"sha": "commit1", "author": "org"})

        name = path.rsplit("/", 1)[-1]
        payload = self.body if name == "model.gguf" else SIDECAR
        digest = self.digest if name == "model.gguf" else SIDECAR_OID

        if request.method == "HEAD":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://cdn.example/blob",
                    "X-Linked-Size": str(len(payload)),
                    "X-Linked-ETag": f'"{digest}"',
                    "X-Repo-Commit": "commit1",
                    "Accept-Ranges": "bytes",
                },
            )

        rng = request.headers.get("Range")
        if rng:
            self.range_requests.append(rng)
        if rng and not self.ignore_range:
            first = int(rng.split("=")[1].split("-")[0])
            return httpx.Response(206, content=payload[first:])
        return httpx.Response(200, content=payload)


def build_manager(tmp_path: Path, fake: FakeHub) -> tuple[downloads.DownloadManager, StateStore]:
    root = tmp_path / "models"
    root.mkdir(exist_ok=True)
    inner = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), follow_redirects=False)
    client = hub.HubClient(client=inner)
    client.configure(base_url="https://hub.example", token=None, enabled=True)
    store = StateStore(tmp_path / "state.json")
    manager = downloads.DownloadManager(
        client=client,
        store=store,
        scan_manager=None,
        roots=lambda: [root],
        layout=lambda: "publisher_repo",
        concurrency=lambda: 1,
    )
    return manager, store


# -- destinations -------------------------------------------------------


def test_the_default_layout_is_publisher_and_repo(tmp_path: Path) -> None:
    """LM Studio's layout, and the de-facto one — so a download lands
    where a scan of an existing library would already look."""
    root, directory = downloads.resolve_destination(
        repo="unsloth/Qwen3.8-27B-GGUF",
        roots=[tmp_path],
        root=None,
        subdirectory=None,
        layout="publisher_repo",
    )
    assert root == tmp_path
    assert directory == tmp_path / "unsloth" / "Qwen3.8-27B-GGUF"


def test_flat_puts_the_file_in_the_root(tmp_path: Path) -> None:
    _, directory = downloads.resolve_destination(
        repo="unsloth/X", roots=[tmp_path], root=None, subdirectory=None, layout="flat"
    )
    assert directory == tmp_path


def test_a_root_that_was_not_configured_is_refused(tmp_path: Path) -> None:
    """Downloads only ever land in a directory the operator nominated.
    That is the whole rule this milestone exists to honour."""
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_destination(
            repo="org/repo",
            roots=[tmp_path],
            root=str(tmp_path.parent / "elsewhere"),
            subdirectory=None,
            layout="publisher_repo",
        )
    assert raised.value.code == "UnknownRoot"


def test_no_roots_configured_is_a_conflict_not_a_guess(tmp_path: Path) -> None:
    """A fresh install has nowhere to put a model, and inventing a
    directory is exactly what this component refuses to do."""
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_destination(
            repo="org/repo", roots=[], root=None, subdirectory=None, layout="publisher_repo"
        )
    assert raised.value.code == "NoRootsConfigured"
    assert raised.value.status == 409


def test_path_traversal_out_of_the_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(downloads.DownloadError) as raised:
        downloads.resolve_destination(
            repo="org/repo",
            roots=[tmp_path],
            root=None,
            subdirectory="../../etc",
            layout="publisher_repo",
        )
    assert raised.value.code == "PathTraversal"


def test_a_subdirectory_overrides_the_layout(tmp_path: Path) -> None:
    _, directory = downloads.resolve_destination(
        repo="org/repo",
        roots=[tmp_path],
        root=None,
        subdirectory="my models/qwen",
        layout="publisher_repo",
    )
    assert directory == tmp_path / "my models" / "qwen"


# -- the transfer -------------------------------------------------------


async def drain(manager: downloads.DownloadManager, record_id: str) -> object:
    job = manager._jobs[record_id]
    if job.task is not None:
        await job.task
    return job.record


async def test_a_download_verifies_then_renames(tmp_path: Path) -> None:
    """Nothing that exists at `destinationPath` was written unverified."""
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    record = await drain(manager, record.id)

    assert record.state is DownloadState.done
    entry = record.files[0]
    assert entry.verified is True
    landed = Path(entry.destinationPath)
    assert landed.read_bytes() == BODY
    assert not landed.with_name(landed.name + ".part").exists()
    assert entry.role is ModelFileRole.weights


async def test_the_destination_is_known_before_a_byte_moves(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    assert record.destinationDirectory
    assert record.bytesTotal == len(BODY)
    assert record.state is DownloadState.queued
    assert record.resolvedCommit == "commit1"
    await drain(manager, record.id)


async def test_a_sidecar_verifies_by_git_blob_hash(tmp_path: Path) -> None:
    """`config.json` has no `lfs` object and therefore no sha256 — but it
    is not unverifiable, and writing it unchecked would be a choice."""
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf", "config.json"]))
    record = await drain(manager, record.id)
    assert record.state is DownloadState.done
    sidecar = next(f for f in record.files if f.path == "config.json")
    assert sidecar.verified is True
    assert Path(sidecar.destinationPath).read_bytes() == SIDECAR


async def test_a_bad_digest_leaves_a_part_file_and_fails(tmp_path: Path) -> None:
    """A file that fails verification is never renamed into place, and
    never silently retried into the same bytes."""
    fake = FakeHub()
    # Upstream publishes a digest its own bytes do not have — the one
    # case where retrying cannot help, and the partial has to be kept
    # for inspection rather than renamed into place.
    fake.digest = "b" * 64
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    record = await drain(manager, record.id)

    assert record.state is DownloadState.failed
    assert record.errorCode == "DigestMismatch"
    # Retried once from clean, then given up on: re-fetching to compute
    # the same wrong hash a third time is not diligence.
    assert record.attempts == downloads.MAX_DIGEST_ATTEMPTS
    entry = record.files[0]
    landed = Path(entry.destinationPath)
    assert not landed.exists()
    assert landed.with_name(landed.name + ".part").exists()


async def test_a_partial_file_is_resumed_from_its_byte_count(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))

    # Pretend a previous run stopped a third of the way in.
    entry = record.files[0]
    part = Path(entry.destinationPath + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(BODY[:1000])

    record = await drain(manager, record.id)
    assert record.state is DownloadState.done
    assert fake.range_requests == ["bytes=1000-"]
    assert Path(entry.destinationPath).read_bytes() == BODY


async def test_a_file_replaced_upstream_discards_the_partial(tmp_path: Path) -> None:
    """The load-bearing step. Publishers requantize under the same
    filename; appending the tail of the new file to the head of the old
    one gives exactly the right number of bytes and a corrupt model.
    """
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))

    entry = record.files[0]
    part = Path(entry.destinationPath + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"bytes from the model that used to be here")

    # Upstream now serves different content under the same path.
    fake.body = b"GGUF" + b"different" * 4000
    fake.digest = hashlib.sha256(fake.body).hexdigest()
    entry.sha256 = DIGEST  # what the record started with

    record = await drain(manager, record.id)
    assert record.state is DownloadState.done
    # `restartedFromZero` is the durable signal; `message` describes the
    # current phase and has moved on to "complete" by now.
    assert record.restartedFromZero is True
    assert Path(entry.destinationPath).read_bytes() == fake.body
    assert fake.range_requests == []  # started over, no Range


async def test_a_server_ignoring_range_restarts_rather_than_appending(tmp_path: Path) -> None:
    """A 200 to a ranged request means the whole file is coming. Writing
    it after what is on disk would be silent corruption."""
    fake = FakeHub()
    fake.ignore_range = True
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))

    entry = record.files[0]
    part = Path(entry.destinationPath + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(BODY[:500])

    record = await drain(manager, record.id)
    assert record.state is DownloadState.done
    assert Path(entry.destinationPath).read_bytes() == BODY


async def test_an_already_complete_file_is_not_refetched(tmp_path: Path) -> None:
    """Re-downloading a file the operator already has is the one thing
    this whole component exists to avoid."""
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    entry = record.files[0]
    destination = Path(entry.destinationPath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(BODY)

    record = await drain(manager, record.id)
    assert record.state is DownloadState.done
    assert fake.range_requests == []
    assert record.files[0].verified is True


# -- control -------------------------------------------------------------


async def test_cancelling_removes_the_part_but_never_a_finished_file(
    tmp_path: Path,
) -> None:
    """The `.part` is ours; the finished file is the operator's."""
    fake = FakeHub()
    manager, store = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    record = await drain(manager, record.id)
    landed = Path(record.files[0].destinationPath)
    assert landed.exists()

    # A stray partial from something else beside it.
    stray = landed.with_name(landed.name + ".part")
    stray.write_bytes(b"partial")

    await manager.cancel(record.id)
    assert landed.exists()  # the operator's file, untouched
    assert not stray.exists()  # ours, removed
    assert manager.get(record.id) is None
    assert store.get_download(record.id) is None


async def test_pausing_keeps_the_partial_file(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    paused = await manager.pause(record.id)
    assert paused.state is DownloadState.paused
    assert "partial files are kept" in (paused.message or "")


async def test_resume_is_refused_on_a_finished_download(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    await drain(manager, record.id)
    with pytest.raises(downloads.DownloadError) as raised:
        await manager.resume(record.id)
    assert raised.value.status == 409


async def test_two_downloads_cannot_write_the_same_file(tmp_path: Path) -> None:
    """Interleaved appends produce a file of the right length that fails
    verification, with nothing in either record explaining why."""
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    first = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    await manager.pause(first.id)

    with pytest.raises(downloads.DownloadError) as raised:
        await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    assert raised.value.code == "DestinationBusy"


async def test_a_missing_file_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    with pytest.raises(downloads.DownloadError) as raised:
        await manager.start(DownloadSpec(repo="org/repo", files=["not-in-the-repo.gguf"]))
    assert raised.value.code == "FileNotInRepo"
    assert raised.value.status == 404


async def test_renaming_is_refused_for_a_multi_file_download(tmp_path: Path) -> None:
    """Renaming one shard of a set breaks the set."""
    fake = FakeHub()
    manager, _ = build_manager(tmp_path, fake)
    with pytest.raises(downloads.DownloadError) as raised:
        await manager.start(
            DownloadSpec(
                repo="org/repo", files=["model.gguf", "config.json"], filename="renamed.gguf"
            )
        )
    assert "shard" in str(raised.value)


# -- persistence ---------------------------------------------------------


async def test_records_survive_a_restart_as_paused(tmp_path: Path) -> None:
    """A 40 GB transfer interrupted by a restart has to be resumable, and
    a record that claims to be downloading after the process that owned
    it died is describing nothing."""
    fake = FakeHub()
    manager, store = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    record.state = DownloadState.downloading
    store.put_download(record)

    revived = StateStore(tmp_path / "state.json")
    revived.load()
    restored = revived.get_download(record.id)
    assert restored is not None
    assert restored.state is DownloadState.paused
    assert "restarted" in (restored.message or "")


async def test_shutdown_parks_a_transfer_rather_than_losing_it(tmp_path: Path) -> None:
    fake = FakeHub()
    manager, store = build_manager(tmp_path, fake)
    record = await manager.start(DownloadSpec(repo="org/repo", files=["model.gguf"]))
    await manager.shutdown()
    stored = store.get_download(record.id)
    assert stored is not None
    assert stored.state in (DownloadState.paused, DownloadState.done)
