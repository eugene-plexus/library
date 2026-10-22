"""Persistent state: separate user-owned profiles and a replaceable model cache.

Three things live here and they are not equally important.

**Profiles are user-owned state.** They are the
operator's tuning work — the flags that took an afternoon to get right —
and a scan cannot reproduce them. `*.profiles.json` stores their versioned
records and model identities separately from the replaceable model cache.
Run intents have a separate journal, managed by `run_operations.py`.

**Download records are persisted for one reason:** a 40 GB transfer
interrupted by a restart has to be resumable, and a record that only
lived in memory would leave a `.part` file on disk with nothing able to
continue it. Anything in flight when the process stops comes back
`paused`, because the transfer that owned it is gone.

That asymmetry decides the behaviour everywhere else:

* An entry survives its file disappearing, as `missing`, because
  dropping it would drop the profiles with it.
* `DELETE /v1/models/{id}` deletes profiles and nothing else, and
  refuses a model that is still present — a present model reappears on
  the next scan, so a delete that looked like it worked would be a lie.
* A moved model reads as a new model. Nothing guesses that a file which
  vanished from one root and appeared in another is the same file,
  because a wrong guess silently applies one model's tuning to another.
  What it does is *say* where the file went (`movedTo`, from the content
  fingerprint) and, when the person clicks, move the profiles there
  (`relocate`), in one transaction and only to a file with the same contents.

## JSON, not YAML

The rest of Eugene Plexus persists YAML, because operators read and edit
it. This file is machine-owned: written on every scan, one record per
model with nested metadata, and nobody hand-edits it. YAML would cost
parse time and gain nothing. `config.yaml` — the part an operator does
edit — stays YAML.

Only the current version is read. What v0.1 saved (the profiles inside the state
file, flash attention as a checkbox) was converted by v0.3.0 and is no longer:
it is refused by name, with the way out. That, unsupported versions and invalid
profile documents make writes unavailable and appear on health; loading never
silently discards them. A corrupt cache is rebuilt by a scan, because the
profiles are in a document of their own.

Writes are atomic (fsynced temp file, then `os.replace`) so a crash mid-write
leaves the previous state rather than a truncated file. Losing a scan
cache to a crash is free; losing the profiles is not.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import fingerprint, profile_storage
from ._causes import plain_cause, v01_data
from ._generated.models import (
    Download,
    DownloadState,
    LibraryModel,
    ModelFileRole,
    ModelFormat,
    ModelProfile,
    ModelProfileSpec,
    ModelStatus,
    OllamaKept,
    RelocateMatch,
)
from ._private_files import write_private_text
from .paths import normalize
from .profile_storage import StorageUnavailable, V01Data
from .scanner import CacheKey

log = logging.getLogger(__name__)

STATE_VERSION = 2


def _now() -> datetime:
    return datetime.now(tz=UTC)


class RelocateRefused(Exception):
    """A relocate that did not happen, and the plain reason (row 22). `status`
    is the HTTP answer: 409 unless an entry is unknown (404)."""

    def __init__(self, title: str, detail: str, *, status: int = 409) -> None:
        super().__init__(detail)
        self.title = title
        self.detail = detail
        self.status = status


@dataclass
class Relocation:
    """What a relocate did: the model the profiles are on now, how many moved,
    the ones that came with a new name, and the stale paths left in them."""

    model: LibraryModel
    moved: int
    renamed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


#: What a `likely` candidate says Eugene could not check.
_NO_FINGERPRINT = "The missing file was never fingerprinted, so its contents cannot be compared."


def _free_name(name: str, taken: set[str]) -> str:
    """`name`, or `name (moved)` when the target model uses it, and `(moved 2)`
    when that is used too."""
    if name not in taken:
        return name
    candidate = f"{name} (moved)"
    number = 2
    while candidate in taken:
        candidate = f"{name} (moved {number})"
        number += 1
    return candidate


def _stale_path_warnings(profile: ModelProfile, old_paths: list[str]) -> list[str]:
    """A profile that names the old file (or one of its files) in its arguments,
    environment or flags moves unchanged, and is listed: Eugene does not edit a
    person's arguments, and the person is told which ones to look at."""

    def names_old(text: str) -> bool:
        flat = text.replace("\\", "/").casefold()
        return any(old.replace("\\", "/").casefold() in flat for old in old_paths if old)

    flag_text = [
        s
        for value in (profile.flags or {}).values()
        for s in (value if isinstance(value, list) else [value])
        if isinstance(s, str)
    ]
    where = [
        ("extra arguments", profile.extraArgs or []),
        ("environment", list((profile.env or {}).values())),
        ("flags", flag_text),
    ]
    return [
        f'The profile "{profile.name}" still names the old path in its {label}.'
        for label, texts in where
        if any(names_old(text) for text in texts)
    ]


class StateStore:
    """Thread-safe owner of the state file.

    One lock over both maps: a scan replacing the model set and a
    profile write from an HTTP handler are genuinely concurrent, and
    they touch the same file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._profiles_path = profile_storage.path_for(path)
        self._storage_error: str | None = None
        self._lock = threading.RLock()
        self._models: dict[str, LibraryModel] = {}
        self._unchecked_ggufs: set[str] = set()
        self._kept: dict[str, OllamaKept] = {}
        self._profiles: dict[str, list[ModelProfile]] = {}
        self._downloads: dict[str, Download] = {}
        self._by_path: dict[str, LibraryModel] = {}
        self._last_scan_at: datetime | None = None

    # -- persistence -----------------------------------------------------

    @property
    def storage_error(self) -> str | None:
        return self._storage_error

    def _preserve(self, path: Path) -> None:
        backup = path.with_name(path.name + ".preserved")
        if path.exists() and not backup.exists():
            data = path.read_bytes()
            fd = os.open(
                backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    def load(self) -> None:
        """Load caches independently; refuse writes if user data is unreadable.

        The first thing found wrong is the reason writes are refused, and it is
        not replaced by whatever else this load finds wrong after it.
        """
        with self._lock:
            self._storage_error = None
            self._models = {}
            self._profiles = {}
            self._downloads = {}
            raw: dict[str, Any] = {}
            if self._path.exists():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("state must be an object")
                    version = raw.get("version")
                    if type(version) is int and version == 1:
                        self._storage_error = v01_data(
                            self._path.name, "version 1 state (profiles kept inside it)"
                        )
                        log.error("%s", self._storage_error)
                        return
                    if type(version) is not int or version != STATE_VERSION:
                        self._storage_error = self._storage_error or (
                            f"Unsupported library state version {version!r}; "
                            "restore matching software and state."
                        )
                        log.error("%s", self._storage_error)
                        return
                except (OSError, ValueError) as exc:
                    log.error("Cannot read state file %s: %s", self._path, exc)
                    self._preserve(self._path)
                    raw = {}
                    if not self._profiles_path.exists():
                        self._storage_error = self._storage_error or (
                            f"{self._path.name} is unreadable: {plain_cause(exc, self._path)}. "
                            "Restore its preserved copy before saving."
                        )
            for entry in raw.get("models") or []:
                try:
                    model = LibraryModel.model_validate(entry)
                except Exception as exc:
                    log.warning("dropping unreadable cached model: %s", exc)
                    continue
                self._models[model.id] = model
            self._kept = {}
            for path, value in (raw.get("keptFrom") or {}).items():
                try:
                    self._kept[normalize(path)] = OllamaKept.model_validate(value)
                except (ValueError, TypeError) as exc:
                    log.warning("Cannot read Keep provenance for %s: %s", path, exc)
            try:
                profiles: dict[str, list[dict[str, Any]]] = {}
                if self._profiles_path.exists():
                    profiles, identities = profile_storage.read(self._profiles_path)
                    for entry in identities:
                        model = LibraryModel.model_validate(entry)
                        if model.id not in self._models:
                            self._models[model.id] = model.model_copy(
                                update={"status": ModelStatus.missing}
                            )
                # Validate all owned records before adopting any: never drop tuning work.
                self._profiles = {
                    model_id: [ModelProfile.model_validate(p) for p in records]
                    for model_id, records in profiles.items()
                }
            except V01Data as exc:
                # v0.1's file is intact and v0.3.0 converts it: no copy to keep.
                self._storage_error = self._storage_error or str(exc)
                self._profiles = {}
            except (OSError, ValueError, KeyError, TypeError, StorageUnavailable) as exc:
                log.error("Cannot read profiles file %s: %s", self._profiles_path, exc)
                if self._profiles_path.exists():
                    self._preserve(self._profiles_path)
                self._storage_error = self._storage_error or (
                    f"Saved profiles are unreadable: {plain_cause(exc, self._profiles_path)}. "
                    "Restore matching data before saving."
                )
                self._profiles = {}
            for entry in raw.get("downloads") or []:
                try:
                    record = Download.model_validate(entry)
                except Exception as exc:
                    log.warning("dropping unreadable download record: %s", exc)
                    continue
                if record.state in (
                    DownloadState.queued,
                    DownloadState.resolving,
                    DownloadState.downloading,
                    DownloadState.verifying,
                ):
                    record.state = DownloadState.paused
                    record.message = (
                        "paused: the process restarted while transferring; resume to continue."
                    )
                self._downloads[record.id] = record
            stamp = raw.get("lastScanAt")
            if isinstance(stamp, str):
                try:
                    self._last_scan_at = datetime.fromisoformat(stamp)
                except ValueError:
                    self._last_scan_at = None
            self._reindex_locked()
            # Cached entries from an older reader have no proof of valid tensor
            # ranges. Revalidate once per process; later scans remain incremental.
            self._unchecked_ggufs = {
                m.id for m in self._models.values() if m.format == ModelFormat.gguf
            }
            if self._storage_error:
                log.error("%s", self._storage_error)

    def _writable(self) -> None:
        if self._storage_error is not None:
            raise StorageUnavailable(self._storage_error)

    def _write_profiles_locked(self) -> None:
        self._writable()
        profile_storage.write(
            self._profiles_path,
            {
                mid: [p.model_dump(mode="json", exclude_none=True) for p in ps]
                for mid, ps in self._profiles.items()
            },
            [
                m.model_dump(mode="json", exclude_none=True)
                for mid, m in self._models.items()
                if mid in self._profiles
            ],
        )

    @contextmanager
    def _profile_transaction(self) -> Iterator[None]:
        with self._lock:
            self._writable()
            previous = copy.deepcopy(self._profiles)
            previous_models = dict(self._models)
            previous_kept = dict(self._kept)
            try:
                yield
                self._write_locked()
                self._write_profiles_locked()
            except BaseException:
                self._profiles = previous
                self._models = previous_models
                self._kept = previous_kept
                self._reindex_locked()
                raise

    def _write_locked(self) -> None:
        self._writable()
        payload = {
            "version": STATE_VERSION,
            "keptFrom": {p: v.model_dump(mode="json") for p, v in self._kept.items()},
            "lastScanAt": self._last_scan_at.isoformat() if self._last_scan_at else None,
            "models": [m.model_dump(mode="json", exclude_none=True) for m in self._models.values()],
            "downloads": [
                d.model_dump(mode="json", exclude_none=True) for d in self._downloads.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(self._path, json.dumps(payload, indent=2))

    # -- the incremental-rescan cache -------------------------------------

    def _reindex_locked(self) -> None:
        """Rebuild the path index after the model set changes.

        Derived rather than persisted: an index that outlived the thing
        it describes is worse than no index, and rebuilding it is one
        pass over a few hundred records.
        """
        self._by_path = {normalize(m.path): m for m in self._models.values()}

    def lookup(self, key: CacheKey) -> LibraryModel | None:
        """Find a stored entry for an unchanged file.

        Compared against the **weights** file, not `sizeBytes`:
        `sizeBytes` is the total across every shard and the projector,
        while the key carries the stat of the one file it was built
        from. Comparing those two would make every sharded or
        multimodal model miss the cache on every scan — which costs
        nothing in a unit test and re-reads gigabytes of headers in the
        field.

        Timestamps compare on whole seconds, because that is what
        survives the round-trip through RFC 3339 in the state file. A
        file rewritten within the same second as its last scan *and* to
        exactly the same size reads as unchanged: an accepted miss, and
        one of the reasons `full: true` exists.
        """
        path, size, mtime_ns = key
        with self._lock:
            model = self._by_path.get(path)
            if model is None or model.status != ModelStatus.present:
                return None
            if model.id in self._unchecked_ggufs:
                return None
            if model.modifiedAt is None:
                return None
            weights = next(
                (f for f in (model.files or []) if f.role == ModelFileRole.weights), None
            )
            if weights is None or weights.sizeBytes != size:
                return None
            if int(model.modifiedAt.timestamp()) != mtime_ns // 1_000_000_000:
                return None
            return model

    # -- models -----------------------------------------------------------

    def list_models(self) -> list[LibraryModel]:
        with self._lock:
            moved = self._moved_to_locked()
            return [self._with_profile_count(m, moved) for m in self._models.values()]

    def get_model(self, model_id: str) -> LibraryModel | None:
        with self._lock:
            model = self._models.get(model_id)
            return self._with_profile_count(model, self._moved_to_locked()) if model else None

    def find_by_path(self, path: str) -> LibraryModel | None:
        with self._lock:
            model = self._by_path.get(normalize(path))
            return self._with_profile_count(model, self._moved_to_locked()) if model else None

    def _with_profile_count(
        self, model: LibraryModel, moved: dict[str, str] | None = None
    ) -> LibraryModel:
        """The model as the API shows it: with its profile count, and for a
        missing one where its file went. Both are worked out here and never
        stored, so neither can outlive what it describes."""
        update: dict[str, Any] = {"profileCount": len(self._profiles.get(model.id, []))}
        if moved is None:
            moved = self._moved_to_locked()
        if model.id in moved:
            update["movedTo"] = moved[model.id]
        return model.model_copy(update=update)

    def _moved_to_locked(self) -> dict[str, str]:
        """`{missing id: present id}` where exactly one present entry has the
        missing entry's fingerprint. Two copies leave the choice to the person."""
        present: dict[str, list[str]] = {}
        for model in self._models.values():
            if model.status == ModelStatus.present and model.fingerprint:
                present.setdefault(model.fingerprint, []).append(model.id)
        return {
            model.id: found[0]
            for model in self._models.values()
            if model.status == ModelStatus.missing
            and model.fingerprint
            and len(found := present.get(model.fingerprint, [])) == 1
        }

    def replace_models(self, found: list[LibraryModel], *, scanned_at: datetime) -> dict[str, int]:
        """Install a completed scan's results.

        Returns per-outcome counts for the scan report. Entries not found
        by this scan become `missing` **if they carry profiles** and are
        dropped otherwise — keeping a profile-less record of a file
        somebody deleted is clutter, keeping one that holds tuning work
        is the whole point.
        """
        with self._lock:
            self._writable()
            previous = dict(self._models)
            now = scanned_at
            added = updated = 0
            next_models: dict[str, LibraryModel] = {}

            for model in found:
                existing = previous.get(model.id)
                kept = self._kept.get(normalize(model.path))
                if kept is not None:
                    model = model.model_copy(update={"keptFrom": kept, "name": kept.servedName})
                elif existing is not None and model.ollama is not None:
                    if existing.name in model.ollama.tags:
                        model = model.model_copy(update={"name": existing.name})
                    elif existing.name != model.name:
                        log.info(
                            "%s is gone from Ollama. This model is now served as %s.",
                            existing.name,
                            model.name,
                        )
                if existing is None:
                    added += 1
                    next_models[model.id] = model.model_copy(
                        update={"firstSeenAt": now, "lastSeenAt": now}
                    )
                    continue
                changed = (
                    existing.status != ModelStatus.present
                    or existing.modifiedAt != model.modifiedAt
                )
                if changed:
                    updated += 1
                next_models[model.id] = model.model_copy(
                    update={"firstSeenAt": existing.firstSeenAt or now, "lastSeenAt": now}
                )

            missing = 0
            for model_id, model in previous.items():
                if model_id in next_models:
                    continue
                if not self._profiles.get(model_id):
                    continue
                missing += 1
                next_models[model_id] = model.model_copy(update={"status": ModelStatus.missing})

            self._models = next_models
            self._unchecked_ggufs.difference_update(m.id for m in found)
            self._last_scan_at = now
            self._reindex_locked()
            self._write_locked()
            return {"added": added, "updated": updated, "missing": missing}

    def add_model(self, model: LibraryModel, *, seen_at: datetime) -> LibraryModel:
        """List one model the library just wrote (a prepared model's
        provenance file) without waiting for the next walk, which will
        find the same file and keep it."""
        with self._lock:
            self._writable()
            existing = self._models.get(model.id)
            first = existing.firstSeenAt if existing and existing.firstSeenAt else seen_at
            self._models[model.id] = model.model_copy(
                update={"firstSeenAt": first, "lastSeenAt": seen_at}
            )
            self._reindex_locked()
            self._write_locked()
            return self._with_profile_count(self._models[model.id])

    def record_kept(
        self, model: LibraryModel, kept: OllamaKept, source_id: str | None
    ) -> LibraryModel:
        """Apply provenance and move profiles only for a proven identical Keep.

        Reuse relocate's transaction and profile collision rules. The source
        remains present in Ollama; only its profiles move to the Library home.
        """
        with self._profile_transaction():
            source = self._models.get(source_id) if source_id else None
            if source is not None:
                self._require_same_file(source, model, model.fingerprint, False)
            self._kept[normalize(model.path)] = kept
            model = model.model_copy(
                update={
                    "name": kept.servedName,
                    "keptFrom": kept,
                    "firstSeenAt": model.firstSeenAt or _now(),
                    "lastSeenAt": _now(),
                }
            )
            self._models[model.id] = model
            if source is not None:
                self.relocate(source.id, model.id, found=model.fingerprint, keep_source=True)
            self._reindex_locked()
            return self._with_profile_count(self._models[model.id])

    def forget_model(self, model_id: str) -> bool:
        """Drop an entry and its profiles; no file is touched here. Forget, for a
        missing model, and Delete once its files are gone (LS8)."""
        with self._profile_transaction():
            if model_id not in self._models and model_id not in self._profiles:
                return False
            removed = self._models.pop(model_id, None)
            if removed is not None:
                self._kept.pop(normalize(removed.path), None)
            self._profiles.pop(model_id, None)
            self._reindex_locked()
            return True

    # -- relocating: a missing model's profiles to its file at a new path ----

    def relocate_candidates(
        self, model_id: str
    ) -> list[tuple[LibraryModel, RelocateMatch, list[str]]]:
        """The present models that may be this missing model's file (row 22):
        `same` for an equal fingerprint, `likely` only when this entry has no
        fingerprint and everything its entry says matches. Nothing else is a
        candidate, and a model that is not missing has none."""
        with self._lock:
            source = self._models.get(model_id)
            if source is None or source.status != ModelStatus.missing:
                return []
            moved = self._moved_to_locked()
            found: list[tuple[LibraryModel, RelocateMatch, list[str]]] = []
            for model in self._models.values():
                if model.status != ModelStatus.present or model.format != source.format:
                    continue
                if source.fingerprint:
                    if model.fingerprint != source.fingerprint:
                        continue
                    found.append((model, RelocateMatch.same, []))
                elif fingerprint.difference(source, model) is None:
                    found.append((model, RelocateMatch.likely, [_NO_FINGERPRINT]))
            found.sort(key=lambda item: (item[1] != RelocateMatch.same, item[0].path))
            return [(self._with_profile_count(m, moved), how, why) for m, how, why in found]

    def check_relocation(self, source_id: str, target_id: str) -> tuple[LibraryModel, LibraryModel]:
        """The two entries, or why they cannot be paired at all: unknown (404),
        the source is still there, the target is not. Reads nothing from disk,
        so the caller asks before it spends a read on a call that cannot work."""
        with self._lock:
            return self._pair_locked(source_id, target_id)

    def _pair_locked(self, source_id: str, target_id: str) -> tuple[LibraryModel, LibraryModel]:
        source = self._models.get(source_id)
        if source is None:
            raise RelocateRefused("Unknown model", f"No model with id {source_id!r}.", status=404)
        target = self._models.get(target_id)
        if target is None:
            raise RelocateRefused("Unknown model", f"No model with id {target_id!r}.", status=404)
        if source.status != ModelStatus.missing:
            raise RelocateRefused(
                "Nothing to move",
                f"This model is still at {source.path}; there is nothing to move.",
            )
        if target.status != ModelStatus.present:
            raise RelocateRefused("File not there", f"{target.path} is no longer there.")
        return source, target

    def relocate(
        self,
        source_id: str,
        target_id: str,
        *,
        found: str | None,
        confirm_likely: bool = False,
        keep_source: bool = False,
    ) -> Relocation:
        """Move a missing model's profiles to the same file at its new path, and
        forget the missing entry (row 22). All of it in one transaction: if the
        profiles cannot be written, nothing has moved.

        `found` is the target's fingerprint **read just now** by the caller, so
        the proof is about the file as it is at the click and not as it was at
        the scan. Refused, with the plain reason (`RelocateRefused`), unless
        the contents match: equal fingerprints, or, for an entry that has none,
        every detail the entries carry and `confirm_likely`.
        """
        with self._profile_transaction():
            if keep_source:
                source, target = self._models[source_id], self._models[target_id]
                if source.ollama is None or target.status != ModelStatus.present:
                    raise RelocateRefused(
                        "Keep refused", "Keep needs a present Ollama source and Library target."
                    )
            else:
                source, target = self._pair_locked(source_id, target_id)
            self._require_same_file(source, target, found, confirm_likely)

            moving = self._profiles.pop(source_id, [])
            kept = self._profiles.setdefault(target_id, [])
            names = {p.name for p in kept}
            ids = {p.id for p in kept}
            # The target keeps its own default; one with no profiles takes the
            # source's, so a model with profiles never has none.
            has_default = any(p.default for p in kept)
            renamed: list[tuple[str, str]] = []
            warnings: list[str] = []
            old_paths = list(dict.fromkeys([source.path, *(f.path for f in source.files or [])]))
            for profile in moving:
                name = _free_name(profile.name, names)
                if name != profile.name:
                    renamed.append((profile.name, name))
                names.add(name)
                profile_id = profile.id if profile.id not in ids else uuid.uuid4().hex[:12]
                ids.add(profile_id)
                moved = profile.model_copy(
                    update={
                        "id": profile_id,
                        "name": name,
                        "default": profile.default and not has_default,
                    }
                )
                kept.append(moved)
                warnings.extend(_stale_path_warnings(moved, old_paths))
            if not kept:
                self._profiles.pop(target_id, None)
            if not keep_source:
                self._models.pop(source_id, None)
            self._reindex_locked()
            model = self._with_profile_count(self._models[target_id])
        return Relocation(model=model, moved=len(moving), renamed=renamed, warnings=warnings)

    @staticmethod
    def _require_same_file(
        source: LibraryModel, target: LibraryModel, found: str | None, confirm_likely: bool
    ) -> None:
        if source.fingerprint:
            if found is None or found != source.fingerprint:
                why = fingerprint.difference(source, target) or "its contents are not the same"
                raise RelocateRefused(
                    "Different file", f"{target.path} is a different file: {why}."
                )
            return
        # No fingerprint to compare: what the entries say can only make it likely.
        difference = fingerprint.difference(source, target)
        if difference is not None:
            raise RelocateRefused(
                "Different file", f"{target.path} is a different file: {difference}."
            )
        if not confirm_likely:
            raise RelocateRefused(
                "Cannot prove it",
                "Eugene cannot prove these are the same file. Confirm to move anyway.",
            )

    @property
    def last_scan_at(self) -> datetime | None:
        with self._lock:
            return self._last_scan_at

    # -- profiles ----------------------------------------------------------

    def list_profiles(self, model_id: str) -> list[ModelProfile]:
        with self._lock:
            profiles = list(self._profiles.get(model_id, []))
        # Default first: it is the one a launch flow reaches for.
        return sorted(profiles, key=lambda p: (not p.default, p.createdAt or _now()))

    def get_profile(self, model_id: str, profile_id: str) -> ModelProfile | None:
        with self._lock:
            for profile in self._profiles.get(model_id, []):
                if profile.id == profile_id:
                    return profile
            return None

    def create_profile(self, model_id: str, spec: ModelProfileSpec) -> ModelProfile:
        """Save a new profile. Raises `ValueError` if the name is taken."""
        with self._profile_transaction():
            existing = self._profiles.setdefault(model_id, [])
            if any(p.name == spec.name for p in existing):
                raise ValueError(f"a profile named {spec.name!r} already exists for this model")

            now = _now()
            # The first profile is the default whether it asked or not:
            # a model with profiles and no default would make every
            # launch flow handle an empty case that need not exist.
            is_default = bool(spec.default) or not existing
            profile = ModelProfile(
                id=uuid.uuid4().hex[:12],
                name=spec.name,
                default=is_default,
                engine=spec.engine,
                mode=spec.mode,
                flags=spec.flags,
                extraArgs=spec.extraArgs,
                env=spec.env,
                notes=spec.notes,
                maxTokens=spec.maxTokens,
                temperature=spec.temperature,
                topP=spec.topP,
                topK=spec.topK,
                minP=spec.minP,
                authorSampling=spec.authorSampling,
                builtBy=spec.builtBy,
                createdAt=now,
                updatedAt=now,
            )
            if is_default:
                self._clear_defaults_locked(model_id, keep=profile.id)
            existing.append(profile)
            return profile

    def replace_profile(
        self, model_id: str, profile_id: str, spec: ModelProfileSpec
    ) -> ModelProfile | None:
        """Whole-document replace. `None` if the profile does not exist,
        `ValueError` if the new name collides with a sibling."""
        with self._profile_transaction():
            profiles = self._profiles.get(model_id)
            if not profiles:
                return None
            index = next((i for i, p in enumerate(profiles) if p.id == profile_id), None)
            if index is None:
                return None
            if any(p.name == spec.name and p.id != profile_id for p in profiles):
                raise ValueError(f"a profile named {spec.name!r} already exists for this model")

            current = profiles[index]
            # Refusing to un-default the only default keeps the "exactly
            # one default" invariant a caller can rely on; clearing it
            # would need a second call to restore.
            is_default = bool(spec.default) or (current.default and len(profiles) == 1)
            # `builtBy` is the one field a replace does not replace by
            # omission: it records a past measurement, not a setting, and
            # every edit path writes a whole profile. Absent keeps it;
            # only an explicit null clears it (library.yaml,
            # `ModelProfileSpec.builtBy`). An edit is what makes the
            # record's numbers stale, and the UI labels them for that
            # rather than losing them.
            built_by = spec.builtBy if "builtBy" in spec.model_fields_set else current.builtBy
            updated = ModelProfile(
                id=current.id,
                name=spec.name,
                default=is_default,
                engine=spec.engine,
                mode=spec.mode,
                flags=spec.flags,
                extraArgs=spec.extraArgs,
                env=spec.env,
                notes=spec.notes,
                maxTokens=spec.maxTokens,
                temperature=spec.temperature,
                topP=spec.topP,
                topK=spec.topK,
                minP=spec.minP,
                authorSampling=spec.authorSampling,
                builtBy=built_by,
                createdAt=current.createdAt,
                updatedAt=_now(),
            )
            profiles[index] = updated
            if is_default:
                self._clear_defaults_locked(model_id, keep=updated.id)
            return updated

    def delete_profile(self, model_id: str, profile_id: str) -> bool:
        with self._profile_transaction():
            profiles = self._profiles.get(model_id)
            if not profiles:
                return False
            remaining = [p for p in profiles if p.id != profile_id]
            if len(remaining) == len(profiles):
                return False

            # Deleting the default promotes the oldest survivor, so a
            # model never ends up with profiles and no default.
            if remaining and not any(p.default for p in remaining):
                oldest = min(range(len(remaining)), key=lambda i: remaining[i].createdAt or _now())
                remaining[oldest] = remaining[oldest].model_copy(update={"default": True})

            if remaining:
                self._profiles[model_id] = remaining
            else:
                self._profiles.pop(model_id, None)
            return True

    def _clear_defaults_locked(self, model_id: str, *, keep: str) -> None:
        profiles = self._profiles.get(model_id, [])
        for index, profile in enumerate(profiles):
            if profile.id != keep and profile.default:
                profiles[index] = profile.model_copy(update={"default": False})

    # -- downloads ---------------------------------------------------------

    def list_downloads(self) -> list[Download]:
        """Every record, live objects included.

        Returned by reference rather than copied, deliberately: the
        download manager mutates a record's progress fields many times a
        second, and a snapshot would either be stale or cost a full model
        copy per chunk written.
        """
        with self._lock:
            return list(self._downloads.values())

    def get_download(self, download_id: str) -> Download | None:
        with self._lock:
            return self._downloads.get(download_id)

    def put_download(self, record: Download) -> None:
        """Insert or update one record and flush the state file.

        Called at phase transitions, not per chunk: writing the state
        file on every megabyte of a 40 GB transfer would be forty
        thousand rewrites of a file whose other contents did not change.
        """
        with self._lock:
            self._writable()
            self._downloads[record.id] = record
            self._write_locked()

    def delete_download(self, download_id: str) -> bool:
        with self._lock:
            self._writable()
            if self._downloads.pop(download_id, None) is None:
                return False
            self._write_locked()
            return True
