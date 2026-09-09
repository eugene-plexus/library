"""Persistent state: the model cache, the profile store, and downloads.

Three things live here and they are not equally important.

**Profiles are the only thing this component owns.** They are the
operator's tuning work — the flags that took an afternoon to get right —
and nothing on disk can reproduce them. Everything else here is a cache
of what a scan found and can be rebuilt by scanning again.

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

## JSON, not YAML

The rest of Eugene Plexus persists YAML, because operators read and edit
it. This file is machine-owned: written on every scan, one record per
model with nested metadata, and nobody hand-edits it. YAML would cost
parse time and gain nothing. `config.yaml` — the part an operator does
edit — stays YAML.

Writes are atomic (temp file, then `os.replace`) so a crash mid-write
leaves the previous state rather than a truncated file. Losing a scan
cache to a crash is free; losing the profiles is not.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._generated.models import (
    Download,
    DownloadState,
    LibraryModel,
    ModelFileRole,
    ModelProfile,
    ModelProfileSpec,
    ModelStatus,
)
from .paths import normalize
from .scanner import CacheKey

log = logging.getLogger(__name__)

STATE_VERSION = 1


def _now() -> datetime:
    return datetime.now(tz=UTC)


class StateStore:
    """Thread-safe owner of the state file.

    One lock over both maps: a scan replacing the model set and a
    profile write from an HTTP handler are genuinely concurrent, and
    they touch the same file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._models: dict[str, LibraryModel] = {}
        self._profiles: dict[str, list[ModelProfile]] = {}
        self._downloads: dict[str, Download] = {}
        self._by_path: dict[str, LibraryModel] = {}
        self._last_scan_at: datetime | None = None

    # -- persistence -----------------------------------------------------

    def load(self) -> None:
        """Read the state file if it exists.

        A corrupt or unreadable state file is logged and treated as
        empty rather than raised. The component must start: its config
        endpoints are how an operator fixes things, and refusing to boot
        over a damaged *cache* would be the degraded-mode rule broken
        for the least important data in the system. Profiles can be lost
        this way, which is why the write is atomic in the first place.
        """
        with self._lock:
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.error(
                    "state file %s is unreadable (%s); starting empty. "
                    "Saved profiles in it are not recoverable by this process — "
                    "the file is left in place, not overwritten, until something is saved.",
                    self._path,
                    exc,
                )
                return
            if not isinstance(raw, dict):
                log.error("state file %s is not a JSON object; starting empty", self._path)
                return

            self._models = {}
            self._profiles = {}
            self._downloads = {}
            for entry in raw.get("models") or []:
                try:
                    model = LibraryModel.model_validate(entry)
                # One unreadable record must not cost the operator the rest.
                except Exception as exc:
                    log.warning("dropping unreadable model record: %s", exc)
                    continue
                self._models[model.id] = model
            for model_id, records in (raw.get("profiles") or {}).items():
                profiles: list[ModelProfile] = []
                for record in records:
                    try:
                        profiles.append(ModelProfile.model_validate(record))
                    except Exception as exc:
                        log.warning("dropping unreadable profile record: %s", exc)
                if profiles:
                    self._profiles[model_id] = profiles

            for entry in raw.get("downloads") or []:
                try:
                    record = Download.model_validate(entry)
                except Exception as exc:
                    log.warning("dropping unreadable download record: %s", exc)
                    continue
                # Nothing is transferring: this process just started, so
                # a record that claims to be in flight is describing a
                # transfer that died with the last one. `paused` is the
                # honest state, and it is the one `resume` accepts.
                if record.state in (
                    DownloadState.queued,
                    DownloadState.resolving,
                    DownloadState.downloading,
                    DownloadState.verifying,
                ):
                    record.state = DownloadState.paused
                    record.message = (
                        "paused: the process restarted while this was transferring. "
                        "The bytes already fetched are still on disk; resume to continue."
                    )
                self._downloads[record.id] = record

            stamp = raw.get("lastScanAt")
            if isinstance(stamp, str):
                try:
                    self._last_scan_at = datetime.fromisoformat(stamp)
                except ValueError:
                    self._last_scan_at = None

            self._reindex_locked()
            log.info(
                "loaded %d models and %d profile sets from %s",
                len(self._models),
                len(self._profiles),
                self._path,
            )

    def _write_locked(self) -> None:
        payload: dict[str, Any] = {
            "version": STATE_VERSION,
            "lastScanAt": self._last_scan_at.isoformat() if self._last_scan_at else None,
            "models": [m.model_dump(mode="json", exclude_none=True) for m in self._models.values()],
            "profiles": {
                model_id: [p.model_dump(mode="json", exclude_none=True) for p in profiles]
                for model_id, profiles in self._profiles.items()
            },
            "downloads": [
                d.model_dump(mode="json", exclude_none=True) for d in self._downloads.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Same directory so `os.replace` stays on one filesystem and is
        # therefore atomic; a temp dir elsewhere would silently degrade
        # to a copy on some platforms.
        temp = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temp, self._path)
        finally:
            temp.unlink(missing_ok=True)

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
            return [self._with_profile_count(m) for m in self._models.values()]

    def get_model(self, model_id: str) -> LibraryModel | None:
        with self._lock:
            model = self._models.get(model_id)
            return self._with_profile_count(model) if model else None

    def find_by_path(self, path: str) -> LibraryModel | None:
        with self._lock:
            model = self._by_path.get(normalize(path))
            return self._with_profile_count(model) if model else None

    def _with_profile_count(self, model: LibraryModel) -> LibraryModel:
        return model.model_copy(update={"profileCount": len(self._profiles.get(model.id, []))})

    def replace_models(self, found: list[LibraryModel], *, scanned_at: datetime) -> dict[str, int]:
        """Install a completed scan's results.

        Returns per-outcome counts for the scan report. Entries not found
        by this scan become `missing` **if they carry profiles** and are
        dropped otherwise — keeping a profile-less record of a file
        somebody deleted is clutter, keeping one that holds tuning work
        is the whole point.
        """
        with self._lock:
            previous = dict(self._models)
            now = scanned_at
            added = updated = 0
            next_models: dict[str, LibraryModel] = {}

            for model in found:
                existing = previous.get(model.id)
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
            self._last_scan_at = now
            self._reindex_locked()
            self._write_locked()
            return {"added": added, "updated": updated, "missing": missing}

    def forget_model(self, model_id: str) -> bool:
        """Drop a missing entry and its profiles. No file is touched."""
        with self._lock:
            if model_id not in self._models and model_id not in self._profiles:
                return False
            self._models.pop(model_id, None)
            self._profiles.pop(model_id, None)
            self._reindex_locked()
            self._write_locked()
            return True

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
        with self._lock:
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
                flags=spec.flags,
                extraArgs=spec.extraArgs,
                env=spec.env,
                notes=spec.notes,
                createdAt=now,
                updatedAt=now,
            )
            if is_default:
                self._clear_defaults_locked(model_id, keep=profile.id)
            existing.append(profile)
            self._write_locked()
            return profile

    def replace_profile(
        self, model_id: str, profile_id: str, spec: ModelProfileSpec
    ) -> ModelProfile | None:
        """Whole-document replace. `None` if the profile does not exist,
        `ValueError` if the new name collides with a sibling."""
        with self._lock:
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
            updated = ModelProfile(
                id=current.id,
                name=spec.name,
                default=is_default,
                engine=spec.engine,
                flags=spec.flags,
                extraArgs=spec.extraArgs,
                env=spec.env,
                notes=spec.notes,
                createdAt=current.createdAt,
                updatedAt=_now(),
            )
            profiles[index] = updated
            if is_default:
                self._clear_defaults_locked(model_id, keep=updated.id)
            self._write_locked()
            return updated

    def delete_profile(self, model_id: str, profile_id: str) -> bool:
        with self._lock:
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
            self._write_locked()
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
            self._downloads[record.id] = record
            self._write_locked()

    def delete_download(self, download_id: str) -> bool:
        with self._lock:
            if self._downloads.pop(download_id, None) is None:
                return False
            self._write_locked()
            return True
