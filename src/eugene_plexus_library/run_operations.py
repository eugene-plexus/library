"""Durable operator intents. Agents execute only jobs assigned to their node.

The journal owns its format independently of the HTTP snapshots. No bearer
credentials are persisted. Checkpoints and leases use SQLite transactions;
external side effects use stable download/profile/runtime identities so replay
after a crash reconciles the result instead of repeating a declaration.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._generated.models import (
    Download,
    DownloadSpec,
    EngineKind,
    LibraryModel,
    ModelProfile,
    PreparedProvenance,
)

TERMINAL = {"ready", "skipped", "failed", "cancelled"}
LEASE_SECONDS = 120


class PreparationIntent(BaseModel):
    """Prepare the model for an engine before running it (LS5). Asked for,
    never implied: Run without it picks an engine that runs the model as it is."""

    model_config = ConfigDict(extra="forbid")
    engine: EngineKind
    contextSize: int | None = Field(
        default=None,
        gt=0,
        description="The context the engine prepares for. Absent: the engine's own "
        "recommendation for the node.",
    )


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str | None = Field(default=None, max_length=128)
    modelId: str | None = None
    download: DownloadSpec | None = None
    preparation: PreparationIntent | None = None

    @model_validator(mode="after")
    def one_source(self) -> Intent:
        if sum(x is not None for x in (self.modelId, self.download)) != 1:
            raise ValueError("choose exactly one of modelId or download")
        return self


class PreparationStatus(BaseModel):
    """Where a preparation is, as the engine's node reports it (LS5)."""

    state: Literal["waiting", "running", "done", "failed", "cancelled"]
    step: str | None = Field(default=None, description="The recipe's own words for where it is.")
    message: str | None = Field(default=None, description="Its last line of output.")
    bytesWritten: int | None = Field(
        default=None, ge=0, description="What it has written beside the model so far."
    )
    bytesNeeded: int | None = Field(
        default=None, ge=0, description="What it expects to write in all, when known."
    )
    warnings: list[str] = Field(
        default_factory=list, description="What the engine's own tools warned about."
    )


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease: str
    step: str = Field(
        pattern=(
            "^(checking|awaiting-install|installing|preparing|settings|launching|loading"
            "|ready|skipped|failed)$"
        )
    )
    engine: str | None = None
    runtime: str | None = None
    runtimeStatus: str | None = None
    install: dict[str, Any] | None = None
    preparation: PreparationStatus | None = None
    error: str | None = None
    failedStep: str | None = None
    loadingSince: int | None = None


class PreparedRequest(BaseModel):
    """The engine's node has prepared the model: list it (LS5). The library
    writes the provenance file beside the entry, as *Add a prepared model*
    does, and the operation goes on with the prepared model."""

    model_config = ConfigDict(extra="forbid")
    lease: str
    name: str = Field(pattern="^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    provenance: PreparedProvenance = Field(
        description="`entry` is the engine's entry file as this library spells it, "
        "inside a Library folder."
    )


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease: str
    engine: str
    contextSize: int | None = Field(default=None, gt=0)


class PreparedFileRequest(BaseModel):
    """A file of the prepared model, which the library writes itself (LS10):
    the engine's node prepares it in a folder of its own and sends it."""

    model_config = ConfigDict(extra="forbid")
    lease: str
    path: str = Field(
        min_length=1,
        max_length=1024,
        description="Where it goes, relative to the run's Library folder with `/` between "
        "folders: inside an engine's own folder at its top (`Strata-data/...`).",
    )


class PreparedFileComplete(PreparedFileRequest):
    """Every byte of the file has been sent: check it, then put it in place."""

    sizeBytes: int = Field(ge=0)
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class PreparedFileState(BaseModel):
    """What the library holds at one path of a preparation."""

    path: str
    sizeBytes: int | None = Field(
        default=None, ge=0, description="The whole file under its own name, when there is one."
    )
    sha256: str | None = Field(default=None, description="Its SHA-256, in hex.")
    receivedBytes: int = Field(
        ge=0, description="What has arrived of a send that is not complete yet."
    )


class Operation(BaseModel):
    """HTTP view; the journal envelope has its own version and schema."""

    id: str
    node: str | None
    intent: Intent
    model: LibraryModel | None
    step: Literal[
        "downloading",
        "checking",
        "awaiting-install",
        "installing",
        "preparing",
        "settings",
        "launching",
        "loading",
        "ready",
        "skipped",
        "failed",
        "cancelled",
    ]
    engine: str | None = None
    runtime: str | None = None
    runtimeStatus: str | None = None
    download: Download | None = None
    install: dict[str, Any] | None = None
    preparation: PreparationStatus | None = None
    preparedFrom: LibraryModel | None = Field(
        default=None,
        description="The model a preparation started from; `model` is then the prepared one.",
    )
    profile: ModelProfile | None = None
    error: str | None = None
    failedStep: str | None = None
    answer: Literal["install", "skip"] | None = None
    startedAt: int
    finishedAt: int | None = None
    loadingSince: int | None = None
    dismissed: bool = False


class OperationList(BaseModel):
    operations: list[Operation]


class ClaimedOperation(Operation):
    lease: str
    leaseUntil: float


class Journal:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = path, clock
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        with self._transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported run journal version {version}; kept {path}")
            if (
                version == 0
                and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
            ):
                raise ValueError(f"unversioned, nonempty run journal; kept {path}")
            db.execute(
                "CREATE TABLE IF NOT EXISTS operations "
                "(id TEXT PRIMARY KEY, intent TEXT NOT NULL, document TEXT NOT NULL)"
            )
            db.execute("PRAGMA user_version=1")
            db.execute(
                "CREATE TABLE IF NOT EXISTS submission_aliases "
                "(id TEXT PRIMARY KEY, intent TEXT NOT NULL, operation_id TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS operation_stage "
                "ON operations(json_extract(document, '$.step'))"
            )

    @contextmanager
    def _transaction(self):  # type: ignore[no-untyped-def]
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def list(
        self, *, all_records: bool = False, pending_only: bool = False
    ) -> list[dict[str, Any]]:
        where = "1" if all_records else "NOT json_extract(document, '$.dismissed')"
        if pending_only:
            where += (
                " AND (json_extract(document, '$.step') "
                "NOT IN ('ready','skipped','failed','cancelled')"
                " OR (json_extract(document, '$.step')='cancelled' "
                "AND json_extract(document, '$.intent.download') IS NOT NULL "
                "AND NOT coalesce(json_extract(document, '$.downloadCancelled'), 0)))"
            )
        query = (
            "SELECT document FROM operations WHERE "
            + where
            + " ORDER BY json_extract(document, '$.step') "
            "IN ('ready','skipped','failed','cancelled'), rowid DESC LIMIT 256"
        )
        with closing(sqlite3.connect(self.path)) as db:
            records = [json.loads(row[0]) for row in db.execute(query)]
        return [r for r in records if all_records or not r.get("dismissed")]

    def get(self, id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT document FROM operations WHERE id=?", (id,)).fetchone()
        if row is None:
            raise HTTPException(404, "No such run operation")
        return json.loads(row[0])  # type: ignore[no-any-return]

    def create(
        self,
        id: str,
        intent: Intent,
        model: dict[str, Any] | Callable[[], dict[str, Any] | None] | None,
    ) -> dict[str, Any]:
        encoded = json.dumps(intent.model_dump(mode="json"), sort_keys=True)
        with self._transaction() as db:
            alias = db.execute(
                "SELECT intent, operation_id FROM submission_aliases WHERE id=?", (id,)
            ).fetchone()
            if alias:
                if alias[0] != encoded:
                    raise HTTPException(409, "This operation ID already names a different intent")
                document: dict[str, Any] = json.loads(
                    db.execute(
                        "SELECT document FROM operations WHERE id=?", (alias[1],)
                    ).fetchone()[0]
                )
                return document
            found = db.execute(
                "SELECT intent, document FROM operations WHERE id=?", (id,)
            ).fetchone()
            if found:
                if found[0] != encoded:
                    raise HTTPException(409, "This operation ID already names a different intent")
                return json.loads(found[1])  # type: ignore[no-any-return]
            active = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT document FROM operations WHERE json_extract(document, '$.step') "
                    "NOT IN ('ready','skipped','failed','cancelled')"
                )
            ]
            for current in active:
                if current["intent"] == intent.model_dump(mode="json"):
                    db.execute(
                        "INSERT INTO submission_aliases VALUES (?,?,?)",
                        (id, encoded, current["id"]),
                    )
                    return current  # type: ignore[no-any-return]
            if len(active) >= 128:
                raise HTTPException(429, "Too many pending run operations")
            # Resolve inputs only for a genuinely new intent. An idempotent retry
            # still returns its receipt after the model/download was removed.
            if callable(model):
                model = model()
            # Keep completed IDs as idempotency receipts; callers may safely retry
            # after an arbitrarily long disconnection. Payloads are small.
            record = {
                "id": id,
                "intent": intent.model_dump(mode="json"),
                "node": intent.node,
                "model": model,
                "step": "checking" if model else "downloading",
                "engine": None,
                "runtime": None,
                "runtimeStatus": None,
                "download": None,
                "install": None,
                "preparation": None,
                "preparedFrom": None,
                "profile": None,
                "error": None,
                "failedStep": None,
                "answer": None,
                "startedAt": int(self.clock() * 1000),
                "finishedAt": None,
                "lease": None,
                "leaseUntil": 0,
                "dismissed": False,
            }
            db.execute("INSERT INTO operations VALUES (?,?,?)", (id, encoded, json.dumps(record)))
            return record

    def change(self, id: str, update: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._transaction() as db:
            row = db.execute("SELECT document FROM operations WHERE id=?", (id,)).fetchone()
            if row is None:
                raise HTTPException(404, "No such run operation")
            record = json.loads(row[0])
            update(record)
            if record["step"] in TERMINAL and record["finishedAt"] is None:
                record["finishedAt"] = int(self.clock() * 1000)
            db.execute("UPDATE operations SET document=? WHERE id=?", (json.dumps(record), id))
            return record  # type: ignore[no-any-return]

    def claim(self, id: str, node: str | None) -> dict[str, Any]:
        def claim(record: dict[str, Any]) -> None:
            if record["node"] != node:
                raise HTTPException(403, "Run belongs to another node")
            if record["step"] in TERMINAL or record["step"] == "downloading":
                raise HTTPException(409, "Run is not executable")
            if record["leaseUntil"] > self.clock():
                raise HTTPException(409, "Run is already claimed")
            record.update(lease=secrets.token_hex(24), leaseUntil=self.clock() + LEASE_SECONDS)

        return self.change(id, claim)

    def verify_lease(self, record: dict[str, Any], node: str | None, lease: str) -> None:
        if record["node"] != node:
            raise HTTPException(403, "Run belongs to another node")
        if (
            record["step"] in TERMINAL
            or record["leaseUntil"] <= self.clock()
            or record["lease"] != lease
        ):
            raise HTTPException(409, "Run was cancelled, completed, or its lease expired")

    def checkpoint(self, id: str, node: str | None, patch: Checkpoint) -> dict[str, Any]:
        def update(record: dict[str, Any]) -> None:
            self.verify_lease(record, node, patch.lease)
            if (
                patch.step in {"installing", "preparing", "loading", "ready"}
                and record["answer"] == "skip"
            ):
                raise HTTPException(409, "Operator skipped installation and launch")
            if (
                record["intent"].get("preparation")
                and patch.step in {"settings", "launching", "loading", "ready"}
                and record.get("preparedFrom") is None
            ):
                # The run goes on with the prepared model, never the original.
                raise HTTPException(409, "The prepared model is not listed yet")
            record.update(patch.model_dump(exclude={"lease"}, exclude_unset=True))
            record.update(lease=None, leaseUntil=0)

        return self.change(id, update)


def public(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in {"lease", "leaseUntil"}}
