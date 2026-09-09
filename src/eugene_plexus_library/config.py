"""Runtime configuration: schema declaration, file-backed state, PATCH apply.

Implements the shared Eugene Plexus config protocol:

* `GET /v1/config/schema` -> field metadata for UI rendering (`as_schema`)
* `GET /v1/config` -> current effective values, secrets redacted
* `PATCH /v1/config` -> partial update, per-key validation

The field that matters is `modelRoots`, a `path_list`. Roots are config
rather than a resource of their own precisely so the generic config
editor renders them with no library-specific UI code — `path_list` was
added to `ConfigValueType` in specs M2 for this, and the UI turns it
into an add/remove list of directory pickers rather than a text field,
because asking someone to comma-separate Windows paths is asking for a
bug report.

`hfToken` is the one `sensitive` field: redacted in `GET`, accepted in
`PATCH`, and sealed at rest with the master key. That machinery was
wired in at M2 with nothing using it, precisely so adding this field
needed no plumbing work.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from . import security
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)

log = logging.getLogger(__name__)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "library": "Model library",
    "scanning": "Scanning",
    "catalogue": "Model catalogue",
    "downloads": "Downloads",
    "guidance": "Hardware guidance",
    "logging": "Logging",
}

FIELDS: list[ConfigField] = [
    ConfigField(
        key="modelRoots",
        label="Model directories",
        description=(
            "Directories to scan for models. Point these at wherever you "
            "already keep models — an LM Studio folder, a drive full of "
            "GGUFs, a HuggingFace cache. Subdirectories are searched, so "
            "one entry covers a whole publisher/repo tree. Nothing is "
            "moved, renamed or copied; these are read only. Order "
            "matters only in that it is the order you see them in, and "
            "downloads will default to the first."
        ),
        category="library",
        valueType=ConfigValueType.path_list,
        default=[],
    ),
    ConfigField(
        key="scanOnStartup",
        label="Scan at startup",
        description=(
            "Walk the model directories when the library starts. Leave "
            "this on unless your models live on a slow network share and "
            "you would rather scan on demand — the scan runs in the "
            "background either way and never blocks the API."
        ),
        category="scanning",
        valueType=ConfigValueType.boolean,
        default=True,
    ),
    ConfigField(
        key="followSymlinks",
        label="Follow symlinked directories",
        description=(
            "Descend into directories that are symlinks. Off by default, "
            "and deliberately: a symlink loop turns a scan into an "
            "infinite walk, and on Linux and macOS a HuggingFace cache "
            "links its snapshots into a content-addressed blob store, "
            "which would report the same model repeatedly under hash "
            "names. Turn it on if you deliberately symlink model "
            "directories together."
        ),
        category="scanning",
        valueType=ConfigValueType.boolean,
        default=False,
    ),
    ConfigField(
        key="catalogueEnabled",
        label="Search and download models",
        description=(
            "Allow this component to reach the upstream model catalogue "
            "to search, describe and download models. Turn it off for an "
            "air-gapped install: nothing outbound is attempted and the "
            "search screen says so plainly instead of timing out. "
            "Scanning the directories you already have is unaffected "
            "either way."
        ),
        category="catalogue",
        valueType=ConfigValueType.boolean,
        default=True,
    ),
    ConfigField(
        key="catalogueBaseUrl",
        label="Catalogue address",
        description=(
            "Where the catalogue lives. Change this only for a private "
            "or mirrored hub that speaks the same API — a regional "
            "mirror, or an enterprise instance. Hardcoding the public "
            "hub would make this component useless behind either."
        ),
        category="catalogue",
        valueType=ConfigValueType.url,
        default="https://huggingface.co",
    ),
    ConfigField(
        key="hfToken",
        label="Catalogue access token",
        description=(
            "A token from your account on the catalogue. Needed for "
            "gated models — the ones whose licence you have to accept on "
            "their own page — and for private repositories. Worth "
            "setting even without either: the hub's own response tells "
            "unauthenticated clients that a token raises the request "
            "limit and speeds up transfers. Stored encrypted."
        ),
        category="catalogue",
        valueType=ConfigValueType.secret,
        sensitive=True,
    ),
    ConfigField(
        key="downloadLayout",
        label="Where downloads land",
        description=(
            "How a downloaded file is filed under the model directory "
            "you picked. `publisher_repo` makes "
            "`<root>/<publisher>/<repo>/<file>`, which is the layout LM "
            "Studio uses and the one most people already have. `flat` "
            "drops the file straight into the root. Either way the file "
            "keeps its own upstream name — nothing is renamed, hashed, "
            "or hidden in a cache."
        ),
        category="downloads",
        valueType=ConfigValueType.enum,
        default="publisher_repo",
        enumValues=["publisher_repo", "flat"],
    ),
    ConfigField(
        key="maxConcurrentDownloads",
        label="Downloads at once",
        description=(
            "How many transfers run simultaneously. One is the default "
            "and usually right: two large downloads sharing a connection "
            "both finish later than one after the other, and the "
            "progress bar you are watching is the one you started "
            "first. Raise it if your link is faster than the hub's "
            "per-connection throughput."
        ),
        category="downloads",
        valueType=ConfigValueType.integer,
        default=1,
        minimum=1,
        maximum=8,
    ),
    ConfigField(
        key="guidanceContextLength",
        label="Assumed context length",
        description=(
            "The context size quant recommendations are calculated at, "
            "when nothing else says. This is the number that decides "
            "which quant gets recommended: the KV cache grows linearly "
            "with it, so a model that fits comfortably at 8k may not fit "
            "at all at 128k. A config field rather than a constant "
            "because both answers are correct for different people."
        ),
        category="guidance",
        valueType=ConfigValueType.integer,
        default=8192,
        minimum=256,
        maximum=1048576,
    ),
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty the library's terminal output is. `DEBUG` names "
            "every file the scanner looks at and why it was or wasn't "
            "treated as a model, which is the fastest way to find out "
            "why something isn't showing up."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
]

# NOTE on what is NOT in this schema:
# The bind port. Ports are owned by the agent topology and passed to
# spawned children via EUGENE_PLEXUS_LIBRARY_BIND_PORT. Two sources of
# truth on a port is the trap where the agent spawns at one and the
# component's own config claims another.

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema() -> ConfigSchema:
    return ConfigSchema(component="library", fields=list(FIELDS), categories=CATEGORY_LABELS)


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """None if valid, otherwise an operator-readable error message."""
    if value is None:
        return None  # null clears to default

    vt = field.valueType

    if vt in (ConfigValueType.string, ConfigValueType.url, ConfigValueType.file_path):
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None and re.search(field.pattern, value) is None:
            return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.path_list:
        if not isinstance(value, list):
            return f"expected a list of paths, got {type(value).__name__}"
        for index, item in enumerate(value):
            if not isinstance(item, str):
                return f"entry {index} is {type(item).__name__}, expected a path string"
            if not item.strip():
                return f"entry {index} is empty"
        # Duplicates are rejected rather than de-duplicated. Two spellings
        # of one directory would scan it twice and attribute its models to
        # whichever root won, and silently dropping one of the operator's
        # entries is worse than telling them.
        from .paths import normalize

        seen: dict[str, int] = {}
        for index, item in enumerate(value):
            key = normalize(item)
            if key in seen:
                return f"entry {index} duplicates entry {seen[key]} ({item!r})"
            seen[key] = index
        return None

    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt in (ConfigValueType.number, ConfigValueType.duration):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config state. Thread-safe for the read/write pattern
    the routes use."""

    def __init__(self, path: Path, *, master_key: bytes | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._pending_restart: set[str] = set()
        self._master_key = master_key

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for key, value in raw.items():
                    if key not in _FIELDS_BY_KEY:
                        continue
                    merged[key] = self._decrypt_loaded(key, value)
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

    def _decrypt_loaded(self, key: str, value: Any) -> Any:
        if not security.is_envelope(value):
            return value
        if self._master_key is None:
            log.warning(
                "config field %r is encrypted on disk but no master key is available; "
                "treating as unset",
                key,
            )
            return None
        try:
            return security.open_envelope(security.Envelope.from_dict(value), self._master_key)
        except ValueError as exc:
            log.warning("config field %r failed to decrypt (%s); treating as unset", key, exc)
            return None

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                out[key] = REDACTED if (field and field.sensitive and value is not None) else value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue
                error = _validate_value(field, new_value)
                if error is not None:
                    rejected.append(ConfigFieldError(key=key, message=error))
                    continue

                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value

                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)

            if applied:
                self._write_locked()

            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(self._pending_restart),
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def model_roots(self) -> list[Path]:
        """Configured roots as paths.

        Non-string junk is filtered rather than raised on: validation
        rejects it at PATCH time, and a hand-edited config file must not
        stop the component from starting — the config endpoints are how
        an operator repairs it.
        """
        raw = self.get("modelRoots") or []
        if not isinstance(raw, list):
            log.warning("modelRoots is %s, expected a list; treating as empty", type(raw).__name__)
            return []
        return [Path(item).expanduser() for item in raw if isinstance(item, str) and item.strip()]

    def catalogue_enabled(self) -> bool:
        return bool(self.get("catalogueEnabled"))

    def catalogue_base_url(self) -> str:
        value = self.get("catalogueBaseUrl")
        return value if isinstance(value, str) and value.strip() else "https://huggingface.co"

    def hf_token(self) -> str | None:
        """The real token, never the redaction.

        `as_document` replaces a sensitive value with the redacted
        marker for the wire; this reads through to what is actually
        stored, and is the only path that should.
        """
        value = self.get("hfToken")
        if not isinstance(value, str) or not value.strip() or value == REDACTED:
            return None
        return value.strip()

    def download_layout(self) -> str:
        value = self.get("downloadLayout")
        return value if value in ("publisher_repo", "flat") else "publisher_repo"

    def max_concurrent_downloads(self) -> int:
        value = self.get("maxConcurrentDownloads")
        return value if isinstance(value, int) and value > 0 else 1

    def guidance_context_length(self) -> int:
        value = self.get("guidanceContextLength")
        return value if isinstance(value, int) and value > 0 else 8192

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        on_disk: dict[str, Any] = {}
        for key, value in self._values.items():
            field = _FIELDS_BY_KEY.get(key)
            if (
                field is not None
                and field.sensitive
                and isinstance(value, str)
                and value
                and self._master_key is not None
            ):
                on_disk[key] = security.seal(value, self._master_key).to_dict()
            else:
                on_disk[key] = value
        with self._path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(on_disk, handle, sort_keys=True, default_flow_style=False)
