"""Runtime configuration: schema declaration, file-backed state, PATCH apply.

Implements the shared Eugene Plexus config protocol:

* `GET /v1/config/schema` -> field metadata for UI rendering (`as_schema`)
* `GET /v1/config` -> current effective values, secrets redacted
* `PATCH /v1/config` -> partial update, per-key validation

The field that matters is `modelRoots`, a `library_folders` list
(2026-09-14; a `path_list` from M2 until then). Roots are config rather
than a resource of their own precisely so the generic config editor
renders them with no library-specific UI code. Each entry is a
`LibraryFolder`: the directory as this host spells it, plus `mounts` --
where other machines find the same directory, one per OS shape -- so
the reach of a folder is stated once here and every node inherits it,
instead of each node's agent carrying a row per folder. A bare string
in a PATCH body or in the file means a folder with no mounts; `GET` answers
in the object form. `folders.py` holds the coercion and the validation.

`catalogueSources` (LS4) is where Discover finds models, and holds the
one secret: each hub's token, redacted in `GET`, accepted in `PATCH`, and
sealed at rest with the master key, per entry (`catalogue_sources.py`). A
file without it gets the default list.

`modelRoots` can take its **default** from the process environment
(`Settings.default_model_roots`), for a host whose layout is fixed
before any config exists — the container image knows its models volume
is at `/models`. It is a default in exactly the sense this protocol
already has: the schema reports it, `null` clears to it, and here an
empty list does too, because "no directories at all" is not something
an operator wants from a library whose image has said where they are.
The default is never persisted: the file holds `[]` while the default
is in effect, so the file records what the operator set and the
environment keeps supplying the rest.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from . import _private_files, ollama_detect, security
from . import catalogue_sources as sources_mod
from ._generated.models import (
    CatalogueSource,
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
    LibraryFolderKind,
    LibraryFolderStatus,
)
from .folders import (
    coerce_folders,
    folder_kind,
    folder_paths,
    folders_from_body,
    folders_from_paths,
    validate_folders,
)
from .paths import is_within, normalize, real_within

log = logging.getLogger(__name__)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "library": "Library",
    "scanning": "Scanning",
    "catalogue": "Model catalogue",
    "downloads": "Downloads",
    "guidance": "Hardware guidance",
    "logging": "Logging",
}

FIELDS: list[ConfigField] = [
    ConfigField(
        key="modelRoots",
        label="Folders",
        description=(
            "The folders this Library catalogues, spelled as the machine the "
            "Library runs on spells them.\n\n"
            "Point them at wherever you already keep models. Subfolders are "
            "searched, and nothing is moved or copied. Downloads go to the first "
            "folder by default, and a node runs a model only from one of these "
            "folders.\n\n"
            "A folder's `kind` is `folder`, or `ollama` for an Ollama model store, which "
            "is never written to.\n\n"
            "Each folder can list its **mounts**, where other machines find it: one "
            "path for Linux and macOS nodes (`/mnt/models`) and one for Windows nodes "
            "(`\\\\NAS\\models`). A folder with no mounts is reached at its own path on other "
            "machines, or not at all. A node that mounts the share elsewhere sets an override "
            "under Library → that node → Folders."
        ),
        category="library",
        valueType=ConfigValueType.library_folders,
        default=[],
    ),
    ConfigField(
        key="findOllama",
        label="Show Ollama's models",
        description=(
            "The Library looks for Ollama's model store on its own machine and lists its "
            "models, read-only.\n\n"
            "They stay where Ollama keeps them, and Ollama still manages them. Turn this "
            "off to hide them. A store somewhere else is added under Folders with the "
            "kind `ollama`."
        ),
        category="library",
        valueType=ConfigValueType.boolean,
        default=True,
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
            "Descend into symlinked directories and Windows directory "
            "junctions. Off by default; turn it on if you deliberately "
            "link model directories together. Directory cycles are "
            "skipped. Linked files are always read, including HuggingFace "
            "snapshot files, and models keep their named paths rather "
            "than the link targets. Explicitly configured roots are "
            "always scanned. Changes take effect on the next scan."
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
        key=sources_mod.SOURCES_KEY,
        label="Where to find models",
        description=(
            "The places Discover searches; every result says which source it came "
            "from, in this list's order.\n\n"
            "A **hub** is Hugging Face or anything that speaks its API, such as a "
            "mirror or an enterprise instance, at its own address. Its access "
            "token is needed for gated and private models, and also makes searches "
            "faster and raises the request limit. Tokens are stored encrypted and "
            "never shown again.\n\n"
            "An **engine** list is the models an engine says it supports, as the node "
            "picked in Discover reports them. Leave its engine empty for every "
            "engine's.\n\n"
            "Turning off Search and download models stops every source here."
        ),
        category="catalogue",
        valueType=ConfigValueType.catalogue_sources,
        default=sources_mod.default_sources(),
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
        label="Context to plan for",
        description=(
            "The context every fit verdict is judged at.\n\nIt applies to "
            "Home's first model, Discover's list and the Library. Run on llama.cpp "
            "never starts a model below it (or below the model's own, when that is "
            "smaller), putting what the GPU cannot hold in system memory, as the "
            "verdict said. The KV cache grows linearly with it, so a model that fits "
            "comfortably at 8k may not fit at all at 128k. Discover can try another "
            "number without changing this one."
        ),
        category="guidance",
        valueType=ConfigValueType.integer,
        default=16384,
        minimum=256,
        maximum=1048576,
    ),
    ConfigField(
        key="starterModelsFile",
        label="Starter model list",
        description=(
            "A YAML file naming the starter models a new install is offered, one "
            "per size class.\n\n"
            "Empty uses the list that ships with this version. Point it at your own "
            "file to recommend something else, or at a file with an empty `classes:` "
            "list to recommend nothing. The path is on the machine the Library runs "
            "on."
        ),
        category="guidance",
        valueType=ConfigValueType.file_path,
    ),
    ConfigField(
        key="starterCpuFloorTokensPerSecond",
        label="Starter CPU speed floor",
        description=(
            "The minimum measured speed for a starter model on a machine with no graphics card.\n\n"
            "In tok/s, from 1 to 100; the default is 4. The measurement belongs to the "
            "named test processor. Your processor may be faster or slower. The largest "
            "entry that fits and reaches this floor is recommended. If none does, the "
            "smallest dense entry that fits is recommended. After the download, "
            "Benchmark measures the speed here."
        ),
        category="guidance",
        valueType=ConfigValueType.number,
        default=4,
        minimum=1,
        maximum=100,
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

ROOTS_KEY = "modelRoots"
DETECTION_SECONDS = 10.0
DEFAULT_ROOTS_VARIABLE = "EUGENE_PLEXUS_LIBRARY_DEFAULT_MODEL_ROOTS"


#: What an unset value does, shown where the control would otherwise be
#: blank (settings never lie, 2026-09-30).
UNSET_MEANS: dict[str, str] = {
    "starterModelsFile": "Not set: uses the starter list shipped with this version.",
    ROOTS_KEY: "No folders: nothing is scanned, and a download has nowhere to go.",
}


def as_schema(
    *, default_roots: Sequence[str] = (), pending: dict[str, Any] | None = None
) -> ConfigSchema:
    """The field metadata, with the roots' default as this install has it.

    A picker rendering the schema shows the default beside the field,
    so an operator in a container sees `/models` and where it came from
    rather than an empty list that the effective config contradicts.
    `pending` is `ConfigStore.pending_restart()`: a field saved but not yet
    in effect says so, and what the running library uses.
    """
    fields: list[ConfigField] = []
    for field in FIELDS:
        update: dict[str, Any] = {}
        if field.key in UNSET_MEANS:
            update["unsetMeans"] = UNSET_MEANS[field.key]
        if field.key == ROOTS_KEY and default_roots:
            update["default"] = folders_from_paths(default_roots)
            update["defaultSource"] = (
                f"Set by this install's environment ({DEFAULT_ROOTS_VARIABLE}), with no "
                "mounts declared. Clearing the list returns to it; adding mounts replaces it "
                "with an explicit list that says the same path."
            )
        if pending and field.key in pending:
            update["pendingRestart"] = True
            if not field.sensitive:
                update["inEffect"] = pending[field.key]
        fields.append(field.model_copy(update=update) if update else field)
    return ConfigSchema(component="library", fields=fields, categories=CATEGORY_LABELS)


def _is_unset(field: ConfigField, value: Any) -> bool:
    """None, or an empty string where empty means nothing: a secret (an
    empty token is no token) and an address (an empty hub address is the
    hub). Those used to be stored as given and shown as a value."""
    if value is None:
        return True
    return (
        isinstance(value, str)
        and not value.strip()
        and field.valueType in (ConfigValueType.secret, ConfigValueType.url)
    )


def _defaults(*, default_roots: Sequence[str] = ()) -> dict[str, Any]:
    # Copies: a default list held as a value must not be the schema's own.
    values = {f.key: copy.deepcopy(f.default) for f in FIELDS if f.default is not None}
    if default_roots:
        values[ROOTS_KEY] = folders_from_paths(default_roots)
    return values


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

    if vt == ConfigValueType.library_folders:
        return validate_folders(value)

    if vt == ConfigValueType.catalogue_sources:
        return sources_mod.validate(value)

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
        if field.key == "starterCpuFloorTokensPerSecond" and not math.isfinite(value):
            return "must be a finite number"
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

    def __init__(
        self,
        path: Path,
        *,
        master_key: bytes | None = None,
        default_roots: Sequence[str] = (),
    ) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._default_roots: list[str] = [r for r in default_roots if r.strip()]
        self._values: dict[str, Any] = self._field_defaults()
        # Whether `modelRoots` is currently the environment's default
        # rather than something the operator set. Decides what the file
        # gets: the default is never written, so the file keeps meaning
        # "what the operator chose" and the variable keeps applying.
        self._roots_defaulted = bool(self._default_roots)
        # What this process runs on for every `requiresRestart` field, as
        # loaded. A field is pending a restart while its saved value
        # differs, and the schema says so (settings never lie, 2026-09-30).
        self._started: dict[str, Any] = dict(self._values)
        self._master_key = master_key
        self._detected: tuple[float, list[Path]] | None = None

    def _field_defaults(self) -> dict[str, Any]:
        return _defaults(default_roots=self._default_roots)

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = self._field_defaults()
                merged[sources_mod.SOURCES_KEY] = self._sources_loaded(raw)
                for key, value in raw.items():
                    field = _FIELDS_BY_KEY.get(key)
                    if field is None or key == sources_mod.SOURCES_KEY:
                        continue
                    value = self._decrypt_loaded(key, value)
                    # A `null` (or, for a secret or an address, an empty
                    # string) in the file is the default, as a PATCH of
                    # null is. `catalogueEnabled: null` used to turn the
                    # catalogue off while GET said null and the schema said
                    # it was on; an empty token read as "<redacted>".
                    if _is_unset(field, value):
                        value = field.default
                    if key == "starterCpuFloorTokensPerSecond":
                        error = _validate_value(field, value)
                        if error:
                            log.warning("%s: %s; using the default %s", key, error, field.default)
                            value = field.default
                    merged[key] = value
                # An empty list on disk is "nothing configured", and the
                # environment's default fills it -- including a file
                # written before the variable existed, which is every
                # container that came up before this default did.
                self._roots_defaulted = bool(self._default_roots) and not merged.get(ROOTS_KEY)
                if self._roots_defaulted:
                    merged[ROOTS_KEY] = folders_from_paths(self._default_roots)
                else:
                    # The file holds folders (a bare path is one with no mounts);
                    # a junk entry is skipped with a warning.
                    merged[ROOTS_KEY] = coerce_folders(merged.get(ROOTS_KEY))
                self._values = merged
            else:
                self._values = self._field_defaults()
                self._roots_defaulted = bool(self._default_roots)
                self._write_locked()
            self._started = dict(self._values)

    def _sources_loaded(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """`catalogueSources` as the file has it, its tokens opened; the
        default list when the file has none."""
        listed = raw.get(sources_mod.SOURCES_KEY)
        if not isinstance(listed, list):
            return sources_mod.default_sources()
        out: list[dict[str, Any]] = []
        for entry in listed:
            if not isinstance(entry, dict):
                continue
            held = dict(entry)
            held.pop("hasToken", None)
            if "token" in held:
                where = f"{sources_mod.SOURCES_KEY}[{held.get('id')}].token"
                token = self._decrypt_loaded(where, held["token"])
                if isinstance(token, str) and token.strip():
                    held["token"] = token.strip()
                else:
                    held.pop("token")
            out.append(held)
        return out

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
                if key == sources_mod.SOURCES_KEY:
                    out[key] = sources_mod.redact(value)
                    continue
                out[key] = REDACTED if (field and field.sensitive and value is not None) else value
            return ConfigDocument.model_validate(out)

    def schema(self) -> ConfigSchema:
        return as_schema(default_roots=self._default_roots, pending=self.pending_restart())

    def pending_restart(self) -> dict[str, Any]:
        """`requiresRestart` fields whose saved value is not the one this
        process runs on, with the value it runs on."""
        with self._lock:
            return {
                f.key: self._started.get(f.key)
                for f in FIELDS
                if f.requiresRestart and self._values.get(f.key) != self._started.get(f.key)
            }

    @property
    def default_roots(self) -> list[str]:
        """What the environment said, whether or not it is in effect."""
        return list(self._default_roots)

    def roots_are_defaulted(self) -> bool:
        """True while `modelRoots` is the environment's default rather
        than something the operator set."""
        with self._lock:
            return self._roots_defaulted

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending_restart: list[str] = []
        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue
                if key == ROOTS_KEY:
                    # The wizard and scripts say a folder by its path: a bare
                    # string is a folder with no mounts, before it is judged.
                    new_value = folders_from_body(new_value)
                error = _validate_value(field, new_value)
                if error is not None:
                    rejected.append(ConfigFieldError(key=key, message=error))
                    continue
                if key == sources_mod.SOURCES_KEY:
                    # Tokens are kept per id across a round trip that was shown
                    # none; null is the default list, keeping them the same way.
                    self._values[key] = sources_mod.merge(
                        new_value if new_value is not None else sources_mod.default_sources(),
                        self._values.get(key),
                    )
                    applied.append(key)
                    continue

                defaults = self._field_defaults()
                if key == ROOTS_KEY and new_value is not None:
                    # Cleaned, and stored in the object form.
                    new_value = coerce_folders(new_value)
                if key == ROOTS_KEY and self._default_roots:
                    # `[]` clears to the default as `null` does: with the
                    # environment naming the directories, "none" is not a
                    # state this install can be in, only "these others".
                    self._roots_defaulted = not new_value
                    if self._roots_defaulted:
                        new_value = None

                if _is_unset(field, new_value):
                    new_value = None
                if new_value is None and defaults.get(key) is not None:
                    self._values[key] = defaults[key]
                else:
                    self._values[key] = new_value

                applied.append(key)
                # This PATCH's keys only, and only while they differ from
                # what the process runs on: the contract's subset of
                # `applied`. It was every restart key saved since start.
                if field.requiresRestart and self._values.get(key) != self._started.get(key):
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(pending_restart),
                pendingRestart=pending_restart,
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def configured_roots(self) -> list[Path]:
        """What `modelRoots` holds, as paths on this host.

        Junk is filtered rather than raised on: validation rejects it at
        PATCH time, and a hand-edited config file must not stop the
        component from starting — the config endpoints are how an
        operator repairs it.
        """
        return [Path(item).expanduser() for item in folder_paths(self.get(ROOTS_KEY))]

    def found_ollama(self) -> Path | None:
        """Ollama's own store on this machine, when `findOllama` is on and no
        folder of kind `ollama` is configured (CX9-C2). Never written to the
        config: turning the setting off is all it takes to remove it."""
        found = self.found_ollama_stores()
        return found[0] if found else None

    def found_ollama_stores(self) -> list[Path]:
        """All automatically found stores, without duplicating configured ones."""
        if not self.get("findOllama"):
            return []
        configured = coerce_folders(self.get(ROOTS_KEY))
        if any(folder_kind(f) == "ollama" for f in configured):
            return []
        found = self._detected_store()
        taken = {normalize(f["path"]) for f in configured}
        return [p for p in found if normalize(p) not in taken]

    def _detected_store(self) -> list[Path]:
        """What detection found, kept for a few seconds: the roots are asked for
        on every route, and a store on a share that has gone away must not be
        stat'ed each time."""
        now = time.perf_counter()
        with self._lock:
            cached = self._detected
            if cached is not None and now - cached[0] < DETECTION_SECONDS:
                return cached[1]
        found = ollama_detect.find_stores()
        with self._lock:
            self._detected = (now, found)
        return found

    def model_roots(self) -> list[Path]:
        """Every folder the Library reads: the configured ones, then Ollama's
        own store when it was found. For scanning and for "is this path in a
        Library folder"; a place to *write* is `writable_roots`."""
        return [*self.configured_roots(), *self.found_ollama_stores()]

    def ollama_roots(self) -> list[Path]:
        """The folders that are Ollama stores: read-only, never written to."""
        roots = [
            Path(f["path"]).expanduser()
            for f in coerce_folders(self.get(ROOTS_KEY))
            if folder_kind(f) == "ollama"
        ]
        # Hiding a store from the catalogue does not make it writable. Also
        # protect a detected store configured as an ordinary folder.
        return list(dict.fromkeys([*roots, *self._detected_store()]))

    def writable_roots(self) -> list[Path]:
        """Where a download, a provenance file or a prepared model may be
        written: every folder but an Ollama store, and none inside one."""
        stores = self.ollama_roots()
        roots: list[Path] = []
        for f in coerce_folders(self.get(ROOTS_KEY)):
            root = Path(f["path"]).expanduser()
            if folder_kind(f) != "ollama" and not any(
                is_within(root, s) or real_within(root, s) for s in stores
            ):
                roots.append(root)
        return roots

    def library_folders(self) -> list[LibraryFolderStatus]:
        """The folders with their mounts, as the wire carries them: the
        configured ones, then Ollama's own store when it was found."""
        folders = [
            LibraryFolderStatus.model_validate({"kind": LibraryFolderKind.folder, **f})
            for f in coerce_folders(self.get(ROOTS_KEY))
        ]
        for found in self.found_ollama_stores():
            folders.append(
                LibraryFolderStatus(
                    path=str(found), mounts=[], kind=LibraryFolderKind.ollama, found=True
                )
            )
        stores = self.ollama_roots()
        for folder in folders:
            store = next(
                (s for s in stores if is_within(folder.path, s) or real_within(folder.path, s)),
                None,
            )
            if store is not None:
                relation = "in" if is_within(folder.path, store) else "a link into"
                folder.readOnly = True
                folder.readOnlyReason = (
                    f"{folder.path} is {relation} the Ollama model store {store}. "
                    "Ollama manages the files there, and the Library never writes there. "
                    "Choose another folder."
                )
                try:
                    list(Path(folder.path).iterdir())
                except OSError as exc:
                    folder.error = f"Cannot read {folder.path}: {exc}"
        return folders

    def catalogue_enabled(self) -> bool:
        return bool(self.get("catalogueEnabled"))

    def catalogue_sources(self) -> list[CatalogueSource]:
        """The sources with their real tokens, never the redaction:
        `as_document` hides them for the wire, and this is the only path
        that reads them."""
        return sources_mod.as_models(self.get(sources_mod.SOURCES_KEY) or [])

    def download_layout(self) -> str:
        value = self.get("downloadLayout")
        return value if value in ("publisher_repo", "flat") else "publisher_repo"

    def max_concurrent_downloads(self) -> int:
        value = self.get("maxConcurrentDownloads")
        return value if isinstance(value, int) and value > 0 else 1

    def starter_models_file(self) -> str | None:
        """The operator's own list, or None for the one in the wheel."""
        value = self.get("starterModelsFile")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def guidance_context_length(self) -> int:
        value = self.get("guidanceContextLength")
        return value if isinstance(value, int) and value > 0 else 16384

    def starter_cpu_floor_tokens_per_second(self) -> float:
        return float(self.get("starterCpuFloorTokensPerSecond"))

    def _sealed(self, value: str | None) -> Any:
        """A secret as the file holds it: sealed when there is a master key."""
        if value and self._master_key is not None:
            return security.seal(value, self._master_key).to_dict()
        return value or None

    def _sealed_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in entry.items() if k != "token"}
        if entry.get("token"):
            out["token"] = self._sealed(entry["token"])
        return out

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        on_disk: dict[str, Any] = {}
        for key, value in self._values.items():
            field = _FIELDS_BY_KEY.get(key)
            if key == sources_mod.SOURCES_KEY:
                on_disk[key] = [self._sealed_entry(e) for e in value or [] if isinstance(e, dict)]
                continue
            if key == ROOTS_KEY and self._roots_defaulted:
                # The environment's default is not the operator's choice,
                # so it is not written down as one. `[]` here reads back
                # as "use the default" (see `load`), which keeps a later
                # change to the variable in force.
                on_disk[key] = []
            elif (
                field is not None
                and field.sensitive
                and isinstance(value, str)
                and value
                and self._master_key is not None
            ):
                on_disk[key] = security.seal(value, self._master_key).to_dict()
            else:
                on_disk[key] = value
        # 0600 and replaced rather than rewritten: without a master key
        # the hub token above is plaintext, and a truncate-then-write
        # killed halfway is a library that forgets every folder. See
        # `_private_files`.
        _private_files.write_private_text(
            self._path, yaml.safe_dump(on_disk, sort_keys=True, default_flow_style=False)
        )
