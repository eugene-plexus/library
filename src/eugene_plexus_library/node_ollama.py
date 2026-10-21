"""Last metadata report per worker; a report never creates a Library model."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ._generated.models import LibraryModel, NodeOllamaReport
from ._private_files import write_private_text

log = logging.getLogger(__name__)


class NodeStores:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.reports: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.reports = {
                    node: NodeOllamaReport.model_validate(report).model_dump(
                        mode="json", exclude_none=True
                    )
                    for node, report in raw.items()
                }
            except (OSError, ValueError, AttributeError) as exc:
                log.warning("Cannot read kept Ollama reports from %s: %s", path, exc)

    def replace(self, node: str, body: dict[str, Any]) -> None:
        with self._lock:
            self._replace(node, body)

    def _replace(self, node: str, body: dict[str, Any]) -> None:
        report = deepcopy(body)
        for store in report["stores"]:
            for model in store["models"]:
                model.pop("libraryModel", None)
        next_reports = {**self.reports, node: {"node": node, **report}}
        write_private_text(self.path, json.dumps(next_reports))
        self.reports = next_reports

    def listing(self, models: Sequence[LibraryModel]) -> dict[str, Any]:
        held = {
            m.keptFrom.digest: m.id
            for m in models
            if m.keptFrom is not None and m.status == "present"
        }
        nodes = deepcopy(list(self.reports.values()))
        for node in nodes:
            for store in node["stores"]:
                for model in store["models"]:
                    if model["digest"] in held:
                        model["libraryModel"] = held[model["digest"]]
        return {"nodes": nodes}
