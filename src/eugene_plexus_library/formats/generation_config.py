"""Report author sampling, without supplying defaults or changing decoding."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .._generated.models import RecommendedSampling, RecommendedSamplingSource

log = logging.getLogger(__name__)


def sampling_values(
    values: dict[str, Any], *, source: RecommendedSamplingSource
) -> RecommendedSampling | None:
    """Keep only values the gateway's sampling surface can carry."""
    kept: dict[str, Any] = {}
    for key, maximum in (("temperature", 2), ("topP", 1), ("topK", None), ("minP", 1)):
        value = values.get(key)
        if value is None:
            continue
        if (
            type(value) not in (int, float)
            or (isinstance(value, float) and not math.isfinite(value))
            or (key == "topK" and not isinstance(value, int))
        ):
            log.warning(
                "The file's %s %r is not a valid %s, so it is not used.",
                key,
                value,
                "whole number" if key == "topK" else "number",
            )
        elif value < 0 or (maximum is not None and value > maximum):
            log.warning(
                "The file's %s %s is outside %s, so it is not used.",
                key,
                f"{value:g}" if isinstance(value, float) else str(value),
                f"0-{maximum}" if maximum is not None else "0 or above",
            )
        else:
            kept[key] = value
    return RecommendedSampling(source=source, **kept) if kept else None


def read(directory: Path) -> RecommendedSampling | None:
    """Read only explicit sampling from generation_config.json with do_sample=true.

    Reread this small sidecar even for a cached model: its mtime is independent
    of the weights, and deleting it or choosing greedy must clear old values.
    """
    path = directory / "generation_config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("Cannot read author sampling from %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        log.warning("Cannot read author sampling from %s: expected a JSON object", path)
        return None
    if raw.get("do_sample") is not True:
        if raw.get("do_sample") is False:
            log.info("%s: the author asks for greedy decoding; not used", path)
        return None
    return sampling_values(
        {
            target: raw.get(key)
            for key, target in (
                ("temperature", "temperature"),
                ("top_p", "topP"),
                ("top_k", "topK"),
                ("min_p", "minP"),
            )
        },
        source=RecommendedSamplingSource.generation_config,
    )
