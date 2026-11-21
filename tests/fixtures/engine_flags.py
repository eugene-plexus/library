"""Regenerate `engine_flags.json` from an agent checkout (library#22).

Run it with the agent's venv, so its dependencies import:

    <agent>/.venv/Scripts/python tests/fixtures/engine_flags.py <agent>/src <agent commit>

It writes, for every flag every engine adapter declares, the class the
agent's `operations.flag_class` gives it. `tests/test_operations_flags.py`
compares the Library's lists with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(agent_src: str, commit: str) -> None:
    sys.path.insert(0, agent_src)
    from eugene_plexus_agent import operations  # type: ignore[import-not-found]
    from eugene_plexus_agent.engines import ADAPTERS  # type: ignore[import-not-found]

    engines = {
        kind.value: {
            field.key: operations.flag_class(kind, field) for field in adapter.flag_schema().fields
        }
        for kind, adapter in sorted(ADAPTERS.items(), key=lambda item: item[0].value)
    }
    doc = {
        "source": (
            f"eugene-plexus/agent {commit}: operations.flag_class(engine, field) for every "
            "field of every adapter flag_schema(). Regenerate with tests/fixtures/engine_flags.py."
        ),
        "engines": engines,
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
