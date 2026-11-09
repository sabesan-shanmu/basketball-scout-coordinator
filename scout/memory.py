"""Long-term memory: an archive of past game plans.

The swarm's factual knowledge is the offline CSV dataset (data/). On top of that,
every finished plan is appended to memory/game_plans.jsonl. Before planning, the
Head Coach is shown the most recent validated plan for the same matchup so it can
stay consistent or explain what changed.

Short-term memory is the LangGraph state shared by the nodes during one run (graph.py).

Both functions take an optional `path` so experiments can use a separate archive.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent.parent / "memory" / "game_plans.jsonl"


def save_plan(my_team: str, opponent: str, plan: str, validated: bool, path: Path = MEMORY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "my_team": my_team,
        "opponent": opponent,
        "validated": validated,
        "plan": plan,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def last_plan(my_team: str, opponent: str, path: Path = MEMORY_FILE) -> dict | None:
    """The most recent validated plan for this matchup, or None."""
    if not path.exists():
        return None

    latest = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["my_team"] == my_team and record["opponent"] == opponent and record["validated"]:
                latest = record
    return latest
