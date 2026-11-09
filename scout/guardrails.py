"""Guardrails around the LLM agents.

1. Input check   - the matchup must name two different teams that are in the dataset.
2. Grounding     - every number in a report must appear in that agent's tool output.
3. Bounds check  - claims that are impossible in an NBA game are rejected.

(The fourth guardrail, the per-role tool allowlist, is enforced in graph.py.)
"""
from __future__ import annotations

import json
import re

from .data import load

NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")
MINUTES_CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes|min)\b", re.IGNORECASE)
PERCENT_CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Small integers ("3 keys", "Q4", list numbering) and years are not statistics.
NON_STAT_NUMBERS = set(range(11)) | {2025, 2026}
MATCH_TOLERANCE = 0.051
MAX_GAME_MINUTES = 48 + 3 * 5   # regulation plus triple overtime
MIN_REPORT_LENGTH = 80


def check_matchup(my_team: str, opponent: str) -> tuple[str, str]:
    """Return the normalized team codes, or raise ValueError with the valid options."""
    my_team, opponent = my_team.upper().strip(), opponent.upper().strip()
    valid_codes = load().team_codes

    unknown = [code for code in (my_team, opponent) if code not in valid_codes]
    if unknown:
        raise ValueError(f"Unknown team code(s) {unknown}. Valid codes: {sorted(valid_codes)}")
    if my_team == opponent:
        raise ValueError("A team cannot scout itself.")
    return my_team, opponent


def _collect_numbers(value, numbers: set[float]) -> None:
    """Add every number found anywhere inside a JSON value (recursively)."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numbers.add(round(float(value), 1))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, numbers)
    elif isinstance(value, list):
        for item in value:
            _collect_numbers(item, numbers)
    elif isinstance(value, str):
        for match in NUMBER.finditer(value):
            numbers.add(round(float(match.group(1)), 1))


def numbers_in_evidence(evidence: list[str]) -> set[float]:
    """All numbers the agent was shown. Evidence items are JSON tool results or plain text."""
    numbers: set[float] = set()
    for item in evidence:
        try:
            _collect_numbers(json.loads(item), numbers)
        except json.JSONDecodeError:
            _collect_numbers(item, numbers)
    # "-4.2 vs league" may be written as "4.2 below league".
    return numbers | {abs(n) for n in numbers}


def ungrounded_numbers(report: str, evidence: list[str]) -> list[str]:
    """Numbers in the report that don't appear in any tool output."""
    allowed = numbers_in_evidence(evidence)
    ungrounded = []
    for match in NUMBER.finditer(report):
        value = float(match.group(1))
        if value.is_integer() and int(value) in NON_STAT_NUMBERS:
            continue
        if not any(abs(value - known) <= MATCH_TOLERANCE for known in allowed):
            ungrounded.append(match.group(0))
    return ungrounded


def impossible_claims(report: str) -> list[str]:
    problems = []
    for match in MINUTES_CLAIM.finditer(report):
        if float(match.group(1)) > MAX_GAME_MINUTES:
            problems.append(f"'{match.group(0)}' exceeds the length of an NBA game")
    for match in PERCENT_CLAIM.finditer(report):
        if float(match.group(1)) > 100:
            problems.append(f"'{match.group(0)}' is not a valid percentage")
    return problems


def validate_report(report: str, evidence: list[str]) -> list[str]:
    """Return the list of problems with a report. An empty list means it passes."""
    problems = []
    if not evidence:
        problems.append("No tools were called, so the report has no data behind it.")
    if len(report.strip()) < MIN_REPORT_LENGTH:
        problems.append("Report is empty or too short.")

    ungrounded = sorted(set(ungrounded_numbers(report, evidence)))
    if ungrounded:
        problems.append(f"Numbers not found in any tool output: {', '.join(ungrounded[:10])}")

    problems.extend(impossible_claims(report))
    return problems
