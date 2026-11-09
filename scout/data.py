"""Load the offline January 2026 CSVs and drop rows that are impossible in an NBA game.

This is the only module that reads the data/ folder. Everything the agents know
comes through here, so the bounds checks below are the first guardrail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REGULATION_TEAM_MINUTES = 240   # 5 players x 48 minutes
OVERTIME_TEAM_MINUTES = 25      # 5 players x 5 minutes
MAX_SHOT_DISTANCE_FT = 90       # the court is 94 ft long


@dataclass
class Dataset:
    """All the tables the tools read from."""

    team_logs: pd.DataFrame
    player_logs: pd.DataFrame
    shots: pd.DataFrame
    team_opponent_stats: pd.DataFrame
    player_defense: pd.DataFrame
    rotations: pd.DataFrame
    dropped_rows: dict[str, int] = field(default_factory=dict)

    @property
    def team_codes(self) -> set[str]:
        return set(self.team_logs["TEAM_ABBREVIATION"])

    def team_id(self, team_code: str) -> int:
        matches = self.team_logs.loc[self.team_logs["TEAM_ABBREVIATION"] == team_code, "TEAM_ID"]
        if matches.empty:
            raise KeyError(f"Unknown team code: {team_code}")
        return int(matches.iloc[0])


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run scripts/fetch_data.py first.")
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    if "GAME_ID" in df:
        df["GAME_ID"] = df["GAME_ID"].str.zfill(10)  # CSV drops the leading zeros
    return df


def _valid_team_rows(team_logs: pd.DataFrame) -> pd.Series:
    """A team plays 240 minutes, plus 25 for each overtime period."""
    extra_minutes = team_logs["MIN"] - REGULATION_TEAM_MINUTES
    return (extra_minutes >= 0) & (extra_minutes % OVERTIME_TEAM_MINUTES <= 1)


def _valid_player_rows(player_logs: pd.DataFrame, team_logs: pd.DataFrame) -> pd.Series:
    """A player can't play longer than the game lasted, and percentages must be 0-1."""
    game_length = team_logs[["GAME_ID", "TEAM_ID", "MIN"]].rename(columns={"MIN": "TEAM_MIN"})
    merged = player_logs.merge(game_length, on=["GAME_ID", "TEAM_ID"], how="left")
    max_minutes = (merged["TEAM_MIN"] / 5).fillna(48).to_numpy()

    within_game = (player_logs["MIN"] >= 0) & (player_logs["MIN"] <= max_minutes + 0.5)
    valid_percentages = (
        player_logs[["FG_PCT", "FG3_PCT", "FT_PCT"]].fillna(0).apply(lambda col: col.between(0, 1)).all(axis=1)
    )
    return within_game & valid_percentages & (player_logs["PTS"] >= 0)


def _drop_invalid_rows(data: Dataset) -> None:
    team_ok = _valid_team_rows(data.team_logs)
    player_ok = _valid_player_rows(data.player_logs, data.team_logs)
    shot_ok = data.shots["SHOT_DISTANCE"].between(0, MAX_SHOT_DISTANCE_FT)

    data.dropped_rows = {
        "team_logs": int((~team_ok).sum()),
        "player_logs": int((~player_ok).sum()),
        "shots": int((~shot_ok).sum()),
    }
    data.team_logs = data.team_logs[team_ok]
    data.player_logs = data.player_logs[player_ok]
    data.shots = data.shots[shot_ok]


@cache
def load() -> Dataset:
    """Load every CSV once per process and remove impossible rows."""
    data = Dataset(
        team_logs=_read_csv("team_game_logs"),
        player_logs=_read_csv("player_game_logs"),
        shots=_read_csv("shots"),
        team_opponent_stats=_read_csv("team_opponent_stats"),
        player_defense=_read_csv("player_defense"),
        rotations=_read_csv("rotations"),
    )
    _drop_invalid_rows(data)
    return data
