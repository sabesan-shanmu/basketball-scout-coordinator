"""One-time download of January 2026 NBA data into local CSV files.

The agents never use the network: they only read the files this script writes
to data/. Re-running skips files that already exist, and the rotation download
resumes where it stopped.

Usage:
    python scripts/fetch_data.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import (
    gamerotation,
    leaguedashptdefend,
    leaguedashteamstats,
    leaguegamelog,
    shotchartdetail,
)

SEASON = "2025-26"
DATE_FROM = "01/01/2026"
DATE_TO = "01/31/2026"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIMEOUT_SECONDS = 60
PAUSE_SECONDS = 0.7  # stay polite to stats.nba.com

JANUARY = {"date_from_nullable": DATE_FROM, "date_to_nullable": DATE_TO, "timeout": TIMEOUT_SECONDS}


def save_once(name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Download a table and save it as data/<name>.csv, unless the file already exists."""
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        print(f"skip {name} (already downloaded)")
        return pd.read_csv(path, dtype={"GAME_ID": str})

    table = fetch()
    table.to_csv(path, index=False)
    print(f"wrote {name}: {len(table)} rows")
    time.sleep(PAUSE_SECONDS)
    return table


def fetch_team_game_logs() -> pd.DataFrame:
    return leaguegamelog.LeagueGameLog(
        season=SEASON, player_or_team_abbreviation="T", **JANUARY
    ).get_data_frames()[0]


def fetch_player_game_logs() -> pd.DataFrame:
    return leaguegamelog.LeagueGameLog(
        season=SEASON, player_or_team_abbreviation="P", **JANUARY
    ).get_data_frames()[0]


def fetch_shots() -> pd.DataFrame:
    """Every shot attempted in January (player_id=0 and team_id=0 mean league-wide)."""
    return shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        context_measure_simple="FGA",
        season_nullable=SEASON,
        season_type_all_star="Regular Season",
        **JANUARY,
    ).get_data_frames()[0]


def fetch_team_opponent_stats() -> pd.DataFrame:
    """What each team allows its opponents to do, per game."""
    return leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON, measure_type_detailed_defense="Opponent", per_mode_detailed="PerGame", **JANUARY
    ).get_data_frames()[0]


def fetch_player_defense() -> pd.DataFrame:
    """Defender tracking: FG% allowed compared with what the shooters normally hit."""
    return leaguedashptdefend.LeagueDashPtDefend(
        season=SEASON, defense_category="Overall", per_mode_simple="PerGame", **JANUARY
    ).get_data_frames()[0]


def fetch_rotations(game_ids: list[str]) -> None:
    """Substitution data: each player's stints (time in, time out) for every game.

    Appends one game at a time so an interrupted download can resume.
    """
    path = DATA_DIR / "rotations.csv"
    already_saved = set()
    if path.exists():
        already_saved = set(pd.read_csv(path, dtype={"GAME_ID": str})["GAME_ID"].str.zfill(10))

    remaining = sorted(set(game_ids) - already_saved)
    for number, game_id in enumerate(remaining, start=1):
        try:
            home_and_away = gamerotation.GameRotation(game_id=game_id, timeout=TIMEOUT_SECONDS).get_data_frames()
        except Exception as exc:  # keep going; a missing game is reported, never invented
            print(f"rotation FAILED for {game_id}: {exc}")
        else:
            stints = pd.concat(home_and_away, ignore_index=True)
            stints["GAME_ID"] = game_id
            stints.to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"rotation {number}/{len(remaining)}: {game_id}")
        time.sleep(PAUSE_SECONDS)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    team_logs = save_once("team_game_logs", fetch_team_game_logs)
    save_once("player_game_logs", fetch_player_game_logs)
    save_once("shots", fetch_shots)
    save_once("team_opponent_stats", fetch_team_opponent_stats)
    save_once("player_defense", fetch_player_defense)

    game_ids = team_logs["GAME_ID"].astype(str).str.zfill(10).unique().tolist()
    fetch_rotations(game_ids)


if __name__ == "__main__":
    main()
