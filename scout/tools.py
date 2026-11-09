"""Analysis tools the agents can call.

Each tool reads only the offline dataset and returns a small dictionary.
Rules every tool follows:
- When data is missing it returns {"found": False, ...} so the agent has no gap to guess at.
- Numbers are rounded to one decimal and percentages are on a 0-100 scale, so the
  validator can match the numbers in a report against the tool output exactly.
"""
from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool

from .data import load

MIN_ZONE_ATTEMPTS = 15        # smaller samples are too noisy to call a zone hot or cold
MIN_CONTESTED_SHOTS = 5.0     # per game, for a defender to be rated
MIN_GAMES = 3
HOT_COLD_THRESHOLD = 5.0      # percentage points above/below league average
NEUTRAL_RANGE = 2.0


# ---------------------------------------------------------------- helpers
def _round(value: float) -> float:
    return round(float(value), 1)


def _percent(made: float, attempts: float) -> float:
    return _round(100 * made / attempts) if attempts else 0.0


def _not_found(what: str) -> dict:
    return {"found": False, "message": f"No January 2026 data for {what}. Do not estimate it."}


def _normalize_team(team: str) -> str | None:
    """Return the upper-case team code, or None if it isn't in the dataset."""
    code = team.upper().strip()
    return code if code in load().team_codes else None


def _shots_by_offense_and_defense() -> pd.DataFrame:
    """Every shot, labelled with the shooting team (OFF) and the defending team (DEF)."""
    data = load()
    team_code_by_id = dict(zip(data.team_logs["TEAM_ID"], data.team_logs["TEAM_ABBREVIATION"]))
    shots = data.shots.copy()
    shots["OFF"] = shots["TEAM_ID"].map(team_code_by_id)
    shots["DEF"] = shots["HTM"].where(shots["OFF"] != shots["HTM"], shots["VTM"])
    return shots


def _zone_summary(shots: pd.DataFrame, zone_column: str = "SHOT_ZONE_BASIC") -> pd.DataFrame:
    """Attempts, FG% and share of attempts per zone, skipping small samples."""
    zones = shots.groupby(zone_column)["SHOT_MADE_FLAG"].agg(made="sum", attempts="count")
    zones = zones[zones["attempts"] >= MIN_ZONE_ATTEMPTS]
    zones["fg_pct"] = (100 * zones["made"] / zones["attempts"]).round(1)
    zones["share_pct"] = (100 * zones["attempts"] / len(shots)).round(1)
    return zones


def _league_fg_pct_by_zone() -> dict[str, float]:
    return _zone_summary(load().shots)["fg_pct"].to_dict()


def _hot_or_cold(difference: float) -> str:
    if difference >= HOT_COLD_THRESHOLD:
        return "hot"
    if difference <= -HOT_COLD_THRESHOLD:
        return "cold"
    return "average"


def _describe_zone(difference: float, side: str) -> str:
    """Spell out what a +/- difference means so the model doesn't have to reason about signs."""
    if abs(difference) < NEUTRAL_RANGE:
        return "about league average"
    shoots_better = difference > 0
    if side == "offense":
        return ("shoots better than league here (strength)" if shoots_better
                else "shoots worse than league here (weakness)")
    return ("opponents shoot better than league here (DEFENSIVE WEAKNESS - attack it)" if shoots_better
            else "opponents shoot worse than league here (defensive strength - avoid)")


def _find_player_shots(player: str) -> pd.DataFrame:
    """Exact name match first, then a partial match."""
    shots = load().shots
    names = shots["PLAYER_NAME"].str.lower()
    query = player.lower().strip()
    exact = shots[names == query]
    return exact if not exact.empty else shots[names.str.contains(query, regex=False)]


def _defender_row(row) -> dict:
    return {
        "player": row.PLAYER_NAME,
        "position": row.PLAYER_POSITION,
        "games": int(row.GP),
        "contested_fga_per_game": _round(row.D_FGA),
        "fg_pct_allowed": _round(100 * row.D_FG_PCT),
        "shooters_normal_fg_pct": _round(100 * row.NORMAL_FG_PCT),
        "pct_plusminus": _round(100 * row.PCT_PLUSMINUS),
    }


# ---------------------------------------------------------------- tools
@tool
def team_summary(team: str) -> dict:
    """January 2026 record and per-game scoring for a team (3-letter code, e.g. 'DEN')."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    logs = load().team_logs
    games = logs[logs["TEAM_ABBREVIATION"] == code]
    opponents = logs[logs["GAME_ID"].isin(games["GAME_ID"]) & (logs["TEAM_ABBREVIATION"] != code)]
    return {
        "found": True,
        "team": code,
        "games": len(games),
        "wins": int((games["WL"] == "W").sum()),
        "losses": int((games["WL"] == "L").sum()),
        "pts_per_game": _round(games["PTS"].mean()),
        "opp_pts_per_game": _round(opponents["PTS"].mean()),
        "fg3a_per_game": _round(games["FG3A"].mean()),
        "fg3_pct": _percent(games["FG3M"].sum(), games["FG3A"].sum()),
        "tov_per_game": _round(games["TOV"].mean()),
    }


@tool
def top_scorers(team: str, n: int = 3) -> dict:
    """A team's top-n January scorers: points, minutes, shot attempts, efficiency."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    logs = load().player_logs
    per_player = logs[logs["TEAM_ABBREVIATION"] == code].groupby("PLAYER_NAME").agg(
        games=("GAME_ID", "nunique"),
        pts=("PTS", "mean"),
        minutes=("MIN", "mean"),
        fga=("FGA", "mean"),
        ast=("AST", "mean"),
        tov=("TOV", "mean"),
        fgm_total=("FGM", "sum"),
        fga_total=("FGA", "sum"),
        fg3m_total=("FG3M", "sum"),
        fg3a_total=("FG3A", "sum"),
    )
    n = max(1, min(n, 6))
    leaders = per_player[per_player["games"] >= MIN_GAMES].nlargest(n, "pts")

    players = [
        {
            "player": name,
            "games": int(row.games),
            "pts_per_game": _round(row.pts),
            "min_per_game": _round(row.minutes),
            "fga_per_game": _round(row.fga),
            "fg_pct": _percent(row.fgm_total, row.fga_total),
            "fg3_pct": _percent(row.fg3m_total, row.fg3a_total),
            "ast_per_game": _round(row.ast),
            "tov_per_game": _round(row.tov),
        }
        for name, row in leaders.iterrows()
    ]
    return {"found": True, "team": code, "players": players}


@tool
def player_shot_profile(player: str) -> dict:
    """Hot/cold zones for one player: FG% and shot share by zone and by distance,
    with the league-average FG% for each zone for comparison."""
    shots = _find_player_shots(player)
    if shots.empty:
        return _not_found(f"player '{player}'")
    if shots["PLAYER_NAME"].nunique() > 1:
        return _not_found(f"player '{player}' (ambiguous name, use the full name)")

    league = _league_fg_pct_by_zone()
    zones = []
    for zone, row in _zone_summary(shots).iterrows():
        league_pct = _round(league.get(zone, row.fg_pct))
        difference = _round(row.fg_pct - league_pct)
        zones.append({
            "zone": zone,
            "attempts": int(row.attempts),
            "fg_pct": _round(row.fg_pct),
            "share_pct": _round(row.share_pct),
            "league_fg_pct": league_pct,
            "vs_league": difference,
            "label": _hot_or_cold(difference),
        })

    distance_splits = [
        {
            "range": distance,
            "attempts": int(row.attempts),
            "fg_pct": _round(row.fg_pct),
            "share_pct": _round(row.share_pct),
        }
        for distance, row in _zone_summary(shots, "SHOT_ZONE_RANGE").iterrows()
    ]
    return {
        "found": True,
        "player": shots["PLAYER_NAME"].iloc[0],
        "total_attempts": len(shots),
        "fg_pct": _percent(shots["SHOT_MADE_FLAG"].sum(), len(shots)),
        "zones": zones,
        "distance_splits": distance_splits,
        "note": f"zones with fewer than {MIN_ZONE_ATTEMPTS} attempts are omitted",
    }


@tool
def team_shot_zones(team: str, side: str = "offense") -> dict:
    """Where a team shoots from (side='offense') or what it allows (side='defense'),
    by zone, versus league-average FG%."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    shots = _shots_by_offense_and_defense()
    column = "OFF" if side == "offense" else "DEF"
    team_shots = shots[shots[column] == code]
    league = _league_fg_pct_by_zone()

    zones = []
    for zone, row in _zone_summary(team_shots).iterrows():
        league_pct = _round(league.get(zone, row.fg_pct))
        difference = _round(row.fg_pct - league_pct)
        zones.append({
            "zone": zone,
            "attempts": int(row.attempts),
            "fg_pct": _round(row.fg_pct),
            "share_pct": _round(row.share_pct),
            "league_fg_pct": league_pct,
            "vs_league": difference,
            "read": _describe_zone(difference, side),
        })
    return {"found": True, "team": code, "side": side, "total_attempts": len(team_shots), "zones": zones}


@tool
def team_defense_profile(team: str) -> dict:
    """What opponents average against this team per game, with league ranks (1 = best defense)."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    data = load()
    stats = data.team_opponent_stats
    rows = stats[stats["TEAM_ID"] == data.team_id(code)]
    if rows.empty:
        return _not_found(f"opponent stats for {code}")
    row = rows.iloc[0]

    profile = {"found": True, "team": code}
    for stat in ("OPP_PTS", "OPP_FG_PCT", "OPP_FG3A", "OPP_FG3_PCT", "OPP_FTA", "OPP_OREB", "OPP_TOV", "OPP_AST"):
        value = row[stat] * 100 if stat.endswith("PCT") else row[stat]
        profile[stat.lower()] = _round(value)
        profile[f"{stat.lower()}_rank"] = int(row[f"{stat}_RANK"])
    return profile


@tool
def weakest_defenders(team: str, n: int = 3) -> dict:
    """Opponent defenders who allow the biggest FG% increase over what shooters normally hit
    (tracking data, January 2026). Higher pct_plusminus = worse defender."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    defense = load().player_defense
    rated = defense[
        (defense["PLAYER_LAST_TEAM_ABBREVIATION"] == code)
        & (defense["D_FGA"] >= MIN_CONTESTED_SHOTS)
        & (defense["GP"] >= MIN_GAMES)
    ].sort_values("PCT_PLUSMINUS", ascending=False)
    if rated.empty:
        return _not_found(f"defender tracking for {code}")

    return {
        "found": True,
        "team": code,
        "weakest": [_defender_row(row) for row in rated.head(n).itertuples()],
        "strongest": [_defender_row(row) for row in rated.tail(2).iloc[::-1].itertuples()],
        "note": f"only defenders contesting at least {MIN_CONTESTED_SHOTS} FGA per game are rated",
    }


@tool
def player_vs_opponent(player: str, opponent: str) -> dict:
    """A player's box-score line in January games against a specific opponent, and their
    January average for comparison. Returns found=False if they did not meet."""
    code = _normalize_team(opponent)
    if not code:
        return _not_found(f"team '{opponent}'")

    logs = load().player_logs
    games = logs[logs["PLAYER_NAME"].str.lower() == player.lower().strip()]
    if games.empty:
        return _not_found(f"player '{player}'")

    name = games["PLAYER_NAME"].iloc[0]
    january_average = {
        "pts": _round(games["PTS"].mean()),
        "fg_pct": _percent(games["FGM"].sum(), games["FGA"].sum()),
        "fg3_pct": _percent(games["FG3M"].sum(), games["FG3A"].sum()),
        "tov": _round(games["TOV"].mean()),
    }
    head_to_head = games[games["MATCHUP"].str.endswith(code)]
    if head_to_head.empty:
        return {
            "found": False,
            "player": name,
            "january_average": january_average,
            "message": (f"{name} did not play {code} in January 2026. Do not invent head-to-head numbers; "
                        "use team_shot_zones/team_defense_profile instead."),
        }

    box_scores = [
        {"date": row.GAME_DATE, "min": _round(row.MIN), "pts": int(row.PTS),
         "fgm": int(row.FGM), "fga": int(row.FGA), "fg3m": int(row.FG3M), "fg3a": int(row.FG3A),
         "ast": int(row.AST), "tov": int(row.TOV), "result": row.WL}
        for row in head_to_head.itertuples()
    ]
    return {"found": True, "player": name, "opponent": code,
            "january_average": january_average, "games": box_scores}


@tool
def rotation_patterns(team: str) -> dict:
    """Substitution patterns: average minutes, when starters first sit,
    and who most often closes games (on the floor at the final buzzer)."""
    code = _normalize_team(team)
    if not code:
        return _not_found(f"team '{team}'")

    data = load()
    stints = data.rotations[data.rotations["TEAM_ID"] == data.team_id(code)].copy()
    if stints.empty:
        return _not_found(f"rotations for {code}")

    # Times are stored in tenths of a second.
    stints["player"] = stints["PLAYER_FIRST"] + " " + stints["PLAYER_LAST"]
    stints["in_min"] = stints["IN_TIME_REAL"] / 600
    stints["out_min"] = stints["OUT_TIME_REAL"] / 600
    stints["length"] = stints["out_min"] - stints["in_min"]
    game_count = stints["GAME_ID"].nunique()

    minutes_per_game = stints.groupby(["player", "GAME_ID"])["length"].sum().groupby("player").mean()
    opening_stints = stints[stints["in_min"] == 0]
    starts = opening_stints.groupby("player")["GAME_ID"].nunique()
    first_sub_out = opening_stints.groupby("player")["out_min"].mean()

    game_end = stints.groupby("GAME_ID")["out_min"].transform("max")
    games_closed = stints[stints["out_min"] >= game_end - 0.01].groupby("player")["GAME_ID"].nunique()

    players = []
    for player in minutes_per_game.nlargest(9).index:
        players.append({
            "player": player,
            "min_per_game": _round(minutes_per_game[player]),
            "starts": int(starts.get(player, 0)),
            "first_sub_out_minute": _round(first_sub_out[player]) if player in first_sub_out else None,
            "closing_pct": _round(100 * games_closed.get(player, 0) / game_count),
        })
    return {"found": True, "team": code, "games": game_count, "players": players}


ALL_TOOLS = {
    t.name: t
    for t in (team_summary, top_scorers, player_shot_profile, team_shot_zones,
              team_defense_profile, weakest_defenders, player_vs_opponent, rotation_patterns)
}


def to_json(result: dict) -> str:
    return json.dumps(result, default=str)
