"""Feature definitions shared by training and inference.

Keeping a single ordered feature list is essential: a model trained with one
ordering must receive exactly the same ordering at prediction time.
"""
from __future__ import annotations

from datetime import datetime

FEATURE_NAMES: list[str] = [
    "home_ppg5",
    "home_gf5",
    "home_ga5",
    "home_btts5",
    "home_corners5",
    "away_ppg5",
    "away_gf5",
    "away_ga5",
    "away_btts5",
    "away_corners5",
    "home_ppg10",
    "away_ppg10",
    "home_rest_days",
    "away_rest_days",
    "h2h_home_ppg",
    "league_home_goals",
    "league_away_goals",
]

DEFAULTS: dict[str, float] = {
    "home_ppg5": 1.35,
    "home_gf5": 1.4,
    "home_ga5": 1.3,
    "home_btts5": 0.5,
    "home_corners5": 5.0,
    "away_ppg5": 1.2,
    "away_gf5": 1.2,
    "away_ga5": 1.4,
    "away_btts5": 0.5,
    "away_corners5": 4.6,
    "home_ppg10": 1.35,
    "away_ppg10": 1.2,
    "home_rest_days": 7.0,
    "away_rest_days": 7.0,
    "h2h_home_ppg": 1.35,
    "league_home_goals": 1.45,
    "league_away_goals": 1.15,
}


def to_vector(metrics: dict) -> list[float]:
    vector: list[float] = []
    for name in FEATURE_NAMES:
        value = metrics.get(name, DEFAULTS[name])
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = DEFAULTS[name]
        if number != number or number in (float("inf"), float("-inf")):
            number = DEFAULTS[name]
        vector.append(round(number, 4))
    return vector


def _rest_days(matches: list[dict], kickoff: datetime) -> float:
    if not matches:
        return 14.0
    last = matches[0].get("kickoff")
    if not last:
        return 14.0
    try:
        parsed = datetime.fromisoformat(str(last).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return 14.0
    return max(2.0, min(30.0, (kickoff - parsed).total_seconds() / 86400.0))


def context_metrics(context: dict) -> dict:
    """Derive the model feature dictionary from a match-context payload."""
    home = context.get("home") or {}
    away = context.get("away") or {}
    home5 = (home.get("profiles") or {}).get("5") or {}
    away5 = (away.get("profiles") or {}).get("5") or {}
    home10 = (home.get("profiles") or {}).get("10") or {}
    away10 = (away.get("profiles") or {}).get("10") or {}
    baseline = context.get("baseline") or {}

    kickoff_raw = (context.get("fixture") or {}).get("kickoff")
    try:
        kickoff = datetime.fromisoformat(str(kickoff_raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        kickoff = datetime.utcnow()

    home_team_id = (home.get("team") or {}).get("id")
    h2h = context.get("h2h") or []
    h2h_points = 0
    h2h_count = 0
    for game in h2h[:5]:
        home_id = (game.get("home") or {}).get("id")
        away_id = (game.get("away") or {}).get("id")
        hs, as_ = game.get("home_score"), game.get("away_score")
        if hs is None or as_ is None:
            continue
        if home_id == home_team_id:
            points = 3 if hs > as_ else (1 if hs == as_ else 0)
        elif away_id == home_team_id:
            points = 3 if as_ > hs else (1 if hs == as_ else 0)
        else:
            continue
        h2h_points += points
        h2h_count += 1

    metrics = {
        "home_ppg5": home5.get("ppg", DEFAULTS["home_ppg5"]),
        "home_gf5": home5.get("avg_goals_for", DEFAULTS["home_gf5"]),
        "home_ga5": home5.get("avg_goals_against", DEFAULTS["home_ga5"]),
        "home_btts5": home5.get("btts_rate", DEFAULTS["home_btts5"]),
        "home_corners5": home5.get("avg_corners_for", DEFAULTS["home_corners5"]),
        "away_ppg5": away5.get("ppg", DEFAULTS["away_ppg5"]),
        "away_gf5": away5.get("avg_goals_for", DEFAULTS["away_gf5"]),
        "away_ga5": away5.get("avg_goals_against", DEFAULTS["away_ga5"]),
        "away_btts5": away5.get("btts_rate", DEFAULTS["away_btts5"]),
        "away_corners5": away5.get("avg_corners_for", DEFAULTS["away_corners5"]),
        "home_ppg10": home10.get("ppg", DEFAULTS["home_ppg10"]),
        "away_ppg10": away10.get("ppg", DEFAULTS["away_ppg10"]),
        "home_rest_days": _rest_days(home.get("matches") or [], kickoff),
        "away_rest_days": _rest_days(away.get("matches") or [], kickoff),
        "h2h_home_ppg": (h2h_points / h2h_count) if h2h_count else DEFAULTS["h2h_home_ppg"],
        "league_home_goals": baseline.get("home_goals_avg", DEFAULTS["league_home_goals"]),
        "league_away_goals": baseline.get("away_goals_avg", DEFAULTS["league_away_goals"]),
    }
    return metrics
