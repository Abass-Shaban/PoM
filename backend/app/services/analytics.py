"""Football analytics: form, venue splits, league baselines and head-to-head.

Everything here is derived from real stored fixtures.  When the sample is thin
we fall back to league-level priors and expose the sample size / prior usage in
the response so the UI can be transparent about data quality.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..errors import NotFoundError, ProviderError
from ..models import Fixture, League, Team, utcnow
from .ingest import ensure_team_history
from .provider.base import H2HGameDTO, MatchSummaryDTO
from .provider.factory import get_provider
from .serialize import fixture_dict, iso_utc, league_dict, team_dict

logger = logging.getLogger("blaque.analytics")

WINDOWS = (5, 10, 15)

# League-level priors used only when the stored sample is insufficient.
PRIOR_HOME_GOALS = 1.45
PRIOR_AWAY_GOALS = 1.15
PRIOR_HOME_CORNERS = 5.4
PRIOR_AWAY_CORNERS = 4.6


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None else None


def _weighted_average(
    rows: Iterable[dict],
    key: str,
    before: datetime,
    half_life_days: float | None = None,
) -> float | None:
    half_life = half_life_days or settings.prediction_decay_days
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(key)
        kickoff = row.get("_kickoff")
        if value is None or kickoff is None:
            continue
        age = max((before - kickoff).total_seconds() / 86400.0, 0.0)
        weight = 0.5 ** (age / half_life)
        numerator += weight * float(value)
        denominator += weight
    return numerator / denominator if denominator else None


def team_matches(
    db: Session,
    team_id: int,
    before: datetime,
    limit: int = 20,
    league_slug: str | None = None,
) -> list[Fixture]:
    stmt = (
        select(Fixture)
        .where(
            Fixture.status == "finished",
            Fixture.kickoff < before,
            or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id),
            Fixture.home_score.is_not(None),
            Fixture.away_score.is_not(None),
        )
        .order_by(Fixture.kickoff.desc())
        .limit(limit)
    )
    if league_slug:
        stmt = stmt.where(Fixture.league_slug == league_slug)
    return list(db.scalars(stmt).unique())


def _team_match_row(fixture: Fixture, team_id: int) -> dict:
    is_home = fixture.home_team_id == team_id
    goals_for = fixture.home_score if is_home else fixture.away_score
    goals_against = fixture.away_score if is_home else fixture.home_score
    if goals_for is None or goals_against is None:
        result = "-"
    elif goals_for > goals_against:
        result = "W"
    elif goals_for == goals_against:
        result = "D"
    else:
        result = "L"
    corners_for = fixture.home_corners if is_home else fixture.away_corners
    corners_against = fixture.away_corners if is_home else fixture.home_corners
    opponent = fixture.away_team if is_home else fixture.home_team
    opponent_id = fixture.away_team_id if is_home else fixture.home_team_id
    return {
        "id": fixture.id,
        "kickoff": iso_utc(fixture.kickoff),
        "_kickoff": fixture.kickoff,
        "is_home": is_home,
        "opponent": team_dict(opponent),
        "opponent_id": opponent_id,
        "score": f"{goals_for}-{goals_against}",
        "scoreline": fixture.scoreline,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": (goals_for - goals_against) if goals_for is not None and goals_against is not None else None,
        "result": result,
        "points": 3 if result == "W" else (1 if result == "D" else 0),
        "btts": bool(goals_for and goals_against),
        "over_2_5": bool(goals_for is not None and goals_against is not None and goals_for + goals_against > 2.5),
        "corners_for": corners_for,
        "corners_against": corners_against,
        "total_corners": (corners_for + corners_against)
        if corners_for is not None and corners_against is not None
        else None,
        "league": fixture.league.name if fixture.league else fixture.league_slug,
        "league_slug": fixture.league_slug,
        "venue": fixture.venue,
    }


def _venue_split(rows: list[dict], at_home: bool) -> dict:
    subset = [r for r in rows if r["is_home"] is at_home]
    played = len(subset)
    wins = sum(1 for r in subset if r["result"] == "W")
    draws = sum(1 for r in subset if r["result"] == "D")
    losses = sum(1 for r in subset if r["result"] == "L")
    goals_for = sum(r["goals_for"] or 0 for r in subset)
    goals_against = sum(r["goals_against"] or 0 for r in subset)
    corner_rows = [r for r in subset if r["corners_for"] is not None and r["corners_against"] is not None]
    corners_for = sum(r["corners_for"] for r in corner_rows)
    corners_against = sum(r["corners_against"] for r in corner_rows)
    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "avg_goals_for": _round(goals_for / played) if played else None,
        "avg_goals_against": _round(goals_against / played) if played else None,
        "corner_sample": len(corner_rows),
        "avg_corners_for": _round(corners_for / len(corner_rows)) if corner_rows else None,
        "avg_corners_against": _round(corners_against / len(corner_rows)) if corner_rows else None,
    }


def team_profile(
    db: Session,
    team: Team,
    matches: list[Fixture],
    before: datetime,
    window: int,
    baseline: dict | None = None,
    league_position: dict | None = None,
) -> dict:
    rows = [_team_match_row(fixture, team.id) for fixture in matches[:window]]
    played = len(rows)
    wins = sum(1 for r in rows if r["result"] == "W")
    draws = sum(1 for r in rows if r["result"] == "D")
    losses = sum(1 for r in rows if r["result"] == "L")
    goals_for = sum(r["goals_for"] or 0 for r in rows)
    goals_against = sum(r["goals_against"] or 0 for r in rows)
    corner_rows = [r for r in rows if r["corners_for"] is not None and r["corners_against"] is not None]
    corners_for = sum(r["corners_for"] for r in corner_rows)
    corners_against = sum(r["corners_against"] for r in corner_rows)

    clean_sheets = sum(1 for r in rows if r["goals_against"] == 0)
    failed_to_score = sum(1 for r in rows if r["goals_for"] == 0)
    btts = sum(1 for r in rows if r["btts"])
    over25 = sum(1 for r in rows if r["over_2_5"])
    points = wins * 3 + draws

    return {
        "window": window,
        "requested_window": window,
        "sample_size": played,
        "complete": played >= window,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "ppg": _round(points / played) if played else None,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "avg_goals_for": _round(goals_for / played) if played else None,
        "avg_goals_against": _round(goals_against / played) if played else None,
        "weighted_goals_for": _round(_weighted_average(rows, "goals_for", before)),
        "weighted_goals_against": _round(_weighted_average(rows, "goals_against", before)),
        "clean_sheets": clean_sheets,
        "clean_sheet_rate": _round(clean_sheets / played) if played else None,
        "failed_to_score": failed_to_score,
        "btts_count": btts,
        "btts_rate": _round(btts / played) if played else None,
        "over_2_5_count": over25,
        "over_2_5_rate": _round(over25 / played) if played else None,
        "corner_sample": len(corner_rows),
        "corners_for": corners_for if corner_rows else None,
        "corners_against": corners_against if corner_rows else None,
        "avg_corners_for": _round(corners_for / len(corner_rows)) if corner_rows else None,
        "avg_corners_against": _round(corners_against / len(corner_rows)) if corner_rows else None,
        "weighted_corners_for": _round(_weighted_average(corner_rows, "corners_for", before)),
        "weighted_corners_against": _round(_weighted_average(corner_rows, "corners_against", before)),
        "form": "".join(r["result"] for r in rows[:5]),
        "home": _venue_split(rows, True),
        "away": _venue_split(rows, False),
        "league_position": league_position,
        "matches": [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
        "team": team_dict(team),
        "baseline_sample": (baseline or {}).get("sample_size"),
    }


def league_baseline(db: Session, league_slug: str, before: datetime, limit: int = 400) -> dict:
    fixtures = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.league_slug == league_slug,
                Fixture.status == "finished",
                Fixture.kickoff < before,
                Fixture.home_score.is_not(None),
                Fixture.away_score.is_not(None),
            )
            .order_by(Fixture.kickoff.desc())
            .limit(limit)
        )
    )
    sample = len(fixtures)
    if sample:
        home_goals = sum(f.home_score or 0 for f in fixtures) / sample
        away_goals = sum(f.away_score or 0 for f in fixtures) / sample
        btts = sum(1 for f in fixtures if (f.home_score or 0) > 0 and (f.away_score or 0) > 0) / sample
        over25 = sum(1 for f in fixtures if (f.home_score or 0) + (f.away_score or 0) > 2.5) / sample
        home_wins = sum(1 for f in fixtures if (f.home_score or 0) > (f.away_score or 0)) / sample
        draws = sum(1 for f in fixtures if f.home_score == f.away_score) / sample
        away_wins = sum(1 for f in fixtures if (f.home_score or 0) < (f.away_score or 0)) / sample
        corner_fixtures = [f for f in fixtures if f.home_corners is not None and f.away_corners is not None]
        corner_sample = len(corner_fixtures)
        home_corners = (
            sum(f.home_corners or 0 for f in corner_fixtures) / corner_sample
            if corner_sample
            else PRIOR_HOME_CORNERS
        )
        away_corners = (
            sum(f.away_corners or 0 for f in corner_fixtures) / corner_sample
            if corner_sample
            else PRIOR_AWAY_CORNERS
        )
    else:
        home_goals, away_goals = PRIOR_HOME_GOALS, PRIOR_AWAY_GOALS
        btts, over25, home_wins, draws, away_wins = 0.5, 0.5, 0.45, 0.25, 0.30
        corner_sample = 0
        home_corners, away_corners = PRIOR_HOME_CORNERS, PRIOR_AWAY_CORNERS

    # Empirical-Bayes shrinkage toward the league prior for small samples.
    goal_weight = max(0.0, (20 - sample) / 20)
    home_goals = home_goals * (1 - goal_weight) + PRIOR_HOME_GOALS * goal_weight
    away_goals = away_goals * (1 - goal_weight) + PRIOR_AWAY_GOALS * goal_weight
    corner_weight = max(0.0, (10 - corner_sample) / 10)
    home_corners = home_corners * (1 - corner_weight) + PRIOR_HOME_CORNERS * corner_weight
    away_corners = away_corners * (1 - corner_weight) + PRIOR_AWAY_CORNERS * corner_weight

    return {
        "league_slug": league_slug,
        "sample_size": sample,
        "corner_sample": corner_sample,
        "priors_used": sample < 20 or corner_sample < 10,
        "home_goals_avg": _round(home_goals),
        "away_goals_avg": _round(away_goals),
        "total_goals_avg": _round(home_goals + away_goals),
        "home_corners_avg": _round(home_corners),
        "away_corners_avg": _round(away_corners),
        "total_corners_avg": _round(home_corners + away_corners),
        "btts_rate": _round(btts),
        "over_2_5_rate": _round(over25),
        "home_win_rate": _round(home_wins),
        "draw_rate": _round(draws),
        "away_win_rate": _round(away_wins),
    }


def h2h_matches(
    db: Session, team_a: int, team_b: int, before: datetime, limit: int = 10
) -> list[dict]:
    fixtures = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status == "finished",
                Fixture.kickoff < before,
                or_(
                    and_(Fixture.home_team_id == team_a, Fixture.away_team_id == team_b),
                    and_(Fixture.home_team_id == team_b, Fixture.away_team_id == team_a),
                ),
            )
            .order_by(Fixture.kickoff.desc())
            .limit(limit)
        ).unique()
    )
    return [_h2h_dict(fixture) for fixture in fixtures]


def _h2h_dict(fixture: Fixture) -> dict:
    if fixture.home_score is None or fixture.away_score is None:
        winner = None
    elif fixture.home_score > fixture.away_score:
        winner = "home"
    elif fixture.home_score < fixture.away_score:
        winner = "away"
    else:
        winner = "draw"
    return {
        "id": fixture.id,
        "date": iso_utc(fixture.kickoff),
        "home": team_dict(fixture.home_team),
        "away": team_dict(fixture.away_team),
        "home_score": fixture.home_score,
        "away_score": fixture.away_score,
        "winner": winner,
        "league": fixture.league.name if fixture.league else fixture.league_slug,
        "league_slug": fixture.league_slug,
    }


def _provider_h2h(
    db: Session, games: list[H2HGameDTO], home_id: int, away_id: int
) -> list[dict]:
    output = []
    for game in games:
        if {game.home_id, game.away_id} != {home_id, away_id}:
            continue
        home_team = db.get(Team, game.home_id)
        away_team = db.get(Team, game.away_id)
        if game.home_score is None or game.away_score is None:
            winner = None
        elif game.home_score > game.away_score:
            winner = "home"
        elif game.home_score < game.away_score:
            winner = "away"
        else:
            winner = "draw"
        output.append(
            {
                "id": game.id,
                "date": iso_utc(game.date),
                "home": team_dict(home_team)
                if home_team
                else {"id": game.home_id, "name": game.home_name, "logo": None},
                "away": team_dict(away_team)
                if away_team
                else {"id": game.away_id, "name": game.away_name, "logo": None},
                "home_score": game.home_score,
                "away_score": game.away_score,
                "winner": winner,
                "league": game.league,
                "league_slug": None,
            }
        )
    return output


def _merge_h2h(stored: list[dict], extra: list[dict], limit: int = 10) -> list[dict]:
    merged: dict = {}
    for game in stored + extra:
        key = game.get("id") or f"{game.get('date')}-{game.get('home', {}).get('id')}"
        if key not in merged:
            merged[key] = game
    ordered = sorted(merged.values(), key=lambda g: g.get("date") or "", reverse=True)
    return ordered[:limit]


def standings_lookup(db: Session, league_slug: str) -> dict[int, dict]:
    provider = get_provider()
    try:
        table = provider.standings(league_slug)
    except ProviderError:
        return {}
    lookup: dict[int, dict] = {}
    for row in table.get("rows") or []:
        lookup[row.team.id] = {
            "rank": row.rank,
            "points": row.points,
            "played": row.played,
            "group": row.group,
            "form": row.form,
        }
    return lookup


def _apply_summary(db: Session, fixture: Fixture, summary: MatchSummaryDTO) -> None:
    home = summary.home_stats or {}
    away = summary.away_stats or {}
    if home:
        fixture.home_corners = int(home["wonCorners"]) if home.get("wonCorners") is not None else fixture.home_corners
        fixture.home_shots = int(home["totalShots"]) if home.get("totalShots") is not None else fixture.home_shots
        fixture.home_shots_on_target = int(home["shotsOnTarget"]) if home.get("shotsOnTarget") is not None else fixture.home_shots_on_target
        fixture.home_possession = home.get("possessionPct", fixture.home_possession)
        fixture.home_fouls = int(home["foulsCommitted"]) if home.get("foulsCommitted") is not None else fixture.home_fouls
        fixture.home_yellow_cards = int(home["yellowCards"]) if home.get("yellowCards") is not None else fixture.home_yellow_cards
        fixture.home_red_cards = int(home["redCards"]) if home.get("redCards") is not None else fixture.home_red_cards
    if away:
        fixture.away_corners = int(away["wonCorners"]) if away.get("wonCorners") is not None else fixture.away_corners
        fixture.away_shots = int(away["totalShots"]) if away.get("totalShots") is not None else fixture.away_shots
        fixture.away_shots_on_target = int(away["shotsOnTarget"]) if away.get("shotsOnTarget") is not None else fixture.away_shots_on_target
        fixture.away_possession = away.get("possessionPct", fixture.away_possession)
        fixture.away_fouls = int(away["foulsCommitted"]) if away.get("foulsCommitted") is not None else fixture.away_fouls
        fixture.away_yellow_cards = int(away["yellowCards"]) if away.get("yellowCards") is not None else fixture.away_yellow_cards
        fixture.away_red_cards = int(away["redCards"]) if away.get("redCards") is not None else fixture.away_red_cards
    if summary.odds:
        fixture.odds_json = summary.odds
    if summary.venue:
        fixture.venue = summary.venue
    fixture.updated_at = utcnow()


def build_match_context(
    db: Session,
    fixture: Fixture,
    window: int = 10,
    *,
    include_summary: bool = True,
    apply_summary: bool = True,
) -> dict:
    home = fixture.home_team
    away = fixture.away_team
    if home is None or away is None:
        raise NotFoundError("One of the teams for this fixture could not be found.")

    warnings: list[str] = []
    # Backfill history when the local database has fewer matches than requested.
    for team in (home, away):
        try:
            ensure_team_history(db, team.id, max(window, 15))
        except Exception:  # noqa: BLE001
            logger.warning("Could not backfill history for %s", team.id, exc_info=True)
            warnings.append(f"Could not refresh {team.name}'s match history from the provider.")

    before = fixture.kickoff
    home_matches = team_matches(db, home.id, before, limit=20)
    away_matches = team_matches(db, away.id, before, limit=20)
    baseline = league_baseline(db, fixture.league_slug, before)

    table = standings_lookup(db, fixture.league_slug)

    summary: MatchSummaryDTO | None = None
    if include_summary:
        try:
            summary = get_provider().summary(fixture.league_slug, fixture.id)
            if apply_summary:
                _apply_summary(db, fixture, summary)
                db.commit()
        except ProviderError as exc:
            warnings.append(f"Match summary unavailable: {exc.message}")
        except Exception:  # noqa: BLE001
            logger.warning("Summary enrichment failed for fixture %s", fixture.id, exc_info=True)
            warnings.append("Detailed match summary could not be loaded.")

    home_profiles = {
        str(w): team_profile(
            db, home, home_matches, before, w, baseline, table.get(home.id)
        )
        for w in WINDOWS
    }
    away_profiles = {
        str(w): team_profile(
            db, away, away_matches, before, w, baseline, table.get(away.id)
        )
        for w in WINDOWS
    }

    if len(home_matches) < window:
        warnings.append(
            f"Only {len(home_matches)} completed {home.name} matches are available before kick-off."
        )
    if len(away_matches) < window:
        warnings.append(
            f"Only {len(away_matches)} completed {away.name} matches are available before kick-off."
        )
    if baseline.get("priors_used"):
        warnings.append("League averages use a small sample and include prior shrinkage.")
    if baseline.get("corner_sample", 0) < 10:
        warnings.append("Corner projections rely on limited corner data.")
    if fixture.status in ("live", "finished"):
        warnings.append(
            "This match is already underway or complete; the model output is a pre-kick-off snapshot."
        )

    stored_h2h = h2h_matches(db, home.id, away.id, before)
    extra_h2h = _provider_h2h(db, summary.h2h, home.id, away.id) if summary else []
    h2h = _merge_h2h(stored_h2h, extra_h2h)

    return {
        "fixture": fixture_dict(fixture),
        "league": league_dict(fixture.league),
        "home": {
            "team": team_dict(home),
            "profiles": home_profiles,
            "matches": home_profiles[str(window)]["matches"],
            "position": table.get(home.id),
        },
        "away": {
            "team": team_dict(away),
            "profiles": away_profiles,
            "matches": away_profiles[str(window)]["matches"],
            "position": table.get(away.id),
        },
        "h2h": h2h,
        "baseline": baseline,
        "summary": {
            "odds": summary.odds if summary else fixture.odds_json,
            "venue": summary.venue if summary else fixture.venue,
            "attendance": summary.attendance if summary else None,
            "last_five": {
                str(team_id): games for team_id, games in (summary.last_five.items() if summary else [])
            },
        },
        "data_quality": {
            "warnings": warnings,
            "home_sample": len(home_matches),
            "away_sample": len(away_matches),
            "h2h_sample": len(h2h),
            "baseline_sample": baseline.get("sample_size"),
            "corner_sample": baseline.get("corner_sample"),
            "window": window,
        },
        "generated_at": iso_utc(utcnow()),
    }


def team_dashboard(db: Session, team_id: int, window: int = 10) -> dict:
    team = db.get(Team, team_id)
    if team is None:
        raise NotFoundError("Team not found.")
    league_slug = team.league_slug
    if league_slug is None:
        league_slug = db.scalar(
            select(Fixture.league_slug)
            .where((Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id))
            .order_by(Fixture.kickoff.desc())
            .limit(1)
        )
    try:
        ensure_team_history(db, team_id, 15)
    except Exception:  # noqa: BLE001
        logger.warning("Team history refresh failed for %s", team_id, exc_info=True)
    before = utcnow()
    matches = team_matches(db, team_id, before, limit=20)
    baseline = league_baseline(db, league_slug, before) if league_slug else None
    table = standings_lookup(db, league_slug) if league_slug else {}
    profiles = {
        str(w): team_profile(db, team, matches, before, w, baseline, table.get(team_id))
        for w in WINDOWS
    }
    league = db.get(League, league_slug) if league_slug else None
    upcoming = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status.in_(["scheduled", "live"]),
                Fixture.kickoff >= before,
                or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id),
            )
            .order_by(Fixture.kickoff.asc())
            .limit(5)
        ).unique()
    )
    return {
        "team": team_dict(team),
        "league": league_dict(league),
        "position": table.get(team_id),
        "profiles": profiles,
        "matches": profiles[str(window)]["matches"],
        "upcoming": [fixture_dict(f) for f in upcoming],
        "baseline": baseline,
    }
