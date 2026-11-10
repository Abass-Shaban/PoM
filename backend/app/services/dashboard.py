"""Dashboard aggregation: fixtures, live scores, league pulse and charts."""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, time as dtime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import Fixture, League, PredictionRecord, SavedPrediction, Team, utcnow
from .ingest import refresh_today
from .ml import model_status
from .runtime import get_featured_slugs
from .serialize import fixture_dict, iso_utc, league_dict

logger = logging.getLogger("blaque.dashboard")

_REFRESH_LOCK = threading.Lock()
_LAST_REFRESH = 0.0
_REFRESH_INTERVAL = 60.0


def _maybe_refresh_today(force: bool = False) -> bool:
    global _LAST_REFRESH
    now = time.time()
    if not force and now - _LAST_REFRESH < _REFRESH_INTERVAL:
        return False
    if not _REFRESH_LOCK.acquire(blocking=False):
        return False
    try:
        if not force and time.time() - _LAST_REFRESH < _REFRESH_INTERVAL:
            return False
        refresh_today()
        _LAST_REFRESH = time.time()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Dashboard live refresh failed", exc_info=True)
        return False
    finally:
        _REFRESH_LOCK.release()


def _fixtures_in_range(db: Session, start: datetime, end: datetime, limit: int = 60) -> list[Fixture]:
    return list(
        db.scalars(
            select(Fixture)
            .where(Fixture.kickoff >= start, Fixture.kickoff < end)
            .order_by(Fixture.kickoff.asc())
            .limit(limit)
        ).unique()
    )


def _league_pulse(db: Session) -> list[dict]:
    leagues = list(
        db.scalars(
            select(League).where(League.is_featured.is_(True)).order_by(League.sort_order.asc())
        )
    )
    if not leagues:
        leagues = list(db.scalars(select(League).order_by(League.sort_order.asc()).limit(12)))
    now = utcnow()
    output = []
    for league in leagues:
        total = db.scalar(
            select(func.count()).select_from(Fixture).where(Fixture.league_slug == league.slug)
        ) or 0
        upcoming = db.scalar(
            select(func.count())
            .select_from(Fixture)
            .where(
                Fixture.league_slug == league.slug,
                Fixture.status == "scheduled",
                Fixture.kickoff >= now,
            )
        ) or 0
        live = db.scalar(
            select(func.count())
            .select_from(Fixture)
            .where(Fixture.league_slug == league.slug, Fixture.status.in_(["live", "suspended"]))
        ) or 0
        output.append(
            {
                **league_dict(league),
                "fixtures": int(total),
                "upcoming": int(upcoming),
                "live": int(live),
            }
        )
    return output


def _chart_payload(db: Session, start: datetime) -> dict:
    fixtures = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status == "finished",
                Fixture.kickoff >= start,
                Fixture.home_score.is_not(None),
                Fixture.away_score.is_not(None),
            )
            .order_by(Fixture.kickoff.desc())
            .limit(3000)
        ).unique()
    )
    by_league: dict[str, dict] = {}
    outcomes = {"home": 0, "draw": 0, "away": 0}
    btts = {"yes": 0, "no": 0}
    over = {"over": 0, "under": 0}
    goals = 0
    for fixture in fixtures:
        hs = fixture.home_score or 0
        away_score = fixture.away_score or 0
        total = hs + away_score
        goals += total
        name = fixture.league.name if fixture.league else fixture.league_slug
        bucket = by_league.setdefault(name, {"league": name, "matches": 0, "goals": 0})
        bucket["matches"] += 1
        bucket["goals"] += total
        if hs > away_score:
            outcomes["home"] += 1
        elif hs == away_score:
            outcomes["draw"] += 1
        else:
            outcomes["away"] += 1
        if hs > 0 and away_score > 0:
            btts["yes"] += 1
        else:
            btts["no"] += 1
        if total > 2.5:
            over["over"] += 1
        else:
            over["under"] += 1
    leagues = [
        {
            **bucket,
            "avg_goals": round(bucket["goals"] / bucket["matches"], 2) if bucket["matches"] else 0,
        }
        for bucket in by_league.values()
    ]
    leagues.sort(key=lambda item: item["matches"], reverse=True)
    return {
        "goals_by_league": leagues[:8],
        "outcomes": outcomes,
        "btts": btts,
        "over_2_5": over,
        "sample_matches": len(fixtures),
        "total_goals": goals,
        "avg_goals": round(goals / len(fixtures), 2) if fixtures else 0,
    }


def build_dashboard(db: Session, day: date | None = None, *, refresh: bool = False) -> dict:
    day = day or utcnow().date()
    is_today = day == utcnow().date()
    refreshed = False
    if is_today and refresh:
        refreshed = _maybe_refresh_today(force=True)
    elif is_today:
        _maybe_refresh_today(force=False)

    start = datetime.combine(day, dtime.min)
    end = start + timedelta(days=1)
    now = utcnow()

    today = _fixtures_in_range(db, start, end, limit=80)
    live = list(
        db.scalars(
            select(Fixture)
            .where(Fixture.status.in_(["live", "suspended"]))
            .order_by(Fixture.kickoff.asc())
            .limit(20)
        ).unique()
    )
    upcoming = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status == "scheduled",
                Fixture.kickoff >= now - timedelta(hours=2),
            )
            .order_by(Fixture.kickoff.asc())
            .limit(12)
        ).unique()
    )
    recent = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status == "finished",
                Fixture.kickoff >= now - timedelta(days=5),
            )
            .order_by(Fixture.kickoff.desc())
            .limit(12)
        ).unique()
    )

    finished_today = [f for f in today if f.status == "finished"]
    live_today = [f for f in today if f.status in ("live", "suspended")]
    goals_today = sum((f.home_score or 0) + (f.away_score or 0) for f in finished_today)
    btts_today = sum(
        1 for f in finished_today if (f.home_score or 0) > 0 and (f.away_score or 0) > 0
    )

    tracked_teams = db.scalar(select(func.count()).select_from(Team)) or 0
    tracked_leagues = db.scalar(select(func.count()).select_from(League)) or 0
    predictions_today = db.scalar(
        select(func.count())
        .select_from(PredictionRecord)
        .where(PredictionRecord.created_at >= start)
    ) or 0
    saved_total = db.scalar(select(func.count()).select_from(SavedPrediction)) or 0

    leagues = _league_pulse(db)
    charts = _chart_payload(db, now - timedelta(days=30))
    model = model_status()

    return {
        "date": day.isoformat(),
        "generated_at": iso_utc(now),
        "refreshed": refreshed,
        "live": [fixture_dict(f) for f in live],
        "today": [fixture_dict(f) for f in today],
        "upcoming": [fixture_dict(f) for f in upcoming],
        "recent": [fixture_dict(f) for f in recent],
        "leagues": leagues,
        "stats": {
            "fixtures_today": len(today),
            "live_now": len(live_today),
            "finished_today": len(finished_today),
            "goals_today": goals_today,
            "avg_goals_today": round(goals_today / len(finished_today), 2) if finished_today else 0,
            "btts_today": btts_today,
            "tracked_teams": int(tracked_teams),
            "tracked_leagues": int(tracked_leagues),
            "predictions_today": int(predictions_today),
            "saved_predictions": int(saved_total),
            "model": {
                "available": model.get("available"),
                "version": model.get("version"),
                "accuracy": (model.get("metrics") or {}).get("accuracy"),
                "log_loss": (model.get("metrics") or {}).get("log_loss"),
            },
        },
        "charts": charts,
        "attribution": "Live football data via ESPN. Analytics and predictions by BLAQUE ANALYSIS.",
    }
