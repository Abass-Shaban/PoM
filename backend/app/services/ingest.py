"""Ingestion: turn normalized provider objects into durable database rows."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import settings
from ..data.leagues import LEAGUES, LeagueMeta
from ..database import session_scope
from ..errors import ProviderError
from ..models import Fixture, League, Team, utcnow
from .provider.base import FixtureDTO, TeamDTO
from .provider.factory import get_provider
from .runtime import get_featured_slugs

logger = logging.getLogger("blaque.ingest")


def _is_domestic(slug: str | None) -> bool:
    return bool(slug) and not str(slug).startswith("uefa.")


def seed_leagues(db: Session) -> int:
    created = 0
    featured = set(get_featured_slugs())
    order = 10
    for slug, meta in LEAGUES.items():
        league = db.get(League, slug)
        if league is None:
            league = League(
                slug=slug,
                name=meta.name,
                short_name=meta.short_name,
                country=meta.country,
                logo=meta.logo,
                is_featured=slug in featured,
                sort_order=meta.sort_order,
            )
            db.add(league)
            created += 1
        else:
            league.name = meta.name
            league.short_name = meta.short_name
            league.country = meta.country
            league.logo = meta.logo
            league.sort_order = meta.sort_order
            league.is_featured = slug in featured
        order += 10
    db.commit()
    return created


def ensure_league(db: Session, slug: str) -> League:
    league = db.get(League, slug)
    if league is None:
        meta: LeagueMeta | None = LEAGUES.get(slug)
        league = League(
            slug=slug,
            name=meta.name if meta else slug,
            short_name=meta.short_name if meta else slug,
            country=meta.country if meta else None,
            logo=meta.logo if meta else None,
            is_featured=slug in get_featured_slugs(),
            sort_order=meta.sort_order if meta else 999,
        )
        db.add(league)
        db.flush()
    return league


def upsert_team(db: Session, team: TeamDTO, league_slug: str | None = None) -> Team:
    target_league = league_slug or team.league_slug
    row = db.get(Team, team.id)
    if row is None:
        row = Team(
            id=team.id,
            name=team.name,
            short_name=team.short_name,
            abbreviation=team.abbreviation,
            logo=team.logo,
            color=team.color,
            venue=team.venue,
            league_slug=target_league,
        )
        # Flush immediately so the new row enters the identity map.  Without
        # this, a second upsert in the same (autoflush-disabled) session would
        # miss the pending row and try to insert a duplicate primary key.
        try:
            with db.begin_nested():
                db.add(row)
                db.flush()
        except IntegrityError:
            # Another session inserted the same team concurrently.
            row = db.get(Team, team.id)
            if row is None:
                raise
        if row is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Could not persist team {team.id}")
        return row
    row.name = team.name or row.name
    row.short_name = team.short_name or row.short_name
    row.abbreviation = team.abbreviation or row.abbreviation
    row.logo = team.logo or row.logo
    row.color = team.color or row.color
    row.venue = team.venue or row.venue
    if target_league and (
        row.league_slug is None
        or (not _is_domestic(row.league_slug) and _is_domestic(target_league))
    ):
        row.league_slug = target_league
    return row


def _stat(stats: dict, *keys: str) -> float | None:
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return value
    return None


def _as_int(value: float | None) -> int | None:
    return int(round(value)) if value is not None else None


def upsert_fixture(db: Session, dto: FixtureDTO) -> Fixture:
    ensure_league(db, dto.league_slug)
    upsert_team(db, dto.home, dto.league_slug)
    upsert_team(db, dto.away, dto.league_slug)

    row = db.get(Fixture, dto.id)
    if row is None:
        row = Fixture(id=dto.id, league_slug=dto.league_slug)
        db.add(row)

    home_stats = dto.stats.get("home") or {}
    away_stats = dto.stats.get("away") or {}

    row.league_slug = dto.league_slug
    row.home_team_id = dto.home.id
    row.away_team_id = dto.away.id
    row.kickoff = dto.kickoff
    row.season = dto.season
    row.round = dto.round
    row.status = dto.status
    row.status_detail = dto.status_detail
    row.minute = dto.minute
    row.home_score = dto.home_score
    row.away_score = dto.away_score
    row.venue = dto.venue
    row.city = dto.city
    row.neutral_site = dto.neutral_site

    if home_stats:
        row.home_corners = _as_int(_stat(home_stats, "wonCorners", "corners"))
        row.home_shots = _as_int(_stat(home_stats, "totalShots", "shots"))
        row.home_shots_on_target = _as_int(_stat(home_stats, "shotsOnTarget"))
        row.home_possession = _stat(home_stats, "possessionPct", "possession")
        row.home_fouls = _as_int(_stat(home_stats, "foulsCommitted"))
        row.home_yellow_cards = _as_int(_stat(home_stats, "yellowCards"))
        row.home_red_cards = _as_int(_stat(home_stats, "redCards"))
    if away_stats:
        row.away_corners = _as_int(_stat(away_stats, "wonCorners", "corners"))
        row.away_shots = _as_int(_stat(away_stats, "totalShots", "shots"))
        row.away_shots_on_target = _as_int(_stat(away_stats, "shotsOnTarget"))
        row.away_possession = _stat(away_stats, "possessionPct", "possession")
        row.away_fouls = _as_int(_stat(away_stats, "foulsCommitted"))
        row.away_yellow_cards = _as_int(_stat(away_stats, "yellowCards"))
        row.away_red_cards = _as_int(_stat(away_stats, "redCards"))

    if dto.stats:
        row.stats_json = dto.stats
    if dto.odds:
        row.odds_json = dto.odds
    row.updated_at = utcnow()
    row.ingested_at = utcnow()
    return row


def _month_sequence(months_back: int, months_ahead: int) -> list[str]:
    today = date.today().replace(day=1)
    months: list[str] = []
    for offset in range(-months_back, months_ahead + 1):
        year = today.year + (today.month - 1 + offset) // 12
        month = (today.month - 1 + offset) % 12 + 1
        months.append(f"{year}{month:02d}")
    return months


def sync_league(
    db: Session,
    slug: str,
    *,
    months_back: int = 1,
    months_ahead: int = 1,
) -> dict:
    provider = get_provider()
    league = ensure_league(db, slug)
    fixtures_written = 0
    teams_written = 0
    errors: list[str] = []

    # Standings are also a rich source of team metadata and the current season.
    try:
        table = provider.standings(slug)
        if table.get("season"):
            league.season = str(table["season"])
        for row in table.get("rows") or []:
            if row.team.id:
                upsert_team(db, row.team, slug)
                teams_written += 1
        if table.get("league", {}).get("logo"):
            league.logo = table["league"]["logo"]
    except ProviderError as exc:
        errors.append(f"standings: {exc.message}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Standings sync failed for %s", slug)
        errors.append(f"standings: {exc}")

    for year_month in _month_sequence(months_back, months_ahead):
        try:
            fixtures = provider.fixtures_by_month(slug, year_month)
            for dto in fixtures:
                upsert_fixture(db, dto)
                fixtures_written += 1
        except ProviderError as exc:
            errors.append(f"{year_month}: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Month sync failed for %s %s", slug, year_month)
            errors.append(f"{year_month}: {exc}")
        finally:
            db.commit()

    db.commit()
    return {
        "league": slug,
        "fixtures": fixtures_written,
        "teams": teams_written,
        "errors": errors,
    }


def sync_featured(
    *,
    months_back: int = 1,
    months_ahead: int = 1,
    slugs: list[str] | None = None,
    max_workers: int = 1,
) -> dict:
    slugs = slugs or get_featured_slugs()
    started = utcnow()
    results: list[dict] = []
    errors: list[str] = []

    def worker(slug: str) -> dict:
        try:
            with session_scope() as db:
                return sync_league(
                    db, slug, months_back=months_back, months_ahead=months_ahead
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("League sync failed for %s", slug)
            return {"league": slug, "fixtures": 0, "teams": 0, "errors": [str(exc)]}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, slug): slug for slug in slugs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            errors.extend(f"{result['league']}: {e}" for e in result.get("errors") or [])

    results.sort(key=lambda item: slugs.index(item["league"]) if item["league"] in slugs else 99)
    return {
        "scope": "featured",
        "leagues": results,
        "fixtures": sum(item.get("fixtures", 0) for item in results),
        "teams": sum(item.get("teams", 0) for item in results),
        "errors": errors,
        "started_at": started.isoformat() + "Z",
        "finished_at": utcnow().isoformat() + "Z",
    }


def refresh_today(slugs: list[str] | None = None) -> dict:
    """Refresh today's scoreboards (live scores, new results)."""
    provider = get_provider()
    today = date.today().strftime("%Y%m%d")
    written = 0
    errors: list[str] = []
    with session_scope() as db:
        for slug in slugs or get_featured_slugs():
            try:
                for dto in provider.fixtures_by_date(slug, today):
                    upsert_fixture(db, dto)
                    written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Today refresh failed for %s: %s", slug, exc)
                errors.append(f"{slug}: {exc}")
        db.commit()
    return {"fixtures": written, "errors": errors}


def sync_date(day: date, slugs: list[str] | None = None) -> dict:
    """Fetch every featured league's scoreboard for a specific date."""
    provider = get_provider()
    day_string = day.strftime("%Y%m%d")
    written = 0
    errors: list[str] = []
    with session_scope() as db:
        for slug in slugs or get_featured_slugs():
            try:
                for dto in provider.fixtures_by_date(slug, day_string):
                    upsert_fixture(db, dto)
                    written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Date sync failed for %s %s: %s", slug, day_string, exc)
                errors.append(f"{slug}: {exc}")
        db.commit()
    return {"date": day.isoformat(), "fixtures": written, "errors": errors}


def ensure_team_history(db: Session, team_id: int, window: int = 15) -> int:
    """Backfill a team's recent fixtures when the local cache is too thin."""
    team = db.get(Team, team_id)
    if team is None:
        return 0
    league_slug = team.league_slug
    if not league_slug:
        league_slug = db.scalar(
            select(Fixture.league_slug)
            .where((Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id))
            .order_by(Fixture.kickoff.desc())
            .limit(1)
        )
    if not league_slug:
        return 0
    count = (
        db.scalar(
            select(func.count())
            .select_from(Fixture)
            .where(
                Fixture.status == "finished",
                (Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id),
            )
        )
        or 0
    )
    if count >= window:
        return 0
    provider = get_provider()
    latest = db.scalar(
        select(func.max(Fixture.kickoff)).where(
            (Fixture.home_team_id == team_id) | (Fixture.away_team_id == team_id)
        )
    )
    anchor = latest or utcnow()
    # European seasons start in July/August; calendar-year leagues are handled
    # by also requesting the previous season below.
    season = anchor.year if anchor.month >= 7 else anchor.year - 1
    written = 0
    for season_year in (season, season - 1):
        try:
            for dto in provider.team_schedule(league_slug, team_id, season_year):
                upsert_fixture(db, dto)
                written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("History backfill failed for team %s season %s: %s", team_id, season_year, exc)
    if written:
        db.commit()
    return written


def fixture_snapshot(fixture: Fixture) -> dict:
    return {
        "id": fixture.id,
        "league": {
            "slug": fixture.league_slug,
            "name": fixture.league.name if fixture.league else fixture.league_slug,
            "logo": fixture.league.logo if fixture.league else None,
        },
        "home": {
            "id": fixture.home_team_id,
            "name": fixture.home_team.name if fixture.home_team else "Home",
            "logo": fixture.home_team.logo if fixture.home_team else None,
        },
        "away": {
            "id": fixture.away_team_id,
            "name": fixture.away_team.name if fixture.away_team else "Away",
            "logo": fixture.away_team.logo if fixture.away_team else None,
        },
        "kickoff": fixture.kickoff.isoformat() + "Z",
        "status": fixture.status,
        "home_score": fixture.home_score,
        "away_score": fixture.away_score,
        "venue": fixture.venue,
    }
