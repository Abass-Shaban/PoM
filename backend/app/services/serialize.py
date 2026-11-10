"""JSON serializers shared by routers and analytics services."""
from __future__ import annotations

from datetime import datetime

from ..models import Fixture, League, Team


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(microsecond=0).isoformat() + "Z"


def team_dict(team: Team | None) -> dict:
    if team is None:
        return {"id": 0, "name": "Unknown", "logo": None, "abbreviation": None}
    return {
        "id": team.id,
        "name": team.name,
        "short_name": team.short_name,
        "abbreviation": team.abbreviation,
        "logo": team.logo,
        "color": team.color,
        "league_slug": team.league_slug,
        "venue": team.venue,
    }


def league_dict(league: League | None) -> dict:
    if league is None:
        return {"slug": "", "name": "Unknown", "logo": None}
    return {
        "slug": league.slug,
        "name": league.name,
        "short_name": league.short_name,
        "country": league.country,
        "logo": league.logo,
        "season": league.season,
        "is_featured": league.is_featured,
    }


def fixture_stats(fixture: Fixture) -> dict | None:
    if fixture.home_corners is None and fixture.home_shots is None:
        return None
    return {
        "corners": {"home": fixture.home_corners, "away": fixture.away_corners},
        "shots": {"home": fixture.home_shots, "away": fixture.away_shots},
        "shots_on_target": {
            "home": fixture.home_shots_on_target,
            "away": fixture.away_shots_on_target,
        },
        "possession": {"home": fixture.home_possession, "away": fixture.away_possession},
        "fouls": {"home": fixture.home_fouls, "away": fixture.away_fouls},
        "yellow_cards": {
            "home": fixture.home_yellow_cards,
            "away": fixture.away_yellow_cards,
        },
        "red_cards": {"home": fixture.home_red_cards, "away": fixture.away_red_cards},
    }


def fixture_dict(fixture: Fixture, *, include_stats: bool = True) -> dict:
    payload = {
        "id": fixture.id,
        "league": league_dict(fixture.league),
        "home": team_dict(fixture.home_team),
        "away": team_dict(fixture.away_team),
        "kickoff": iso_utc(fixture.kickoff),
        "date": fixture.kickoff.strftime("%Y-%m-%d"),
        "time": fixture.kickoff.strftime("%H:%M"),
        "season": fixture.season,
        "status": fixture.status,
        "status_detail": fixture.status_detail,
        "minute": fixture.minute,
        "home_score": fixture.home_score,
        "away_score": fixture.away_score,
        "scoreline": fixture.scoreline,
        "venue": fixture.venue,
        "city": fixture.city,
        "neutral_site": fixture.neutral_site,
        "is_live": fixture.status in ("live", "suspended"),
        "is_finished": fixture.is_finished,
        "odds": fixture.odds_json,
        "updated_at": iso_utc(fixture.updated_at),
    }
    if include_stats:
        payload["stats"] = fixture_stats(fixture)
    return payload
