"""ESPN public soccer API adapter.

ESPN exposes a public, key-free JSON API used by its own soccer pages.  It
provides fixtures, live scores, standings, head-to-head history and detailed
match statistics (including corners and possession).  Responses are cached in
the database and stale data is served if the upstream service is unavailable.

No match data in this application is generated or mocked: when the provider
cannot be reached the API returns an explicit error and the UI surfaces it.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from ...config import settings
from ...data.leagues import LEAGUES
from ...errors import NotFoundError, ProviderError
from ..runtime import get_ttl
from .base import (
    FixtureDTO,
    FootballProvider,
    H2HGameDTO,
    MatchSummaryDTO,
    StandingRowDTO,
    TeamDTO,
)
from .cache import get_cached, make_key, set_cached

logger = logging.getLogger("blaque.provider.espn")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 BLAQUE-ANALYSIS/1.0"
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "—", "N/A"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def american_to_probability(moneyline: float | None) -> float | None:
    if moneyline is None:
        return None
    ml = float(moneyline)
    if ml == 0:
        return None
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


class ESPNProvider(FootballProvider):
    name = "espn"
    capabilities = {
        "fixtures": True,
        "live_scores": True,
        "standings": True,
        "head_to_head": True,
        "team_history": True,
        "match_statistics": True,
        "corners": True,
        "possession": True,
        "odds": True,
        "expected_goals": False,
        "lineups": False,
        "requires_api_key": False,
        "attribution": "Data provided by ESPN",
    }

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.http_timeout, connect=10.0),
            headers={"User-Agent": _UA, "Accept": "application/json"},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=8),
        )
        self._semaphore = threading.BoundedSemaphore(settings.http_concurrency)
        self._last_error: str | None = None
        self._last_error_at: datetime | None = None
        self._league_cache: dict[str, dict] = {}

    # ------------------------------------------------------------- transport
    def _get(
        self,
        url: str,
        *,
        params: dict | None = None,
        cache_key: str,
        ttl: int,
    ) -> dict:
        cached, fresh = get_cached(cache_key)
        if fresh and cached is not None:
            return cached

        last_exc: Exception | None = None
        for attempt in range(max(1, settings.http_retries)):
            try:
                with self._semaphore:
                    response = self._client.get(url, params=params)
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ProviderError("The data provider returned an invalid response.") from exc
                    set_cached(cache_key, payload, ttl, url=str(response.url), status_code=200)
                    self._last_error = None
                    return payload
                if response.status_code == 404:
                    raise NotFoundError("That match could not be found at the data provider.")
                if response.status_code in (429, 500, 502, 503, 504):
                    last_exc = ProviderError(
                        f"The data provider is temporarily unavailable (HTTP {response.status_code})."
                    )
                    time.sleep(min(1.2 * (2**attempt), 6.0))
                    continue
                raise ProviderError(
                    f"The data provider rejected the request (HTTP {response.status_code})."
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                time.sleep(min(1.0 * (2**attempt), 5.0))
            except NotFoundError:
                raise
            except ProviderError:
                raise

        self._last_error = str(last_exc) if last_exc else "unknown provider error"
        self._last_error_at = datetime.now(timezone.utc)
        if cached is not None:
            logger.warning("Serving stale cache for %s after provider failure", cache_key)
            return cached
        raise ProviderError(
            "Live football data is temporarily unavailable. Please try again shortly."
        )

    def _site(self, slug: str, path: str) -> str:
        return f"{settings.espn_site_base}/{slug}/{path}"

    def _league_meta(self, slug: str, payload: dict | None = None) -> dict:
        if slug in self._league_cache:
            return self._league_cache[slug]
        meta = LEAGUES.get(slug)
        if meta:
            data = {
                "slug": slug,
                "name": meta.name,
                "short_name": meta.short_name,
                "country": meta.country,
                "logo": meta.logo,
                "season": None,
            }
        else:
            data = {
                "slug": slug,
                "name": slug,
                "short_name": slug,
                "country": None,
                "logo": None,
                "season": None,
            }
        if payload:
            leagues = payload.get("leagues") or []
            if leagues:
                first = leagues[0]
                data["name"] = first.get("name") or data["name"]
                data["short_name"] = first.get("abbreviation") or data["short_name"]
                logos = first.get("logos") or []
                if logos:
                    data["logo"] = logos[0].get("href") or data["logo"]
                season = first.get("season") or {}
                if isinstance(season, dict):
                    data["season"] = season.get("displayName") or season.get("slug")
        self._league_cache[slug] = data
        return data

    # -------------------------------------------------------------- normalize
    def _team(self, competitor: dict, league_slug: str) -> TeamDTO:
        team = competitor.get("team") or {}
        logos = team.get("logos") or []
        logo = team.get("logo") or (logos[0].get("href") if logos else None)
        venue = team.get("venue")
        return TeamDTO(
            id=int(team.get("id") or 0),
            name=team.get("displayName") or team.get("name") or "Unknown",
            short_name=team.get("shortDisplayName") or team.get("name"),
            abbreviation=team.get("abbreviation"),
            logo=logo,
            color=team.get("color"),
            venue=venue.get("fullName") if isinstance(venue, dict) else None,
            league_slug=league_slug,
        )

    @staticmethod
    def _status(competition: dict) -> tuple[str, str | None, str | None]:
        status = competition.get("status") or {}
        stype = status.get("type") or {}
        name = (stype.get("name") or "").upper()
        state = stype.get("state")
        if "POSTPONED" in name:
            status_value = "postponed"
        elif "CANCEL" in name:
            status_value = "canceled"
        elif "SUSPEND" in name:
            status_value = "suspended"
        elif state == "in":
            status_value = "live"
        elif state == "post" or stype.get("completed"):
            status_value = "finished"
        else:
            status_value = "scheduled"
        detail = stype.get("shortDetail") or stype.get("detail")
        minute = status.get("displayClock") if state == "in" else ("FT" if status_value == "finished" else None)
        return status_value, detail, minute

    @staticmethod
    def _score(competitor: dict) -> int | None:
        score = competitor.get("score")
        if isinstance(score, dict):
            score = score.get("value", score.get("displayValue"))
        return _int(score)

    @staticmethod
    def _statistics(competitor: dict) -> dict:
        output: dict[str, float] = {}
        for stat in competitor.get("statistics") or []:
            name = stat.get("name")
            if not name:
                continue
            value = _num(stat.get("displayValue", stat.get("value")))
            if value is not None:
                output[name] = value
        return output

    def _normalize_event(self, event: dict, league_slug: str) -> FixtureDTO | None:
        competitions = event.get("competitions") or []
        if not competitions:
            return None
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if home is None or away is None:
            return None
        home_team = self._team(home, league_slug)
        away_team = self._team(away, league_slug)
        if not home_team.id or not away_team.id:
            return None
        status, detail, minute = self._status(competition)
        venue = competition.get("venue") or {}
        address = venue.get("address") or {}
        season = event.get("season") or {}
        home_stats = self._statistics(home)
        away_stats = self._statistics(away)
        stats = {
            "home": home_stats,
            "away": away_stats,
        }
        return FixtureDTO(
            id=int(event.get("id")),
            league_slug=league_slug,
            kickoff=_parse_dt(event.get("date")) or datetime.utcnow(),
            home=home_team,
            away=away_team,
            status=status,
            status_detail=detail,
            minute=minute,
            season=season.get("slug") or str(season.get("year") or "") or None,
            round=None,
            home_score=self._score(home),
            away_score=self._score(away),
            venue=venue.get("fullName"),
            city=address.get("city"),
            neutral_site=bool(competition.get("neutralSite")),
            stats=stats,
        )

    def _normalize_events(self, payload: dict, league_slug: str) -> list[FixtureDTO]:
        fixtures: list[FixtureDTO] = []
        for event in payload.get("events") or []:
            try:
                normalized = self._normalize_event(event, league_slug)
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed event for %s", league_slug, exc_info=True)
                continue
            if normalized:
                fixtures.append(normalized)
        return fixtures

    # ----------------------------------------------------------------- public
    def fixtures_by_month(self, league_slug: str, year_month: str) -> list[FixtureDTO]:
        key = make_key("espn", "scoreboard", league_slug, year_month)
        now = datetime.utcnow()
        try:
            ym_dt = datetime.strptime(year_month, "%Y%m")
            historical = ym_dt < now.replace(day=1)
        except ValueError:
            historical = False
        ttl = 24 * 3600 if historical else get_ttl("cache_ttl_fixtures")
        payload = self._get(
            self._site(league_slug, "scoreboard"),
            params={"dates": year_month},
            cache_key=key,
            ttl=ttl,
        )
        self._league_meta(league_slug, payload)
        return self._normalize_events(payload, league_slug)

    def fixtures_by_date(self, league_slug: str, day: str | None = None) -> list[FixtureDTO]:
        params = {"dates": day} if day else None
        today = datetime.utcnow().strftime("%Y%m%d")
        is_today = day is None or day == today
        ttl = get_ttl("cache_ttl_live") if is_today else get_ttl("cache_ttl_fixtures")
        key = make_key("espn", "scoreboard", league_slug, day or "today")
        payload = self._get(
            self._site(league_slug, "scoreboard"),
            params=params,
            cache_key=key,
            ttl=ttl,
        )
        self._league_meta(league_slug, payload)
        return self._normalize_events(payload, league_slug)

    def live(self, league_slugs: list[str] | None = None) -> list[FixtureDTO]:
        results: list[FixtureDTO] = []
        for slug in league_slugs or list(LEAGUES.keys()):
            try:
                results.extend(
                    f for f in self.fixtures_by_date(slug) if f.status in ("live", "suspended")
                )
            except ProviderError:
                logger.warning("Live scoreboard unavailable for %s", slug)
        results.sort(key=lambda f: f.kickoff)
        return results

    def standings(self, league_slug: str) -> dict:
        key = make_key("espn", "standings", league_slug)
        payload = self._get(
            f"https://site.api.espn.com/apis/v2/sports/soccer/{league_slug}/standings",
            cache_key=key,
            ttl=get_ttl("cache_ttl_standings"),
        )
        meta = self._league_meta(league_slug, payload)
        rows: list[StandingRowDTO] = []
        season = None
        groups = payload.get("children") or []
        if not groups and payload.get("standings"):
            groups = [payload]
        for group in groups:
            standings = group.get("standings") or {}
            season = season or standings.get("seasonDisplayName") or standings.get("season")
            group_name = group.get("name") if group is not payload else None
            for entry in standings.get("entries") or []:
                team_data = entry.get("team") or {}
                stats_list = entry.get("stats") or []
                stats = {
                    stat.get("name"): stat.get("value")
                    for stat in stats_list
                    if stat.get("name")
                }
                logos = team_data.get("logos") or []
                note = entry.get("note") or {}
                rows.append(
                    StandingRowDTO(
                        rank=int(stats.get("rank") or 0),
                        team=TeamDTO(
                            id=int(team_data.get("id") or 0),
                            name=team_data.get("displayName") or "Unknown",
                            short_name=team_data.get("shortDisplayName"),
                            abbreviation=team_data.get("abbreviation"),
                            logo=(logos[0].get("href") if logos else None),
                            color=team_data.get("color"),
                            league_slug=league_slug,
                        ),
                        played=int(stats.get("gamesPlayed") or 0),
                        wins=int(stats.get("wins") or 0),
                        draws=int(stats.get("ties") or 0),
                        losses=int(stats.get("losses") or 0),
                        goals_for=int(stats.get("pointsFor") or 0),
                        goals_against=int(stats.get("pointsAgainst") or 0),
                        goal_diff=int(stats.get("pointDifferential") or 0),
                        points=int(stats.get("points") or 0),
                        form=stats.get("form") if isinstance(stats.get("form"), str) else None,
                        note=note.get("description"),
                        group=group_name,
                    )
                )
        rows.sort(key=lambda row: (row.group or "", row.rank))
        return {
            "league": meta,
            "season": season,
            "groups": sorted({row.group for row in rows if row.group}),
            "rows": rows,
        }

    def teams(self, league_slug: str) -> list[TeamDTO]:
        key = make_key("espn", "teams", league_slug)
        payload = self._get(
            self._site(league_slug, "teams"),
            cache_key=key,
            ttl=get_ttl("cache_ttl_teams"),
        )
        output: list[TeamDTO] = []
        for sport in payload.get("sports") or []:
            for league in sport.get("leagues") or []:
                for item in league.get("teams") or []:
                    team = item.get("team") or item
                    logos = team.get("logos") or []
                    output.append(
                        TeamDTO(
                            id=int(team.get("id") or 0),
                            name=team.get("displayName") or "Unknown",
                            short_name=team.get("shortDisplayName"),
                            abbreviation=team.get("abbreviation"),
                            logo=(logos[0].get("href") if logos else team.get("logo")),
                            color=team.get("color"),
                            venue=(team.get("venue") or {}).get("fullName")
                            if isinstance(team.get("venue"), dict)
                            else None,
                            league_slug=league_slug,
                        )
                    )
        return output

    def team_schedule(
        self, league_slug: str, team_id: int, season: int | None = None
    ) -> list[FixtureDTO]:
        params = {"season": season} if season else None
        key = make_key("espn", "schedule", league_slug, team_id, season or "current")
        payload = self._get(
            self._site(league_slug, f"teams/{team_id}/schedule"),
            params=params,
            cache_key=key,
            ttl=get_ttl("cache_ttl_schedule"),
        )
        fixtures: list[FixtureDTO] = []
        for event in payload.get("events") or []:
            event_league = (event.get("league") or {}).get("slug") or league_slug
            try:
                normalized = self._normalize_event(event, event_league)
            except (KeyError, TypeError, ValueError):
                continue
            if normalized:
                fixtures.append(normalized)
        fixtures.sort(key=lambda f: f.kickoff)
        return fixtures

    def summary(self, league_slug: str, event_id: int) -> MatchSummaryDTO:
        key = make_key("espn", "summary", league_slug, event_id)
        payload = self._get(
            self._site(league_slug, "summary"),
            params={"event": event_id},
            cache_key=key,
            ttl=get_ttl("cache_ttl_summary"),
        )
        home_stats: dict = {}
        away_stats: dict = {}
        for team_block in (payload.get("boxscore") or {}).get("teams") or []:
            team = team_block.get("team") or {}
            team_id = int(team.get("id") or 0)
            stats = {
                stat.get("name"): _num(stat.get("displayValue", stat.get("value")))
                for stat in team_block.get("statistics") or []
                if stat.get("name")
            }
            stats = {k: v for k, v in stats.items() if v is not None}
            if team_id:
                if not home_stats:
                    home_stats = stats
                else:
                    away_stats = stats
        # The boxscore order is not guaranteed; identify by team id if possible.
        header = payload.get("header") or {}
        competitions = header.get("competitions") or []
        if competitions:
            for competitor in competitions[0].get("competitors") or []:
                team_id = int((competitor.get("team") or {}).get("id") or 0)
                block = next(
                    (
                        b
                        for b in (payload.get("boxscore") or {}).get("teams") or []
                        if int((b.get("team") or {}).get("id") or 0) == team_id
                    ),
                    None,
                )
                if block is None:
                    continue
                stats = {
                    stat.get("name"): _num(stat.get("displayValue", stat.get("value")))
                    for stat in block.get("statistics") or []
                    if stat.get("name")
                }
                stats = {k: v for k, v in stats.items() if v is not None}
                if competitor.get("homeAway") == "home":
                    home_stats = stats
                else:
                    away_stats = stats

        h2h: list[H2HGameDTO] = []
        for series in payload.get("seasonseries") or []:
            if series.get("type") != "head-to-head":
                continue
            for event in series.get("events") or []:
                competitors = event.get("competitors") or []
                if len(competitors) < 2:
                    continue
                home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                h2h.append(
                    H2HGameDTO(
                        id=_int(event.get("id")),
                        date=_parse_dt(event.get("date")),
                        home_id=int((home.get("team") or {}).get("id") or 0),
                        away_id=int((away.get("team") or {}).get("id") or 0),
                        home_name=(home.get("team") or {}).get("displayName") or "Unknown",
                        away_name=(away.get("team") or {}).get("displayName") or "Unknown",
                        home_score=_int(home.get("score")),
                        away_score=_int(away.get("score")),
                        league=(event.get("league") or {}).get("name")
                        if isinstance(event.get("league"), dict)
                        else None,
                        status=(event.get("statusType") or {}).get("state"),
                    )
                )

        last_five: dict[int, list[dict]] = {}
        for block in payload.get("lastFiveGames") or []:
            team_id = int((block.get("team") or {}).get("id") or 0)
            if not team_id:
                continue
            games = []
            for event in block.get("events") or []:
                games.append(
                    {
                        "id": _int(event.get("id")),
                        "date": event.get("gameDate"),
                        "result": event.get("gameResult"),
                        "score": event.get("score"),
                        "home_team_id": _int(event.get("homeTeamId")),
                        "away_team_id": _int(event.get("awayTeamId")),
                        "home_score": _int(event.get("homeTeamScore")),
                        "away_score": _int(event.get("awayTeamScore")),
                        "opponent": (event.get("opponent") or {}).get("displayName"),
                        "league": event.get("leagueName"),
                        "at_vs": event.get("atVs"),
                    }
                )
            last_five[team_id] = games

        odds = self._parse_odds(payload)
        game_info = payload.get("gameInfo") or {}
        venue = (game_info.get("venue") or {}).get("fullName")
        attendance = _int(game_info.get("attendance"))
        return MatchSummaryDTO(
            home_stats=home_stats,
            away_stats=away_stats,
            h2h=h2h,
            last_five=last_five,
            odds=odds,
            venue=venue,
            attendance=attendance,
        )

    @staticmethod
    def _parse_odds(payload: dict) -> dict | None:
        pick = payload.get("pickcenter") or []
        if not pick:
            return None
        book = pick[0]
        home_ml = _num((book.get("homeTeamOdds") or {}).get("moneyLine"))
        away_ml = _num((book.get("awayTeamOdds") or {}).get("moneyLine"))
        draw_ml = _num((book.get("drawOdds") or {}).get("moneyLine"))
        if home_ml is None and away_ml is None:
            return None
        probs = {
            "home": american_to_probability(home_ml),
            "draw": american_to_probability(draw_ml),
            "away": american_to_probability(away_ml),
        }
        valid = {k: v for k, v in probs.items() if v is not None}
        total = sum(valid.values()) if valid else 0
        implied = {k: (v / total if total else None) for k, v in valid.items()}
        return {
            "provider": (book.get("provider") or {}).get("name"),
            "home_ml": home_ml,
            "draw_ml": draw_ml,
            "away_ml": away_ml,
            "over_under": _num(book.get("overUnder")),
            "over_odds": _num(book.get("overOdds")),
            "under_odds": _num(book.get("underOdds")),
            "implied": implied,
            "details": book.get("details"),
        }

    def check(self) -> dict:
        started = time.perf_counter()
        try:
            self.fixtures_by_date("eng.1")
            return {
                "ok": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "last_error": self._last_error,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "last_error": str(exc),
            }
