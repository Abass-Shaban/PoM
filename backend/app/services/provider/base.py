"""Data-transfer objects and the provider interface.

The rest of the application only speaks in these normalized objects, which
makes it possible to add another football data provider (API-Football,
football-data.org, ...) without touching the analytics layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class TeamDTO:
    id: int
    name: str
    short_name: str | None = None
    abbreviation: str | None = None
    logo: str | None = None
    color: str | None = None
    venue: str | None = None
    league_slug: str | None = None


@dataclass(slots=True)
class FixtureDTO:
    id: int
    league_slug: str
    kickoff: datetime
    home: TeamDTO
    away: TeamDTO
    status: str = "scheduled"
    status_detail: str | None = None
    minute: str | None = None
    season: str | None = None
    round: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    venue: str | None = None
    city: str | None = None
    neutral_site: bool = False
    stats: dict = field(default_factory=dict)
    odds: dict | None = None


@dataclass(slots=True)
class StandingRowDTO:
    rank: int
    team: TeamDTO
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_diff: int = 0
    points: int = 0
    form: str | None = None
    note: str | None = None
    group: str | None = None


@dataclass(slots=True)
class H2HGameDTO:
    id: int | None
    date: datetime | None
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    home_score: int | None
    away_score: int | None
    league: str | None = None
    status: str | None = None


@dataclass(slots=True)
class MatchSummaryDTO:
    home_stats: dict = field(default_factory=dict)
    away_stats: dict = field(default_factory=dict)
    h2h: list[H2HGameDTO] = field(default_factory=list)
    last_five: dict[int, list[dict]] = field(default_factory=dict)
    odds: dict | None = None
    venue: str | None = None
    attendance: int | None = None


class FootballProvider(ABC):
    name: str = "base"
    capabilities: dict = {}

    @abstractmethod
    def fixtures_by_month(self, league_slug: str, year_month: str) -> list[FixtureDTO]: ...

    @abstractmethod
    def fixtures_by_date(self, league_slug: str, day: str | None = None) -> list[FixtureDTO]: ...

    @abstractmethod
    def standings(self, league_slug: str) -> dict: ...

    @abstractmethod
    def teams(self, league_slug: str) -> list[TeamDTO]: ...

    @abstractmethod
    def team_schedule(
        self, league_slug: str, team_id: int, season: int | None = None
    ) -> list[FixtureDTO]: ...

    @abstractmethod
    def summary(self, league_slug: str, event_id: int) -> MatchSummaryDTO: ...

    def health(self) -> dict:
        return {"provider": self.name, "capabilities": self.capabilities}
