"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (portable across SQLite/PostgreSQL/MySQL)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class League(Base):
    __tablename__ = "leagues"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(80))
    country: Mapped[str | None] = mapped_column(String(80))
    logo: Mapped[str | None] = mapped_column(String(500))
    season: Mapped[str | None] = mapped_column(String(32))
    is_featured: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    fixtures: Mapped[list["Fixture"]] = relationship(back_populates="league")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<League {self.slug} {self.name!r}>"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_slug: Mapped[str | None] = mapped_column(
        ForeignKey("leagues.slug", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    short_name: Mapped[str | None] = mapped_column(String(100))
    abbreviation: Mapped[str | None] = mapped_column(String(10))
    logo: Mapped[str | None] = mapped_column(String(500))
    color: Mapped[str | None] = mapped_column(String(16))
    venue: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    league: Mapped[League | None] = relationship()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Team {self.id} {self.name!r}>"


class Fixture(Base):
    __tablename__ = "fixtures"
    __table_args__ = (
        Index("ix_fixtures_kickoff_status", "kickoff", "status"),
        Index("ix_fixtures_home_kickoff", "home_team_id", "kickoff"),
        Index("ix_fixtures_away_kickoff", "away_team_id", "kickoff"),
        Index("ix_fixtures_league_kickoff", "league_slug", "kickoff"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_slug: Mapped[str] = mapped_column(
        ForeignKey("leagues.slug", ondelete="CASCADE"), nullable=False, index=True
    )
    home_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    away_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    kickoff: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    season: Mapped[str | None] = mapped_column(String(32))
    round: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="scheduled", nullable=False)
    status_detail: Mapped[str | None] = mapped_column(String(80))
    minute: Mapped[str | None] = mapped_column(String(20))
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(200))
    city: Mapped[str | None] = mapped_column(String(120))
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)

    # Core match statistics (null when the match has not been played or the
    # provider does not expose the metric).
    home_corners: Mapped[int | None] = mapped_column(Integer)
    away_corners: Mapped[int | None] = mapped_column(Integer)
    home_shots: Mapped[int | None] = mapped_column(Integer)
    away_shots: Mapped[int | None] = mapped_column(Integer)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer)
    home_possession: Mapped[float | None] = mapped_column(Float)
    away_possession: Mapped[float | None] = mapped_column(Float)
    home_fouls: Mapped[int | None] = mapped_column(Integer)
    away_fouls: Mapped[int | None] = mapped_column(Integer)
    home_yellow_cards: Mapped[int | None] = mapped_column(Integer)
    away_yellow_cards: Mapped[int | None] = mapped_column(Integer)
    home_red_cards: Mapped[int | None] = mapped_column(Integer)
    away_red_cards: Mapped[int | None] = mapped_column(Integer)
    stats_json: Mapped[dict | None] = mapped_column(JSON)
    odds_json: Mapped[dict | None] = mapped_column(JSON)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    league: Mapped[League] = relationship(back_populates="fixtures", lazy="selectin")
    home_team: Mapped[Team] = relationship(
        foreign_keys=[home_team_id], lazy="selectin"
    )
    away_team: Mapped[Team] = relationship(
        foreign_keys=[away_team_id], lazy="selectin"
    )

    @property
    def is_finished(self) -> bool:
        return self.status == "finished" and self.home_score is not None

    @property
    def scoreline(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        return f"{self.home_score}-{self.away_score}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Fixture {self.id} {self.home_team_id} v {self.away_team_id}>"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)

    favorite_teams: Mapped[list["FavoriteTeam"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    favorite_fixtures: Mapped[list["FavoriteFixture"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    saved_predictions: Mapped[list["SavedPrediction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class FavoriteTeam(Base):
    __tablename__ = "favorite_teams"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="favorite_teams")
    team: Mapped[Team] = relationship(lazy="selectin")


class FavoriteFixture(Base):
    __tablename__ = "favorite_fixtures"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="favorite_fixtures")
    fixture: Mapped[Fixture] = relationship(lazy="selectin")


class SavedPrediction(Base):
    __tablename__ = "saved_predictions"
    __table_args__ = (UniqueConstraint("user_id", "fixture_id", name="uq_saved_user_fixture"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False, index=True
    )
    window: Mapped[int] = mapped_column(Integer, default=10)
    model_version: Mapped[str | None] = mapped_column(String(80))
    fixture_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    prediction: Mapped[dict] = mapped_column(JSON, nullable=False)
    note: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="saved_predictions")
    fixture: Mapped[Fixture] = relationship(lazy="selectin")


class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    __table_args__ = (Index("ix_prediction_records_created", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    model_version: Mapped[str | None] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ApiCache(Base):
    __tablename__ = "api_cache"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    url: Mapped[str | None] = mapped_column(String(1000))
    status_code: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    hits: Mapped[int] = mapped_column(Integer, default=0)


class SyncRun(Base):
    __tablename__ = "sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(80), default="featured")
    status: Mapped[str] = mapped_column(String(20), default="running")
    triggered_by: Mapped[str | None] = mapped_column(String(120))
    detail: Mapped[dict | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    detail: Mapped[dict | None] = mapped_column(JSON)
    ip: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
