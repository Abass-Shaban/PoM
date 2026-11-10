"""Application configuration for BLAQUE ANALYSIS.

All values can be supplied through environment variables or a ``.env`` file in
the project root.  The defaults are safe for local development; production
deployments should override ``SECRET_KEY`` and ``DATABASE_URL``.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- general
    app_name: str = "BLAQUE ANALYSIS"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    secret_key: str = "dev-only-change-me-6f1c0b8f0f5a4f4e9a3d2c1b7e8f9a0c"
    access_token_expire_minutes: int = 60 * 8
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    admin_email: str = ""
    admin_password: str = ""
    login_rate_limit: int = 12
    login_rate_window_seconds: int = 300

    # --------------------------------------------------------------- database
    # SQLite by default (zero-config).  Set a PostgreSQL/MySQL URL in
    # production, e.g. postgresql+psycopg://blaque:blaque@localhost:5432/blaque
    database_url: str = f"sqlite:///{(DATA_DIR / 'blaque.db').as_posix()}"
    db_echo: bool = False

    # --------------------------------------------------------------- provider
    provider: str = "espn"
    espn_site_base: str = "https://site.api.espn.com/apis/site/v2/sports/soccer"
    espn_core_base: str = "https://sports.core.api.espn.com/v2/sports/soccer"
    http_timeout: float = 25.0
    http_retries: int = 3
    http_concurrency: int = 6

    # Featured leagues (ESPN slugs).  Order is used in the UI.
    featured_leagues: str = (
        "eng.1,esp.1,ita.1,ger.1,fra.1,uefa.champions,"
        "uefa.europa,usa.1,ned.1,por.1,tur.1,bra.1"
    )

    # ------------------------------------------------------------------ cache
    cache_ttl_live: int = 45
    cache_ttl_fixtures: int = 300
    cache_ttl_standings: int = 6 * 60 * 60
    cache_ttl_summary: int = 60 * 60
    cache_ttl_schedule: int = 6 * 60 * 60
    cache_ttl_teams: int = 24 * 60 * 60

    # ------------------------------------------------------------------- sync
    sync_on_startup: bool = True
    sync_months_back: int = 1
    sync_months_ahead: int = 1
    sync_min_fixtures: int = 80

    # ------------------------------------------------------------------ model
    model_path: str = str(MODEL_DIR / "blaque_model.joblib")
    model_meta_path: str = str(MODEL_DIR / "blaque_model_meta.json")
    model_min_samples: int = 180
    model_blend_weight: float = 0.35
    model_random_state: int = 42

    # ------------------------------------------------------------- prediction
    prediction_goals_cap: int = 10
    prediction_window: int = 10
    prediction_decay_days: float = 180.0

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def league_slugs(self) -> list[str]:
        return [s.strip() for s in self.featured_leagues.split(",") if s.strip()]

    def ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    if settings.environment == "production" and "change-me" in settings.secret_key:
        raise RuntimeError(
            "SECRET_KEY must be set to a strong random value in production."
        )
    return settings


def generate_secret() -> str:
    """Utility used by the setup script / documentation."""
    return secrets.token_urlsafe(48)


settings = get_settings()
