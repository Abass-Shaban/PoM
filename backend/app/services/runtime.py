"""Runtime configuration overrides editable from the admin dashboard.

Values are persisted in ``system_settings`` and cached in memory so hot paths
(cache TTL lookups, featured-league lists) do not hit the database.
"""
from __future__ import annotations

import threading

from sqlalchemy.orm import Session

from ..config import settings
from ..models import SystemSetting

_LOCK = threading.RLock()
_OVERRIDES: dict = {}

_ALLOWED_TTL_KEYS = {
    "cache_ttl_live",
    "cache_ttl_fixtures",
    "cache_ttl_standings",
    "cache_ttl_summary",
    "cache_ttl_schedule",
    "cache_ttl_teams",
}
_ALLOWED_KEYS = _ALLOWED_TTL_KEYS | {
    "sync_on_startup",
    "model_blend_weight",
    "featured_leagues",
}


def load_overrides(db: Session) -> dict:
    global _OVERRIDES
    row = db.get(SystemSetting, "runtime")
    value = dict(row.value) if row and isinstance(row.value, dict) else {}
    value = {k: v for k, v in value.items() if k in _ALLOWED_KEYS}
    with _LOCK:
        _OVERRIDES = value
    return value


def update_overrides(db: Session, updates: dict) -> dict:
    clean = {k: v for k, v in updates.items() if k in _ALLOWED_KEYS and v is not None}
    with _LOCK:
        merged = {**_OVERRIDES, **clean}
    row = db.get(SystemSetting, "runtime")
    if row is None:
        row = SystemSetting(key="runtime", value=merged)
        db.add(row)
    else:
        row.value = merged
    db.commit()
    with _LOCK:
        _OVERRIDES = merged
    return merged


def get_ttl(name: str) -> int:
    with _LOCK:
        value = _OVERRIDES.get(name)
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return int(getattr(settings, name))


def get_blend_weight() -> float:
    with _LOCK:
        value = _OVERRIDES.get("model_blend_weight")
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 0.8))
    return settings.model_blend_weight


def get_featured_slugs() -> list[str]:
    with _LOCK:
        value = _OVERRIDES.get("featured_leagues")
    if isinstance(value, list) and value:
        return [str(v) for v in value if v]
    return settings.league_slugs


def sync_on_startup() -> bool:
    with _LOCK:
        value = _OVERRIDES.get("sync_on_startup")
    if isinstance(value, bool):
        return value
    return settings.sync_on_startup


def snapshot() -> dict:
    with _LOCK:
        return dict(_OVERRIDES)
