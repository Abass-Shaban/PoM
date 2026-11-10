"""Persistent HTTP response cache used to protect the upstream provider."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select

from ...database import SessionLocal
from ...models import ApiCache, utcnow

logger = logging.getLogger("blaque.cache")


def make_key(*parts: Any) -> str:
    return ":".join(str(p) for p in parts if p is not None)[:255]


def get_cached(key: str) -> tuple[dict | None, bool]:
    """Return ``(payload, is_fresh)``.  Stale payloads are returned so the API
    can degrade gracefully when the upstream provider is unavailable."""
    with SessionLocal() as db:
        row = db.get(ApiCache, key)
        if row is None:
            return None, False
        try:
            payload = json.loads(row.payload)
        except json.JSONDecodeError:
            return None, False
        if row.expires_at > utcnow():
            row.hits += 1
            db.commit()
            return payload, True
        return payload, False


def set_cached(
    key: str,
    payload: dict,
    ttl_seconds: int,
    *,
    url: str | None = None,
    status_code: int = 200,
) -> None:
    if ttl_seconds <= 0:
        return
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    expires = utcnow() + timedelta(seconds=ttl_seconds)
    with SessionLocal() as db:
        row = db.get(ApiCache, key)
        if row is None:
            row = ApiCache(key=key, payload=serialized, expires_at=expires)
            db.add(row)
        row.payload = serialized
        row.expires_at = expires
        row.url = url
        row.status_code = status_code
        try:
            db.commit()
        except Exception:  # concurrent writers (SQLite) — last write wins
            db.rollback()
            logger.debug("cache write race for %s", key)


def clear_cache(prefix: str | None = None) -> int:
    with SessionLocal() as db:
        stmt = delete(ApiCache)
        if prefix:
            stmt = stmt.where(ApiCache.key.like(f"{prefix}%"))
        result = db.execute(stmt)
        db.commit()
        return int(result.rowcount or 0)


def cache_stats() -> dict:
    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(ApiCache)) or 0
        fresh = (
            db.scalar(
                select(func.count()).select_from(ApiCache).where(ApiCache.expires_at > utcnow())
            )
            or 0
        )
        hits = db.scalar(select(func.coalesce(func.sum(ApiCache.hits), 0))) or 0
        newest = db.scalar(select(func.max(ApiCache.created_at)))
    return {
        "entries": int(total),
        "fresh": int(fresh),
        "hits": int(hits),
        "newest": newest.isoformat() + "Z" if isinstance(newest, datetime) else None,
    }
