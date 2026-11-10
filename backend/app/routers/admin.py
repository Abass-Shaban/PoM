"""Administrator routes: users, data sync, model training, settings and audit."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import audit, require_admin
from ..errors import ConflictError, NotFoundError, ValidationError
from ..models import (
    AuditLog,
    Fixture,
    League,
    PredictionRecord,
    SavedPrediction,
    SyncRun,
    Team,
    User,
    utcnow,
)
from ..schemas import AdminUserUpdate, LeagueUpdate, SettingsUpdate, SyncRequest, UserOut
from ..services.ingest import seed_leagues, sync_featured, sync_league
from ..services.ml import model_status, reload_model, train_model
from ..services.provider.cache import cache_stats, clear_cache
from ..services.provider.factory import get_provider
from ..services.runtime import get_blend_weight, get_featured_slugs, snapshot, update_overrides
from ..services.serialize import fixture_dict, iso_utc, league_dict

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats")
def admin_stats(db: Session = Depends(get_db)) -> dict:
    today = datetime.combine(utcnow().date(), datetime.min.time())
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    active_users = db.scalar(select(func.count()).select_from(User).where(User.is_active.is_(True))) or 0
    admins = db.scalar(select(func.count()).select_from(User).where(User.role == "admin")) or 0
    fixtures = db.scalar(select(func.count()).select_from(Fixture)) or 0
    teams = db.scalar(select(func.count()).select_from(Team)) or 0
    leagues = db.scalar(select(func.count()).select_from(League)) or 0
    predictions = db.scalar(select(func.count()).select_from(PredictionRecord)) or 0
    predictions_today = db.scalar(
        select(func.count()).select_from(PredictionRecord).where(PredictionRecord.created_at >= today)
    ) or 0
    saved = db.scalar(select(func.count()).select_from(SavedPrediction)) or 0
    last_sync = db.scalar(select(SyncRun).order_by(SyncRun.id.desc()).limit(1))
    recent_activity = list(
        db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(10))
    )
    return {
        "users": {"total": int(total_users), "active": int(active_users), "admins": int(admins)},
        "data": {
            "fixtures": int(fixtures),
            "teams": int(teams),
            "leagues": int(leagues),
        },
        "predictions": {"total": int(predictions), "today": int(predictions_today), "saved": int(saved)},
        "cache": cache_stats(),
        "model": model_status(),
        "last_sync": {
            "id": last_sync.id,
            "status": last_sync.status,
            "scope": last_sync.scope,
            "started_at": iso_utc(last_sync.started_at),
            "finished_at": iso_utc(last_sync.finished_at),
            "detail": last_sync.detail,
        }
        if last_sync
        else None,
        "recent_activity": [
            {
                "id": row.id,
                "action": row.action,
                "detail": row.detail,
                "ip": row.ip,
                "created_at": iso_utc(row.created_at),
            }
            for row in recent_activity
        ],
    }


@router.get("/users")
def list_users(
    q: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(User)
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where((User.email.ilike(pattern)) | (User.full_name.ilike(pattern)))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(db.scalars(stmt.order_by(User.created_at.desc()).offset(offset).limit(limit)))
    return {
        "items": [
            {
                **UserOut.model_validate(row).model_dump(mode="json"),
                "saved_predictions": db.scalar(
                    select(func.count())
                    .select_from(SavedPrediction)
                    .where(SavedPrediction.user_id == row.id)
                )
                or 0,
            }
            for row in rows
        ],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


@router.patch("/users/{user_id}")
def update_user(
    user_id: int,
    payload: AdminUserUpdate,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found.")
    if payload.role is not None:
        if user.id == admin.id and payload.role != "admin":
            raise ConflictError("You cannot remove your own administrator role.")
        if user.role == "admin" and payload.role != "admin":
            admin_count = db.scalar(
                select(func.count()).select_from(User).where(User.role == "admin")
            ) or 0
            if admin_count <= 1:
                raise ConflictError("At least one administrator account is required.")
        user.role = payload.role
    if payload.is_active is not None:
        if user.id == admin.id and not payload.is_active:
            raise ConflictError("You cannot deactivate your own account.")
        user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    audit(
        db,
        "admin.user.update",
        user=admin,
        request=request,
        detail={"target_user": user.id, "role": user.role, "is_active": user.is_active},
    )
    db.commit()
    return {"ok": True, "user": UserOut.model_validate(user).model_dump(mode="json")}


@router.get("/predictions")
def admin_predictions(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict:
    rows = list(
        db.scalars(select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit))
    )
    items = []
    for row in rows:
        fixture = db.get(Fixture, row.fixture_id)
        user = db.get(User, row.user_id) if row.user_id else None
        payload = row.payload or {}
        items.append(
            {
                "id": row.id,
                "fixture_id": row.fixture_id,
                "fixture": fixture_dict(fixture) if fixture else None,
                "user": {"id": user.id, "email": user.email} if user else None,
                "model_version": row.model_version,
                "probabilities": payload.get("probabilities"),
                "most_likely": payload.get("most_likely"),
                "expected": payload.get("expected"),
                "created_at": iso_utc(row.created_at),
            }
        )
    return {"items": items, "total": len(items)}


@router.get("/settings")
def get_settings(db: Session = Depends(get_db)) -> dict:
    seed_leagues(db)
    leagues = list(db.scalars(select(League).order_by(League.sort_order.asc())))
    provider = get_provider()
    return {
        "runtime": snapshot(),
        "defaults": {
            "cache_ttl_live": settings.cache_ttl_live,
            "cache_ttl_fixtures": settings.cache_ttl_fixtures,
            "cache_ttl_standings": settings.cache_ttl_standings,
            "cache_ttl_summary": settings.cache_ttl_summary,
            "cache_ttl_schedule": settings.cache_ttl_schedule,
            "cache_ttl_teams": settings.cache_ttl_teams,
            "sync_on_startup": settings.sync_on_startup,
            "model_blend_weight": get_blend_weight(),
            "featured_leagues": get_featured_slugs(),
        },
        "provider": {
            "name": provider.name,
            "capabilities": provider.capabilities,
            "site_base": settings.espn_site_base,
            "key_required": False,
        },
        "model": model_status(),
        "cache": cache_stats(),
        "leagues": [
            {
                **league_dict(league),
                "fixtures": db.scalar(
                    select(func.count()).select_from(Fixture).where(Fixture.league_slug == league.slug)
                )
                or 0,
            }
            for league in leagues
        ],
    }


@router.put("/settings")
def save_settings(
    payload: SettingsUpdate,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    updates = payload.model_dump(exclude_none=True)
    if "featured_leagues" in updates:
        known = {league.slug for league in db.scalars(select(League))}
        unknown = [slug for slug in updates["featured_leagues"] if slug not in known]
        if unknown:
            raise ValidationError(f"Unknown league slugs: {', '.join(unknown)}")
        for league in db.scalars(select(League)):
            league.is_featured = league.slug in updates["featured_leagues"]
        db.commit()
    merged = update_overrides(db, updates)
    audit(db, "admin.settings.update", user=admin, request=request, detail=updates)
    db.commit()
    return {"ok": True, "runtime": merged}


@router.patch("/leagues/{slug}")
def update_league(
    slug: str,
    payload: LeagueUpdate,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    league = db.get(League, slug)
    if league is None:
        raise NotFoundError("League not found.")
    league.is_featured = payload.is_featured
    db.commit()
    current = get_featured_slugs()
    if payload.is_featured and slug not in current:
        current.append(slug)
    elif not payload.is_featured and slug in current:
        current.remove(slug)
    update_overrides(db, {"featured_leagues": current})
    audit(
        db,
        "admin.league.update",
        user=admin,
        request=request,
        detail={"slug": slug, "is_featured": payload.is_featured},
    )
    db.commit()
    return {"ok": True, "league": league_dict(league)}


@router.post("/sync")
def run_sync(
    payload: SyncRequest,
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    run = SyncRun(
        scope=payload.scope,
        status="running",
        triggered_by=admin.email,
        detail={"league": payload.league},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        if payload.scope == "league":
            if not payload.league:
                raise ValidationError("A league slug is required for a league sync.")
            if db.get(League, payload.league) is None:
                raise NotFoundError("League not found.")
            result = sync_league(
                db,
                payload.league,
                months_back=payload.months_back,
                months_ahead=payload.months_ahead,
            )
        else:
            result = sync_featured(
                months_back=payload.months_back,
                months_ahead=payload.months_ahead,
            )
        run.status = "completed"
        run.detail = result
        run.finished_at = utcnow()
        db.commit()
        audit(db, "admin.sync", user=admin, request=request, detail={"scope": payload.scope, "league": payload.league})
        db.commit()
        return {"ok": True, "run_id": run.id, "result": result}
    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.detail = {"error": str(exc)}
        run.finished_at = utcnow()
        db.commit()
        raise


@router.get("/sync-runs")
def sync_runs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict:
    rows = list(db.scalars(select(SyncRun).order_by(SyncRun.id.desc()).limit(limit)))
    return {
        "items": [
            {
                "id": row.id,
                "scope": row.scope,
                "status": row.status,
                "triggered_by": row.triggered_by,
                "detail": row.detail,
                "started_at": iso_utc(row.started_at),
                "finished_at": iso_utc(row.finished_at),
            }
            for row in rows
        ]
    }


@router.post("/model/train")
def train(
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    result = train_model(db)
    audit(db, "admin.model.train", user=admin, request=request, detail=result)
    db.commit()
    return {"ok": True, "result": result, "status": model_status()}


@router.post("/model/reload")
def reload(
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    reload_model()
    audit(db, "admin.model.reload", user=admin, request=request)
    db.commit()
    return {"ok": True, "status": model_status()}


@router.get("/audit")
def audit_log(
    limit: int = Query(default=100, ge=1, le=500),
    action: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    stmt = select(AuditLog)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    rows = list(db.scalars(stmt.order_by(AuditLog.id.desc()).limit(limit)))
    users = {user.id: user.email for user in db.scalars(select(User))}
    return {
        "items": [
            {
                "id": row.id,
                "action": row.action,
                "user": users.get(row.user_id),
                "detail": row.detail,
                "ip": row.ip,
                "created_at": iso_utc(row.created_at),
            }
            for row in rows
        ],
        "total": len(rows),
    }


@router.post("/cache/clear")
def clear_api_cache(
    request: Request,
    prefix: str | None = Query(default=None, max_length=80),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> dict:
    count = clear_cache(prefix)
    audit(db, "admin.cache.clear", user=admin, request=request, detail={"prefix": prefix, "removed": count})
    db.commit()
    return {"ok": True, "removed": count}
