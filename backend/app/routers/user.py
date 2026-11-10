"""Authenticated user routes: favorites and saved predictions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import audit, get_current_user
from ..errors import ConflictError, NotFoundError
from ..models import (
    FavoriteFixture,
    FavoriteTeam,
    Fixture,
    PredictionRecord,
    SavedPrediction,
    Team,
    User,
    utcnow,
)
from ..schemas import SavePredictionRequest
from ..services.analysis import analyze_fixture
from ..services.serialize import fixture_dict, iso_utc, team_dict

router = APIRouter(prefix="/api", tags=["user"])


def _settlement(saved: SavedPrediction) -> dict | None:
    fixture = saved.fixture
    if fixture is None or fixture.status != "finished":
        return None
    if fixture.home_score is None or fixture.away_score is None:
        return None
    if fixture.home_score > fixture.away_score:
        actual = "home"
    elif fixture.home_score < fixture.away_score:
        actual = "away"
    else:
        actual = "draw"
    pick = (saved.prediction or {}).get("most_likely", {}).get("outcome")
    return {
        "actual": actual,
        "actual_label": {"home": "Home win", "draw": "Draw", "away": "Away win"}[actual],
        "pick": pick,
        "hit": pick == actual,
        "scoreline": fixture.scoreline,
    }


def _saved_dict(saved: SavedPrediction) -> dict:
    return {
        "id": saved.id,
        "fixture_id": saved.fixture_id,
        "window": saved.window,
        "note": saved.note,
        "model_version": saved.model_version,
        "created_at": iso_utc(saved.created_at),
        "updated_at": iso_utc(saved.updated_at),
        "fixture": saved.fixture_snapshot,
        "prediction": saved.prediction,
        "settlement": _settlement(saved),
    }


@router.get("/favorites/teams")
def list_favorite_teams(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    rows = list(
        db.scalars(
            select(FavoriteTeam)
            .where(FavoriteTeam.user_id == user.id)
            .order_by(FavoriteTeam.created_at.desc())
        )
    )
    return {
        "items": [
            {**team_dict(row.team), "favorited_at": iso_utc(row.created_at)}
            for row in rows
            if row.team
        ]
    }


@router.post("/favorites/teams/{team_id}", status_code=201)
def add_favorite_team(
    team_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    team = db.get(Team, team_id)
    if team is None:
        raise NotFoundError("Team not found.")
    existing = db.get(FavoriteTeam, (user.id, team_id))
    if existing is not None:
        return {"ok": True, "already": True, "team": team_dict(team)}
    db.add(FavoriteTeam(user_id=user.id, team_id=team_id))
    db.commit()
    audit(db, "favorite.team.add", user=user, request=request, detail={"team_id": team_id})
    db.commit()
    return {"ok": True, "team": team_dict(team)}


@router.delete("/favorites/teams/{team_id}")
def remove_favorite_team(
    team_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    db.execute(
        delete(FavoriteTeam).where(
            FavoriteTeam.user_id == user.id, FavoriteTeam.team_id == team_id
        )
    )
    db.commit()
    audit(db, "favorite.team.remove", user=user, request=request, detail={"team_id": team_id})
    db.commit()
    return {"ok": True}


@router.get("/favorites/fixtures")
def list_favorite_fixtures(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    rows = list(
        db.scalars(
            select(FavoriteFixture)
            .where(FavoriteFixture.user_id == user.id)
            .order_by(FavoriteFixture.created_at.desc())
        )
    )
    return {
        "items": [
            {**fixture_dict(row.fixture), "favorited_at": iso_utc(row.created_at)}
            for row in rows
            if row.fixture
        ]
    }


@router.post("/favorites/fixtures/{fixture_id}", status_code=201)
def add_favorite_fixture(
    fixture_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    fixture = db.get(Fixture, fixture_id)
    if fixture is None:
        raise NotFoundError("Fixture not found.")
    existing = db.get(FavoriteFixture, (user.id, fixture_id))
    if existing is not None:
        return {"ok": True, "already": True, "fixture": fixture_dict(fixture)}
    db.add(FavoriteFixture(user_id=user.id, fixture_id=fixture_id))
    db.commit()
    audit(db, "favorite.fixture.add", user=user, request=request, detail={"fixture_id": fixture_id})
    db.commit()
    return {"ok": True, "fixture": fixture_dict(fixture)}


@router.delete("/favorites/fixtures/{fixture_id}")
def remove_favorite_fixture(
    fixture_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    db.execute(
        delete(FavoriteFixture).where(
            FavoriteFixture.user_id == user.id, FavoriteFixture.fixture_id == fixture_id
        )
    )
    db.commit()
    audit(db, "favorite.fixture.remove", user=user, request=request, detail={"fixture_id": fixture_id})
    db.commit()
    return {"ok": True}


@router.get("/predictions/saved")
def list_saved(
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    rows = list(
        db.scalars(
            select(SavedPrediction)
            .where(SavedPrediction.user_id == user.id)
            .order_by(SavedPrediction.updated_at.desc())
            .limit(limit)
        )
    )
    return {"items": [_saved_dict(row) for row in rows], "total": len(rows)}


@router.post("/predictions/saved", status_code=201)
def save_prediction(
    payload: SavePredictionRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    fixture = db.get(Fixture, payload.fixture_id)
    if fixture is None:
        raise NotFoundError("Fixture not found.")
    analysis = analyze_fixture(db, payload.fixture_id, payload.window, user=user, record=True)
    prediction = analysis["prediction"]
    snapshot = analysis["fixture"]
    existing = db.scalar(
        select(SavedPrediction).where(
            SavedPrediction.user_id == user.id,
            SavedPrediction.fixture_id == payload.fixture_id,
        )
    )
    if existing is None:
        existing = SavedPrediction(
            user_id=user.id,
            fixture_id=payload.fixture_id,
            window=payload.window,
            model_version=prediction["model"]["version"],
            fixture_snapshot=snapshot,
            prediction=prediction,
            note=payload.note,
        )
        db.add(existing)
        message = "Prediction saved."
    else:
        existing.window = payload.window
        existing.model_version = prediction["model"]["version"]
        existing.fixture_snapshot = snapshot
        existing.prediction = prediction
        existing.note = payload.note
        existing.updated_at = utcnow()
        message = "Saved prediction updated."
    db.commit()
    db.refresh(existing)
    audit(
        db,
        "prediction.save",
        user=user,
        request=request,
        detail={"fixture_id": payload.fixture_id, "window": payload.window},
    )
    db.commit()
    return {"ok": True, "message": message, "item": _saved_dict(existing)}


@router.get("/predictions/saved/{saved_id}")
def get_saved(
    saved_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    saved = db.get(SavedPrediction, saved_id)
    if saved is None or saved.user_id != user.id:
        raise NotFoundError("Saved prediction not found.")
    return _saved_dict(saved)


@router.post("/predictions/saved/{saved_id}/refresh")
def refresh_saved(
    saved_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    saved = db.get(SavedPrediction, saved_id)
    if saved is None or saved.user_id != user.id:
        raise NotFoundError("Saved prediction not found.")
    analysis = analyze_fixture(db, saved.fixture_id, saved.window, user=user, record=True)
    saved.prediction = analysis["prediction"]
    saved.fixture_snapshot = analysis["fixture"]
    saved.model_version = analysis["prediction"]["model"]["version"]
    saved.updated_at = utcnow()
    db.commit()
    db.refresh(saved)
    audit(db, "prediction.refresh", user=user, request=request, detail={"saved_id": saved_id})
    db.commit()
    return {"ok": True, "item": _saved_dict(saved)}


@router.delete("/predictions/saved/{saved_id}")
def delete_saved(
    saved_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    saved = db.get(SavedPrediction, saved_id)
    if saved is None or saved.user_id != user.id:
        raise NotFoundError("Saved prediction not found.")
    db.delete(saved)
    db.commit()
    audit(db, "prediction.delete", user=user, request=request, detail={"saved_id": saved_id})
    db.commit()
    return {"ok": True}


@router.get("/predictions/history")
def prediction_history(
    limit: int = Query(default=30, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    rows = list(
        db.scalars(
            select(PredictionRecord)
            .where(PredictionRecord.user_id == user.id)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
        )
    )
    items = []
    for row in rows:
        fixture = db.get(Fixture, row.fixture_id)
        items.append(
            {
                "id": row.id,
                "fixture_id": row.fixture_id,
                "model_version": row.model_version,
                "created_at": iso_utc(row.created_at),
                "fixture": fixture_dict(fixture) if fixture else None,
                "probabilities": (row.payload or {}).get("probabilities"),
                "most_likely": (row.payload or {}).get("most_likely"),
            }
        )
    return {"items": items, "total": len(items)}


@router.get("/predictions/summary")
def prediction_summary(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    saved = list(
        db.scalars(select(SavedPrediction).where(SavedPrediction.user_id == user.id))
    )
    settled = [item for item in saved if _settlement(item) is not None]
    hits = sum(1 for item in settled if (_settlement(item) or {}).get("hit"))
    generated = db.scalar(
        select(func.count())
        .select_from(PredictionRecord)
        .where(PredictionRecord.user_id == user.id)
    ) or 0
    return {
        "saved": len(saved),
        "settled": len(settled),
        "hits": hits,
        "accuracy": round(hits / len(settled), 4) if settled else None,
        "generated": int(generated),
    }
