"""High-level analysis orchestration used by the API routers."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..errors import NotFoundError
from ..models import Fixture, PredictionRecord, User
from .analytics import build_match_context
from .predictor import predict_match
from .serialize import fixture_dict


def analyze_fixture(
    db: Session,
    fixture_id: int,
    window: int = 10,
    *,
    user: User | None = None,
    record: bool = False,
) -> dict:
    fixture = db.get(Fixture, fixture_id)
    if fixture is None:
        raise NotFoundError("That fixture could not be found. It may not be in the tracked competitions yet.")
    context = build_match_context(db, fixture, window)
    prediction = predict_match(context)
    if record:
        db.add(
            PredictionRecord(
                fixture_id=fixture.id,
                user_id=user.id if user else None,
                model_version=prediction["model"]["version"],
                payload=prediction,
            )
        )
        db.commit()
    return {
        "fixture": context["fixture"],
        "context": context,
        "prediction": prediction,
    }


def quick_prediction(db: Session, fixture_id: int, window: int = 10) -> dict:
    fixture = db.get(Fixture, fixture_id)
    if fixture is None:
        raise NotFoundError("Fixture not found.")
    context = build_match_context(db, fixture, window, include_summary=False, apply_summary=False)
    prediction = predict_match(context)
    return {"fixture": fixture_dict(fixture), "prediction": prediction}
