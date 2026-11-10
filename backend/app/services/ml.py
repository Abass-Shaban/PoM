"""Scikit-learn model: training, evaluation, persistence and inference.

The model learns the relationship between pre-match form/venue/corner features
and the three-way result.  It is blended with the Dixon-Coles Poisson model;
when no model has been trained the app transparently falls back to the
statistical model alone.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import session_scope
from ..models import Fixture, utcnow
from .features import FEATURE_NAMES, DEFAULTS, to_vector

logger = logging.getLogger("blaque.ml")

_LOCK = threading.RLock()
_ARTIFACT: dict | None = None
_LOAD_ERROR: str | None = None

PRIOR_HOME = 1.45
PRIOR_AWAY = 1.15


def _stats(history: deque, window: int) -> dict:
    rows = list(history)[:window]
    if not rows:
        return {
            "ppg": 1.35,
            "gf": 1.35,
            "ga": 1.25,
            "btts": 0.5,
            "corners": 4.8,
        }
    played = len(rows)
    points = sum(r["points"] for r in rows)
    goals_for = sum(r["gf"] for r in rows)
    goals_against = sum(r["ga"] for r in rows)
    btts = sum(1 for r in rows if r["gf"] > 0 and r["ga"] > 0)
    corner_rows = [r["corners"] for r in rows if r.get("corners") is not None]
    return {
        "ppg": points / played,
        "gf": goals_for / played,
        "ga": goals_against / played,
        "btts": btts / played,
        "corners": (sum(corner_rows) / len(corner_rows)) if corner_rows else 4.8,
    }


def _baseline(history: deque) -> tuple[float, float]:
    rows = list(history)
    sample = len(rows)
    if not sample:
        return PRIOR_HOME, PRIOR_AWAY
    home = sum(r[0] for r in rows) / sample
    away = sum(r[1] for r in rows) / sample
    weight = max(0.0, (20 - sample) / 20)
    return (
        home * (1 - weight) + PRIOR_HOME * weight,
        away * (1 - weight) + PRIOR_AWAY * weight,
    )


def build_training_data(db: Session, max_samples: int = 9000) -> tuple[np.ndarray, np.ndarray, list]:
    fixtures = list(
        db.scalars(
            select(Fixture)
            .where(
                Fixture.status == "finished",
                Fixture.home_score.is_not(None),
                Fixture.away_score.is_not(None),
            )
            .order_by(Fixture.kickoff.asc())
        ).unique()
    )
    team_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    h2h_history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=5))
    league_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
    last_match: dict[int, datetime] = {}

    features: list[list[float]] = []
    labels: list[int] = []
    dates: list[datetime] = []

    for fixture in fixtures:
        home_id = fixture.home_team_id
        away_id = fixture.away_team_id
        hs = int(fixture.home_score or 0)
        as_ = int(fixture.away_score or 0)
        kickoff = fixture.kickoff

        home_hist = team_history[home_id]
        away_hist = team_history[away_id]
        if len(home_hist) >= 3 and len(away_hist) >= 3:
            h5 = _stats(home_hist, 5)
            a5 = _stats(away_hist, 5)
            h10 = _stats(home_hist, 10)
            a10 = _stats(away_hist, 10)
            lh, la = _baseline(league_history[fixture.league_slug])
            pair = tuple(sorted((home_id, away_id)))
            h2h_points = 0
            h2h_count = 0
            for h_id, h_score, a_score in list(h2h_history[pair])[:5]:
                if h_id == home_id:
                    pts = 3 if h_score > a_score else (1 if h_score == a_score else 0)
                else:
                    pts = 3 if a_score > h_score else (1 if h_score == a_score else 0)
                h2h_points += pts
                h2h_count += 1
            home_rest = 14.0
            away_rest = 14.0
            if home_id in last_match:
                home_rest = max(2.0, min(30.0, (kickoff - last_match[home_id]).total_seconds() / 86400))
            if away_id in last_match:
                away_rest = max(2.0, min(30.0, (kickoff - last_match[away_id]).total_seconds() / 86400))
            metrics = {
                "home_ppg5": h5["ppg"],
                "home_gf5": h5["gf"],
                "home_ga5": h5["ga"],
                "home_btts5": h5["btts"],
                "home_corners5": h5["corners"],
                "away_ppg5": a5["ppg"],
                "away_gf5": a5["gf"],
                "away_ga5": a5["ga"],
                "away_btts5": a5["btts"],
                "away_corners5": a5["corners"],
                "home_ppg10": h10["ppg"],
                "away_ppg10": a10["ppg"],
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "h2h_home_ppg": (h2h_points / h2h_count) if h2h_count else DEFAULTS["h2h_home_ppg"],
                "league_home_goals": lh,
                "league_away_goals": la,
            }
            features.append(to_vector(metrics))
            labels.append(0 if hs > as_ else (1 if hs == as_ else 2))
            dates.append(kickoff)

        home_points = 3 if hs > as_ else (1 if hs == as_ else 0)
        away_points = 3 if as_ > hs else (1 if hs == as_ else 0)
        home_corners = fixture.home_corners
        away_corners = fixture.away_corners
        team_history[home_id].appendleft(
            {"gf": hs, "ga": as_, "points": home_points, "corners": home_corners}
        )
        team_history[away_id].appendleft(
            {"gf": as_, "ga": hs, "points": away_points, "corners": away_corners}
        )
        pair = tuple(sorted((home_id, away_id)))
        h2h_history[pair].appendleft((home_id, hs, as_))
        league_history[fixture.league_slug].appendleft((hs, as_))
        last_match[home_id] = kickoff
        last_match[away_id] = kickoff

    if len(features) > max_samples:
        features = features[-max_samples:]
        labels = labels[-max_samples:]
        dates = dates[-max_samples:]
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int), dates


def _brier(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    for index, label in enumerate(y_true):
        if label in classes:
            onehot[index, int(np.where(classes == label)[0][0])] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict_proba(X)
    preds = model.predict(X)
    classes = np.asarray(model.classes_)
    labels = [int(c) for c in classes]
    result = {
        "samples": int(len(y)),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "log_loss": round(float(log_loss(y, probs, labels=labels)), 4),
        "brier": round(_brier(y, probs, classes), 4),
    }
    matrix = np.zeros((3, 3), dtype=int)
    for truth, pred in zip(y, preds):
        matrix[int(truth), int(pred)] += 1
    result["confusion_matrix"] = matrix.tolist()
    result["class_distribution"] = {
        "home_win": int(np.sum(y == 0)),
        "draw": int(np.sum(y == 1)),
        "away_win": int(np.sum(y == 2)),
    }
    return result


def _algorithm_label(model) -> str:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    return type(estimator).__name__


def train_model(db: Session | None = None) -> dict:
    if db is not None:
        return _train_model(db)
    with session_scope() as session:
        return _train_model(session)


def _train_model(db: Session) -> dict:
    X, y, dates = build_training_data(db)
    if len(X) < settings.model_min_samples:
        return {
            "trained": False,
            "reason": "insufficient_data",
            "message": (
                f"At least {settings.model_min_samples} completed matches with "
                f"team history are required; found {len(X)}."
            ),
            "samples": int(len(X)),
        }
    split = max(int(len(X) * 0.8), 1)
    if len(X) - split < 30:
        split = max(len(X) - 30, 1)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    if len(set(y_train.tolist())) < 3:
        return {
            "trained": False,
            "reason": "insufficient_classes",
            "message": "The stored history does not contain all three outcomes yet.",
            "samples": int(len(X)),
        }

    candidates = {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, C=0.7)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
            max_depth=None,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=settings.model_random_state,
        ),
    }
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        results[name] = _evaluate(candidate, X_val, y_val)
        fitted[name] = candidate

    best_name = min(results, key=lambda key: results[key]["log_loss"])
    best = fitted[best_name]
    # Refit the winning algorithm on all available data for production use.
    best.fit(X, y)
    validation = results[best_name]
    version = "ml-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    artifact = {
        "model": best,
        "features": FEATURE_NAMES,
        "metrics": validation,
        "comparison": results,
        "algorithm": best_name,
        "version": version,
        "trained_at": utcnow().isoformat() + "Z",
        "samples": int(len(X)),
        "validation_samples": int(len(X_val)),
        "class_distribution": validation.get("class_distribution"),
    }
    path = Path(settings.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    meta = {k: v for k, v in artifact.items() if k != "model"}
    meta["model_file"] = str(path)
    meta["available"] = True
    meta_path = Path(settings.model_meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with _LOCK:
        global _ARTIFACT, _LOAD_ERROR
        _ARTIFACT = artifact
        _LOAD_ERROR = None
    logger.info("Trained model %s (%s) on %s matches", version, best_name, len(X))
    return {
        "trained": True,
        "version": version,
        "algorithm": best_name,
        "metrics": validation,
        "comparison": results,
        "samples": int(len(X)),
        "validation_samples": int(len(X_val)),
        "features": FEATURE_NAMES,
    }


def _load_artifact() -> dict | None:
    global _ARTIFACT, _LOAD_ERROR
    if _ARTIFACT is not None:
        return _ARTIFACT
    path = Path(settings.model_path)
    if not path.exists():
        return None
    with _LOCK:
        if _ARTIFACT is not None:
            return _ARTIFACT
        try:
            artifact = joblib.load(path)
            if not isinstance(artifact, dict) or "model" not in artifact:
                raise ValueError("Unexpected model artifact format")
            if artifact.get("features") != FEATURE_NAMES:
                raise ValueError("Model feature schema does not match this application version")
            _ARTIFACT = artifact
            _LOAD_ERROR = None
            return artifact
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load ML model")
            _LOAD_ERROR = str(exc)
            return None


def model_status() -> dict:
    artifact = _load_artifact()
    meta: dict = {}
    path = Path(settings.model_meta_path)
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    if artifact is None:
        return {
            "available": False,
            "reason": _LOAD_ERROR or "not_trained",
            "message": "Train the scikit-learn model from the admin dashboard to blend ML with the statistical model.",
            "features": FEATURE_NAMES,
        }
    return {
        "available": True,
        "version": artifact.get("version"),
        "algorithm": artifact.get("algorithm"),
        "trained_at": artifact.get("trained_at"),
        "samples": artifact.get("samples"),
        "validation_samples": artifact.get("validation_samples"),
        "metrics": artifact.get("metrics"),
        "comparison": artifact.get("comparison"),
        "features": FEATURE_NAMES,
        "model_file": meta.get("model_file"),
    }


def predict_from_metrics(metrics: dict) -> dict | None:
    artifact = _load_artifact()
    if artifact is None:
        return None
    try:
        vector = np.asarray([to_vector(metrics)], dtype=float)
        probs = artifact["model"].predict_proba(vector)[0]
        classes = list(artifact["model"].classes_)
        mapping = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for label, probability in zip(classes, probs):
            if int(label) == 0:
                mapping["home"] = float(probability)
            elif int(label) == 1:
                mapping["draw"] = float(probability)
            else:
                mapping["away"] = float(probability)
        total = sum(mapping.values()) or 1.0
        mapping = {k: max(0.0001, v / total) for k, v in mapping.items()}
        total = sum(mapping.values())
        mapping = {k: v / total for k, v in mapping.items()}
        return {
            "probabilities": mapping,
            "version": artifact.get("version"),
            "algorithm": artifact.get("algorithm"),
            "trained_at": artifact.get("trained_at"),
        }
    except Exception:  # noqa: BLE001
        logger.exception("ML inference failed; falling back to Poisson")
        return None


def reload_model() -> None:
    global _ARTIFACT, _LOAD_ERROR
    with _LOCK:
        _ARTIFACT = None
        _LOAD_ERROR = None
