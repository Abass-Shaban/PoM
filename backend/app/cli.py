"""Command-line utilities for BLAQUE ANALYSIS.

Examples
--------
    python -m backend.app.cli sync --months-back 15 --months-ahead 1
    python -m backend.app.cli train-model
    python -m backend.app.cli create-admin --email admin@example.com --password "S3curePass"
    python -m backend.app.cli status
"""
from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import func, select

from .config import settings
from .database import SessionLocal, init_db
from .models import Fixture, League, SyncRun, Team, User, utcnow
from .security import hash_password, validate_password_strength
from .services.ingest import refresh_today, seed_leagues, sync_featured, sync_league
from .services.ml import model_status, train_model
from .services.runtime import load_overrides


def _prepare() -> None:
    settings.ensure_dirs()
    init_db()
    with SessionLocal() as db:
        seed_leagues(db)
        load_overrides(db)


def cmd_sync(args: argparse.Namespace) -> int:
    _prepare()
    print(
        f"Syncing real fixtures from ESPN "
        f"(months back={args.months_back}, ahead={args.months_ahead}) ..."
    )
    if args.league:
        with SessionLocal() as db:
            result = sync_league(
                db, args.league, months_back=args.months_back, months_ahead=args.months_ahead
            )
        print(json.dumps(result, indent=2))
    else:
        result = sync_featured(
            months_back=args.months_back,
            months_ahead=args.months_ahead,
            slugs=args.leagues or None,
            max_workers=args.workers,
        )
        refresh_today()
        with SessionLocal() as db:
            db.add(
                SyncRun(
                    scope="cli",
                    status="completed",
                    triggered_by="cli",
                    detail=result,
                    finished_at=utcnow(),
                )
            )
            db.commit()
        print(
            f"Synced {result['fixtures']} fixtures across "
            f"{len(result['leagues'])} leagues ({result['teams']} team rows)."
        )
        if result["errors"]:
            print("Warnings:")
            for error in result["errors"][:25]:
                print(f"  - {error}")
    with SessionLocal() as db:
        print(
            "Database now holds "
            f"{db.scalar(select(func.count()).select_from(Fixture)) or 0} fixtures, "
            f"{db.scalar(select(func.count()).select_from(Team)) or 0} teams, "
            f"{db.scalar(select(func.count()).select_from(League)) or 0} leagues."
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _prepare()
    print("Training scikit-learn model from stored completed matches ...")
    with SessionLocal() as db:
        result = train_model(db)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("trained") else 1


def cmd_create_admin(args: argparse.Namespace) -> int:
    _prepare()
    email = args.email.lower().strip()
    try:
        validate_password_strength(args.password)
    except Exception as exc:  # noqa: BLE001
        print(f"Password rejected: {exc}", file=sys.stderr)
        return 2
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(
                email=email,
                full_name=args.name,
                password_hash=hash_password(args.password),
                role="admin",
            )
            db.add(user)
            message = "Administrator account created."
        else:
            user.role = "admin"
            user.is_active = True
            if args.reset_password:
                user.password_hash = hash_password(args.password)
            message = "Existing account promoted to administrator."
        db.commit()
        print(f"{message} {email}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    _prepare()
    with SessionLocal() as db:
        payload = {
            "database": settings.database_url,
            "counts": {
                "leagues": db.scalar(select(func.count()).select_from(League)) or 0,
                "teams": db.scalar(select(func.count()).select_from(Team)) or 0,
                "fixtures": db.scalar(select(func.count()).select_from(Fixture)) or 0,
                "users": db.scalar(select(func.count()).select_from(User)) or 0,
            },
            "model": model_status(),
        }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blaque",
        description="BLAQUE ANALYSIS maintenance commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Fetch real fixtures and statistics from ESPN.")
    sync.add_argument("--months-back", type=int, default=15)
    sync.add_argument("--months-ahead", type=int, default=1)
    sync.add_argument("--league", default=None, help="Sync a single league slug.")
    sync.add_argument(
        "--leagues",
        nargs="*",
        default=None,
        help="Limit the sync to specific league slugs.",
    )
    sync.add_argument("--workers", type=int, default=1, help="Concurrent league workers (SQLite should use 1).")
    sync.set_defaults(func=cmd_sync)

    train = sub.add_parser("train-model", help="Train the scikit-learn result model.")
    train.set_defaults(func=cmd_train)

    admin = sub.add_parser("create-admin", help="Create or promote an administrator.")
    admin.add_argument("--email", required=True)
    admin.add_argument("--password", required=True)
    admin.add_argument("--name", default="BLAQUE Administrator")
    admin.add_argument("--reset-password", action="store_true")
    admin.set_defaults(func=cmd_create_admin)

    status = sub.add_parser("status", help="Show database and model status.")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
