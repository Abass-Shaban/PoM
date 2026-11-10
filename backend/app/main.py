"""FastAPI application entry point for BLAQUE ANALYSIS."""
from __future__ import annotations

import logging
import mimetypes
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import FRONTEND_DIR, settings
from .database import SessionLocal, init_db
from .errors import AppError
from .models import Fixture, SyncRun, User, utcnow
from .routers import admin, auth, public, user
from .security import hash_password
from .services.ingest import refresh_today, seed_leagues, sync_featured
from .services.runtime import load_overrides, sync_on_startup

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("blaque")


def _bootstrap_admin() -> None:
    if not settings.admin_email or not settings.admin_password:
        return
    with SessionLocal() as db:
        email = settings.admin_email.lower().strip()
        existing = db.scalar(select(User).where(User.email == email))
        if existing is None:
            db.add(
                User(
                    email=email,
                    full_name="BLAQUE Administrator",
                    password_hash=hash_password(settings.admin_password),
                    role="admin",
                )
            )
            db.commit()
            logger.info("Bootstrapped administrator account %s", email)
        elif existing.role != "admin":
            existing.role = "admin"
            db.commit()


def _initial_sync() -> None:
    """Warm the database with real fixtures in the background on first boot."""
    run_id: int | None = None
    try:
        with SessionLocal() as db:
            count = db.scalar(select(func.count()).select_from(Fixture)) or 0
            run = SyncRun(scope="startup", status="running", triggered_by="system")
            db.add(run)
            db.commit()
            db.refresh(run)
            run_id = run.id
        months_back = 15 if count < settings.sync_min_fixtures else settings.sync_months_back
        logger.info("Starting background sync (months_back=%s)", months_back)
        result = sync_featured(
            months_back=months_back,
            months_ahead=max(settings.sync_months_ahead, 1),
        )
        refresh_today()
        with SessionLocal() as db:
            run = db.get(SyncRun, run_id)
            if run:
                run.status = "completed"
                run.detail = result
                run.finished_at = utcnow()
                db.commit()
        logger.info("Background sync complete: %s fixtures", result.get("fixtures"))
    except Exception:  # noqa: BLE001
        logger.exception("Background sync failed")
        try:
            with SessionLocal() as db:
                run = db.get(SyncRun, run_id) if run_id is not None else None
                if run:
                    run.status = "failed"
                    run.finished_at = utcnow()
                    db.commit()
        except Exception:  # noqa: BLE001
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings.ensure_dirs()
    init_db()
    with SessionLocal() as db:
        seed_leagues(db)
        load_overrides(db)
    _bootstrap_admin()
    if sync_on_startup():
        thread = threading.Thread(target=_initial_sync, name="blaque-initial-sync", daemon=True)
        thread.start()
    logger.info("%s v%s started (provider=%s)", settings.app_name, settings.app_version, settings.provider)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="BLAQUE ANALYSIS API",
    description=(
        "Football analytics and prediction API. Fixtures, results, standings and "
        "match statistics are sourced from ESPN's public soccer API. Predictions "
        "combine a Dixon-Coles Poisson model with a scikit-learn classifier."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "img-src 'self' data: https://a.espncdn.com https://*.espncdn.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' data: https://fonts.gstatic.com; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):  # noqa: ARG001
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):  # noqa: ARG001
    details = []
    for error in exc.errors():
        details.append(
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part != "body"),
                "message": error.get("msg"),
                "type": error.get("type"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Some of the information you entered is invalid.",
                "details": details,
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):  # noqa: ARG001
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"http_{exc.status_code}", "message": message}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):  # noqa: ARG001
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Something went wrong on our side. Please try again.",
            }
        },
    )


app.include_router(auth.router)
app.include_router(public.router)
app.include_router(user.router)
app.include_router(admin.router)

# ---------------------------------------------------------------- static files
for folder in ("assets", "css", "js", "vendor"):
    directory = FRONTEND_DIR / folder
    if directory.exists():
        app.mount(f"/{folder}", StaticFiles(directory=directory), name=folder)

INDEX_FILE = FRONTEND_DIR / "index.html"


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = FRONTEND_DIR / "assets" / "favicon.svg"
    if svg.exists():
        return FileResponse(svg, media_type="image/svg+xml")
    return JSONResponse(status_code=204, content=None)


@app.get("/", include_in_schema=False)
def index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE, media_type="text/html", headers={"Cache-Control": "no-cache"})
    return JSONResponse(
        status_code=200,
        content={
            "name": "BLAQUE ANALYSIS API",
            "message": "Frontend not found. Place the frontend files in /frontend.",
            "docs": "/docs",
        },
    )


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "not_found", "message": "Endpoint not found."}},
        )
    candidate = (FRONTEND_DIR / full_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIR.resolve())
    except ValueError:
        candidate = INDEX_FILE
    if candidate.is_file():
        media_type, _ = mimetypes.guess_type(str(candidate))
        return FileResponse(candidate, media_type=media_type)
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE, media_type="text/html", headers={"Cache-Control": "no-cache"})
    return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "Not found."}})
