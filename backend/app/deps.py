"""Shared FastAPI dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .errors import AuthError, ForbiddenError
from .models import AuditLog, User
from .security import decode_token

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def _user_from_credentials(
    credentials: HTTPAuthorizationCredentials | None, db: Session
) -> User | None:
    if credentials is None or not credentials.credentials:
        return None
    payload = decode_token(credentials.credentials)
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthError("Invalid authentication token.") from exc
    user = db.get(User, user_id)
    if user is None:
        raise AuthError("This account no longer exists.")
    if not user.is_active:
        raise ForbiddenError("This account has been deactivated.")
    return user


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    user = _user_from_credentials(credentials, db)
    if user is None:
        raise AuthError("Please sign in to continue.")
    return user


def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User | None:
    return _user_from_credentials(credentials, db)


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if not user.is_admin:
        raise ForbiddenError("Administrator access is required.")
    return user


def audit(
    db: Session,
    action: str,
    *,
    user: User | None = None,
    request: Request | None = None,
    detail: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            user_id=user.id if user else None,
            action=action,
            detail=detail,
            ip=client_ip(request) if request else None,
        )
    )
