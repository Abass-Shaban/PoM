"""Authentication routes: register, login, profile and password management."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import audit, client_ip, get_current_user
from ..errors import AuthError, ConflictError
from ..models import User, utcnow
from ..schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenOut,
    UpdateProfileRequest,
    UserOut,
)
from ..security import (
    create_access_token,
    hash_password,
    login_limiter,
    validate_password_strength,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

_DUMMY_HASH = hash_password("blaque-dummy-password-1")


def _issue(user: User) -> TokenOut:
    token, expires_in = create_access_token(user.id, user.email, user.role)
    return TokenOut(
        access_token=token,
        expires_in=expires_in,
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenOut, status_code=201)
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)) -> TokenOut:
    email = payload.email.lower().strip()
    existing = db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise ConflictError("An account with this email address already exists.")

    is_first_user = (db.scalar(select(func.count()).select_from(User)) or 0) == 0
    role = "admin" if is_first_user or (settings.admin_email and email == settings.admin_email.lower()) else "user"
    user = User(
        email=email,
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    audit(db, "auth.register", user=user, request=request, detail={"role": role})
    db.commit()
    return _issue(user)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> TokenOut:
    email = payload.email.lower().strip()
    key = f"{client_ip(request)}:{email}"
    login_limiter.check(key)
    user = db.scalar(select(User).where(User.email == email))
    valid = verify_password(payload.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not valid:
        login_limiter.hit(key)
        raise AuthError("Incorrect email or password.")
    if not user.is_active:
        raise AuthError("This account has been deactivated. Please contact an administrator.")
    login_limiter.reset(key)
    user.last_login_at = utcnow()
    db.commit()
    audit(db, "auth.login", user=user, request=request)
    db.commit()
    return _issue(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)


@router.patch("/me", response_model=UserOut)
def update_me(
    payload: UpdateProfileRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    if payload.full_name is not None:
        user.full_name = payload.full_name
    db.commit()
    db.refresh(user)
    audit(db, "profile.update", user=user, request=request)
    db.commit()
    return UserOut.model_validate(user)


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not verify_password(payload.current_password, user.password_hash):
        raise AuthError("Your current password is incorrect.")
    if payload.current_password == payload.new_password:
        raise AuthError("Your new password must be different from the current password.")
    validate_password_strength(payload.new_password)
    user.password_hash = hash_password(payload.new_password)
    db.commit()
    audit(db, "profile.change_password", user=user, request=request)
    db.commit()
    return {"ok": True, "message": "Password updated."}


@router.post("/logout")
def logout(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    audit(db, "auth.logout", user=user, request=request)
    db.commit()
    return {"ok": True}
