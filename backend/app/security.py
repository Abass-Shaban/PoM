"""Password hashing, JWT issuing/validation and login throttling."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from .config import settings
from .errors import AuthError, RateLimitError

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, n, r, p, salt, digest = stored.split("$")
        if algorithm != "scrypt":
            return False
        computed = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_unb64(digest)),
        )
        return hmac.compare_digest(computed, _unb64(digest))
    except (ValueError, TypeError):
        return False


def validate_password_strength(password: str) -> None:
    from .errors import ValidationError

    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValidationError("Password must be 128 characters or fewer.")
    if not any(c.isalpha() for c in password):
        raise ValidationError("Password must contain at least one letter.")
    if not any(c.isdigit() for c in password):
        raise ValidationError("Password must contain at least one number.")


def create_access_token(user_id: int, email: str, role: str) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": "blaque-analysis",
        "type": "access",
    }
    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return token, int(expires_delta.total_seconds())


def decode_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
            issuer="blaque-analysis",
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Your session has expired. Please sign in again.") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid authentication token.") from exc
    if payload.get("type") != "access" or not payload.get("sub"):
        raise AuthError("Invalid authentication token.")
    return payload


class LoginRateLimiter:
    """Small in-process sliding-window limiter for credential endpoints.

    For multi-worker deployments, swap this for Redis.  It intentionally keeps
    only hashed keys in memory.
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, cutoff: float) -> deque[float]:
        bucket = self._events[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return bucket

    def check(self, key: str) -> None:
        cutoff = time.time() - settings.login_rate_window_seconds
        with self._lock:
            bucket = self._prune(key, cutoff)
            if len(bucket) >= settings.login_rate_limit:
                retry_after = int(settings.login_rate_window_seconds - (time.time() - bucket[0]))
                raise RateLimitError(
                    "Too many attempts. Please wait before trying again.",
                    details={"retry_after": max(retry_after, 1)},
                )

    def hit(self, key: str) -> None:
        with self._lock:
            self._events[key].append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


login_limiter = LoginRateLimiter()
