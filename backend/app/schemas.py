"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .security import validate_password_strength


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("full_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Please enter your full name.")
        return value

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        validate_password_strength(value)
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UpdateProfileRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=120)

    @field_validator("full_name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Please enter your full name.")
        return value


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        validate_password_strength(value)
        return value


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class SavePredictionRequest(BaseModel):
    fixture_id: int = Field(gt=0)
    window: int = Field(default=10, ge=5, le=15)
    note: str | None = Field(default=None, max_length=500)


class AdminUserUpdate(BaseModel):
    role: Literal["user", "admin"] | None = None
    is_active: bool | None = None


class SyncRequest(BaseModel):
    scope: Literal["featured", "league"] = "featured"
    league: str | None = Field(default=None, max_length=64)
    months_back: int = Field(default=1, ge=0, le=24)
    months_ahead: int = Field(default=1, ge=0, le=12)


class SettingsUpdate(BaseModel):
    cache_ttl_live: int | None = Field(default=None, ge=10, le=3600)
    cache_ttl_fixtures: int | None = Field(default=None, ge=30, le=86400)
    cache_ttl_standings: int | None = Field(default=None, ge=300, le=604800)
    sync_on_startup: bool | None = None
    model_blend_weight: float | None = Field(default=None, ge=0.0, le=0.8)
    featured_leagues: list[str] | None = None


class LeagueUpdate(BaseModel):
    is_featured: bool

