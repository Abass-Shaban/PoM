"""Provider factory (singleton)."""
from __future__ import annotations

import threading

from ...config import settings
from ...errors import ProviderError
from .base import FootballProvider
from .espn import ESPNProvider

_LOCK = threading.Lock()
_PROVIDER: FootballProvider | None = None


def get_provider() -> FootballProvider:
    global _PROVIDER
    if _PROVIDER is None:
        with _LOCK:
            if _PROVIDER is None:
                name = (settings.provider or "espn").lower()
                if name == "espn":
                    _PROVIDER = ESPNProvider()
                else:
                    raise ProviderError(
                        f"Unsupported data provider '{name}'. Supported providers: espn."
                    )
    return _PROVIDER


def reset_provider() -> None:
    global _PROVIDER
    with _LOCK:
        _PROVIDER = None
