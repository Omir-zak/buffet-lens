"""In-process TTL cache; yfinance and web search are slow and rate-limited, so repeat lookups skip the network."""

import time
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._items[key]  # expired
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._items.clear()
