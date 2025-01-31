"""Module for caching frequently accessed data."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class Cache:
  """Simple in-memory cache with TTL."""

  def __init__(self):
    self._cache: Dict[str, Dict[str, Any]] = {}

  def get(self, key: str) -> Optional[Any]:
    """Get value from cache if not expired."""
    if key in self._cache:
      item = self._cache[key]
      if datetime.now() < item["expires_at"]:
        return item["value"]
      del self._cache[key]
    return None

  def set(self, key: str, value: Any, ttl_seconds: int = 300):
    """Set value in cache with TTL."""
    self._cache[key] = {"value": value, "expires_at": datetime.now() + timedelta(seconds=ttl_seconds)}

  def delete(self, key: str):
    """Delete value from cache."""
    if key in self._cache:
      del self._cache[key]


deps_cache = Cache()
