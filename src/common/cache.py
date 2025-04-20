"""Module for caching frequently accessed data with Redis support.

Examples:
    Basic usage with default Redis connection:
        >>> from common.cache import Cache
        >>> cache = Cache(prefix="app")
        >>> cache.set("user:123", {"name": "John", "role": "admin"})
        >>> user = cache.get("user:123")
        >>> print(user)
        {'name': 'John', 'role': 'admin'}

    Using in-memory cache fallback:
        >>> cache = Cache(prefix="app", use_redis=False)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Pattern, Set

import redis

from config import settings


class BaseCache:
  """Base cache interface."""

  def get(self, key: str) -> Optional[Any]:
    """Get value from cache if not expired."""
    raise NotImplementedError

  def set(self, key: str, value: Any, ttl_seconds: int = 300):
    """Set value in cache with TTL."""
    raise NotImplementedError

  def delete(self, key: str):
    """Delete value from cache."""
    raise NotImplementedError


class MemoryCache(BaseCache):
  """Simple in-memory cache with TTL.

  Examples:
      >>> cache = MemoryCache()
      >>> cache.set("session:123", {"user_id": 456}, ttl_seconds=600)
      >>> cache.get("session:123")
      {'user_id': 456}
  """

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


class ValidationRule:
  """Validation rule for cache entries.

  Examples:
      >>> # Create a rule that validates user objects have required fields
      >>> def validate_user(user):
      ...     return isinstance(user, dict) and "id" in user and "name" in user
      >>>
      >>> rule = ValidationRule("user:.*", validate_user)
      >>> rule.applies_to("user:123")
      True
      >>> rule.applies_to("product:456")
      False
      >>> rule.validate({"id": 123, "name": "John"})
      True
      >>> rule.validate({"name": "John"})  # Missing 'id'
      False
  """

  def __init__(self, key_pattern: str, validator: Callable[[Any], bool]):
    """Initialize validation rule.

    Args:
        key_pattern: Regex pattern for keys this rule applies to
        validator: Function that takes a value and returns True if valid
    """
    self.key_pattern = re.compile(key_pattern)
    self.validator = validator

  def applies_to(self, key: str) -> bool:
    """Check if this rule applies to the given key."""
    return bool(self.key_pattern.match(key))

  def validate(self, value: Any) -> bool:
    """Validate the value."""
    return self.validator(value)


class InvalidationRule:
  """Rule for invalidating related cache entries.

  Examples:
      >>> # Create a rule that invalidates user lists when any user changes
      >>> rule = InvalidationRule("user:[0-9]+", "user:list.*")
      >>> rule.is_trigger("user:123")
      True
      >>> rule.is_trigger("product:456")
      False
  """

  def __init__(self, trigger_key_pattern: str, invalidate_key_pattern: str):
    """Initialize invalidation rule.

    Args:
        trigger_key_pattern: Regex pattern for keys that trigger invalidation
        invalidate_key_pattern: Regex pattern for keys to invalidate
    """
    self.trigger_pattern = re.compile(trigger_key_pattern)
    self.invalidate_pattern = re.compile(invalidate_key_pattern)

  def is_trigger(self, key: str) -> bool:
    """Check if this key triggers invalidation."""
    return bool(self.trigger_pattern.match(key))

  def get_invalidation_pattern(self) -> Pattern:
    """Get the pattern for keys to invalidate."""
    return self.invalidate_pattern


class RedisCache(BaseCache):
  """Enhanced Redis-based cache with validation and invalidation rules.

  Examples:
      Basic usage:
          >>> cache = RedisCache(prefix="myapp")
          >>> cache.set("counter", 42)
          >>> cache.get("counter")
          42

      With validation:
          >>> # Add validation rule for positive numbers
          >>> cache.add_validation_rule("counter:.*", lambda x: isinstance(x, int) and x > 0)
          >>> cache.set("counter:visitors", 10)  # Passes validation
          True
          >>> cache.set("counter:visitors", -5)  # Fails validation
          False

      With invalidation rules:
          >>> # When a product changes, invalidate category cache
          >>> cache.add_invalidation_rule("product:[0-9]+", "category:.*")
          >>> cache.set("category:electronics", {"name": "Electronics", "count": 15})
          >>> cache.set("product:123", {"name": "Laptop", "price": 999})  # Invalidates category cache
  """

  def __init__(self, prefix: str = "", **redis_kwargs):
    """Initialize Redis connection.

    Args:
        prefix: Prefix for all cache keys to avoid collisions
        redis_kwargs: Arguments to pass to redis.Redis constructor

    Examples:
        >>> # Default connection to localhost
        >>> cache = RedisCache(prefix="app")
        >>>
        >>> # Connect to a specific Redis server
        >>> cache = RedisCache(
        ...     prefix="app",
        ...     host="redis.example.com",
        ...     port=6380,
        ...     password="secret"
        ... )
    """
    self.prefix = prefix
    self.validation_rules: List[ValidationRule] = []
    self.invalidation_rules: List[InvalidationRule] = []
    self.dependencies: Dict[str, Set[str]] = {}  # key -> set of dependent keys

    # Use environment variables or fallback to defaults
    redis_kwargs.setdefault("host", getattr(settings, "redis_host", "localhost"))
    redis_kwargs.setdefault("port", getattr(settings, "redis_port", 6379))
    redis_kwargs.setdefault("db", getattr(settings, "redis_db", 0))
    redis_kwargs.setdefault("password", getattr(settings, "redis_password", None))
    redis_kwargs.setdefault("decode_responses", True)  # Auto-decode binary responses

    self.redis = redis.Redis(**redis_kwargs)
    # Test connection on init
    try:
      self.redis.ping()
    except redis.ConnectionError:
      # Fallback to memory cache if Redis is not available
      self.redis = None
      self._fallback = MemoryCache()
      print("Warning: Redis connection failed. Using in-memory cache as fallback.")

  def _build_key(self, key: str) -> str:
    """Build cache key with prefix."""
    return f"{self.prefix}:{key}" if self.prefix else key

  def add_validation_rule(self, key_pattern: str, validator: Callable[[Any], bool]):
    """Add a validation rule.

    Args:
        key_pattern: Regex pattern for keys this rule applies to
        validator: Function that takes a value and returns True if valid

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>>
        >>> # Add validation for user objects
        >>> def validate_user(user):
        ...     return isinstance(user, dict) and "email" in user
        >>>
        >>> cache.add_validation_rule("user:.*", validate_user)
        >>>
        >>> # Valid user data passes validation
        >>> cache.set("user:123", {"email": "user@example.com", "name": "Test User"})
        True
        >>>
        >>> # Invalid user data fails validation
        >>> cache.set("user:456", {"name": "No Email User"})
        False
    """
    self.validation_rules.append(ValidationRule(key_pattern, validator))

  def add_invalidation_rule(self, trigger_key_pattern: str, invalidate_key_pattern: str):
    """Add an invalidation rule.

    Args:
        trigger_key_pattern: Regex pattern for keys that trigger invalidation
        invalidate_key_pattern: Regex pattern for keys to invalidate

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>>
        >>> # When any user is updated, invalidate the user list
        >>> cache.add_invalidation_rule("user:[0-9]+", "users:list")
        >>>
        >>> # Set up initial data
        >>> cache.set("users:list", ["user1", "user2"])
        >>>
        >>> # When a user is updated, the list is automatically invalidated
        >>> cache.set("user:1", {"name": "Updated User"})
        >>> cache.get("users:list")  # Returns None as it was invalidated
    """
    self.invalidation_rules.append(InvalidationRule(trigger_key_pattern, invalidate_key_pattern))

  def add_dependency(self, key: str, dependent_key: str):
    """Add a dependency between keys.

    When key is modified or deleted, dependent_key will be invalidated.

    Args:
        key: The key that triggers invalidation when modified
        dependent_key: The key to invalidate

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>>
        >>> # Set up dependency: product depends on inventory
        >>> cache.add_dependency("inventory:1001", "product:electronics:1001")
        >>>
        >>> # Set initial data
        >>> cache.set("product:electronics:1001", {"name": "Laptop", "in_stock": True})
        >>> cache.set("inventory:1001", {"quantity": 5})
        >>>
        >>> # Update inventory (triggers invalidation of the product)
        >>> cache.set("inventory:1001", {"quantity": 0})
        >>>
        >>> # Product data is invalidated
        >>> cache.get("product:electronics:1001")  # Returns None
    """
    if key not in self.dependencies:
      self.dependencies[key] = set()
    self.dependencies[key].add(dependent_key)

  def _validate_value(self, key: str, value: Any) -> bool:
    """Validate value against applicable rules."""
    for rule in self.validation_rules:
      if rule.applies_to(key) and not rule.validate(value):
        return False
    return True

  def _invalidate_dependent_keys(self, key: str):
    """Invalidate keys based on dependencies and rules."""
    # Check direct dependencies
    if key in self.dependencies:
      for dependent_key in self.dependencies[key]:
        self.delete(dependent_key)

    # Check invalidation rules
    for rule in self.invalidation_rules:
      if rule.is_trigger(key):
        # Find all matching keys for invalidation
        if self.redis:
          pattern = self._build_key("*")
          all_keys = self.redis.keys(pattern)
          invalidate_pattern = rule.get_invalidation_pattern()

          for full_key in all_keys:
            # Remove prefix for pattern matching
            clean_key = full_key
            if self.prefix and clean_key.startswith(f"{self.prefix}:"):
              clean_key = clean_key[len(f"{self.prefix}:") :]

            if invalidate_pattern.match(clean_key):
              self.delete(clean_key)

  def get(self, key: str) -> Optional[Any]:
    """Get value from Redis cache.

    Args:
        key: Cache key to retrieve

    Returns:
        The cached value or None if not found

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>> cache.set("settings", {"theme": "dark"})
        >>> cache.get("settings")
        {'theme': 'dark'}
        >>> cache.get("nonexistent")
        None
    """
    if self.redis is None:
      return self._fallback.get(key)

    redis_key = self._build_key(key)
    value = self.redis.get(redis_key)

    if value is None:
      return None

    # If value is stored in Redis JSON format, parse it
    try:
      return json.loads(value)
    except (json.JSONDecodeError, TypeError):
      # Not JSON, return as is
      return value

  def set(self, key: str, value: Any, ttl_seconds: int = 300, validate: bool = True) -> bool:
    """Set value in Redis cache with TTL.

    Args:
        key: Cache key
        value: Value to store
        ttl_seconds: Time to live in seconds
        validate: Whether to validate against rules

    Returns:
        bool: True if set was successful, False if validation failed

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>>
        >>> # Basic usage
        >>> cache.set("session:abc", {"user_id": 123})
        True
        >>>
        >>> # With custom TTL (10 minutes)
        >>> cache.set("temp:data", [1, 2, 3], ttl_seconds=600)
        True
        >>>
        >>> # With validation rule
        >>> cache.add_validation_rule("user:.*", lambda x: "name" in x)
        >>> cache.set("user:123", {"name": "John"})  # Valid
        True
        >>> cache.set("user:456", {})  # Invalid
        False
    """
    if validate and not self._validate_value(key, value):
      return False

    if self.redis is None:
      self._fallback.set(key, value, ttl_seconds)
      return True

    redis_key = self._build_key(key)

    # Serialize complex objects to JSON
    if not isinstance(value, (str, int, float, bool)) and value is not None:
      value = json.dumps(value)

    self.redis.set(redis_key, value, ex=ttl_seconds)

    # Invalidate related keys
    self._invalidate_dependent_keys(key)
    return True

  def delete(self, key: str):
    """Delete value from Redis cache.

    Args:
        key: Cache key to delete

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>> cache.set("temp", "value")
        >>> cache.get("temp")
        'value'
        >>> cache.delete("temp")
        >>> cache.get("temp")
        None
    """
    if self.redis is None:
      self._fallback.delete(key)
      return

    redis_key = self._build_key(key)
    self.redis.delete(redis_key)

    # Invalidate dependent keys
    self._invalidate_dependent_keys(key)

  def invalidate_pattern(self, pattern: str):
    """Invalidate all keys matching a pattern.

    Args:
        pattern: Redis key pattern (e.g., "user:*")

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>> cache.set("temp:1", "value1")
        >>> cache.set("temp:2", "value2")
        >>> cache.set("other:1", "value3")
        >>>
        >>> # Delete all temp keys
        >>> cache.invalidate_pattern("temp:*")
        >>>
        >>> cache.get("temp:1")  # None
        >>> cache.get("temp:2")  # None
        >>> cache.get("other:1")  # Still exists
        'value3'
    """
    if self.redis is None:
      # Limited implementation for fallback mode
      return

    full_pattern = self._build_key(pattern)
    keys = self.redis.keys(full_pattern)

    if keys:
      self.redis.delete(*keys)

  def get_all_keys(self, pattern: str = "*") -> List[str]:
    """Get all keys matching a pattern.

    Args:
        pattern: Redis key pattern

    Returns:
        List of matching keys (with prefix removed)

    Examples:
        >>> cache = RedisCache(prefix="app")
        >>> cache.set("user:1", {"name": "John"})
        >>> cache.set("user:2", {"name": "Jane"})
        >>> cache.set("product:1", {"name": "Laptop"})
        >>>
        >>> # Get all user keys
        >>> cache.get_all_keys("user:*")
        ['user:1', 'user:2']
        >>>
        >>> # Get all keys
        >>> all_keys = cache.get_all_keys()
        >>> sorted(all_keys)  # Output order may vary
        ['product:1', 'user:1', 'user:2']
    """
    if self.redis is None:
      return []

    full_pattern = self._build_key(pattern)
    keys = self.redis.keys(full_pattern)

    # Remove prefix from keys
    if self.prefix:
      prefix_len = len(self.prefix) + 1  # +1 for the colon
      return [k[prefix_len:] if k.startswith(f"{self.prefix}:") else k for k in keys]
    return keys


# For backwards compatibility, expose the same interface
class Cache(RedisCache):
  """Cache implementation that adapts to the environment.

  Uses Redis if available, falls back to in-memory cache.

  Examples:
      # Basic usage with Redis (production environment)
      >>> cache = Cache(prefix="myapp")
      >>> cache.set("config", {"debug": False})
      >>> cache.get("config")
      {'debug': False}

      # Force in-memory cache (development environment)
      >>> memory_cache = Cache(prefix="myapp", use_redis=False)
      >>> memory_cache.set("temp", "value")
      >>> memory_cache.get("temp")
      'value'

      # Advanced usage with validation and invalidation
      >>> cache = Cache(prefix="myapp")
      >>>
      >>> # Add validation rule
      >>> cache.add_validation_rule(
      ...     "user:.*",
      ...     lambda user: isinstance(user, dict) and "email" in user
      ... )
      >>>
      >>> # Add invalidation rule
      >>> cache.add_invalidation_rule("user:[0-9]+", "users:list")
      >>>
      >>> # Add direct dependency
      >>> cache.add_dependency("inventory:101", "product:101")
  """

  def __init__(self, prefix: str = "", use_redis: bool = True, **redis_kwargs):
    """Initialize the cache.

    Args:
        prefix: Prefix for all cache keys
        use_redis: Whether to try using Redis or force in-memory
        redis_kwargs: Arguments to pass to redis.Redis constructor

    Examples:
        >>> # Default usage - tries Redis first, falls back to memory cache
        >>> cache = Cache(prefix="myapp")
        >>>
        >>> # Force in-memory cache (useful for testing)
        >>> memory_cache = Cache(prefix="myapp", use_redis=False)
        >>>
        >>> # Connect to a specific Redis server
        >>> remote_cache = Cache(
        ...     prefix="myapp",
        ...     host="redis.example.com",
        ...     port=6380,
        ...     password="secret",
        ...     db=1
        ... )
    """
    if use_redis:
      super().__init__(prefix=prefix, **redis_kwargs)
    else:
      self.redis = None
      self._fallback = MemoryCache()
      self.validation_rules = []
      self.invalidation_rules = []
      self.dependencies = {}


# For backwards compatibility
deps_cache = Cache()
