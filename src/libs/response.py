import time
from typing import Any, Dict, Generic, List, Optional, TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T")
ListT = TypeVar("ListT", bound=List)


class LibResponse(BaseModel, Generic[T]):
  """
  A generic response model for all library functions.
  Provides a consistent interface for success and error handling.
  """

  success: bool
  data: Optional[T] = None
  errors: Optional[List[Dict[str, Any]]] = None
  meta: Optional[Dict[str, Any]] = None

  @classmethod
  def success_response(cls, data: T, meta: Optional[Dict[str, Any]] = None) -> "LibResponse[T]":
    """Basic success response with data"""
    return cls(success=True, data=data, meta=meta)

  @classmethod
  def error_response(cls, errors: List[Dict[str, Any]]) -> "LibResponse[None]":
    """Error response with error details"""
    return cast(LibResponse[None], cls(success=False, errors=errors, data=None))

  @classmethod
  def paginated_response(cls, data: List[T], total: int, page: int, page_size: int) -> "LibResponse[List[T]]":
    """Response with pagination metadata"""
    response = cls(
      success=True,
      data=data,  # type: ignore
      meta={
        "pagination": {
          "total": total,
          "page": page,
          "page_size": page_size,
          "total_pages": (total + page_size - 1) // page_size,
          "has_next": page * page_size < total,
          "has_previous": page > 1,
        }
      },
    )
    return cast("LibResponse[List[T]]", response)

  @classmethod
  def timed_response(cls, data: T, execution_time_ms: float) -> "LibResponse[T]":
    """Response with performance metrics"""
    return cls(success=True, data=data, meta={"performance": {"execution_time_ms": execution_time_ms}})

  @classmethod
  def warning_response(cls, data: T, warnings: List[Dict[str, Any]]) -> "LibResponse[T]":
    """Response with non-fatal warnings"""
    return cls(success=True, data=data, meta={"warnings": warnings})

  @classmethod
  def partial_success(cls, data: T, succeeded: List[Any], failed: List[Dict[str, Any]]) -> "LibResponse[T]":
    """Response for operations where some items succeeded and others failed"""
    return cls(
      success=True,  # Operation completed but with some failures
      data=data,
      meta={"partial_results": {"succeeded_count": len(succeeded), "failed_count": len(failed), "failed_items": failed}},
    )

  @classmethod
  def rate_limited_response(cls, data: T, remaining_requests: int, reset_at: int) -> "LibResponse[T]":
    """Response with rate limiting information"""
    return cls(success=True, data=data, meta={"rate_limit": {"remaining": remaining_requests, "reset_at": reset_at}})

  @classmethod
  def cached_response(cls, data: T, cache_hit: bool, ttl: int) -> "LibResponse[T]":
    """Response with cache information"""
    return cls(success=True, data=data, meta={"cache": {"hit": cache_hit, "ttl": ttl, "expired_at": int(time.time()) + ttl}})

  @classmethod
  def traced_response(cls, data: T, request_id: str, trace_id: Optional[str] = None) -> "LibResponse[T]":
    """Response with tracing information for debugging"""
    return cls(success=True, data=data, meta={"trace": {"request_id": request_id, "trace_id": trace_id}})

  @classmethod
  def auth_response(cls, data: T, token_expires_in: int) -> "LibResponse[T]":
    """Response with authentication state information"""
    return cls(
      success=True,
      data=data,
      meta={
        "auth": {
          "token_expires_in": token_expires_in,
          "requires_refresh": token_expires_in < 300,  # less than 5 minutes
        }
      },
    )

  @classmethod
  def localized_response(cls, data: T, locale: str, available_translations: List[str]) -> "LibResponse[T]":
    """Response with localization information"""
    return cls(success=True, data=data, meta={"localization": {"current_locale": locale, "available": available_translations}})

  @classmethod
  def with_metadata(cls, data: T, **kwargs) -> "LibResponse[T]":
    """Flexible method to add any custom metadata"""
    return cls(success=True, data=data, meta=kwargs)

  def add_meta(self, key: str, value: Any) -> "LibResponse[T]":
    """Add or update metadata after response creation"""
    if self.meta is None:
      self.meta = {}
    self.meta[key] = value
    return self

  def is_successful(self) -> bool:
    """Check if the operation was successful"""
    return self.success

  def has_data(self) -> bool:
    """Check if the response contains data"""
    return self.data is not None

  def has_errors(self) -> bool:
    """Check if the response contains errors"""
    return not self.success and self.errors is not None and len(self.errors) > 0

  def first_error(self) -> Optional[Dict[str, Any]]:
    """Get the first error if any"""
    if self.errors and len(self.errors) > 0:
      return self.errors[0]
    return None

  def get_meta_value(self, key: str, default: Any = None) -> Any:
    """Safely get a value from metadata"""
    if not self.meta:
      return default
    return self.meta.get(key, default)
