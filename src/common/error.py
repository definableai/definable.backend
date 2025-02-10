"""Common error handling module."""

from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError


class BaseCustomError(Exception):
  """Base class for all custom errors."""

  def __init__(self, message: str, error_code: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
    self.message = message
    self.error_code = error_code
    self.status_code = status_code
    self.details = details or {}
    super().__init__(self.message)


class DatabaseError(BaseCustomError):
  """Database related errors."""

  def __init__(self, message: str, original_error: Optional[SQLAlchemyError] = None, details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="DATABASE_ERROR",
      status_code=500,
      details={"original_error": str(original_error) if original_error else None, **(details or {})},
    )


class ValidationCustomError(BaseCustomError):
  """Validation related errors."""

  def __init__(self, message: str, original_error: Optional[ValidationError] = None, details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="VALIDATION_ERROR",
      status_code=422,
      details={"validation_errors": original_error.errors() if original_error else None, **(details or {})},
    )


class NotFoundError(BaseCustomError):
  """Resource not found error."""

  def __init__(self, message: str, resource_type: str, resource_id: Any, details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="NOT_FOUND",
      status_code=404,
      details={"resource_type": resource_type, "resource_id": str(resource_id), **(details or {})},
    )


class AuthenticationError(BaseCustomError):
  """Authentication related errors."""

  def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
    super().__init__(message=message, error_code="AUTHENTICATION_ERROR", status_code=401, details=details)


def handle_exception(error: Exception) -> BaseCustomError:
  """Convert various exceptions to BaseCustomError."""
  if isinstance(error, BaseCustomError):
    return error
  elif isinstance(error, SQLAlchemyError):
    return DatabaseError(message="Database operation failed", original_error=error)
  elif isinstance(error, ValidationError):
    return ValidationCustomError(message="Validation failed", original_error=error)
  elif isinstance(error, HTTPException):
    return BaseCustomError(message=error.detail, error_code="HTTP_ERROR", status_code=error.status_code)
  else:
    return BaseCustomError(message="Internal server error", error_code="INTERNAL_SERVER_ERROR")
