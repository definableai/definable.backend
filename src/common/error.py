"""Common error handling module."""

from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from .logger import log


class BaseCustomError(Exception):
  """Base class for all custom errors."""

  def __init__(self, message: str, error_code: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
    self.message = message
    self.error_code = error_code
    self.status_code = status_code
    self.details = details or {}
    log.error(message, error_code=error_code, status_code=status_code, details=self.details)
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


class BillingError(BaseCustomError):
  """Base class for billing-related errors."""

  def __init__(self, message: str, error_code: str = "BILLING_ERROR", status_code: int = 400, details: Optional[Dict[str, Any]] = None):
    super().__init__(message=message, error_code=error_code, status_code=status_code, details=details)


class InsufficientCreditsError(BillingError):
  """Exception raised when user doesn't have enough credits."""

  def __init__(self, message: str = "Insufficient credits", details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="INSUFFICIENT_CREDITS",
      status_code=402,  # Payment Required
      details=details,
    )


class ChargeNotFoundError(BillingError):
  """Exception raised when a charge is not found."""

  def __init__(self, message: str = "Charge not found", details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="CHARGE_NOT_FOUND",
      status_code=404,  # Not Found
      details=details,
    )


class InvalidTransactionError(BillingError):
  """Exception raised when a transaction is invalid."""

  def __init__(self, message: str = "Invalid transaction", details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="INVALID_TRANSACTION",
      status_code=400,  # Bad Request
      details=details,
    )


class ChargeError(BillingError):
  """Base exception for charge-related errors."""

  def __init__(
    self, message: str = "Charge operation failed", error_code: str = "CHARGE_ERROR", status_code: int = 400, details: Optional[Dict[str, Any]] = None
  ):
    super().__init__(
      message=message,
      error_code=error_code,
      status_code=status_code,
      details=details,
    )


class DuplicateChargeError(ChargeError):
  """Exception for duplicate charge attempts."""

  def __init__(self, message: str = "Duplicate charge detected", details: Optional[Dict[str, Any]] = None):
    super().__init__(
      message=message,
      error_code="DUPLICATE_CHARGE_ERROR",
      status_code=409,  # Conflict is appropriate for duplicates
      details=details,
    )


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
  elif isinstance(error, ChargeError):
    return error
  elif isinstance(error, DuplicateChargeError):
    return error
  elif isinstance(error, InsufficientCreditsError):
    return error
  elif isinstance(error, ChargeNotFoundError):
    return error
  elif isinstance(error, InvalidTransactionError):
    return error
  else:
    return BaseCustomError(message="Internal server error", error_code="INTERNAL_SERVER_ERROR")
