from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Awaitable, Callable
from datetime import datetime
import time
import uuid

from common.logger import log


class Middleware(BaseHTTPMiddleware):
  """Middleware to track request duration and correlation IDs."""

  def __init__(self, app: FastAPI):
    super().__init__(app)
    self.logger = log

  def format_timestamp(self, ts: float) -> str:
    """Format timestamp with microseconds."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

  async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    # Get or generate correlation ID
    correlation_id = request.headers.get("x-log-id") or str(uuid.uuid4())

    # Start timing and format timestamp
    start_time = time.time()
    formatted_start_time = self.format_timestamp(start_time)

    # Add correlation ID to request state
    request.state.correlation_id = correlation_id

    # Create bound logger with correlation ID and formatted start time
    req_logger = self.logger.bind(correlation_id=correlation_id, request_start_time=formatted_start_time)

    # Log request start
    req_logger.info(
      "Incoming request",
      timestamp=formatted_start_time,
    )

    try:
      response = await call_next(request)

      # Calculate duration and format end timestamp
      end_time = time.time()
      duration = end_time - start_time
      formatted_end_time = self.format_timestamp(end_time)

      # Log request completion
      req_logger.info(
        "Request completed",
        duration=f"{duration:.3f}s",
        duration_ms=int(duration * 1000),
        status_code=response.status_code,
        timestamp=formatted_end_time,
      )

      # Add tracking headers
      response.headers.update({
        "X-Request-ID": correlation_id,
        "X-Request-Duration": f"{duration:.3f}s",
        "X-Request-Duration-MS": str(int(duration * 1000)),
      })

      return response

    except Exception as e:
      # Calculate duration and format end timestamp for failed requests
      end_time = time.time()
      duration = end_time - start_time
      formatted_end_time = self.format_timestamp(end_time)

      req_logger.exception("Request failed", exc_info=e, duration=f"{duration:.3f}s", duration_ms=int(duration * 1000), timestamp=formatted_end_time)
      raise
