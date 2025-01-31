"""
This module contains the rate limiting middleware.
"""

import time
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


class Middleware(BaseHTTPMiddleware):
  """Rate limiting middleware for FastAPI applications.

  This middleware implements a sliding window rate limiter that tracks requests
  by client IP address. It will reject requests that exceed the configured
  rate limit with a 429 status code.

  Args:
      app (FastAPI): The FastAPI application instance
      max_requests (int, optional): Maximum number of requests allowed per window. Defaults to 10.
      window_size (int, optional): Size of the sliding window in seconds. Defaults to 60.

  Attributes:
      max_requests (int): Maximum number of requests allowed per window
      window_size (int): Size of the sliding window in seconds
      ip_requests (dict[str, list[float]]): Dictionary storing request timestamps by IP
  """

  def __init__(self, app: FastAPI, max_requests: int = 100, window_size: int = 60):
    super().__init__(app)
    self.max_requests = max_requests
    self.window_size = window_size
    # Stores request timestamps by IP, e.g. { "127.0.0.1": [timestamp1, timestamp2, ...] }
    self.ip_requests: dict[str, list[float]] = {}

  async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Process incoming request and apply rate limiting.

    Tracks request timestamps for each client IP in a sliding window. If a client
    exceeds the configured rate limit, their request is rejected with a 429 status.

    Args:
        request (Request): The incoming HTTP request
        call_next (Callable[[Request], Awaitable[Response]]): Function to process the request

    Returns:
        Response: Either the normal response or a 429 if rate limited

    Raises:
        HTTPException: If the client IP cannot be determined
    """
    client_ip = request.client.host if request.client else None
    if client_ip is None:
      raise HTTPException(status_code=400, detail="Invalid client IP")

    current_time = time.time()
    window_start = current_time - self.window_size

    # Get the list of request timestamps for this IP (or create an empty one).
    request_times = self.ip_requests.get(client_ip, [])

    # Filter out timestamps that are older than the current window.
    request_times = [t for t in request_times if t > window_start]

    # Check if the client has exceeded the rate limit.
    if len(request_times) >= self.max_requests:
      return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    # Otherwise, add this request timestamp and store it back.
    request_times.append(current_time)
    self.ip_requests[client_ip] = request_times

    # Proceed with the normal request handling
    response = await call_next(request)
    return response
