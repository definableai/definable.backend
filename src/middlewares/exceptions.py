import sys
import traceback

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config.settings import settings
from services.__base.acquire import Acquire


class Middleware(BaseHTTPMiddleware):
  """Middleware to handle service exceptions and WebSocket broadcasting."""

  def __init__(self, app, acquire: Acquire):
    super().__init__(app)
    self.ws_manager = acquire.ws_manager
    self.logger = acquire.logger

  async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
    try:
      # Call the route handler
      response = await call_next(request)
      return response

    except Exception:
      # Get the full traceback
      exc_type, exc_value, exc_traceback = sys.exc_info()

      # Get the stack trace
      stack_trace = []
      for frame in traceback.extract_tb(exc_traceback):
        stack_trace.append({"filename": frame.filename, "line": frame.line, "lineno": frame.lineno, "name": frame.name})

      error_info = {
        "error_type": exc_type.__name__,  # type: ignore
        "error_message": str(exc_value),
        "traceback": stack_trace,
      }

      if settings.environment == "dev":
        return JSONResponse(status_code=500, content=error_info)
      return JSONResponse(status_code=500, content="Internal server error")

  async def _get_iterator(self, content: bytes):
    yield content
