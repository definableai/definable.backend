"""
This module contains the main application setup.
"""

import argparse
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from services.__base.manager import Manager

sentry_sdk.init(
  dsn=settings.sentry_dsn,
  # Add data like request headers and IP for users,
  # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
  send_default_pii=True,
  # Enable sending logs to Sentry
  enable_logs=True,
  # Set traces_sample_rate to 1.0 to capture 100%
  # of transactions for tracing.
  traces_sample_rate=1.0,
  # Set profile_session_sample_rate to 1.0 to profile 100%
  # of profile sessions.
  profile_session_sample_rate=1.0,
  # Set profile_lifecycle to "trace" to automatically
  # run the profiler on when there is an active transaction
  profile_lifecycle="trace",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Startup
  yield
  # Shutdown


app = FastAPI(lifespan=lifespan)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allow specific origins
  allow_credentials=True,
  allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
  allow_headers=["*"],  # Allow all headers
)


manager = Manager(app)
manager.register_middlewares()
manager.register_services()


def main():
  parser = argparse.ArgumentParser(description="Run the FastAPI application.")
  parser.add_argument("--dev", action="store_true", help="Run in development mode with Uvicorn.")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
  parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
  parser.add_argument("--workers", type=int, default=4, help="Number of workers to run the server on.")
  parser.add_argument("--timeout", type=int, default=600, help="Worker timeout in seconds.")
  args = parser.parse_args()

  if args.dev:
    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=True)
  else:
    import subprocess

    try:
      subprocess.run([
        "gunicorn",
        "src.app:app",
        "--log-level",
        "info",
        "--workers",
        f"{args.workers}",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--bind",
        f"{args.host}:{args.port}",
        "--timeout",
        f"{args.timeout}",
        "--keep-alive",
        "10",
        "--graceful-timeout",
        "60",
        "--pythonpath",
        "src",
      ])
    except KeyboardInterrupt:
      print("\nServer has been shut down gracefully.")
    except Exception as e:
      print(f"An error occurred: {e}")


if __name__ == "__main__":
  main()
