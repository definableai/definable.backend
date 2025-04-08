"""
This module contains the main application setup.
"""

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.__base.manager import Manager


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
  args = parser.parse_args()

  if args.dev:
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
  else:
    import subprocess

    try:
      subprocess.run([
        "gunicorn",
        "src.app:app",
        "--workers",
        "4",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--bind",
        "0.0.0.0:8000",
        "--pythonpath",
        "src",
      ])
    except KeyboardInterrupt:
      print("\nServer has been shut down gracefully.")
    except Exception as e:
      print(f"An error occurred: {e}")


if __name__ == "__main__":
  main()
