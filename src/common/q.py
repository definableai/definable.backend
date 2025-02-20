from functools import wraps
from typing import Optional

from celery import Celery

from config.settings import settings

# Create the Celery instance directly
celery_app = Celery("app", broker=settings.celery_broker_url, backend=settings.celery_result_backend)
# Configure Celery
celery_app.conf.update(
  task_serializer="json",
  accept_content=["json"],
  result_serializer="json",
  timezone="UTC",
  enable_utc=True,
  task_track_started=True,
  task_time_limit=3600,
)


def task(*args, **kwargs):
  """Task decorator."""

  def decorator(func):
    @celery_app.task(*args, **kwargs)
    @wraps(func)
    async def wrapper(*args, **kwargs):
      return await func(*args, **kwargs)

    return wrapper

  return decorator


def submit_task(task_name: str, *args, countdown: Optional[int] = None, **kwargs):
  """Submit a task."""
  return celery_app.send_task(task_name, args=args, kwargs=kwargs, countdown=countdown)
