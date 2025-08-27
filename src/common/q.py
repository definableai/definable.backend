from functools import wraps
from typing import Optional

from celery import Celery

from config.settings import settings

from .logger import log as logger

# Create the Celery instance directly
logger.info(
  "Initializing Celery application", broker_url=settings.celery_broker_url[:50] + "...", backend=settings.celery_result_backend[:50] + "..."
)

celery_app = Celery("app", broker=settings.celery_broker_url, backend=settings.celery_result_backend)

# Configure Celery
logger.info("Configuring Celery application", task_serializer="json", timezone="UTC", task_time_limit=3600)

celery_app.conf.update(
  task_serializer="json",
  accept_content=["json"],
  result_serializer="json",
  timezone="UTC",
  enable_utc=True,
  task_track_started=True,
  task_time_limit=3600,
  worker_pool="threads" if settings.environment == "development" else "prefork",
  worker_concurrency=2,
  worker_prefetch_multiplier=1,
)

logger.info("Celery application configured successfully")


def task(*args, **kwargs):
  """Task decorator - only supports sync functions now."""

  def decorator(func):
    task_name = func.__name__
    logger.info("Registering Celery task", task_name=task_name, args=args, kwargs=kwargs)

    @celery_app.task(*args, **kwargs)
    @wraps(func)
    def wrapper(*task_args, **task_kwargs):
      logger.info("Starting task execution", task_name=task_name, args_count=len(task_args), kwargs_keys=list(task_kwargs.keys()))

      try:
        # All tasks must be synchronous now
        result = func(*task_args, **task_kwargs)
        logger.info("Task completed successfully", task_name=task_name)
        return result
      except Exception as e:
        logger.error("Task execution failed", task_name=task_name, error=str(e), error_type=type(e).__name__)
        raise

    logger.info("Task registered successfully", task_name=task_name)
    return wrapper

  return decorator


def submit_task(task_name: str, *args, countdown: Optional[int] = None, **kwargs):
  """Submit a task."""
  logger.info("Submitting task to queue", task_name=task_name, args_count=len(args), kwargs_keys=list(kwargs.keys()), countdown=countdown)

  try:
    result = celery_app.send_task(task_name, args=args, kwargs=kwargs, countdown=countdown)
    logger.info("Task submitted successfully", task_name=task_name, task_id=result.id if hasattr(result, "id") else "unknown")
    return result
  except Exception as e:
    logger.error("Failed to submit task", task_name=task_name, error=str(e), error_type=type(e).__name__)
    raise
