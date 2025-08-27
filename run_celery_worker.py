#!/usr/bin/env python3
"""
Script to run Celery workers for background task processing.

Usage:
    python run_celery_worker.py [options]

Examples:
    # Run with default settings
    python run_celery_worker.py

    # Run with custom concurrency
    python run_celery_worker.py --concurrency 4

    # Run with specific log level
    python run_celery_worker.py --loglevel info

    # Run with specific queues
    python run_celery_worker.py --queues kb_processing,kb_indexing
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import after adding src to path
from common.q import celery_app

# Import tasks to register them with Celery
import tasks.kb_tasks  # noqa: F401


def main():
  parser = argparse.ArgumentParser(description="Run Celery worker for KB service tasks")

  parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent worker processes (default: 2)")

  parser.add_argument("--loglevel", choices=["debug", "info", "warning", "error", "critical"], default="info", help="Logging level (default: info)")

  parser.add_argument("--queues", type=str, default="celery", help="Comma-separated list of queues to consume from (default: celery)")

  parser.add_argument("--hostname", type=str, default=None, help="Set custom hostname for the worker")

  parser.add_argument(
    "--max-tasks-per-child", type=int, default=1000, help="Maximum number of tasks a worker process can execute before being recycled (default: 1000)"
  )

  parser.add_argument(
    "--prefetch-multiplier",
    type=int,
    default=1,
    help="How many messages to prefetch at a time multiplied by the number of concurrent processes (default: 1)",
  )

  parser.add_argument(
    "--pool",
    choices=["prefork", "eventlet", "gevent", "threads"],
    default="threads",
    help="Pool implementation (default: threads for Windows compatibility)",  # noqa: E501
  )

  parser.add_argument("--autoscale", type=str, default=None, help="Enable autoscaling by providing max,min number of processes (e.g., 10,3)")

  args = parser.parse_args()

  print("=" * 60)
  print("🚀 Starting Celery Worker for KB Service")
  print("=" * 60)
  print(f"📊 Concurrency: {args.concurrency}")
  print(f"📝 Log Level: {args.loglevel}")
  print(f"📋 Queues: {args.queues}")
  print(f"🔄 Pool: {args.pool}")
  print(f"🔄 Max tasks per child: {args.max_tasks_per_child}")
  print(f"📥 Prefetch multiplier: {args.prefetch_multiplier}")

  if args.autoscale:
    print(f"⚡ Autoscale: {args.autoscale}")

  print("=" * 60)

  # Build worker arguments
  worker_args = [
    "--loglevel",
    args.loglevel,
    "--queues",
    args.queues,
    "--pool",
    args.pool,
    "--max-tasks-per-child",
    str(args.max_tasks_per_child),
    "--prefetch-multiplier",
    str(args.prefetch_multiplier),
  ]

  if args.hostname:
    worker_args.extend(["--hostname", args.hostname])
  else:
    # Set a default hostname
    worker_args.extend(["--hostname", "kb-worker@%h"])

  if args.autoscale:
    worker_args.extend(["--autoscale", args.autoscale])
  else:
    worker_args.extend(["--concurrency", str(args.concurrency)])

  try:
    # Start the worker
    celery_app.worker_main(["worker"] + worker_args)
  except KeyboardInterrupt:
    print("\n🛑 Stopping Celery worker...")
    sys.exit(0)
  except Exception as e:
    print(f"❌ Error starting worker: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
