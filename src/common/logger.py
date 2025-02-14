import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class LogConfig:
  """Logging configuration."""

  def __init__(self, log_level: str = "INFO", json_logs: bool = True, log_file: Optional[str] = None):
    self.log_level = log_level
    self.json_logs = json_logs
    self.log_file = log_file

  def setup_logger(self):
    """Configure logger with the specified settings."""
    # Remove default handler
    logger.remove()

    # Custom log format for JSON logging
    def json_formatter(record):
      log_data = {
        "timestamp": record["time"].timestamp(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "extra": record["extra"],
      }

      if record["exception"]:
        exc = record["exception"]
        log_data["exception"] = {
          "type": exc.__class__.__name__,
          "value": str(exc),
          "traceback": exc.traceback,
        }
      return json.dumps(log_data)

    # Console handler
    logger.patch(json_formatter)
    logger.add(sys.stdout, level=self.log_level, serialize=False)

    # File handler if log_file is specified
    if self.log_file:
      log_path = Path(self.log_file)
      log_path.parent.mkdir(parents=True, exist_ok=True)
      logger.add(
        str(log_path),
        format=json_formatter if self.json_logs else "{time} | {level} | {message}",
        level=self.log_level,
        rotation="500 MB",
        retention="10 days",
        compression="gz",
        serialize=False,
      )


class Logger:
  """Logger wrapper for application logging."""

  def __init__(self, context: Optional[Dict[str, Any]] = None):
    self.context = context or {}

  def bind(self, **kwargs) -> "Logger":
    """Create a new logger instance with additional context."""
    new_context = {**self.context, **kwargs}
    return Logger(new_context)

  def _log(self, level: str, message: str, **kwargs):
    """Internal logging method."""
    extra = {**self.context, **kwargs}
    getattr(logger, level)(message, **extra)

  def debug(self, message: str, **kwargs):
    """Log debug message."""
    self._log("debug", message, **kwargs)

  def info(self, message: str, **kwargs):
    """Log info message."""
    self._log("info", message, **kwargs)

  def warning(self, message: str, **kwargs):
    """Log warning message."""
    self._log("warning", message, **kwargs)

  def error(self, message: str, **kwargs):
    """Log error message."""
    self._log("error", message, **kwargs)

  def exception(self, message: str, exc_info: Exception, **kwargs):
    """Log exception with traceback."""
    self._log("exception", message, exc_info=exc_info, **kwargs)


# Create default logger instance
log = Logger()

# Setup default configuration
default_config = LogConfig()
default_config.setup_logger()
