import json
import os
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

  def add_caller_info(self, record):
    """
    Patch function that adds a 'caller_info' key to each record.
    The value is formatted as: <parent-module>/<module>/<function>:<line>
    """
    file_path = record["file"].path  # e.g. "D:/work/zyeta.backend/src/common/logger.py"
    # Extract the parent directory name and the module (file) name
    parent_module = os.path.basename(os.path.dirname(file_path))
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Build the caller info using the function name and line number from the record
    caller_info = f"{parent_module}/{module_name}/{record['function']}:{record['line']}"
    record["extra"]["caller_info"] = caller_info
    return record

  def file_formatter(self, record):
    """Custom formatter for file logs that outputs JSON with escaped curly braces."""
    log_data = {
      "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
      "level": record["level"].name,
      "caller": record["extra"].get("caller_info", ""),
      "message": record["message"],
      "line": record["line"],
      "extra": record["extra"],
    }
    if record["exception"]:
      exc = record["exception"]
      log_data["exception"] = {
        "type": exc.__class__.__name__,
        "value": str(exc),
        "traceback": exc.__traceback__,
      }
    # Dump JSON and escape curly braces so that they are not processed by Loguru's formatter.
    raw = json.dumps(log_data)
    escaped = raw.replace("{", "{{").replace("}", "}}")
    return escaped + "\n"

  def setup_logger(self):
    """Configure logger with different handlers for console and file."""
    # Remove default handler
    logger.remove()
    # Apply the patch to add caller_info to each record.
    logger.configure(patcher=self.add_caller_info)

    # Add a console handler
    # Console handler with custom timestamp formatting.
    console_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[caller_info]} | {message}"
    logger.add(sys.stdout, level=self.log_level, format=console_format)

    # Add a file handler that uses the JSON formatter
    if self.log_file:
      log_path = Path(self.log_file)
      log_path.parent.mkdir(parents=True, exist_ok=True)
      logger.add(
        str(log_path),
        level=self.log_level,
        rotation="500 MB",
        retention="10 days",
        compression="gz",
        format=self.file_formatter if self.json_logs else "{time} | {level} | {extra[caller_info]} | {message}",
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
    """Internal logging method.

    Using logger.opt(depth=2) ensures that the module, line, and other caller information
    reflect the place where the log method was invoked, not inside this wrapper.
    """
    extra = {**self.context, **kwargs}
    getattr(logger.opt(depth=2), level)(message, **extra)

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
default_config = LogConfig(log_file="logs/app.log")
default_config.setup_logger()
