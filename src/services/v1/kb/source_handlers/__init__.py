from typing import Dict, Type

from .base import BaseSourceHandler
from .file import FileSourceHandler
from .url import URLSourceHandler

# Registry of source handlers
SOURCE_HANDLERS: Dict[str, Type[BaseSourceHandler]] = {
  "file": FileSourceHandler,
  "url": URLSourceHandler,
}


def get_source_handler(source_type: str, config: Dict) -> BaseSourceHandler:
  """Get appropriate source handler based on source type."""
  handler_class = SOURCE_HANDLERS.get(source_type)
  if not handler_class:
    raise ValueError(f"No handler found for source type: {source_type}")
  return handler_class(config)
