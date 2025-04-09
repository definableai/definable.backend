from abc import ABC, abstractmethod
from typing import Any, Dict

from models import KBDocumentModel


class BaseSourceHandler(ABC):
  """Base class for all source handlers."""

  def __init__(self, config: Dict[str, Any]):
    self.config = config

  @abstractmethod
  async def validate_metadata(self, metadata: Dict[str, Any], **kwargs) -> bool:
    """Validate source metadata."""
    pass

  @abstractmethod
  async def preprocess(self, document: KBDocumentModel, **kwargs) -> None:
    """Preprocess the source (e.g., upload file, validate URLs)."""
    pass

  @abstractmethod
  async def extract_content(self, document: KBDocumentModel, **kwargs) -> str:
    """Extract content from the source."""
    pass

  @abstractmethod
  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    pass
