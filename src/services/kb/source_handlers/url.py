from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from common.websocket import WebSocketManager
from libs.firecrawl import firecrawl

from ..model import DocumentStatus, KBDocumentModel
from .base import BaseSourceHandler


class URLOperation(str, Enum):
  """URL operation types."""

  SCRAPE = "scrape"
  CRAWL = "crawl"
  MAP = "map"


class URLMetadata(BaseModel):
  """URL metadata schema."""

  operation: URLOperation = Field(..., description="URL operation type")
  urls: List[str] = Field(..., description="List of URLs to process")
  settings: Dict = Field(..., description="Operation-specific settings")


def clean_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
  """Remove empty or zero values from settings."""

  def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in d.items():
      # Handle nested dictionaries
      if isinstance(value, dict):
        nested_cleaned = clean_dict(value)
        if nested_cleaned:  # Only add if not empty
          cleaned[key] = nested_cleaned
      # Handle lists
      elif isinstance(value, list) and not value:
        continue
      # Handle zero values
      elif value == 0:
        continue
      # Handle empty strings
      elif isinstance(value, str) and not value.strip():
        continue
      # Keep other values
      else:
        cleaned[key] = value
    return cleaned

  return clean_dict(settings)


class URLSourceHandler(BaseSourceHandler):
  """Handler for URL-based sources."""

  def __init__(self, config: Dict):
    super().__init__(config)
    self.max_urls = config.get("max_urls", 100)
    self.timeout = config.get("timeout", 30)

  async def validate_metadata(self, metadata: Dict, **kwargs) -> bool:
    """Validate URL metadata."""
    try:
      # Validate URLs
      result = urlparse(url=metadata["url"])
      if not all([result.scheme, result.netloc]):
        raise ValueError(f"Invalid URL: {metadata['url']}")
      return True

    except Exception as e:
      raise ValueError(f"Invalid URL metadata: {str(e)}")

  async def preprocess(self, document: KBDocumentModel, **kwargs) -> None:
    """Preprocess URLs (validate accessibility)."""
    pass

  async def extract_content(self, document: KBDocumentModel, **kwargs) -> str:
    """Extract content from URLs."""
    try:
      metadata = document.source_metadata
      session = cast(AsyncSession, kwargs.get("session"))
      ws_manager = cast(WebSocketManager, kwargs.get("ws_manager"))
      operation = metadata["operation"]
      url = metadata["url"]
      settings = {**metadata.get("settings", {})}
      settings = clean_settings(settings)
      if operation == "scrape":
        # Single URL scraping
        content = firecrawl.scrape_url(url=url, settings=settings)
        return content

      elif operation == "crawl":
        # Web crawling
        results = await firecrawl.crawl(
          url=url,
          settings=settings,
        )
        if results["success"]:
          crawl_id = results["id"]
        else:
          raise ValueError(f"Crawling failed: {document.id} url: {url}")

        seen_scrape_ids: set[str] = set()
        parent_content = ""

        while True:
          scrape_results = await firecrawl.get_crawl_status(crawl_id)
          if scrape_results["success"]:
            for scrape in scrape_results["data"]:
              if scrape["metadata"]["scrapeId"] not in seen_scrape_ids:
                # if the scrape is the parent url, then we add the content to the parent content
                if scrape["metadata"]["url"] == url:
                  parent_content = scrape["markdown"]
                  continue
                if not scrape["markdown"]:
                  continue
                # prepare the metadata
                metadata["url"] = scrape["metadata"]["url"]
                metadata["is_parent"] = False
                metadata["parent_id"] = str(document.id)

                child_doc = KBDocumentModel(
                  title=scrape["metadata"]["title"],
                  description=scrape["metadata"]["title"],
                  kb_id=document.kb_id,
                  content=scrape["markdown"],
                  source_type_id=2,
                  source_metadata=metadata,
                  extraction_status=DocumentStatus.COMPLETED,
                  indexing_status=DocumentStatus.PENDING,
                  extraction_completed_at=datetime.now(),
                )
                session.add(child_doc)
                await session.commit()
                await ws_manager.broadcast(
                  kwargs.get("org_id"),
                  {"id": str(child_doc.id), "extraction_status": child_doc.extraction_status},
                  "kb",
                  "write",
                )
                seen_scrape_ids.add(scrape["metadata"]["scrapeId"])
          else:
            raise ValueError(f"Crawling failed: {document.id} url: {url}")

          if scrape_results["status"] == "completed":
            break

        return parent_content

      elif operation == "map":
        # Sitemap processing
        results = await firecrawl.map_url(
          url=url,
          settings=settings,
        )
        return "\n\n".join(r.content for r in results)

      else:
        raise ValueError(f"Unsupported operation: {operation}")

    except Exception as e:
      raise ValueError(f"Content extraction failed: {str(e)}")

  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    # No cleanup needed for URL processing
    pass
