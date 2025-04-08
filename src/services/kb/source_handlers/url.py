import copy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, cast
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession

from common.websocket import WebSocketManager
from libs.firecrawl.v1 import firecrawl
from models import DocumentStatus, KBDocumentModel

from .base import BaseSourceHandler


class URLOperation(str, Enum):
  """URL operation types."""

  SCRAPE = "scrape"
  CRAWL = "crawl"
  MAP = "map"


def clean_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
  """Remove empty or zero values from settings."""
  cleaned = {}
  for key, value in settings.items():
    if key == "maxDepth" and value == 0:
      continue
    if key == "excludePaths" and value == []:
      continue
    if key == "includePaths" and value == []:
      continue
    cleaned[key] = value
  return cleaned


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
      metadata = copy.deepcopy(document.source_metadata)
      session = cast(AsyncSession, kwargs.get("session"))
      ws_manager = cast(WebSocketManager, kwargs.get("ws_manager"))
      operation = metadata["operation"]
      start_url = metadata["url"]
      settings = clean_settings(metadata.get("settings", {}))
      metadata["settings"] = settings

      if operation == "scrape":
        # Single URL scraping
        content = firecrawl.scrape_url(url=start_url, settings=settings)
        return content

      elif operation == "crawl":
        # Web crawling
        results = await firecrawl.crawl(
          url=start_url,
          params=settings,
        )
        if results["success"]:
          crawl_id = results["id"]
        else:
          raise ValueError(f"Crawling failed: {document.id} url: {start_url}")

        seen_scrape_ids: set[str] = set()
        parent_content = ""

        while True:
          scrape_results = await firecrawl.get_crawl_status(crawl_id)
          if scrape_results["success"]:
            for scrape in scrape_results["data"]:
              if scrape["metadata"]["scrapeId"] not in seen_scrape_ids:
                # if the scrape is the parent url, then we add the content to the parent content
                if scrape["metadata"]["url"] == start_url or scrape["metadata"]["url"][:-1] == metadata["url"]:
                  parent_content = scrape["markdown"]
                  continue
                if scrape["markdown"].strip() == "":
                  print(f"Skipping empty content: {scrape['metadata']['url']}")
                  continue
                # prepare the metadata
                metadata["url"] = scrape["metadata"]["url"]
                metadata["is_parent"] = False
                metadata["parent_id"] = str(document.id)
                print(metadata)

                child_doc = KBDocumentModel(
                  title=scrape["metadata"]["title"],
                  description=scrape["metadata"]["title"],
                  kb_id=document.kb_id,
                  content=scrape["markdown"],
                  source_type_id=2,
                  source_id=document.source_id,
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
            raise ValueError(f"Crawling failed: {document.id} url: {start_url}")

          if scrape_results["status"] == "completed":
            break

        return parent_content

      elif operation == "map":
        # Sitemap processing
        results = await firecrawl.map_url(
          url=start_url,
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
