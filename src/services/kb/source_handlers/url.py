import copy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, cast
from urllib.parse import urlparse
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.websocket import WebSocketManager
from libs.firecrawl.v1 import firecrawl
from models import DocumentStatus, KBDocumentModel, KBFolder

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

  def _calculate_document_size(self, content: str) -> Dict[str, Any]:
    """Calculate various size metrics for the document content."""
    if not content:
      return {"size": 0, "size_characters": 0, "size_words": 0, "size_lines": 0}

    # Calculate different size metrics
    size_bytes = len(content.encode("utf-8"))
    size_characters = len(content)
    size_words = len(content.split())
    size_lines = len(content.splitlines())

    return {"size": size_bytes, "size_characters": size_characters, "size_words": size_words, "size_lines": size_lines}

  async def _create_folder_for_url_document(self, document: KBDocumentModel, session: AsyncSession) -> Optional[UUID]:
    """Create a folder for the URL document with the document title as folder name."""
    try:
      # Get the parent folder ID from the document
      parent_folder_id = document.folder_id

      # Check if a folder with the same name already exists
      if parent_folder_id:
        # Check in the specified parent folder
        existing_query = select(KBFolder).where(
          KBFolder.kb_id == document.kb_id, KBFolder.parent_id == parent_folder_id, KBFolder.name == document.title
        )
      else:
        # Check at root level
        existing_query = select(KBFolder).where(KBFolder.kb_id == document.kb_id, KBFolder.parent_id.is_(None), KBFolder.name == document.title)

      result = await session.execute(existing_query)
      existing_folder = result.scalar_one_or_none()

      if existing_folder:
        # Return the existing folder ID
        return existing_folder.id

      # Create new folder with the document title
      new_folder = KBFolder(
        name=document.title,
        parent_id=parent_folder_id,
        kb_id=document.kb_id,
        folder_info={"created_by": "url_document", "source_url": document.source_metadata.get("url", "")},
      )

      session.add(new_folder)
      await session.commit()
      await session.refresh(new_folder)

      return new_folder.id
    except Exception as e:
      # Log the error but don't break the process
      print(f"Error creating folder for URL document: {str(e)}")
      # Return the original folder_id if folder creation fails
      return document.folder_id

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

      # Create folder for organizing documents
      folder_id = await self._create_folder_for_url_document(document, session)
      if folder_id:
        # Update the document to use the new folder
        document.folder_id = folder_id
        session.add(document)
        await session.commit()

      if operation == "scrape":
        # Single URL scraping - create 1 document in the folder
        content = firecrawl.scrape_url(url=start_url, settings=settings)

        # Calculate and add size information to metadata
        size_info = self._calculate_document_size(content)
        metadata.update(size_info)

        # Update the document with size information
        document.source_metadata = metadata
        document.folder_id = folder_id
        session.add(document)
        await session.commit()

        return content

      elif operation == "crawl":
        # Web crawling - create multiple documents in the folder
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
                # prepare the metadata for child document
                child_metadata = copy.deepcopy(metadata)
                child_metadata["url"] = scrape["metadata"]["url"]
                child_metadata["is_parent"] = False
                child_metadata["parent_id"] = str(document.id)

                # Calculate and add size information for child document
                child_size_info = self._calculate_document_size(scrape["markdown"])
                child_metadata.update(child_size_info)

                title = scrape["metadata"]["url"].replace(start_url, "").strip("/").strip()
                # Create child document in the same folder
                child_doc = KBDocumentModel(
                  title=title,
                  description=scrape["metadata"]["url"],
                  kb_id=document.kb_id,
                  folder_id=folder_id,  # Place in the created folder
                  content=scrape["markdown"],
                  source_type_id=2,
                  source_metadata=child_metadata,
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

        # Calculate and add size information for parent document
        parent_size_info = self._calculate_document_size(parent_content)
        metadata.update(parent_size_info)

        # Update the parent document to be in the folder with size info
        document.folder_id = folder_id
        document.source_metadata = metadata
        session.add(document)
        await session.commit()

        return parent_content

      elif operation == "map":
        # Sitemap processing
        results = await firecrawl.map_url(
          url=start_url,
          settings=settings,
        )

        # Combine all content for size calculation
        combined_content = "\n\n".join(r.content for r in results)

        # Calculate and add size information to metadata
        size_info = self._calculate_document_size(combined_content)
        metadata.update(size_info)

        # Update the document to be placed in the folder with size info
        document.folder_id = folder_id
        document.source_metadata = metadata
        session.add(document)
        await session.commit()

        return combined_content

      else:
        raise ValueError(f"Unsupported operation: {operation}")

    except Exception as e:
      raise ValueError(f"Content extraction failed: {str(e)}")

  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    # No cleanup needed for URL processing
    pass
