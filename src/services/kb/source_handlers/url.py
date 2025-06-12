from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.definable.document.reader.arxiv_reader import ArxivReader
from libs.definable.document.reader.base import Reader
from libs.definable.document.reader.firecrawl_reader import FirecrawlReader
from libs.definable.document.reader.pdf_reader import PDFUrlReader
from libs.definable.document.reader.website_reader import WebsiteReader
from libs.definable.document.reader.youtube_reader import YouTubeReader
from models import KBDocumentModel, KBFolder

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


class URLReaderFactory:
  """Factory class for selecting appropriate URL readers."""

  @classmethod
  def get_reader(cls, url: str, **kwargs) -> Reader:
    """Get appropriate reader based on URL pattern."""

    # Parse URL
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # YouTube URLs - exact domain matching only
    if domain in ["youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"] or domain.endswith(".youtube.com"):
      return YouTubeReader(chunk=False, **kwargs)

    # arXiv URLs
    if "arxiv.org" in domain:
      return ArxivReader(chunk=False, **kwargs)

    # PDF URLs (check if URL ends with .pdf)
    if path.endswith(".pdf"):
      return PDFUrlReader(chunk=False, **kwargs)

    # Default to website reader for general web content
    return WebsiteReader(chunk=False, **kwargs)

  @classmethod
  def get_url_type(cls, url: str) -> str:
    """Determine the type of URL for logging/metadata."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # Fix YouTube detection here too
    if domain in ["youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"] or domain.endswith(".youtube.com"):
      return "youtube"
    elif "arxiv.org" in domain:
      return "arxiv"
    elif path.endswith(".pdf"):
      return "pdf_url"
    else:
      return "website"


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
      url = metadata.get("url")
      if not url:
        raise ValueError("URL not found in metadata")

      # Basic URL validation
      parsed = urlparse(url)
      if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL format")

      return True

    except Exception as e:
      raise ValueError(f"Invalid URL metadata: {str(e)}")

  async def preprocess(self, document: KBDocumentModel, **kwargs) -> None:
    """Preprocess the URL."""
    pass

  async def extract_content(self, document: KBDocumentModel, **kwargs) -> str:
    """Extract content from URL using appropriate reader based on operation."""
    try:
      metadata = document.source_metadata
      url = metadata.get("url")
      operation = metadata.get("operation", "scrape")
      settings = metadata.get("settings", {})

      if not url:
        raise ValueError("URL not found in metadata")

      # Get URL type for special handling
      url_type = URLReaderFactory.get_url_type(url)

      # Handle special URL types (YouTube, arXiv, PDF) with their dedicated readers
      if url_type in ["youtube", "arxiv", "pdf_url"]:
        reader = URLReaderFactory.get_reader(url)
        documents = await reader.async_read(url)
        content = "\n\n".join(doc.content for doc in documents)

        # Update metadata
        metadata["url_type"] = url_type
        metadata["processed_by"] = reader.__class__.__name__
        return content

      # Handle general web content based on operation
      if operation == URLOperation.MAP:
        # Use firecrawl API directly for mapping
        try:
          from libs.firecrawl.v1 import firecrawl

          urls = firecrawl.map_url(url, settings)
          content = f"Mapped URLs from {url}:\n" + "\n".join(urls)
          metadata["mapped_urls"] = urls
          metadata["url_count"] = len(urls)
        except Exception as e:
          raise ValueError(f"Failed to map URL: {str(e)}")

      elif operation in [URLOperation.SCRAPE, URLOperation.CRAWL]:
        # Use FirecrawlReader for scrape and crawl operations
        try:
          filtered_settings = {}

          if operation == URLOperation.CRAWL:
            # For crawl, structure parameters correctly
            for key, value in settings.items():
              if key in ["maxDepth", "limit", "includePaths", "excludePaths", "ignoreSitemap", "allowBackwardLinks"]:
                filtered_settings[key] = value

            # Add scrapeOptions for crawl
            if "scrapeOptions" in settings:
              scrape_opts = settings["scrapeOptions"]
              filtered_settings["scrapeOptions"] = {}
              for key, value in scrape_opts.items():
                if key in ["excludeTags", "includeTags", "onlyMainContent", "formats", "waitFor", "timeout"]:
                  filtered_settings["scrapeOptions"][key] = value
          else:
            # For scrape, use flat structure
            for key, value in settings.items():
              if key in ["excludeTags", "includeTags", "onlyMainContent", "waitFor", "timeout", "formats"]:
                filtered_settings[key] = value

          print(f"Filtered firecrawl settings: {filtered_settings}")

          # Determine mode based on operation
          mode = "scrape" if operation == URLOperation.SCRAPE else "crawl"

          # Initialize FirecrawlReader with filtered settings
          reader = FirecrawlReader(
            api_key=None,  # Will use default from settings
            params=filtered_settings,
            mode=mode,
            chunk=False,
          )

          # Extract content
          documents = await reader.async_read(url)
          content = "\n\n".join(doc.content for doc in documents)

          # Update metadata
          metadata["pages_processed"] = len(documents)

        except Exception as e:
          # Fallback to WebsiteReader if FirecrawlReader fails
          print(f"FirecrawlReader failed, falling back to WebsiteReader: {str(e)}")

          # Extract relevant settings for WebsiteReader
          max_depth = settings.get("maxDepth", 1) if operation == URLOperation.CRAWL else 1
          max_links = settings.get("limit", 10) if operation == URLOperation.CRAWL else 1

          reader = WebsiteReader(max_depth=max_depth, max_links=max_links, chunk=False)

          if operation == URLOperation.SCRAPE:
            documents = await reader.async_read(url)
          else:  # CRAWL
            documents = await reader.async_read(url)

          content = "\n\n".join(doc.content for doc in documents)
          metadata["fallback_used"] = "WebsiteReader"

      else:
        raise ValueError(f"Unsupported operation: {operation}")

      # Update metadata with processing info
      metadata["url_type"] = url_type
      metadata["operation_completed"] = operation
      metadata["processed_by"] = reader.__class__.__name__ if "reader" in locals() else "FirecrawlAPI"
      metadata["settings_used"] = filtered_settings if "filtered_settings" in locals() else settings

      return content

    except Exception as e:
      raise ValueError(f"Failed to extract URL content: {str(e)}")

  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    # Most URL readers don't need cleanup
    pass
