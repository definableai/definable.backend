from typing import Any, Dict, List, Optional

from agno.document.base import Document
from agno.document.reader.firecrawl_reader import FirecrawlReader
from agno.tools.firecrawl import FirecrawlTools

from common.logger import logger
from config.settings import settings


class FirecrawlService(FirecrawlReader):
  def __init__(self, params: Optional[Dict[str,Any]] = None):
    super().__init__(api_key=settings.firecrawl_api_key, params=params)

  async def async_scrape_url(self, url: str) -> List[Document]:
    """Asynchronously scrape a URL using FirecrawlReader"""
    return await super().async_scrape(url)

  async def async_crawl_url(
      self,
      url: str,
      params: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
    """Asynchronously crawl a URL using FirecrawlReader"""
    # Set mode to crawl
    self.mode = "crawl"
    self.params = params
    return await super().async_crawl(url)


# Create a singleton instance
firecrawl = FirecrawlService()

class Firecrawl(FirecrawlTools):
  def __init__(self, params: Optional[Dict[str,Any]] = None):
    super().__init__(api_key=settings.firecrawl_api_key, params=params)
    logger.info("Firecrawl tool initialized")

  def scrape_website(self, url: str) -> str:
    return super().scrape_website(url)

  def crawl_website(self, url: str, limit: int | None = None) -> str:
    return super().crawl_website(url, limit)

# create a singleton instance
firecrawl_tool = Firecrawl()
