from typing import Any, Dict, List, Optional

import httpx
from firecrawl import FirecrawlApp

from config.settings import settings


class Firecrawl:
  def __init__(self):
    self.app = FirecrawlApp(api_key=settings.firecrawl_api_key)
    self.base_url = "https://api.firecrawl.dev/v1"
    self.token = "Bearer " + settings.firecrawl_api_key

  def map_url(self, url: str, settings: Optional[Dict[str, Any]] = None) -> List[str]:
    return self.app.map_url(url, settings or {})

  def scrape_url(self, url: str, settings: Optional[Dict[str, Any]] = None) -> str:
    result = self.app.scrape_url(url, settings or {})
    if result["metadata"]["statusCode"] == 200:
      return result["markdown"]
    else:
      raise Exception("Failed to scrape URL")

  async def crawl(
    self,
    url: str,
    params: Dict[str, Any],
  ):
    """
    Crawl a URL with custom event handlers.

    Args:
        url: The URL to crawl
        on_document: Callback for document events
        on_error: Callback for error events
        on_done: Callback for completion event
        settings: Crawl settings (limit, exclude_paths, etc.)
    """
    try:
      params["url"] = url
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{self.base_url}/crawl",
          json=params,
          headers={"Authorization": self.token},
        )
        response.raise_for_status()
        print(response.json())
        return response.json()

    except Exception as e:
      raise e

  async def get_crawl_status(self, crawl_id: str) -> dict[str, Any]:
    return self.app.check_crawl_status(crawl_id)


firecrawl = Firecrawl()
