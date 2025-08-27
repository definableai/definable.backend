import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
  import httpx
  from firecrawl import FirecrawlApp
except ImportError:
  raise ImportError("firecrawl-py is required. Install it with: pip install firecrawl-py")

from config.settings import settings


class Firecrawl:
  """Firecrawl wrapper for web scraping and crawling operations."""

  def __init__(self, api_key: Optional[str] = None):
    """Initialize Firecrawl wrapper.

    Args:
        api_key: Optional API key. If not provided, will use settings.firecrawl_api_key
    """
    self.api_key = api_key or settings.firecrawl_api_key
    if not self.api_key:
      raise ValueError("Firecrawl API key is required. Set it in settings or pass it directly.")

    self.app = FirecrawlApp(api_key=self.api_key)
    self.base_url = "https://api.firecrawl.dev/v1"
    self.logger = logging.getLogger(__name__)

    # Rate limiting settings
    self.rate_limit_delay = 1.0  # Delay between requests
    self.max_retries = 3
    self.retry_delay = 2.0

  def _get_headers(self) -> Dict[str, str]:
    """Get headers for API requests."""
    return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

  def _validate_url(self, url: str) -> None:
    """Validate URL format."""
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
      raise ValueError(f"Invalid URL: {url}")

  def _clean_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty or None values from settings."""
    return {k: v for k, v in settings.items() if v is not None and v != "" and v != []}

  def _standardize_response(self, response: Any, url: str, operation: str) -> Dict[str, Any]:
    """Standardize response format across different operations."""
    if isinstance(response, dict):
      # Standard response format
      return {"success": True, "data": response, "url": url, "operation": operation, "timestamp": time.time()}
    elif isinstance(response, str):
      # String response (markdown content)
      return {
        "success": True,
        "data": {"markdown": response, "metadata": {"url": url, "statusCode": 200, "operation": operation}},
        "url": url,
        "operation": operation,
        "timestamp": time.time(),
      }
    else:
      # List or other response
      return {"success": True, "data": response, "url": url, "operation": operation, "timestamp": time.time()}

  async def scrape_url(self, url: str, settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Scrape content from a single URL.

    Args:
        url: The URL to scrape
        settings: Scraping configuration options

    Returns:
        Extracted content as markdown

    Raises:
        ValueError: If scraping fails or URL is invalid
    """
    try:
      self._validate_url(url)
      settings = self._clean_settings(settings or {})

      self.logger.info(f"Scraping URL: {url}")

      # Use the official FirecrawlApp for scraping
      result = self.app.scrape_url(url, params=settings)

      if not result:
        raise ValueError("No result returned from Firecrawl")

      # Extract markdown content
      if isinstance(result, dict):
        if result.get("metadata", {}).get("statusCode") != 200:
          error_msg = result.get("metadata", {}).get("error", "Unknown error")
          raise ValueError(f"Failed to scrape URL: {error_msg}")

        content = result.get("markdown", "")
        if not content:
          # Fallback to other content types
          content = result.get("html", "") or result.get("rawHtml", "")
      else:
        content = str(result)

      if not content.strip():
        raise ValueError("No content extracted from URL")

      self.logger.info(f"Successfully scraped {len(content)} characters from {url}")
      return content

    except Exception as e:
      self.logger.error(f"Error scraping URL {url}: {str(e)}")
      raise ValueError(f"Failed to scrape URL {url}: {str(e)}")

  def map_url(self, url: str, settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Map and discover URLs from a website.

    Args:
        url: The URL to map
        settings: Mapping configuration options

    Returns:
        List of discovered URLs with metadata

    Raises:
        ValueError: If mapping fails or URL is invalid
    """
    try:
      self._validate_url(url)
      settings = self._clean_settings(settings or {})

      self.logger.info(f"Mapping URL: {url}")

      # Use the official FirecrawlApp for mapping
      result = self.app.map_url(url, params=settings)

      if not result:
        raise ValueError("No result returned from Firecrawl")

      # Standardize the response format
      mapped_urls = []
      if isinstance(result, list):
        for item in result:
          if isinstance(item, str):
            mapped_urls.append({"url": item, "status": "discovered", "content": f"URL discovered: {item}"})
          elif isinstance(item, dict):
            mapped_urls.append({
              "url": item.get("url", ""),
              "title": item.get("title", ""),
              "status": "discovered",
              "content": item.get("description", f"URL discovered: {item.get('url', '')}"),
            })
      elif isinstance(result, dict) and "links" in result:
        # Handle case where result contains links array
        for link in result["links"]:
          if isinstance(link, str):
            mapped_urls.append({"url": link, "status": "discovered", "content": f"URL discovered: {link}"})
          elif isinstance(link, dict):
            mapped_urls.append({
              "url": link.get("url", ""),
              "title": link.get("title", ""),
              "status": "discovered",
              "content": link.get("description", f"URL discovered: {link.get('url', '')}"),
            })

      self.logger.info(f"Successfully mapped {len(mapped_urls)} URLs from {url}")
      return mapped_urls

    except Exception as e:
      self.logger.error(f"Error mapping URL {url}: {str(e)}")
      raise ValueError(f"Failed to map URL {url}: {str(e)}")

  async def crawl(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl multiple pages starting from a URL.

    Args:
        url: Starting URL for crawling
        params: Crawling parameters including limit, excludePaths, etc.

    Returns:
        Dictionary containing crawl results with 'success', 'data', and 'status' keys

    Raises:
        ValueError: If crawling fails or configuration is invalid
    """
    try:
      self._validate_url(url)

      # Extract crawling parameters
      limit = params.get("limit", 10)
      exclude_paths = params.get("excludePaths", [])
      include_paths = params.get("includePaths", [])
      scrape_options = params.get("scrapeOptions", {})

      self.logger.info(f"Starting crawl of {url} with limit {limit}")

      # Prepare crawl parameters for Firecrawl
      crawl_params = {"limit": limit, "scrapeOptions": scrape_options}

      if exclude_paths:
        crawl_params["excludePaths"] = exclude_paths
      if include_paths:
        crawl_params["includePaths"] = include_paths

      # Clean the parameters
      crawl_params = self._clean_settings(crawl_params)

      # Use async HTTP client for crawling
      async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{self.base_url}/crawl", json={**crawl_params, "url": url}, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
          raise ValueError(f"Crawl failed: {result.get('error', 'Unknown error')}")

        # Get the crawl ID to check status
        crawl_id = result.get("id")
        if not crawl_id:
          raise ValueError("No crawl ID returned")

        # Poll for completion
        max_wait_time = 300  # 5 minutes
        wait_time = 0
        poll_interval = 5

        while wait_time < max_wait_time:
          await asyncio.sleep(poll_interval)
          wait_time += poll_interval

          status_response = await client.get(f"{self.base_url}/crawl/{crawl_id}", headers=self._get_headers())
          status_response.raise_for_status()
          status_result = status_response.json()

          status = status_result.get("status")
          if status == "completed":
            # Format the response to match expected structure
            scraped_data = []
            data = status_result.get("data", [])

            for i, item in enumerate(data):
              scraped_data.append({
                "markdown": item.get("markdown", ""),
                "metadata": {
                  "url": item.get("metadata", {}).get("sourceURL", url),
                  "scrapeId": f"scrape_{i}",
                  "statusCode": item.get("metadata", {}).get("statusCode", 200),
                  "title": item.get("metadata", {}).get("title", ""),
                },
              })

            return {"success": True, "data": scraped_data, "status": "completed", "total": len(scraped_data)}
          elif status == "failed":
            error_msg = status_result.get("error", "Crawl failed")
            raise ValueError(f"Crawl failed: {error_msg}")
          elif status in ["scraping", "processing"]:
            # Continue waiting
            continue
          else:
            raise ValueError(f"Unknown crawl status: {status}")

        raise ValueError("Crawl timed out")

    except httpx.HTTPError as e:
      self.logger.error(f"HTTP error during crawl: {str(e)}")
      return {"success": False, "error": f"HTTP error: {str(e)}", "status": "failed"}
    except Exception as e:
      self.logger.error(f"Error crawling {url}: {str(e)}")
      return {"success": False, "error": str(e), "status": "failed"}

  async def get_crawl_status(self, crawl_id: str) -> Dict[str, Any]:
    """
    Get the status of a crawl operation.

    Args:
        crawl_id: The ID of the crawl operation

    Returns:
        Dictionary containing crawl status information
    """
    try:
      result = self.app.check_crawl_status(crawl_id)
      return self._standardize_response(result, "", "status_check")
    except Exception as e:
      self.logger.error(f"Error getting crawl status: {str(e)}")
      raise ValueError(f"Failed to get crawl status: {str(e)}")

  def search(self, query: str, settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Perform a web search and return results with content.

    Args:
        query: Search query
        settings: Search configuration options

    Returns:
        List of search results with content

    Raises:
        ValueError: If search fails
    """
    try:
      settings = self._clean_settings(settings or {})

      self.logger.info(f"Searching for: {query}")

      # Use the official FirecrawlApp for searching
      result = self.app.search(query, params=settings)

      if not result:
        raise ValueError("No result returned from Firecrawl search")

      # Standardize search results
      search_results = []
      if isinstance(result, list):
        for item in result:
          search_results.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "markdown": item.get("markdown", ""),
            "metadata": item.get("metadata", {}),
            "snippet": item.get("snippet", ""),
          })

      self.logger.info(f"Successfully found {len(search_results)} search results")
      return search_results

    except Exception as e:
      self.logger.error(f"Error searching for '{query}': {str(e)}")
      raise ValueError(f"Failed to search: {str(e)}")

  async def extract_structured_data(self, url: str, schema: Dict[str, Any], settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract structured data from a URL using AI.

    Args:
        url: The URL to extract data from
        schema: JSON schema defining the structure to extract
        settings: Extraction configuration options

    Returns:
        Dictionary containing extracted structured data

    Raises:
        ValueError: If extraction fails
    """
    try:
      self._validate_url(url)
      settings = self._clean_settings(settings or {})

      self.logger.info(f"Extracting structured data from: {url}")

      # Prepare extraction parameters
      extract_params = {"schema": schema, **settings}

      # Use async HTTP client for extraction
      async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{self.base_url}/extract", json={"url": url, **extract_params}, headers=self._get_headers())
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
          raise ValueError(f"Extraction failed: {result.get('error', 'Unknown error')}")

        return self._standardize_response(result.get("data", {}), url, "extract")

    except httpx.HTTPError as e:
      self.logger.error(f"HTTP error during extraction: {str(e)}")
      raise ValueError(f"HTTP error during extraction: {str(e)}")
    except Exception as e:
      self.logger.error(f"Error extracting structured data from {url}: {str(e)}")
      raise ValueError(f"Failed to extract structured data: {str(e)}")

  async def close(self):
    """Close connections and cleanup resources."""
    # Firecrawl doesn't require explicit cleanup, but this maintains consistency
    pass

  async def __aenter__(self):
    """Async context manager entry."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit."""
    await self.close()


# Create singleton instance
firecrawl = Firecrawl()
