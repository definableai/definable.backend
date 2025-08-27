import os
import platform
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


class Crawl4AI:
  """Crawl4AI wrapper for web scraping and crawling operations."""

  def __init__(self):
    """Initialize Crawl4AI wrapper with platform-specific configurations."""
    # Get platform-specific configurations
    browser_config_params = self._get_platform_config()

    self.browser_config = BrowserConfig(**browser_config_params)
    self._crawler = None

  def _get_platform_config(self) -> Dict[str, Any]:
    """Get platform-specific browser configuration."""
    system = platform.system().lower()

    # Base configuration
    config = {
      "headless": True,
      "browser_type": "chromium",
      "use_persistent_context": False,
      "viewport_width": 1920,
      "viewport_height": 1080,
      "accept_downloads": False,
      "java_script_enabled": True,
    }

    # Platform-specific user agents
    if system == "windows":
      config["user_agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    elif system == "darwin":  # macOS
      config["user_agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    elif system == "linux":
      config["user_agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    else:
      # Fallback for unknown systems
      config["user_agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    # Platform-specific browser configurations
    if system == "linux":
      # Linux often runs in containers or headless environments
      config.update({
        "extra_args": [
          "--no-sandbox",
          "--disable-dev-shm-usage",
          "--disable-gpu",
          "--disable-background-timer-throttling",
          "--disable-backgrounding-occluded-windows",
          "--disable-renderer-backgrounding",
          "--disable-features=TranslateUI",
          "--disable-ipc-flooding-protection",
        ]
      })
    elif system == "darwin":
      # macOS specific configurations
      config.update({
        "extra_args": [
          "--disable-dev-shm-usage",
          "--disable-features=TranslateUI",
        ]
      })
    elif system == "windows":
      # Windows specific configurations
      config.update({
        "extra_args": [
          "--disable-features=TranslateUI",
          "--disable-background-timer-throttling",
        ]
      })

    # Docker/Container detection and additional safety flags
    if self._is_running_in_container():
      if "extra_args" not in config:
        config["extra_args"] = []

      # Ensure we have a list to extend
      extra_args = config.get("extra_args", [])
      if isinstance(extra_args, list):
        extra_args.extend([
          "--no-sandbox",
          "--disable-setuid-sandbox",
          "--disable-dev-shm-usage",
          "--disable-accelerated-2d-canvas",
          "--no-first-run",
          "--no-zygote",
          "--single-process",
          "--disable-gpu",
        ])
        config["extra_args"] = extra_args

    return config

  def _is_running_in_container(self) -> bool:
    """Detect if running in a container environment."""
    try:
      # Check for Docker container
      with open("/proc/1/cgroup", "r") as f:
        return "docker" in f.read() or "containerd" in f.read()
    except (FileNotFoundError, PermissionError):
      # Not Linux or can't read cgroup, check other indicators
      pass

    # Check environment variables that indicate containerization
    container_indicators = [
      "DOCKER_CONTAINER",
      "KUBERNETES_SERVICE_HOST",
      "CONTAINER",
      "PODMAN_USERNS",
    ]

    for indicator in container_indicators:
      if indicator in os.environ:
        return True

    # Check if running as PID 1 (common in containers)
    try:
      return os.getpid() == 1
    except OSError:
      pass

    return False

  async def _get_crawler(self) -> AsyncWebCrawler:
    """Get or create crawler instance with fallback configurations."""
    if self._crawler is None:
      try:
        self._crawler = AsyncWebCrawler(config=self.browser_config)
      except Exception as e:
        # Check if browser is available
        if not self._check_browser_availability():
          error_msg = f"Browser not found on {platform.system()}.\n{self._get_browser_installation_hint()}"
          raise ValueError(error_msg)

        # Fallback to minimal configuration if initial setup fails
        print(f"Primary browser configuration failed: {e}")
        print("Attempting fallback configuration...")

        try:
          fallback_config = self._get_fallback_config()
          self._crawler = AsyncWebCrawler(config=BrowserConfig(**fallback_config))
        except Exception as fallback_error:
          error_msg = (
            f"Both primary and fallback browser configurations failed.\n"
            f"Primary error: {e}\n"
            f"Fallback error: {fallback_error}\n"
            f"{self._get_browser_installation_hint()}"
          )
          raise ValueError(error_msg)

    return self._crawler

  def _get_fallback_config(self) -> Dict[str, Any]:
    """Get minimal fallback configuration for problematic environments."""
    return {
      "headless": True,
      "browser_type": "chromium",
      "use_persistent_context": False,
      "viewport_width": 1280,
      "viewport_height": 720,
      "accept_downloads": False,
      "java_script_enabled": True,
      "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      "extra_args": [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-accelerated-2d-canvas",
        "--no-first-run",
        "--no-zygote",
        "--disable-gpu",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor",
      ],
    }

  def _check_browser_availability(self) -> bool:
    """Check if required browser binaries are available."""
    import shutil

    system = platform.system().lower()

    # Common browser executable names by platform
    browser_executables = {
      "linux": ["chromium-browser", "chromium", "google-chrome", "google-chrome-stable"],
      "darwin": ["Google Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
      "windows": ["chrome.exe", "chromium.exe", "msedge.exe"],
    }

    executables = browser_executables.get(system, browser_executables["linux"])

    for executable in executables:
      if shutil.which(executable) or os.path.exists(executable):
        return True

    return False

  def _get_browser_installation_hint(self) -> str:
    """Get platform-specific browser installation instructions."""
    system = platform.system().lower()

    hints = {
      "linux": """
To install Chrome/Chromium on Linux:
- Ubuntu/Debian: sudo apt-get install chromium-browser
- CentOS/RHEL: sudo yum install chromium
- Or download from: https://www.google.com/chrome/
""",
      "darwin": """
To install Chrome on macOS:
- Download from: https://www.google.com/chrome/
- Or use Homebrew: brew install --cask google-chrome
""",
      "windows": """
To install Chrome on Windows:
- Download from: https://www.google.com/chrome/
- Or use winget: winget install Google.Chrome
""",
    }

    return hints.get(system, hints["linux"])

  async def _retry_with_backoff(self, operation, max_retries: int = 3, base_delay: float = 1.0):
    """Retry an operation with exponential backoff."""
    import asyncio

    for attempt in range(max_retries):
      try:
        return await operation()
      except Exception as e:
        if attempt == max_retries - 1:
          raise e

        # Calculate delay with exponential backoff
        delay = base_delay * (2**attempt)

        # Platform-specific retry delays
        system = platform.system().lower()
        if system == "linux" or self._is_running_in_container():
          delay *= 1.5  # Longer delays for containers
        elif system == "windows":
          delay *= 0.8  # Shorter delays for Windows

        print(f"Operation failed (attempt {attempt + 1}/{max_retries}): {e}")
        print(f"Retrying in {delay:.1f} seconds...")
        await asyncio.sleep(delay)

  async def _close_crawler(self):
    """Close crawler instance."""
    if self._crawler is not None:
      await self._crawler.close()
      self._crawler = None

  def _create_run_config(self, settings: Dict[str, Any]) -> CrawlerRunConfig:
    """Create crawler run configuration from settings."""
    # Use minimal configuration for Crawl4AI 0.7.3 compatibility
    # Most advanced parameters are not supported in this version
    try:
      return CrawlerRunConfig(word_count_threshold=10)
    except TypeError:
      # If word_count_threshold is also not supported, use empty config
      return CrawlerRunConfig()

  def _clean_content(self, content: str) -> str:
    """Clean and normalize extracted content."""
    if not content:
      return ""

    # Remove excessive whitespace
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
      line = line.strip()
      if line:
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

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
      # Validate URL
      parsed = urlparse(url)
      if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")

      settings = settings or {}

      # Define the scraping operation
      async def scraping_operation():
        crawler = await self._get_crawler()
        run_config = self._create_run_config(settings)

        # Start crawler if not already started
        await crawler.start()

        # Perform scraping
        result = await crawler.arun(url, config=run_config)

        if not result.success:
          raise ValueError(f"Crawl failed: {result.error_message}")

        return result

      # Execute with retry logic
      result = await self._retry_with_backoff(scraping_operation)

      # Extract content based on format preference
      content = ""
      if result.markdown:
        content = result.markdown
      elif result.cleaned_html:
        content = result.cleaned_html
      else:
        content = result.html

      cleaned_content = self._clean_content(content)
      if not cleaned_content:
        raise ValueError("No content extracted from URL")

      return cleaned_content

    except Exception as e:
      raise ValueError(f"Failed to scrape URL {url}: {str(e)}")

  async def crawl(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crawl multiple pages starting from a URL.

    Args:
        url: Starting URL for crawling
        params: Crawling parameters including maxDepth, limit, etc.

    Returns:
        Dictionary containing crawl results with 'success', 'data', and 'status' keys

    Raises:
        ValueError: If crawling fails or configuration is invalid
    """
    try:
      # Validate URL
      parsed = urlparse(url)
      if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")

      # Extract crawling parameters
      max_depth = params.get("maxDepth", 1)
      limit = params.get("limit", 10)
      include_paths = params.get("includePaths", [])
      exclude_paths = params.get("excludePaths", [])
      scrape_options = params.get("scrapeOptions", {})

      # Start crawling process
      crawler = await self._get_crawler()
      await crawler.start()

      discovered_urls = await self._discover_urls(url, max_depth, limit, include_paths, exclude_paths)

      # Scrape discovered URLs
      scraped_data: List[Dict[str, Any]] = []
      for discovered_url in discovered_urls[:limit]:
        try:
          content = await self.scrape_url(discovered_url, scrape_options)
          if content.strip():
            scraped_data.append({
              "markdown": content,
              "metadata": {
                "url": discovered_url,
                "scrapeId": f"scrape_{len(scraped_data)}",
                "statusCode": 200,
              },
            })
        except Exception as e:
          print(f"Failed to scrape {discovered_url}: {str(e)}")
          continue

      return {
        "success": True,
        "data": scraped_data,
        "status": "completed",
        "total": len(scraped_data),
      }

    except Exception as e:
      return {
        "success": False,
        "error": str(e),
        "status": "failed",
      }

  async def _discover_urls(self, start_url: str, max_depth: int, limit: int, include_paths: List[str], exclude_paths: List[str]) -> List[str]:
    """
    Discover URLs for crawling using breadth-first search.

    Args:
        start_url: Starting URL
        max_depth: Maximum crawling depth
        limit: Maximum number of URLs to discover
        include_paths: Paths to include (if specified, only these are included)
        exclude_paths: Paths to exclude

    Returns:
        List of discovered URLs
    """
    discovered: set[str] = set()
    to_visit = [(start_url, 0)]  # (url, depth)
    visited = set()

    base_domain = urlparse(start_url).netloc

    while to_visit and len(discovered) < limit:
      current_url, depth = to_visit.pop(0)

      if current_url in visited or depth > max_depth:
        continue

      visited.add(current_url)

      # Extract links from current page
      try:
        crawler = await self._get_crawler()

        # Configure to extract links
        link_config = CrawlerRunConfig(
          word_count_threshold=1,
        )

        result = await crawler.arun(current_url, config=link_config)

        if result.success and result.links:
          for link_data in result.links:
            if isinstance(link_data, dict) and "href" in link_data:
              link_url = link_data["href"]
            else:
              link_url = str(link_data)

            # Resolve relative URLs
            absolute_url = urljoin(current_url, link_url)
            parsed_link = urlparse(absolute_url)

            # Filter URLs
            if (
              parsed_link.netloc == base_domain
              and absolute_url not in visited
              and self._should_include_url(absolute_url, include_paths, exclude_paths)
            ):
              discovered.add(absolute_url)
              if depth < max_depth:
                to_visit.append((absolute_url, depth + 1))

      except Exception as e:
        print(f"Failed to extract links from {current_url}: {str(e)}")
        continue

    return list(discovered)

  def _should_include_url(self, url: str, include_paths: List[str], exclude_paths: List[str]) -> bool:
    """Check if URL should be included based on path filters."""
    parsed = urlparse(url)
    path = parsed.path

    # Check exclude paths first
    for exclude_pattern in exclude_paths:
      if exclude_pattern and exclude_pattern in path:
        return False

    # If include paths specified, URL must match at least one
    if include_paths:
      return any(include_pattern and include_pattern in path for include_pattern in include_paths)

    return True

  async def map_url(self, url: str, settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
      # Validate URL
      parsed = urlparse(url)
      if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")

      settings = settings or {}
      # Note: include_subdomains and ignore_sitemap could be used for future enhancements
      _ = settings.get("includeSubdomains", True)
      _ = settings.get("ignoreSitemap", False)

      # Discover URLs using lightweight crawling
      discovered_urls = await self._discover_urls(
        url,
        max_depth=2,  # Limited depth for mapping
        limit=100,  # More URLs for mapping
        include_paths=[],
        exclude_paths=[],
      )

      # Create mapping results
      results = []
      for discovered_url in discovered_urls:
        results.append({
          "url": discovered_url,
          "status": "discovered",
          "content": f"URL discovered: {discovered_url}",  # Lightweight mapping
        })

      return results

    except Exception as e:
      raise ValueError(f"Failed to map URL {url}: {str(e)}")

  async def close(self):
    """Close the crawler and cleanup resources."""
    await self._close_crawler()

  async def __aenter__(self):
    """Async context manager entry."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit."""
    await self.close()


# Create singleton instance
crawl4ai = Crawl4AI()
