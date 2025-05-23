import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.libs.firecrawl.v1.basic import Firecrawl, firecrawl


class TestFirecrawl:
    """Tests for the Firecrawl class."""

    def test_initialization(self):
        """Test that the Firecrawl class initializes correctly."""
        fc = Firecrawl()
        assert fc.base_url == "https://api.firecrawl.dev/v1"
        assert fc.token.startswith("Bearer ")
        assert fc.app is not None

    @patch('firecrawl.FirecrawlApp.map_url')
    def test_map_url(self, mock_map_url):
        """Test the map_url method."""
        # Setup mock
        expected_urls = ["https://example.com/page1", "https://example.com/page2"]
        mock_map_url.return_value = expected_urls

        # Execute
        test_url = "https://example.com"
        result = firecrawl.map_url(test_url)

        # Verify
        assert result == expected_urls
        mock_map_url.assert_called_once_with(test_url, {})

    @patch('firecrawl.FirecrawlApp.map_url')
    def test_map_url_with_settings(self, mock_map_url):
        """Test the map_url method with custom settings."""
        # Setup mock
        expected_urls = ["https://example.com/page1", "https://example.com/page2"]
        mock_map_url.return_value = expected_urls

        # Execute
        test_url = "https://example.com"
        test_settings = {"limit": 10, "exclude_paths": ["/admin", "/login"]}
        result = firecrawl.map_url(test_url, test_settings)

        # Verify
        assert result == expected_urls
        mock_map_url.assert_called_once_with(test_url, test_settings)

    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_success(self, mock_scrape_url):
        """Test successful scrape_url method."""
        # Setup mock
        expected_markdown = "# Test Page\n\nThis is a test page content."
        mock_scrape_url.return_value = {
            "markdown": expected_markdown,
            "metadata": {"statusCode": 200}
        }

        # Execute
        test_url = "https://example.com"
        result = firecrawl.scrape_url(test_url)

        # Verify
        assert result == expected_markdown
        mock_scrape_url.assert_called_once_with(test_url, {})

    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_failure(self, mock_scrape_url):
        """Test scrape_url method when the request fails."""
        # Setup mock
        mock_scrape_url.return_value = {
            "markdown": "",
            "metadata": {"statusCode": 404}
        }

        # Execute and verify
        test_url = "https://example.com/not-found"
        with pytest.raises(Exception) as excinfo:
            firecrawl.scrape_url(test_url)

        assert "Failed to scrape URL" in str(excinfo.value)
        mock_scrape_url.assert_called_once_with(test_url, {})

    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_with_settings(self, mock_scrape_url):
        """Test scrape_url method with custom settings."""
        # Setup mock
        expected_markdown = "# Test Page\n\nThis is a test page content."
        mock_scrape_url.return_value = {
            "markdown": expected_markdown,
            "metadata": {"statusCode": 200}
        }

        # Execute
        test_url = "https://example.com"
        test_settings = {"element_selectors": ["article", "main"]}
        result = firecrawl.scrape_url(test_url, test_settings)

        # Verify
        assert result == expected_markdown
        mock_scrape_url.assert_called_once_with(test_url, test_settings)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_success(self, mock_post):
        """Test successful crawl method."""
        # Setup mock
        expected_response = {
            "crawl_id": "test-crawl-id",
            "status": "started",
            "message": "Crawl started successfully"
        }
        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_post.return_value = mock_response

        # Execute
        test_url = "https://example.com"
        test_params = {
            "limit": 10,
            "exclude_paths": ["/admin"],
            "callbacks": {"on_document": "http://localhost:8000/callback"}
        }

        result = await firecrawl.crawl(test_url, test_params)

        # Verify
        assert result == expected_response
        mock_post.assert_called_once()
        # Check URL
        assert mock_post.call_args[0][0] == "https://api.firecrawl.dev/v1/crawl"
        # Check json payload includes URL and params
        payload = mock_post.call_args[1]["json"]
        assert payload["url"] == test_url
        assert payload["limit"] == 10
        assert payload["exclude_paths"] == ["/admin"]
        # Check headers
        assert mock_post.call_args[1]["headers"]["Authorization"].startswith("Bearer ")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_api_error(self, mock_post):
        """Test crawl method when the API returns an error."""
        # Setup mock to raise exception
        mock_post.side_effect = httpx.HTTPError("API error")

        # Execute and verify
        test_url = "https://example.com"
        test_params = {"limit": 10}

        with pytest.raises(httpx.HTTPError) as excinfo:
            await firecrawl.crawl(test_url, test_params)

        assert "API error" in str(excinfo.value)

    @pytest.mark.asyncio
    @patch('firecrawl.FirecrawlApp.check_crawl_status')
    async def test_get_crawl_status(self, mock_check_status):
        """Test get_crawl_status method."""
        # Setup mock
        expected_status = {
            "crawl_id": "test-crawl-id",
            "status": "completed",
            "urls_processed": 10,
            "urls_failed": 0
        }
        mock_check_status.return_value = expected_status

        # Execute
        test_crawl_id = "test-crawl-id"
        result = await firecrawl.get_crawl_status(test_crawl_id)

        # Verify
        assert result == expected_status
        mock_check_status.assert_called_once_with(test_crawl_id)