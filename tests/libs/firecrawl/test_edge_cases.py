"""
Edge case tests for the firecrawl library.

These tests validate the firecrawl library's behavior under various edge cases
and error conditions.
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.libs.firecrawl.v1.basic import firecrawl


class TestFirecrawlEdgeCases:
    """Edge case tests for the Firecrawl class."""
    
    @patch('firecrawl.FirecrawlApp.map_url')
    def test_map_url_empty_result(self, mock_map_url):
        """Test map_url when it returns an empty list of URLs."""
        # Setup mock
        mock_map_url.return_value = []
        
        # Execute
        test_url = "https://example.com/no-links"
        result = firecrawl.map_url(test_url)
        
        # Verify
        assert isinstance(result, list)
        assert len(result) == 0
    
    @patch('firecrawl.FirecrawlApp.map_url')
    def test_map_url_with_invalid_url(self, mock_map_url):
        """Test map_url with an invalid URL format."""
        # Setup mock to raise exception
        mock_map_url.side_effect = ValueError("Invalid URL format")
        
        # Execute and verify
        with pytest.raises(ValueError) as excinfo:
            firecrawl.map_url("not-a-valid-url")
        
        assert "Invalid URL format" in str(excinfo.value)
    
    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_empty_content(self, mock_scrape_url):
        """Test scrape_url when it returns empty content but successful status."""
        # Setup mock
        mock_scrape_url.return_value = {
            "markdown": "",
            "metadata": {"statusCode": 200}
        }
        
        # Execute
        test_url = "https://example.com/empty"
        result = firecrawl.scrape_url(test_url)
        
        # Verify
        assert result == ""
    
    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_server_error(self, mock_scrape_url):
        """Test scrape_url when the server returns a 5xx error."""
        # Setup mock
        mock_scrape_url.return_value = {
            "markdown": "",
            "metadata": {"statusCode": 500}
        }
        
        # Execute and verify
        test_url = "https://example.com/server-error"
        with pytest.raises(Exception) as excinfo:
            firecrawl.scrape_url(test_url)
        
        assert "Failed to scrape URL" in str(excinfo.value)
    
    @patch('firecrawl.FirecrawlApp.scrape_url')
    def test_scrape_url_missing_metadata(self, mock_scrape_url):
        """Test scrape_url when the response is missing metadata."""
        # Setup mock with incomplete response
        mock_scrape_url.return_value = {
            "markdown": "Some content"
            # Missing metadata
        }
        
        # Execute and verify
        test_url = "https://example.com"
        with pytest.raises(Exception) as excinfo:
            firecrawl.scrape_url(test_url)
        
        # Should raise KeyError when trying to access missing metadata
        assert "metadata" in str(excinfo.value).lower() or "key" in str(excinfo.value).lower()
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_with_malformed_response(self, mock_post):
        """Test crawl when API returns malformed JSON."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Execute and verify
        test_url = "https://example.com"
        test_params = {"limit": 10}
        
        with pytest.raises(ValueError) as excinfo:
            await firecrawl.crawl(test_url, test_params)
        
        assert "Invalid JSON" in str(excinfo.value)
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_timeout(self, mock_post):
        """Test crawl when request times out."""
        # Setup mock
        mock_post.side_effect = httpx.TimeoutException("Request timed out")
        
        # Execute and verify
        test_url = "https://example.com"
        test_params = {"limit": 10}
        
        with pytest.raises(httpx.TimeoutException) as excinfo:
            await firecrawl.crawl(test_url, test_params)
        
        assert "timed out" in str(excinfo.value).lower()
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_connection_error(self, mock_post):
        """Test crawl when connection fails."""
        # Setup mock
        mock_post.side_effect = httpx.ConnectError("Connection failed")
        
        # Execute and verify
        test_url = "https://example.com"
        test_params = {"limit": 10}
        
        with pytest.raises(httpx.ConnectError) as excinfo:
            await firecrawl.crawl(test_url, test_params)
        
        assert "connection" in str(excinfo.value).lower()
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_crawl_http_403(self, mock_post):
        """Test crawl when API returns 403 Forbidden."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden", 
            request=MagicMock(), 
            response=MagicMock(status_code=403)
        )
        mock_post.return_value = mock_response
        
        # Execute and verify
        test_url = "https://example.com"
        test_params = {"limit": 10}
        
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await firecrawl.crawl(test_url, test_params)
        
        assert "403" in str(excinfo.value)
    
    @pytest.mark.asyncio
    @patch('firecrawl.FirecrawlApp.check_crawl_status')
    async def test_get_crawl_status_not_found(self, mock_check_status):
        """Test get_crawl_status when crawl ID is not found."""
        # Setup mock
        mock_check_status.side_effect = ValueError("Crawl ID not found")
        
        # Execute and verify
        with pytest.raises(ValueError) as excinfo:
            await firecrawl.get_crawl_status("nonexistent-id")
        
        assert "not found" in str(excinfo.value).lower()
    
    @pytest.mark.parametrize("invalid_input,expected_error", [
        (None, TypeError),
        ("", ValueError),
        ("   ", ValueError),
        (123, TypeError),
        ({"url": "https://example.com"}, TypeError),
    ])
    @patch('firecrawl.FirecrawlApp.map_url')
    def test_map_url_invalid_inputs(self, mock_map_url, invalid_input, expected_error):
        """Test map_url with various invalid inputs."""
        # Mock the FirecrawlApp to validate input types
        if invalid_input is None:
            mock_map_url.side_effect = TypeError("Argument 'url' must be a string, not NoneType")
        elif invalid_input == "":
            mock_map_url.side_effect = ValueError("URL cannot be empty")
        elif invalid_input == "   ":
            mock_map_url.side_effect = ValueError("URL contains only whitespace")
        elif isinstance(invalid_input, int):
            mock_map_url.side_effect = TypeError(f"Argument 'url' must be a string, not {type(invalid_input).__name__}")
        elif isinstance(invalid_input, dict):
            mock_map_url.side_effect = TypeError(f"Argument 'url' must be a string, not {type(invalid_input).__name__}")
            
        # Execute and verify
        with pytest.raises(expected_error):
            firecrawl.map_url(invalid_input) 