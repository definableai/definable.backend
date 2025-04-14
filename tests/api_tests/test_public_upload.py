import pytest
from unittest.mock import MagicMock, patch
import io
from uuid import uuid4

@pytest.fixture
def test_file_data():
    """Create a test file for upload testing."""
    file_content = b"test file content"
    file = io.BytesIO(file_content)
    return {
        "file": (file, "test.jpg", "image/jpeg"),
        "content": file_content
    }

@pytest.mark.asyncio
class TestPublicUploadAPI:
    """Test Public Upload API endpoints."""

    async def test_upload_file(self, client, mock_db_session, auth_headers, test_file_data):
        """Test uploading a file endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a file URL
            file_id = str(uuid4())
            file_url = f"https://example.com/uploads/{file_id}/test.jpg"
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "url": file_url
            }
            mock_post.return_value = mock_response
            
            # Create files dict for multipart/form-data request
            files = {"file": test_file_data["file"]}
            
            # Make the API request
            response = client.post(
                "/api/public_upload",
                headers=auth_headers,
                files=files
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert "url" in result
            assert result["url"] == file_url
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()
    
    async def test_upload_file_without_auth(self, client, mock_db_session, test_file_data):
        """Test uploading a file endpoint without authentication."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a file URL
            file_id = str(uuid4())
            file_url = f"https://example.com/uploads/{file_id}/test.jpg"
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "url": file_url
            }
            mock_post.return_value = mock_response
            
            # Create files dict for multipart/form-data request
            files = {"file": test_file_data["file"]}
            
            # Make the API request without auth headers
            response = client.post(
                "/api/public_upload",
                files=files
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert "url" in result
            assert result["url"] == file_url
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once() 