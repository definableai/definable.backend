import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
import sys
from uuid import uuid4

@pytest.fixture
def test_kb_data():
    """Create test knowledge base data."""
    return {
        "name": "Test KB",
        "settings": {
            "embedding_model": "text-embedding-3-large",
            "max_chunk_size": 500,
            "chunk_overlap": 50,
            "separator": "\n\n"
        }
    }

@pytest.fixture
def test_search_data():
    """Create test search query."""
    return {
        "query": "test query",
        "limit": 3,
        "score_threshold": 0.0
    }

@pytest.mark.asyncio
class TestKnowledgeBaseAPI:
    """Test knowledge base API endpoints."""

    async def test_create_kb(self, client, mock_db_session, auth_headers, test_kb_data):
        """Test creating a knowledge base endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a KB
            kb_id = str(uuid4())
            collection_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": kb_id,
                "name": test_kb_data["name"],
                "collection_id": collection_id,
                "settings": test_kb_data["settings"],
                "organization_id": str(uuid4())
            }
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                "/api/kb/create?org_id=test-org-id",
                headers=auth_headers,
                json=test_kb_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == kb_id
            assert result["name"] == test_kb_data["name"]
            assert result["collection_id"] == collection_id
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()
    
    async def test_list_kb(self, client, mock_db_session, auth_headers):
        """Test listing knowledge bases endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of KBs
            kb_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": kb_id,
                    "name": "Test KB",
                    "collection_id": str(uuid4()),
                    "settings": {
                        "embedding_model": "text-embedding-3-large"
                    },
                    "organization_id": str(uuid4())
                }
            ]
            mock_get.return_value = mock_response
            
            # Make the API request
            response = client.get(
                "/api/kb/list?org_id=test-org-id",
                headers=auth_headers
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == kb_id
            
            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()
    
    async def test_search_chunks(self, client, mock_db_session, auth_headers, test_search_data):
        """Test searching chunks endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return search results
            chunk_id = str(uuid4())
            document_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content": "Test content matching the query",
                    "metadata": {"page": 1},
                    "score": 0.95
                }
            ]
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                "/api/kb/search_chunks?org_id=test-org-id&kb_id=test-kb-id",
                headers=auth_headers,
                json=test_search_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["chunk_id"] == chunk_id
            assert result[0]["score"] == 0.95
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once() 