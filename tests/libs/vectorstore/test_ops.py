"""
Tests for the vectorstore module.

Note: These tests use mocking to test the business logic of the `create_vectorstore` 
function without requiring the pgvector extension to be installed. These tests verify:

1. Parameter validation and error handling
2. Collection naming conventions
3. API integration with langchain-postgres
4. Error handling for various scenarios

For full integration tests that validate actual vector operations with PostgreSQL,
see test_ops_integration.py (requires pgvector extension to be installed in the database).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from src.libs.vectorstore.v1.ops import create_vectorstore


class TestVectorstore:
    """Tests for the vectorstore operations."""
    
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    @patch("src.libs.vectorstore.v1.ops.async_session")
    async def test_create_vectorstore_success(self, mock_session, mock_embeddings, mock_pg_vector):
        """Test successful creation of a vectorstore."""
        # Set up mocks
        mock_vectorstore = AsyncMock()
        mock_vectorstore.acreate_collection = AsyncMock()
        mock_vectorstore.aget_collection = AsyncMock()
        
        mock_pg_vector.return_value = mock_vectorstore
        
        # Mock session context
        session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        
        # Mock collection with UUID
        collection = MagicMock()
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        collection.uuid = test_uuid
        mock_vectorstore.aget_collection.return_value = collection
        
        # Call the function
        result = await create_vectorstore("test_org", "test_collection")
        
        # Assertions
        assert result == test_uuid
        mock_pg_vector.assert_called_once()
        mock_embeddings.assert_called_once_with(model="text-embedding-3-large")
        mock_vectorstore.acreate_collection.assert_called_once()
        mock_vectorstore.aget_collection.assert_called_once_with(session_instance)
        
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    async def test_create_vectorstore_handles_pgvector_error(self, mock_embeddings, mock_pg_vector):
        """Test error handling when PGVector raises an exception."""
        # Set up mock to raise exception
        mock_pg_vector.side_effect = Exception("PGVector error")
        
        # Call the function and expect an exception
        with pytest.raises(Exception) as excinfo:
            await create_vectorstore("test_org", "test_collection")
        
        # Check the exception message includes our original error
        assert "PGVector error" in str(excinfo.value)
        assert "libs.pg_vector.create.create_vectorstore" in str(excinfo.value)
        
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    @patch("src.libs.vectorstore.v1.ops.async_session")
    async def test_create_vectorstore_handles_creation_error(self, mock_session, mock_embeddings, mock_pg_vector):
        """Test error handling when create_collection raises an exception."""
        # Set up mocks
        mock_vectorstore = AsyncMock()
        mock_vectorstore.acreate_collection.side_effect = Exception("Creation error")
        mock_pg_vector.return_value = mock_vectorstore
        
        # Call the function and expect an exception
        with pytest.raises(Exception) as excinfo:
            await create_vectorstore("test_org", "test_collection")
        
        # Check the exception message includes our original error
        assert "Creation error" in str(excinfo.value)
        assert "libs.pg_vector.create.create_vectorstore" in str(excinfo.value)
        
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    @patch("src.libs.vectorstore.v1.ops.async_session")
    async def test_create_vectorstore_handles_collection_error(self, mock_session, mock_embeddings, mock_pg_vector):
        """Test error handling when aget_collection raises an exception."""
        # Set up mocks
        mock_vectorstore = AsyncMock()
        mock_vectorstore.acreate_collection = AsyncMock()
        mock_vectorstore.aget_collection.side_effect = Exception("Collection error")
        mock_pg_vector.return_value = mock_vectorstore
        
        # Mock session context
        session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        
        # Call the function and expect an exception
        with pytest.raises(Exception) as excinfo:
            await create_vectorstore("test_org", "test_collection")
        
        # Check the exception message includes our original error
        assert "Collection error" in str(excinfo.value)
        assert "libs.pg_vector.create.create_vectorstore" in str(excinfo.value)
        
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    @patch("src.libs.vectorstore.v1.ops.async_session")
    async def test_create_vectorstore_collection_name_format(self, mock_session, mock_embeddings, mock_pg_vector):
        """Test that the collection name is formatted correctly."""
        # Set up mocks
        mock_vectorstore = AsyncMock()
        mock_vectorstore.acreate_collection = AsyncMock()
        mock_vectorstore.aget_collection = AsyncMock()
        
        mock_pg_vector.return_value = mock_vectorstore
        
        # Mock session context
        session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        
        # Mock collection with UUID
        collection = MagicMock()
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        collection.uuid = test_uuid
        mock_vectorstore.aget_collection.return_value = collection
        
        # Call the function with specific org and collection names
        org_name = "test_organization"
        collection_name = "test_vectors"
        await create_vectorstore(org_name, collection_name)
        
        # Assert that PGVector was called with the correctly formatted collection name
        expected_collection_name = f"{org_name}_{collection_name}"
        mock_pg_vector.assert_called_once()
        call_kwargs = mock_pg_vector.call_args.kwargs
        assert call_kwargs["collection_name"] == expected_collection_name
        
    @pytest.mark.asyncio
    @patch("src.libs.vectorstore.v1.ops.PGVector")
    @patch("src.libs.vectorstore.v1.ops.OpenAIEmbeddings")
    @patch("src.libs.vectorstore.v1.ops.async_session")
    async def test_create_vectorstore_embeddings_model(self, mock_session, mock_embeddings, mock_pg_vector):
        """Test that the correct embeddings model is used."""
        # Set up mocks
        mock_vectorstore = AsyncMock()
        mock_vectorstore.acreate_collection = AsyncMock()
        mock_vectorstore.aget_collection = AsyncMock()
        
        mock_pg_vector.return_value = mock_vectorstore
        
        # Mock session context
        session_instance = AsyncMock()
        mock_session.return_value.__aenter__.return_value = session_instance
        
        # Mock collection with UUID
        collection = MagicMock()
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        collection.uuid = test_uuid
        mock_vectorstore.aget_collection.return_value = collection
        
        # Call the function
        await create_vectorstore("test_org", "test_collection")
        
        # Assert that OpenAIEmbeddings was called with the correct model
        mock_embeddings.assert_called_once_with(model="text-embedding-3-large") 