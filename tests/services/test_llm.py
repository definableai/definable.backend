import pytest
import pytest_asyncio
import sys
import os
import json
from uuid import UUID, uuid4
from datetime import datetime
from typing import Dict, Any, Optional, Union

from pydantic import BaseModel
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException, status

# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.

    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# Store original functions before patching
original_async_session = None
original_select = None

# Only patch if we're running unit tests, not integration tests
if not is_integration_test():
    # Save original functions
    from sqlalchemy.ext.asyncio import AsyncSession as OrigAsyncSession
    from sqlalchemy import select as orig_select
    original_async_session = OrigAsyncSession
    original_select = orig_select

    # Patch the necessary dependencies for testing
    patch('sqlalchemy.ext.asyncio.AsyncSession', MagicMock()).start()
    patch('sqlalchemy.select', lambda *args: MagicMock()).start()

# Import the real service and schema
from src.services.llm.service import LLMService
from src.services.llm.schema import LLMCreate, LLMResponse, LLMUpdate

# Only mock modules for unit tests
if not is_integration_test():
    # Create mock modules before any imports
    sys.modules["database"] = MagicMock()
    sys.modules["database.postgres"] = MagicMock()
    sys.modules["database.models"] = MagicMock()
    sys.modules["src.database"] = MagicMock()
    sys.modules["src.database.postgres"] = MagicMock()
    sys.modules["src.database.models"] = MagicMock()
    sys.modules["config"] = MagicMock()
    sys.modules["config.settings"] = MagicMock()
    sys.modules["src.config"] = MagicMock()
    sys.modules["src.config.settings"] = MagicMock()
    sys.modules["src.services.__base.acquire"] = MagicMock()
    sys.modules["dependencies.security"] = MagicMock()


# Pydantic models for type checking
class LLMConfigModel(BaseModel):
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None


class LLMModel(BaseModel):
  name: str
  provider: str
  version: str
  is_active: bool = True
  config: Optional[Dict[str, Any]] = None

  model_config = {"arbitrary_types_allowed": True}


# Mock models
class MockLLMModel:
    def __init__(self, model_id=None, name="gpt-4", provider="openai", version="v1",
                 is_active=True, config=None, props=None):
        self.id = model_id or uuid4()
        self.name = name
        self.provider = provider
        self.version = version
        self.is_active = is_active
        self.config = config or {"temperature": 0.7, "max_tokens": 2048}
        self.props = props or {}

    def __eq__(self, other):
        if not isinstance(other, MockLLMModel):
            return False
        return (
            self.id == other.id and
            self.name == other.name and
            self.provider == other.provider and
            self.version == other.version
        )


class MockResponse(BaseModel):
  id: Optional[UUID] = None
  name: Optional[str] = None
  provider: Optional[str] = None
  version: Optional[str] = None
  is_active: Optional[bool] = None
  config: Optional[Dict[str, Any]] = None
  props: Optional[Dict[str, Any]] = None
  created_at: Optional[Union[datetime, str]] = None
  updated_at: Optional[Union[datetime, str]] = None
  message: Optional[str] = None

  model_config = {"arbitrary_types_allowed": True, "extra": "allow"}


@pytest.fixture
def mock_user():
    """Create a mock user with proper permissions."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "org_id": uuid4(),
    }


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()

    # Create a properly structured mock result for database queries
    scalar_result = MagicMock()  # This will be returned by scalar_one_or_none

    scalars_result = MagicMock()  # This will be returned by scalars()
    all_result = MagicMock()      # This will be returned by all()
    scalars_result.all = MagicMock(return_value=all_result)

    execute_result = MagicMock()
    execute_result.scalar_one_or_none = MagicMock(return_value=scalar_result)
    execute_result.scalars = MagicMock(return_value=scalars_result)

    # Make execute return the mock result structure
    session.execute = AsyncMock(return_value=execute_result)

    return session


@pytest.fixture
def mock_llm_model():
    """Create a mock LLM model."""
    return MockLLMModel()


@pytest.fixture
def mock_multiple_llm_models():
    """Create multiple mock LLM models."""
    return [
        MockLLMModel(name="gpt-4", provider="openai", version="v1"),
        MockLLMModel(name="gemini-pro", provider="google", version="v1"),
        MockLLMModel(name="claude-3", provider="anthropic", version="sonnet")
    ]


@pytest.fixture
def mock_acquire():
    """Create a mock Acquire object."""
    acquire_mock = MagicMock()
    acquire_mock.logger = MagicMock()
    return acquire_mock


@pytest.fixture
def llm_service(mock_acquire):
    """Create the real LLM service with mocked dependencies."""
    return LLMService(acquire=mock_acquire)


@pytest.mark.asyncio
class TestLLMService:
    """Test LLM service with mocks."""

    async def test_post_add_success(self, llm_service, mock_db_session, mock_llm_model):
        """Test creating a new LLM model successfully."""
        # Setup
        model_data = LLMCreate(
            name="gpt-4",
            provider="openai",
            version="v1",
            is_active=True,
            config={"temperature": 0.7, "max_tokens": 2048},
            props={}
        )

        # Mock database query to return None (model doesn't exist yet)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Mock the refresh behavior to set the model attributes
        async def refresh_side_effect(model):
            model.id = mock_llm_model.id
        mock_db_session.refresh.side_effect = refresh_side_effect

        # Execute
        response = await llm_service.post_add(
            model_data=model_data,
            session=mock_db_session
        )

        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        assert mock_db_session.refresh.called

        assert isinstance(response, LLMResponse)
        assert response.name == model_data.name
        assert response.provider == model_data.provider
        assert response.version == model_data.version
        assert response.is_active == model_data.is_active
        assert response.config == model_data.config
        assert response.props == model_data.props

    async def test_post_add_already_exists(self, llm_service, mock_db_session, mock_llm_model):
        """Test creating a model that already exists."""
        # Setup
        model_data = LLMCreate(
            name="gpt-4",
            provider="openai",
            version="v1",
            is_active=True,
            config={"temperature": 0.7, "max_tokens": 2048},
            props={}
        )

        # Mock database query to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm_model

        # Execute and Assert
        with pytest.raises(HTTPException) as exc_info:
            await llm_service.post_add(
                model_data=model_data,
                session=mock_db_session
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Model already exists" in exc_info.value.detail

    async def test_post_update_success(self, llm_service, mock_db_session, mock_llm_model):
        """Test updating an existing LLM model."""
        # Setup
        model_id = mock_llm_model.id
        update_data = LLMUpdate(
            name="gpt-4-turbo",
            is_active=False,
            config={"temperature": 0.9},
            props={"updated": True}
        )

        # Mock database query to return the model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm_model

        # Execute
        response = await llm_service.post_update(
            model_id=model_id,
            model_data=update_data,
            session=mock_db_session
        )

        # Assert
        assert mock_db_session.commit.called
        assert mock_db_session.refresh.called

        # Check that model was updated
        assert mock_llm_model.name == update_data.name
        assert mock_llm_model.is_active == update_data.is_active
        assert mock_llm_model.config == update_data.config
        assert mock_llm_model.props == update_data.props

        # Check response
        assert isinstance(response, LLMResponse)
        assert response.name == update_data.name
        assert response.is_active == update_data.is_active
        assert response.config == update_data.config
        assert response.props == update_data.props

    async def test_post_update_not_found(self, llm_service, mock_db_session):
        """Test updating a non-existent model."""
        # Setup
        model_id = uuid4()
        update_data = LLMUpdate(name="new-name", props={})

        # Mock database query to return None
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Execute and Assert
        with pytest.raises(HTTPException) as exc_info:
            await llm_service.post_update(
                model_id=model_id,
                model_data=update_data,
                session=mock_db_session
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Model not found" in exc_info.value.detail

    async def test_delete_remove_success(self, llm_service, mock_db_session, mock_llm_model):
        """Test deleting an LLM model."""
        # Setup
        model_id = mock_llm_model.id

        # Mock database query to return the model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm_model

        # Execute
        response = await llm_service.delete_remove(
            model_id=model_id,
            session=mock_db_session
        )

        # Assert
        assert mock_db_session.delete.called
        assert mock_db_session.commit.called

        assert isinstance(response, dict)
        assert response["message"] == "Model deleted successfully"

    async def test_delete_remove_not_found(self, llm_service, mock_db_session):
        """Test deleting a non-existent model."""
        # Setup
        model_id = uuid4()

        # Mock database query to return None
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Execute and Assert
        with pytest.raises(HTTPException) as exc_info:
            await llm_service.delete_remove(
                model_id=model_id,
                session=mock_db_session
            )

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Model not found" in exc_info.value.detail

    async def test_get_list(self, llm_service, mock_db_session, mock_user, mock_multiple_llm_models):
        """Test listing all LLM models."""
        # Setup
        org_id = mock_user["org_id"]

        # Mock database query to return models
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_multiple_llm_models

        # Execute
        response = await llm_service.get_list(
            org_id=org_id,
            session=mock_db_session,
            user=mock_user
        )

        # Assert
        assert isinstance(response, list)
        assert len(response) == len(mock_multiple_llm_models)
        assert all(isinstance(item, LLMResponse) for item in response)

        # Check model names are present in the response
        model_names = [model.name for model in response]
        assert "gpt-4" in model_names
        assert "gemini-pro" in model_names
        assert "claude-3" in model_names


# ============================================================================
# INTEGRATION TESTS - RUN WITH: INTEGRATION_TEST=1 pytest tests/services/test_llm.py
# ============================================================================

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select
    from models import LLMModel


@pytest.fixture
def test_model_data():
    """Create test model data for integration tests."""
    return [
        {
            "name": "gpt-4-integration",
            "provider": "openai",
            "version": "v1",
            "is_active": True,
            "config": {"temperature": 0.7, "max_tokens": 2048},
            "props": {"integration_test": True}
        },
        {
            "name": "claude-integration",
            "provider": "anthropic",
            "version": "sonnet",
            "is_active": True,
            "config": {"temperature": 0.5, "max_tokens": 4096},
            "props": {"integration_test": True}
        }
    ]


@pytest_asyncio.fixture
async def db_integration_setup(setup_test_db, db_session, test_model_data):
    """Setup the database with test model data for integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    created_models = []
    session = None

    try:
        # Get session from the generator without exhausting it
        session_gen = db_session.__aiter__()
        session = await session_gen.__anext__()

        # Import text here to ensure it's available
        from sqlalchemy import text

        # Create test tables if they don't exist
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS llm_models (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                provider VARCHAR(255) NOT NULL,
                version VARCHAR(100) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                config JSONB,
                props JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, provider, version)
            )
        """))

        # Commit table creation
        await session.commit()

        # Clean existing test data
        await session.execute(text("DELETE FROM llm_models WHERE name LIKE '%integration%'"))
        await session.commit()

        # Add new test data
        for model_data in test_model_data:
            # Create models directly using SQL to avoid ORM issues
            await session.execute(
                text("""
                    INSERT INTO llm_models (name, provider, version, is_active, config, props)
                    VALUES (:name, :provider, :version, :is_active, :config, :props)
                    RETURNING id
                """),
                {
                    "name": model_data["name"],
                    "provider": model_data["provider"],
                    "version": model_data["version"],
                    "is_active": model_data["is_active"],
                    "config": json.dumps(model_data["config"]),
                    "props": json.dumps(model_data["props"])
                }
            )

        await session.commit()

        # Query for created models to return them
        result = await session.execute(
            text("SELECT * FROM llm_models WHERE name LIKE '%integration%'")
        )
        # fetchall() already returns a list in SQLAlchemy 2.0 with asyncpg, no need to await
        rows = result.fetchall()

        # Convert rows to model objects
        for row in rows:
            model = MockLLMModel(
                model_id=row.id,
                name=row.name,
                provider=row.provider,
                version=row.version,
                is_active=row.is_active,
                config=row.config,
                props=row.props
            )
            created_models.append(model)

    except Exception as e:
        if session:
            await session.rollback()
        print(f"Error in db_integration_setup: {e}")
        raise
    finally:
        # Explicitly close the session we used for setup
        if session:
            await session.close()

    # Return the created models and session generator to the test
    yield {
        "models": created_models,
        "db_session": db_session  # Return the session generator for test use
    }

    # Clean up after tests
    try:
        # Get a new session for cleanup
        cleanup_session = None
        async for cleanup_session in db_session:
            try:
                await cleanup_session.execute(text("DELETE FROM llm_models WHERE name LIKE '%integration%'"))
                await cleanup_session.commit()
                break  # Only process the first yielded session
            except Exception as e:
                print(f"Error in cleanup: {e}")
                await cleanup_session.rollback()
            finally:
                # Always close the cleanup session
                if cleanup_session:
                    await cleanup_session.close()
    except Exception as e:
        print(f"Could not acquire session for cleanup: {e}")


@pytest.mark.asyncio
class TestLLMServiceIntegration:
    """Integration tests for LLM service with real database."""

    # Apply skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_get_list_integration(self, llm_service, db_integration_setup, mock_user):
        """Test listing all LLM models from the database."""
        org_id = mock_user["org_id"]
        db_session = db_integration_setup["db_session"]

        # Get a new session
        session = None
        async for s in db_session:
            session = s
            try:
                # Execute
                response = await llm_service.get_list(
                    org_id=org_id,
                    session=session,
                    user=mock_user
                )

                # Assert
                assert isinstance(response, list)
                assert len(response) == 2
                assert all(isinstance(item, LLMResponse) for item in response)

                # Check model data
                model_names = [model.name for model in response]
                assert "gpt-4-integration" in model_names
                assert "claude-integration" in model_names
                break
            except Exception as e:
                print(f"Error in test_get_list_integration: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()

    async def test_post_add_integration(self, llm_service, db_integration_setup):
        """Test adding a new LLM model to the database."""
        # Setup
        model_data = LLMCreate(
            name="gemini-pro-integration",
            provider="google",
            version="v1",
            is_active=True,
            config={"temperature": 0.8, "max_tokens": 3072},
            props={"integration_test": True}
        )
        db_session = db_integration_setup["db_session"]

        # Get a new session
        session = None
        async for s in db_session:
            session = s
            try:
                # Execute
                response = await llm_service.post_add(
                    model_data=model_data,
                    session=session
                )

                # Assert
                assert isinstance(response, LLMResponse)
                assert response.name == model_data.name
                assert response.provider == model_data.provider
                assert response.config == model_data.config
                assert response.props == model_data.props
                assert response.id is not None

                # Verify in database
                query = select(LLMModel).where(LLMModel.name == model_data.name)
                result = await session.execute(query)
                db_model = result.scalar_one_or_none()
                assert db_model is not None
                assert db_model.name == model_data.name
                break
            except Exception as e:
                print(f"Error in test_post_add_integration: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()

    async def test_post_update_integration(self, llm_service, db_integration_setup):
        """Test updating an LLM model in the database."""
        # Setup - get the ID from a real model
        model_id = db_integration_setup["models"][0].id
        update_data = LLMUpdate(
            name="gpt-4-updated",
            is_active=False,
            config={"temperature": 0.3},
            props={"updated": True, "integration_test": True}
        )
        db_session = db_integration_setup["db_session"]

        # Get a new session
        session = None
        async for s in db_session:
            session = s
            try:
                # Execute
                response = await llm_service.post_update(
                    model_id=model_id,
                    model_data=update_data,
                    session=session
                )

                # Assert
                assert isinstance(response, LLMResponse)
                assert response.name == update_data.name
                assert response.is_active == update_data.is_active
                assert response.config["temperature"] == update_data.config["temperature"]
                assert response.props == update_data.props

                # Verify in database
                query = select(LLMModel).where(LLMModel.id == model_id)
                result = await session.execute(query)
                db_model = result.scalar_one_or_none()
                assert db_model is not None
                assert db_model.name == update_data.name
                assert db_model.is_active == update_data.is_active
                assert db_model.props == update_data.props
                break
            except Exception as e:
                print(f"Error in test_post_update_integration: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()

    async def test_delete_remove_integration(self, llm_service, db_integration_setup):
        """Test deleting an LLM model from the database."""
        # Setup - get the ID from a real model
        model_id = db_integration_setup["models"][0].id
        db_session = db_integration_setup["db_session"]

        # Get a new session
        session = None
        async for s in db_session:
            session = s
            try:
                # Execute
                response = await llm_service.delete_remove(
                    model_id=model_id,
                    session=session
                )

                # Assert
                assert isinstance(response, LLMResponse)
                assert response.id == model_id

                # Verify in database that it's gone
                query = select(LLMModel).where(LLMModel.id == model_id)
                result = await session.execute(query)
                db_model = result.scalar_one_or_none()
                assert db_model is None
                break
            except Exception as e:
                print(f"Error in test_delete_remove_integration: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()


@pytest.mark.asyncio
class TestLLMServiceErrorHandling:
    """Tests for error handling in the LLM service."""

    # Apply skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_add_invalid_model_data(self, llm_service):
        """Test handling of invalid model data."""
        # Create invalid model data (missing required fields)
        from pydantic import ValidationError

        # Test with missing required field (provider)
        with pytest.raises(ValidationError):
            LLMCreate(
                name="invalid-model",
                # provider is missing
                version="v1",
                config={},
                props={}
            )

        # Test with invalid field type
        with pytest.raises(ValidationError):
            LLMCreate(
                name="invalid-model",
                provider="openai",
                version="v1",
                is_active="not-a-boolean",  # Should be boolean
                config={},
                props={}
            )

    async def test_db_transaction_rollback(self, llm_service, db_session, monkeypatch):
        """Test that transactions roll back properly on error."""
        # Get a new session
        session = None
        async for s in db_session:
            session = s
            try:
                # First create the llm_models table if it doesn't exist
                from sqlalchemy import text
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS llm_models (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        provider VARCHAR(255) NOT NULL,
                        version VARCHAR(100) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        config JSONB,
                        props JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, provider, version)
                    )
                """))
                await session.commit()

                # First make sure the test record doesn't exist
                await session.execute(text("""
                    DELETE FROM llm_models WHERE name = 'error-test-model'
                    AND provider = 'test-provider' AND version = 'v1'
                """))
                await session.commit()

                # Setup test data
                model_data = LLMCreate(
                    name="error-test-model",
                    provider="test-provider",
                    version="v1",
                    is_active=True,
                    config={"temperature": 0.7},
                    props={"test": True}
                )

                # Create a function that simulates a database error during transaction
                async def simulate_error():
                    # Start a transaction
                    transaction = await session.begin()
                    try:
                        # Insert record
                        await session.execute(
                            text("""
                                INSERT INTO llm_models (name, provider, version, is_active, config, props)
                                VALUES (:name, :provider, :version, :is_active, :config, :props)
                            """),
                            {
                                "name": model_data.name,
                                "provider": model_data.provider,
                                "version": model_data.version,
                                "is_active": model_data.is_active,
                                "config": json.dumps(model_data.config or {}),
                                "props": json.dumps(model_data.props or {})
                            }
                        )

                        # Verify the record exists within the transaction
                        check_result = await session.execute(
                            text("SELECT COUNT(*) FROM llm_models WHERE name = :name AND provider = :provider AND version = :version"),
                            {"name": model_data.name, "provider": model_data.provider, "version": model_data.version}
                        )
                        count = check_result.scalar()
                        assert count == 1, "Record should exist before rollback"

                        # Now simulate an error that should trigger rollback
                        raise Exception("Simulated database error")
                    except Exception as e:
                        # Explicitly rollback the transaction
                        await transaction.rollback()
                        raise e

                # Execute and expect an exception
                with pytest.raises(Exception) as exc_info:
                    await simulate_error()

                # Verify the error message
                assert "Simulated database error" in str(exc_info.value)

                # Verify the model was not added (transaction rolled back)
                query = text("SELECT COUNT(*) FROM llm_models WHERE name = :name AND provider = :provider AND version = :version")
                result = await session.execute(query, {
                    "name": model_data.name,
                    "provider": model_data.provider,
                    "version": model_data.version
                })
                count = result.scalar()
                assert count == 0, "Record should not exist after rollback"
                break
            except Exception as e:
                print(f"Error in test_db_transaction_rollback: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()


@pytest.mark.asyncio
class TestLLMServicePerformance:
    """Performance tests for the LLM service."""

    # Apply skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Performance tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_bulk_model_creation(self, llm_service, db_session):
        """Test creating multiple LLM models in bulk."""
        session = None
        async for s in db_session:
            session = s
            try:
                # First create the llm_models table if it doesn't exist
                from sqlalchemy import text
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS llm_models (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        provider VARCHAR(255) NOT NULL,
                        version VARCHAR(100) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        config JSONB,
                        props JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, provider, version)
                    )
                """))
                await session.commit()

                # Clean any existing test data
                await session.execute(text("DELETE FROM llm_models WHERE provider = 'perf-test'"))
                await session.commit()

                # Create 5 test models with different names
                model_count = 5
                created_models = []

                for i in range(model_count):
                    # Insert directly with SQL to avoid ORM issues
                    result = await session.execute(
                        text("""
                            INSERT INTO llm_models (name, provider, version, is_active, config, props)
                            VALUES (:name, :provider, :version, :is_active, :config, :props)
                            RETURNING id, name, provider, version, is_active, config, props
                        """),
                        {
                            "name": f"perf-test-model-{i}",
                            "provider": "perf-test",
                            "version": f"v{i}",
                            "is_active": True,
                            "config": json.dumps({"temperature": 0.7 + (i * 0.1)}),
                            "props": json.dumps({"perf_test": True})
                        }
                    )
                    model_row = result.fetchone()
                    created_models.append(model_row)

                # Commit after inserting all models
                await session.commit()

                # Verify all models were created
                assert len(created_models) == model_count

                # Verify all models exist in the database
                result = await session.execute(
                    text("SELECT * FROM llm_models WHERE provider = 'perf-test'")
                )
                db_models = result.fetchall()
                assert len(db_models) == model_count

                # Clean up test data
                await session.execute(text("DELETE FROM llm_models WHERE provider = 'perf-test'"))
                await session.commit()
                break
            except Exception as e:
                print(f"Error in test_bulk_model_creation: {e}")
                raise
            finally:
                # Ensure session is properly closed
                if session:
                    await session.close()
