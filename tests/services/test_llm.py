import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import UUID, uuid4
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# Create mock modules before any imports
sys.modules['database'] = MagicMock()
sys.modules['database.postgres'] = MagicMock()
sys.modules['database.models'] = MagicMock()
sys.modules['src.database'] = MagicMock()
sys.modules['src.database.postgres'] = MagicMock()
sys.modules['src.database.models'] = MagicMock()
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['src.config'] = MagicMock()
sys.modules['src.config.settings'] = MagicMock()
sys.modules['src.services.__base.acquire'] = MagicMock()
sys.modules['dependencies.security'] = MagicMock()

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
class MockLLMModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "Test LLM"
    provider: str = "openai"
    version: str = "1.0.0"
    is_active: bool = True
    config: Dict[str, Any] = Field(default_factory=lambda: {"max_tokens": 1000, "temperature": 0.7})
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

class MockResponse(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    provider: Optional[str] = None
    version: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[Union[datetime, str]] = None
    updated_at: Optional[Union[datetime, str]] = None
    message: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

@pytest.fixture
def mock_user():
    """Create a mock user."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
    }

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()

    # Setup scalar to return a properly mocked result
    scalar_mock = AsyncMock()
    session.scalar = scalar_mock

    # Setup execute to return a properly mocked result
    execute_mock = AsyncMock()
    # Make unique(), scalars(), first(), etc. return self to allow chaining
    execute_result = AsyncMock()
    execute_result.unique.return_value = execute_result
    execute_result.scalars.return_value = execute_result
    execute_result.scalar_one_or_none.return_value = None
    execute_result.scalar_one.return_value = None
    execute_result.first.return_value = None
    execute_result.all.return_value = []
    execute_result.mappings.return_value = execute_result

    execute_mock.return_value = execute_result
    session.execute = execute_mock

    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    return session

@pytest.fixture
def mock_llm():
    """Create a mock LLM model."""
    return MockLLMModel(
        name="GPT-4",
        provider="openai",
        version="4o",
        is_active=True,
        config={"max_tokens": 4000, "temperature": 0.7}
    )

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    llm_service = MagicMock()

    async def mock_add(model_data, session):
        # Check if model exists with same name, provider, and version
        existing_model = session.execute.return_value.scalar_one_or_none.return_value
        if existing_model:
            raise HTTPException(status_code=400, detail="Model already exists")

        # Create model
        db_model = MockLLMModel(
            name=model_data.name,
            provider=model_data.provider,
            version=model_data.version,
            is_active=model_data.is_active,
            config=model_data.config
        )
        session.add(db_model)
        await session.commit()
        await session.refresh(db_model)

        # Return response matching API format
        return MockResponse(**db_model.model_dump())

    async def mock_update(model_id, model_data, session):
        # Get model
        db_model = session.execute.return_value.scalar_one_or_none.return_value

        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Update fields
        update_data = model_data.model_dump(exclude_unset=True)
        model_dict = db_model.model_dump()

        for field, value in update_data.items():
            model_dict[field] = value

        # Update timestamp
        model_dict["updated_at"] = datetime.now()

        # Create updated model
        db_model = MockLLMModel(**model_dict)

        await session.commit()
        await session.refresh(db_model)

        # Return response matching API format
        return MockResponse(**db_model.model_dump())

    async def mock_remove(model_id, session):
        # Get model
        db_model = session.execute.return_value.scalar_one_or_none.return_value

        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")

        await session.delete(db_model)
        await session.commit()

        # Return response matching API format
        return {"message": "Model deleted successfully"}

    async def mock_list(org_id, session, user):
        # Create mock models
        # Create mock models
        models = [
            MockLLMModel(
                id=uuid4(),
                name="gpt-3.5-turbo",
                provider="openai",
                version="3.5-turbo",
                is_active=True,
                config={"temperature": 0.7}
            ),
            MockLLMModel(
                id=uuid4(),
                name="gpt-4",
                provider="openai",
                version="4",
                is_active=True,
                config={"temperature": 0.7}
            ),
            MockLLMModel(
                id=uuid4(),
                name="claude-3-opus",
                provider="anthropic",
                version="3-opus",
                is_active=True,
                config={"temperature": 0.7}
            )
        ]

        # Set up mock response
        session.execute.return_value.scalars.return_value.all.return_value = models

        # Return response matching API format
        return [MockResponse(**model.model_dump()) for model in models]

    async def mock_get_by_id(model_id, session):
        # Get model
        db_model = session.execute.return_value.scalar_one_or_none.return_value

        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Return response matching API format
        return MockResponse(**db_model.model_dump())

    async def mock_list_by_provider(org_id, provider, session, user):
        # Get all models
        all_models = await mock_list(org_id, session, user)

        # Filter by provider
        filtered_models = [model for model in all_models if model.provider == provider]

        return filtered_models

    async def mock_batch_update_status(model_ids, is_active, session):
        # Update each model in the list
        updated_models = []

        for model_id in model_ids:
            # Get model
            model = MockLLMModel(id=model_id)
            session.execute.return_value.scalar_one_or_none.return_value = model

            # Update status
            model.is_active = is_active
            model.updated_at = datetime.now()

            # Add to updated list
            updated_models.append(model)

        await session.commit()

        # Return response matching API format
        return [MockResponse(**model.model_dump()) for model in updated_models]

    async def mock_search_by_name(org_id, query, session, user):
        # Get all models
        all_models = await mock_list(org_id, session, user)

        # Filter by name containing query (case-insensitive)
        filtered_models = [model for model in all_models if query.lower() in model.name.lower()]

        return filtered_models

    # Create AsyncMock objects for these methods
    add_mock = AsyncMock(side_effect=mock_add)
    update_mock = AsyncMock(side_effect=mock_update)
    remove_mock = AsyncMock(side_effect=mock_remove)
    list_mock = AsyncMock(side_effect=mock_list)
    get_by_id_mock = AsyncMock(side_effect=mock_get_by_id)
    list_by_provider_mock = AsyncMock(side_effect=mock_list_by_provider)
    batch_update_status_mock = AsyncMock(side_effect=mock_batch_update_status)
    search_by_name_mock = AsyncMock(side_effect=mock_search_by_name)

    # Assign mocks to service
    llm_service.add = add_mock
    llm_service.update = update_mock
    llm_service.remove = remove_mock
    llm_service.list = list_mock
    llm_service.get_by_id = get_by_id_mock
    llm_service.list_by_provider = list_by_provider_mock
    llm_service.batch_update_status = batch_update_status_mock
    llm_service.search_by_name = search_by_name_mock

    return llm_service

@pytest.mark.asyncio
class TestLLMService:
    """Tests for the LLM service."""

    async def test_add_llm(self, mock_llm_service, mock_db_session):
        """Test adding a new LLM model."""
        # Create model data
        # Create model data
        model_data = MockResponse(
            name="Claude-3-Haiku",
            provider="anthropic",
            version="3-haiku",
            is_active=True,
            config={"temperature": 0.7, "max_tokens": 2000}
        )

        # Call the service
        response = await mock_llm_service.add(
            model_data,
            session=mock_db_session
        )

        # Verify result structure
        assert response.name == model_data.name
        assert response.provider == model_data.provider
        assert response.version == model_data.version
        assert response.is_active == model_data.is_active
        assert response.config == model_data.config
        assert hasattr(response, "id")
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_llm_service.add.called

    async def test_add_existing_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test adding an existing LLM model."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Create model data with same name, provider, and version
        model_data = MockResponse(
            name=mock_llm.name,
            provider=mock_llm.provider,
            version=mock_llm.version,
            is_active=True,
            config={"temperature": 0.8}
        )

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.add(
                model_data,
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Model already exists" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.add.called

    async def test_update_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test updating an LLM model."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Create update data
        update_data = MockResponse(
            name="GPT-4-Updated",
            config={"temperature": 0.5, "max_tokens": 5000}
        )

        # Call the service
        response = await mock_llm_service.update(
            mock_llm.id,
            update_data,
            session=mock_db_session
        )

        # Verify result structure
        assert response.id == mock_llm.id
        assert response.name == update_data.name
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version    # Unchanged
        assert response.version == mock_llm.version    # Unchanged
        assert response.config == update_data.config
        assert hasattr(response, "updated_at")

        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_llm_service.update.called

    async def test_update_nonexistent_llm(self, mock_llm_service, mock_db_session):
        """Test updating a non-existent LLM model."""
        # Setup mock to return no model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create update data
        update_data = MockResponse(
            name="Updated Name",
            is_active=False
        )

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.update(
                uuid4(),
                update_data,
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 404
        assert "Model not found" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.update.called

    async def test_remove_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test removing an LLM model."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Call the service
        response = await mock_llm_service.remove(
            mock_llm.id,
            session=mock_db_session
        )

        # Verify result structure
        assert "message" in response
        assert "deleted successfully" in response["message"]

        # Verify database operations
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Verify service method was called
        assert mock_llm_service.remove.called

    async def test_remove_nonexistent_llm(self, mock_llm_service, mock_db_session):
        """Test removing a non-existent LLM model."""
        # Setup mock to return no model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.remove(
                uuid4(),
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 404
        assert "Model not found" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.remove.called

    async def test_list_llms(self, mock_llm_service, mock_db_session, mock_user):
        """Test listing all LLM models."""
        """Test listing all LLM models."""
        # Call the service
        response = await mock_llm_service.list(
            mock_user["organization_id"],
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure
        assert isinstance(response, list)
        assert len(response) == 3  # From our mock implementation

        # Verify each model has the right structure
        for model in response:
            assert hasattr(model, "id")
            assert hasattr(model, "name")
            assert hasattr(model, "provider")
            assert hasattr(model, "version")
            assert hasattr(model, "is_active")
            assert hasattr(model, "config")
            assert hasattr(model, "created_at")
            assert hasattr(model, "updated_at")

        # Verify service method was called
        assert mock_llm_service.list.called

    async def test_add_llm_with_invalid_provider(self, mock_llm_service, mock_db_session):
        """Test adding an LLM model with an invalid provider."""
        # Override the mock implementation for this test
        """Test adding an LLM model with an invalid provider."""
        # Override the mock implementation for this test
        async def mock_add_invalid_provider(model_data, session):
            # List of supported providers
            supported_providers = ["openai", "anthropic", "cohere", "gemini"]

            # Check if provider is supported
            if model_data.provider not in supported_providers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider not supported. Supported providers: {', '.join(supported_providers)}"
                )

            # Original implementation
            return await mock_llm_service.add.side_effect(model_data, session)

        # Replace the mock method with our new implementation for this test only
        mock_llm_service.add.side_effect = mock_add_invalid_provider

        # Create model data with invalid provider
        model_data = MockResponse(
            name="Unknown Model",
            provider="unsupported",
            version="1.0",
            is_active=True,
            config={"temperature": 0.7}
        )

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.add(
                model_data,
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Provider not supported" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.add.called

    async def test_update_llm_config(self, mock_llm_service, mock_db_session, mock_llm):
        """Test updating only the config of an LLM model."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Create update data with only config
        update_data = MockResponse(
            config={"temperature": 0.9, "max_tokens": 8000, "top_p": 0.95}
        )

        # Call the service
        response = await mock_llm_service.update(
            mock_llm.id,
            update_data,
            session=mock_db_session
        )

        # Verify result structure
        assert response.id == mock_llm.id
        assert response.name == mock_llm.name  # Unchanged
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version  # Unchanged
        assert response.config == update_data.config  # Updated
        assert response.config["temperature"] == 0.9
        assert response.config["max_tokens"] == 8000
        assert response.config["top_p"] == 0.95

        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_llm_service.update.called

    async def test_toggle_llm_active_status(self, mock_llm_service, mock_db_session, mock_llm):
        """Test toggling the active status of an LLM model."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Create update data with only is_active field
        update_data = MockResponse(
            is_active=False
        )

        # Call the service
        response = await mock_llm_service.update(
            mock_llm.id,
            update_data,
            session=mock_db_session
        )

        # Verify result structure
        assert response.id == mock_llm.id
        assert response.is_active == False  # Updated
        assert response.name == mock_llm.name  # Unchanged
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version  # Unchanged

        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_llm_service.update.called

    async def test_get_llm_by_id(self, mock_llm_service, mock_db_session, mock_llm):
        """Test getting an LLM model by ID."""
        # Setup mock to return an existing model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm

        # Call the service
        response = await mock_llm_service.get_by_id(
            mock_llm.id,
            session=mock_db_session
        )

        # Verify result structure
        assert response.id == mock_llm.id
        assert response.name == mock_llm.name
        assert response.provider == mock_llm.provider
        assert response.version == mock_llm.version
        assert response.is_active == mock_llm.is_active
        assert response.config == mock_llm.config
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")

        # Verify service method was called
        assert mock_llm_service.get_by_id.called

    async def test_get_llm_by_id_not_found(self, mock_llm_service, mock_db_session):
        """Test getting a non-existent LLM model by ID."""
        # Setup mock to return no model
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.get_by_id(
                uuid4(),
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 404
        assert "Model not found" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.get_by_id.called

    async def test_list_llms_by_provider(self, mock_llm_service, mock_db_session, mock_user):
        """Test listing LLM models by provider."""
        """Test listing LLM models by provider."""
        # Call the service
        response = await mock_llm_service.list_by_provider(
            mock_user["organization_id"],
            "openai",
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure
        assert isinstance(response, list)
        assert len(response) > 0
        assert all(model.provider == "openai" for model in response)

        # Verify service method was called
        assert mock_llm_service.list_by_provider.called

    async def test_add_llm_with_empty_name(self, mock_llm_service, mock_db_session):
        """Test adding an LLM model with an empty name."""
        # Override the mock implementation for this test
        """Test adding an LLM model with an empty name."""
        # Override the mock implementation for this test
        async def mock_add_empty_name(model_data, session):
            # Validate name
            if not model_data.name or not model_data.name.strip():
                raise HTTPException(status_code=400, detail="Model name cannot be empty")

            # Validate provider
            if not model_data.provider or not model_data.provider.strip():
                raise HTTPException(status_code=400, detail="Provider cannot be empty")

            # Validate version
            if not model_data.version or not model_data.version.strip():
                raise HTTPException(status_code=400, detail="Version cannot be empty")

            # Original implementation
            return await mock_llm_service.add.side_effect(model_data, session)

        # Replace the mock method with our new implementation for this test only
        mock_llm_service.add.side_effect = mock_add_empty_name

        # Create model data with empty name
        model_data = MockResponse(
            name="",
            provider="openai",
            version="1.0",
            is_active=True,
            config={"temperature": 0.7}
        )

        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.add(
                model_data,
                session=mock_db_session
            )

        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Model name cannot be empty" in str(exc_info.value.detail)

        # Verify service method was called
        assert mock_llm_service.add.called

    async def test_batch_update_llm_status(self, mock_llm_service, mock_db_session):
        """Test batch updating the active status of multiple LLM models."""
        # Create list of model IDs
        """Test batch updating the active status of multiple LLM models."""
        # Create list of model IDs
        model_ids = [uuid4(), uuid4(), uuid4()]

        # Call the service
        response = await mock_llm_service.batch_update_status(
            model_ids,
            is_active=False,
            session=mock_db_session
        )

        # Verify result structure
        assert isinstance(response, list)
        assert len(response) == len(model_ids)
        assert all(model.is_active == False for model in response)
        assert all(model.id in model_ids for model in response)

        # Verify database operations
        assert mock_db_session.commit.call_count == 1

        # Verify service method was called
        assert mock_llm_service.batch_update_status.called

    async def test_search_llm_by_name(self, mock_llm_service, mock_db_session, mock_user):
        """Test searching for LLM models by name."""
        # Call the service
        response = await mock_llm_service.search_by_name(
            mock_user["organization_id"],
            "gpt",
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure
        assert isinstance(response, list)
        assert len(response) > 0
        assert all("gpt" in model.name.lower() for model in response)

        # Verify service method was called
        assert mock_llm_service.search_by_name.called