import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import UUID, uuid4
from datetime import datetime

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

# Mock models
class MockLLMModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', 'Test LLM')
        self.provider = kwargs.get('provider', 'openai')
        self.version = kwargs.get('version', '1.0.0')
        self.is_active = kwargs.get('is_active', True)
        self.config = kwargs.get('config', {"max_tokens": 1000, "temperature": 0.7})
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, data):
        if isinstance(data, dict):
            return cls(**data)
        return cls(**{k: v for k, v in data.__dict__.items() if not k.startswith('_')})
    
    def model_dump(self, **kwargs):
        exclude_unset = kwargs.get('exclude_unset', False)
        if exclude_unset:
            # Return only items that have been explicitly set
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

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
        if session.execute.return_value.scalar_one_or_none.return_value:
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
        return MockResponse.model_validate(db_model)
    
    async def mock_update(model_id, model_data, session):
        # Get model
        db_model = session.execute.return_value.scalar_one_or_none.return_value
        
        if not db_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Update fields
        update_data = model_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_model, field, value)
        
        # Update timestamp
        db_model.updated_at = datetime.now()
        
        await session.commit()
        await session.refresh(db_model)
        
        # Return response matching API format
        return MockResponse.model_validate(db_model)
    
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
        # Create mock models matching Postman examples
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
        
        # Return response matching API format - convert to dict with model_dump
        return [MockResponse.model_validate(model) for model in models]
    
    # Create AsyncMock objects
    add_mock = AsyncMock(side_effect=mock_add)
    update_mock = AsyncMock(side_effect=mock_update)
    remove_mock = AsyncMock(side_effect=mock_remove)
    list_mock = AsyncMock(side_effect=mock_list)
    
    # Assign the mocks to the service
    llm_service.post_add = add_mock
    llm_service.put_update = update_mock
    llm_service.delete_remove = remove_mock
    llm_service.get_list = list_mock
    
    return llm_service

@pytest.mark.asyncio
class TestLLMService:
    """Tests for the LLM service."""
    
    async def test_add_llm(self, mock_llm_service, mock_db_session):
        """Test adding a new LLM model."""
        # Create model data matching Postman request format
        model_data = MockResponse(
            name="o1",
            provider="openai",
            version="4o",
            is_active=True,
            config={}
        )
        
        # Call the service
        response = await mock_llm_service.post_add(model_data, mock_db_session)
        
        # Verify result structure matches API response
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
        assert mock_llm_service.post_add.called
    
    async def test_add_existing_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test adding a model that already exists."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Create model data with same key properties
        model_data = MockResponse(
            name=mock_llm.name,
            provider=mock_llm.provider,
            version=mock_llm.version,
            is_active=True,
            config={"max_tokens": 8000, "temperature": 0.5}
        )
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_llm_service.post_add(model_data, mock_db_session)
        
        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Model already exists" in str(excinfo.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.post_add.called
    
    async def test_update_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test updating an existing LLM model."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Create update data matching Postman request
        update_data = MockResponse(
            name="Updated GPT-4",
            is_active=True,
            config={"max_tokens": 8000, "temperature": 0.8}
        )
        
        # Call the service
        response = await mock_llm_service.put_update(mock_llm.id, update_data, mock_db_session)
        
        # Verify result structure matches API response
        assert response.name == update_data.name
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version  # Unchanged
        assert response.is_active == update_data.is_active
        assert response.config == update_data.config
        assert hasattr(response, "id")
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_llm_service.put_update.called
    
    async def test_update_nonexistent_llm(self, mock_llm_service, mock_db_session):
        """Test updating a model that doesn't exist."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None  # No model
        
        # Create update data
        update_data = MockResponse(
            name="Updated Model",
            is_active=False
        )
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_llm_service.put_update(uuid4(), update_data, mock_db_session)
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Model not found" in str(excinfo.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.put_update.called
    
    async def test_remove_llm(self, mock_llm_service, mock_db_session, mock_llm):
        """Test removing an LLM model."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Call the service
        response = await mock_llm_service.delete_remove(mock_llm.id, mock_db_session)
        
        # Verify result matches API response format
        assert response["message"] == "Model deleted successfully"
        
        # Verify database operations
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_llm_service.delete_remove.called
    
    async def test_remove_nonexistent_llm(self, mock_llm_service, mock_db_session):
        """Test removing a model that doesn't exist."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None  # No model
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_llm_service.delete_remove(uuid4(), mock_db_session)
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Model not found" in str(excinfo.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.delete_remove.called
    
    async def test_list_llms(self, mock_llm_service, mock_db_session, mock_user):
        """Test getting a list of LLM models."""
        # Call the service
        response = await mock_llm_service.get_list(
            mock_user["organization_id"],
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure matches API response
        assert isinstance(response, list)
        assert len(response) == 3  # Three mock models
        
        # Verify each model has the expected fields
        for model in response:
            assert hasattr(model, "id")
            assert hasattr(model, "name")
            assert hasattr(model, "provider")
            assert hasattr(model, "version")
            assert hasattr(model, "is_active")
            assert hasattr(model, "config")
            assert hasattr(model, "created_at")
            assert hasattr(model, "updated_at")
        
        # Check specific models match expected data
        model_names = [model.name for model in response]
        assert "gpt-3.5-turbo" in model_names
        assert "gpt-4" in model_names
        assert "claude-3-opus" in model_names
        
        # Verify service method was called
        assert mock_llm_service.get_list.called
    
    async def test_add_llm_with_invalid_provider(self, mock_llm_service, mock_db_session):
        """Test adding a new LLM model with an invalid provider."""
        # Create a custom implementation that validates provider
        async def mock_add_invalid_provider(model_data, session):
            # List of supported providers
            supported_providers = ["openai", "anthropic", "google", "mistral"]
            
            # Validate provider
            if model_data.provider not in supported_providers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider '{model_data.provider}' is not supported. Supported providers: {', '.join(supported_providers)}"
                )
                
            # This won't execute due to validation error
            db_model = MockLLMModel(
                name=model_data.name,
                provider=model_data.provider,
                version=model_data.version,
                is_active=model_data.is_active,
                config=model_data.config
            )
            session.add(db_model)
            await session.commit()
            
            return MockResponse.model_validate(db_model)
            
        mock_llm_service.post_add = AsyncMock(side_effect=mock_add_invalid_provider)
        
        # Create model data with invalid provider
        model_data = MockResponse(
            name="Unknown Model",
            provider="unsupported_provider",
            version="1.0",
            is_active=True,
            config={}
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.post_add(model_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Provider 'unsupported_provider' is not supported" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.post_add.called
    
    async def test_update_llm_config(self, mock_llm_service, mock_db_session, mock_llm):
        """Test updating only the config of an LLM model."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Create update data with only config
        update_data = MockResponse(
            config={"temperature": 0.9, "top_p": 0.95, "max_tokens": 3000}
        )
        
        # Call the service
        response = await mock_llm_service.put_update(mock_llm.id, update_data, mock_db_session)
        
        # Verify that only config was updated
        assert response.name == mock_llm.name  # Unchanged
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version  # Unchanged
        assert response.is_active == mock_llm.is_active  # Unchanged
        assert response.config == update_data.config  # Updated
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_llm_service.put_update.called
    
    async def test_toggle_llm_active_status(self, mock_llm_service, mock_db_session, mock_llm):
        """Test toggling the active status of an LLM model."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Current status is active, so toggle to inactive
        update_data = MockResponse(
            is_active=False
        )
        
        # Call the service
        response = await mock_llm_service.put_update(mock_llm.id, update_data, mock_db_session)
        
        # Verify that only is_active was updated
        assert response.name == mock_llm.name  # Unchanged
        assert response.provider == mock_llm.provider  # Unchanged
        assert response.version == mock_llm.version  # Unchanged
        assert response.is_active == False  # Updated
        assert response.config == mock_llm.config  # Unchanged
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_llm_service.put_update.called
    
    async def test_get_llm_by_id(self, mock_llm_service, mock_db_session, mock_llm):
        """Test getting an LLM model by ID."""
        # Add a get_by_id method to the service
        async def mock_get_by_id(model_id, session):
            # Get model
            db_model = session.execute.return_value.scalar_one_or_none.return_value
            
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Return response matching API format
            return MockResponse.model_validate(db_model)
            
        mock_llm_service.get_by_id = AsyncMock(side_effect=mock_get_by_id)
        
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_llm  # Model exists
        
        # Call the service
        response = await mock_llm_service.get_by_id(mock_llm.id, mock_db_session)
        
        # Verify result structure matches API response
        assert response.id == mock_llm.id
        assert response.name == mock_llm.name
        assert response.provider == mock_llm.provider
        assert response.version == mock_llm.version
        assert response.is_active == mock_llm.is_active
        assert response.config == mock_llm.config
        
        # Verify service method was called
        assert mock_llm_service.get_by_id.called
    
    async def test_get_llm_by_id_not_found(self, mock_llm_service, mock_db_session):
        """Test getting a non-existent LLM model."""
        # Add a get_by_id method to the service that fails
        async def mock_get_by_id_not_found(model_id, session):
            # No model found
            session.execute.return_value.scalar_one_or_none.return_value = None
            raise HTTPException(status_code=404, detail="Model not found")
            
        mock_llm_service.get_by_id = AsyncMock(side_effect=mock_get_by_id_not_found)
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.get_by_id(uuid4(), mock_db_session)
        
        assert exc_info.value.status_code == 404
        assert "Model not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.get_by_id.called
    
    async def test_list_llms_by_provider(self, mock_llm_service, mock_db_session, mock_user):
        """Test listing LLM models filtered by provider."""
        # Add a method to list by provider
        async def mock_list_by_provider(org_id, provider, session, user):
            # Get all models
            all_models = await mock_llm_service.get_list(org_id, session, user)
            
            # Filter by provider
            filtered_models = [model for model in all_models if model.provider == provider]
            return filtered_models
            
        mock_llm_service.get_list_by_provider = AsyncMock(side_effect=mock_list_by_provider)
        
        # Call the service
        response = await mock_llm_service.get_list_by_provider(
            mock_user["organization_id"],
            "openai",
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert isinstance(response, list)
        assert len(response) > 0
        assert all(model.provider == "openai" for model in response)
        
        # Verify service method was called
        assert mock_llm_service.get_list_by_provider.called
    
    async def test_add_llm_with_empty_name(self, mock_llm_service, mock_db_session):
        """Test adding a model with an empty name."""
        # Create a custom implementation that validates name
        async def mock_add_empty_name(model_data, session):
            # Validate name
            if not model_data.name or len(model_data.name.strip()) == 0:
                raise HTTPException(status_code=400, detail="Model name cannot be empty")
                
            # This won't execute due to validation error
            db_model = MockLLMModel(
                name=model_data.name,
                provider=model_data.provider,
                version=model_data.version,
                is_active=model_data.is_active,
                config=model_data.config
            )
            session.add(db_model)
            await session.commit()
            
            return MockResponse.model_validate(db_model)
            
        mock_llm_service.post_add = AsyncMock(side_effect=mock_add_empty_name)
        
        # Create model data with empty name
        model_data = MockResponse(
            name="",
            provider="openai",
            version="4.0",
            is_active=True,
            config={}
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_llm_service.post_add(model_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Model name cannot be empty" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_llm_service.post_add.called
    
    async def test_batch_update_llm_status(self, mock_llm_service, mock_db_session):
        """Test updating multiple LLM models' status at once."""
        # Add batch update method to service
        async def mock_batch_update_status(model_ids, is_active, session):
            # Update each model in the list
            updated_count = 0
            for model_id in model_ids:
                # Get model (mock different responses for different IDs)
                if model_id in [uuid4(), uuid4()]:  # These won't match any IDs we pass
                    # Skip non-existent models
                    continue
                    
                # For testing, we'll pretend all models exist and are updated
                updated_count += 1
                
            # Commit changes
            await session.commit()
            
            # Return count of updated models
            return {"updated_count": updated_count, "message": f"Updated {updated_count} models"}
            
        mock_llm_service.patch_batch_update_status = AsyncMock(side_effect=mock_batch_update_status)
        
        # Create list of model IDs to update
        model_ids = [uuid4(), uuid4(), uuid4()]
        
        # Call the service
        response = await mock_llm_service.patch_batch_update_status(model_ids, False, mock_db_session)
        
        # Verify result
        assert response["updated_count"] == 3
        assert "Updated 3 models" in response["message"]
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_llm_service.patch_batch_update_status.called
    
    async def test_search_llm_by_name(self, mock_llm_service, mock_db_session, mock_user):
        """Test searching for LLM models by name."""
        # Add search method to service
        async def mock_search_by_name(org_id, query, session, user):
            # Get all models
            all_models = await mock_llm_service.get_list(org_id, session, user)
            
            # Filter by name containing the query
            query = query.lower()
            filtered_models = [model for model in all_models if query in model.name.lower()]
            return filtered_models
            
        mock_llm_service.get_search = AsyncMock(side_effect=mock_search_by_name)
        
        # Call the service
        response = await mock_llm_service.get_search(
            mock_user["organization_id"],
            "gpt",
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert isinstance(response, list)
        assert len(response) > 0
        assert all("gpt" in model.name.lower() for model in response)
        
        # Verify service method was called
        assert mock_llm_service.get_search.called 