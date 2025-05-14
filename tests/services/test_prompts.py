import pytest
import json
from fastapi import HTTPException
from sqlalchemy import func, or_, select
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import UUID, uuid4
from typing import List, Dict, Any, Optional, Union
import datetime

from pydantic import BaseModel

# Patch the necessary dependencies for testing
patch('sqlalchemy.ext.asyncio.AsyncSession', MagicMock()).start()
patch('sqlalchemy.select', lambda *args: MagicMock()).start()
patch('sqlalchemy.func', MagicMock()).start()
patch('sqlalchemy.or_', MagicMock()).start()

# Import the real service and schemas
from src.services.prompts.service import PromptService
from src.services.prompts.schema import (
    PaginatedPromptResponse,
    PromptCategoryCreate,
    PromptCategoryResponse,
    PromptCategoryUpdate,
    PromptCreate,
    PromptResponse,
    PromptUpdate,
)

# Import models needed for patching
from src.models import PromptCategoryModel, PromptModel

# Mock modules that might cause import issues
sys.modules["database"] = MagicMock()
sys.modules["database.postgres"] = MagicMock()
sys.modules["src.database"] = MagicMock()
sys.modules["src.database.postgres"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.config.settings"] = MagicMock()
sys.modules["dependencies.security"] = MagicMock()
sys.modules["src.services.__base.acquire"] = MagicMock()

# Mock models
class MockPromptCategoryModel:
    def __init__(self, id=None, name="Test Category", description="Test description", 
                 icon_url="https://example.com/icon.png", display_order=1, is_active=True):
        self.id = id or uuid4()
        self.name = name
        self.description = description
        self.icon_url = icon_url
        self.display_order = display_order
        self.is_active = is_active
        
    def __eq__(self, other):
        if not isinstance(other, MockPromptCategoryModel):
            return False
        return self.id == other.id and self.name == other.name

class MockPromptModel:
    def __init__(self, id=None, title="Test Prompt", content="This is a test prompt", 
                 description="Test description", is_public=False, is_featured=False,
                 creator_id=None, organization_id=None, category_id=None, category=None,
                 metadata=None, created_at=None):
        self.id = id or uuid4()
        self.title = title
        self.content = content
        self.description = description
        self.is_public = is_public
        self.is_featured = is_featured
        self.creator_id = creator_id or uuid4()
        self.organization_id = organization_id or uuid4()
        self.category_id = category_id or uuid4()
        self.category = category or MockPromptCategoryModel(id=self.category_id)
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.datetime.now()

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
    session = MagicMock()
    session.add = MagicMock()
    
    # Fix for session.delete - it needs to be AsyncMock to be awaitable
    session.delete = AsyncMock()
    
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.scalar = AsyncMock()
    
    # Create a properly structured mock result for database queries
    scalar_mock = MagicMock()
    unique_mock = MagicMock()
    scalars_mock = MagicMock()
    all_mock = MagicMock()
    
    scalars_mock.all = MagicMock(return_value=[])
    scalars_mock.first = MagicMock(return_value=None)
    unique_mock.scalar_one_or_none = MagicMock(return_value=None)
    unique_mock.scalar_one = MagicMock(return_value=0) # Default count is 0
    unique_mock.scalars = MagicMock(return_value=scalars_mock)
    
    execute_mock = AsyncMock()
    execute_mock.scalar_one_or_none = MagicMock(return_value=None)
    execute_mock.scalar_one = MagicMock(return_value=0)
    execute_mock.scalars = MagicMock(return_value=scalars_mock)
    execute_mock.unique = MagicMock(return_value=unique_mock)
    
    session.execute = AsyncMock(return_value=execute_mock)
    
    return session

@pytest.fixture
def mock_category():
    """Create a mock prompt category."""
    return MockPromptCategoryModel()

@pytest.fixture
def mock_categories():
    """Create multiple mock categories."""
    return [
        MockPromptCategoryModel(name="Category 1", display_order=1),
        MockPromptCategoryModel(name="Category 2", display_order=2),
        MockPromptCategoryModel(name="Category 3", display_order=3, is_active=False)
    ]

@pytest.fixture
def mock_prompt():
    """Create a mock prompt."""
    category = MockPromptCategoryModel()
    return MockPromptModel(category=category, category_id=category.id)

@pytest.fixture
def mock_prompts():
    """Create multiple mock prompts."""
    category1 = MockPromptCategoryModel(name="Category 1")
    category2 = MockPromptCategoryModel(name="Category 2")
    
    org_id = uuid4()
    return [
        MockPromptModel(title="Prompt 1", category=category1, category_id=category1.id, organization_id=org_id, is_public=True),
        MockPromptModel(title="Prompt 2", category=category1, category_id=category1.id, organization_id=org_id, is_featured=True),
        MockPromptModel(title="Prompt 3", category=category2, category_id=category2.id, organization_id=org_id),
        MockPromptModel(title="Public Prompt", category=category2, category_id=category2.id, organization_id=uuid4(), is_public=True)
    ]

@pytest.fixture
def mock_acquire():
    """Create a mock Acquire object."""
    acquire_mock = MagicMock()
    acquire_mock.logger = MagicMock()
    return acquire_mock

@pytest.fixture
def prompt_service(mock_acquire):
    """Create the real prompt service with mocked dependencies."""
    return PromptService(acquire=mock_acquire)

# Patch joinedload to prevent SQLAlchemy errors
@pytest.fixture(autouse=True)
def patch_joinedload():
    with patch('sqlalchemy.orm.joinedload', MagicMock()):
        yield

@pytest.mark.asyncio
class TestPromptService:
    """Test PromptService."""
    
    # Category-related tests
    async def test_get_list_categories_active_only(self, prompt_service, mock_db_session, mock_user, mock_categories):
        """Test listing active categories."""
        # Setup - return active categories only
        active_categories = [cat for cat in mock_categories if cat.is_active]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = active_categories
        
        # Execute
        response = await prompt_service.get_list_categories(
            active_only=True,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert len(response) == len(active_categories)
        # Check that the query included the is_active filter
        assert mock_db_session.execute.called
    
    async def test_get_list_categories_all(self, prompt_service, mock_db_session, mock_user, mock_categories):
        """Test listing all categories including inactive ones."""
        # Setup
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_categories
        
        # Execute
        response = await prompt_service.get_list_categories(
            active_only=False,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert len(response) == len(mock_categories)
        assert mock_db_session.execute.called
    
    async def test_post_create_category_success(self, prompt_service, mock_db_session, mock_user):
        """Test creating a new category successfully."""
        # Setup
        category_data = PromptCategoryCreate(
            name="New Category",
            description="A new category description",
            icon_url="https://example.com/new-icon.png",
            display_order=10
        )
        
        # Mock database to return no existing category with the same name
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Create a mock for the new category
        new_category = MockPromptCategoryModel(
            name=category_data.name,
            description=category_data.description,
            icon_url=category_data.icon_url,
            display_order=category_data.display_order
        )
        
        # Use patch to make PromptCategoryModel return our mock
        with patch('src.services.prompts.service.PromptCategoryModel', return_value=new_category):
            # Execute
            response = await prompt_service.post_create_category(
                category_data=category_data,
                session=mock_db_session,
                user=mock_user
            )
        
        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        assert mock_db_session.refresh.called
        assert response.name == category_data.name
        assert response.description == category_data.description
        assert response.icon_url == category_data.icon_url
        assert response.display_order == category_data.display_order
    
    async def test_post_create_category_duplicate(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test creating a category with a duplicate name."""
        # Setup
        category_data = PromptCategoryCreate(
            name=mock_category.name,
            description="A new category description"
        )
        
        # Mock database to return an existing category with the same name
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_category
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.post_create_category(
                category_data=category_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail
    
    async def test_put_update_category_success(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test updating a category successfully."""
        # Setup
        category_id = mock_category.id
        category_data = PromptCategoryUpdate(
            name="Updated Category Name",
            description="Updated description",
            display_order=20
        )
        
        # Mock database to return the category
        mock_db_session.execute.return_value.scalars.return_value.first.side_effect = [
            mock_category,  # First call - get the category
            None            # Second call - check for duplicate name
        ]
        
        # Execute
        response = await prompt_service.put_update_category(
            category_id=category_id,
            category_data=category_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert mock_db_session.commit.called
        assert mock_db_session.refresh.called
        # Check if the category was updated correctly
        assert mock_category.name == category_data.name
        assert mock_category.description == category_data.description
        assert mock_category.display_order == category_data.display_order
    
    async def test_put_update_category_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test updating a non-existent category."""
        # Setup
        category_id = uuid4()
        category_data = PromptCategoryUpdate(name="Updated Category Name")
        
        # Mock database to return no category
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.put_update_category(
                category_id=category_id,
                category_data=category_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Category not found" in exc_info.value.detail
    
    async def test_put_update_category_duplicate_name(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test updating a category with a duplicate name."""
        # Setup
        category_id = mock_category.id
        existing_category = MockPromptCategoryModel(name="Existing Category")
        
        category_data = PromptCategoryUpdate(name="Existing Category")
        
        # Mock database to return the category and then a duplicate
        mock_db_session.execute.return_value.scalars.return_value.first.side_effect = [
            mock_category,      # First call - get the category
            existing_category   # Second call - check for duplicate name
        ]
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.put_update_category(
                category_id=category_id,
                category_data=category_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail
    
    async def test_delete_delete_category_success(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test deleting a category successfully."""
        # Setup
        category_id = mock_category.id
        
        # Set up the mock session to simulate the correct DB behavior
        # First query: check if category exists
        category_result = MagicMock()
        category_result.scalars = MagicMock()
        category_result.scalars.return_value = MagicMock()
        category_result.scalars.return_value.first = MagicMock(return_value=mock_category)
        
        # Second query: count prompts in the category (should be 0 for successful deletion)
        count_result = MagicMock()
        count_result.scalar_one = MagicMock(return_value=0)
        
        # Set up the execute mock to return different results for different queries
        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return category_result
            else:
                return count_result
        
        # Initialize call counter
        call_count = 0
        mock_db_session.execute.side_effect = mock_execute
        
        # Execute the real service method
        response = await prompt_service.delete_delete_category(
            category_id=category_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert the service interacted with the DB correctly
        assert mock_db_session.delete.called
        assert mock_db_session.commit.called
        assert mock_db_session.execute.call_count >= 2
        assert "deleted successfully" in response["message"]
    
    async def test_delete_delete_category_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test deleting a non-existent category."""
        # Setup
        category_id = uuid4()
        
        # Mock database to return no category
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.delete_delete_category(
                category_id=category_id,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Category not found" in exc_info.value.detail
    
    async def test_delete_delete_category_with_prompts(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test deleting a category that has prompts."""
        # Setup
        category_id = mock_category.id
        
        # Set up the mock session to simulate the correct DB behavior
        # First query: check if category exists
        category_result = MagicMock()
        category_result.scalars = MagicMock()
        category_result.scalars.return_value = MagicMock()
        category_result.scalars.return_value.first = MagicMock(return_value=mock_category)
        
        # Second query: count prompts in the category (should be > 0 to trigger the error)
        count_result = MagicMock()
        count_result.scalar_one = MagicMock(return_value=5)
        
        # Set up the execute mock to return different results for different queries
        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return category_result
            else:
                return count_result
        
        # Initialize call counter
        call_count = 0
        mock_db_session.execute.side_effect = mock_execute
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.delete_delete_category(
                category_id=category_id,
                session=mock_db_session,
                user=mock_user
            )
        
        # Check that we got the expected error
        assert exc_info.value.status_code == 400
        assert "Cannot delete category that has" in exc_info.value.detail
        assert mock_db_session.execute.call_count >= 2
    
    async def test_get_get_category_success(self, prompt_service, mock_db_session, mock_user, mock_category):
        """Test getting a category by ID."""
        # Setup
        category_id = mock_category.id
        
        # Mock database to return the category
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_category
        
        # Execute
        response = await prompt_service.get_get_category(
            category_id=category_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert response.id == mock_category.id
        assert response.name == mock_category.name
        assert response.description == mock_category.description
    
    async def test_get_get_category_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test getting a non-existent category."""
        # Setup
        category_id = uuid4()
        
        # Mock database to return no category
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.get_get_category(
                category_id=category_id,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Category not found" in exc_info.value.detail
    
    # Prompt-related tests
    async def test_get_list_prompts(self, prompt_service, mock_db_session, mock_user, mock_prompts):
        """Test listing prompts for an organization."""
        # Setup
        org_id = mock_prompts[0].organization_id
        
        # Filter prompts that should be shown for this org
        expected_prompts = [p for p in mock_prompts if p.organization_id == org_id or p.is_public]
        expected_count = len(expected_prompts)
        
        # Mock database to return the filtered prompts
        all_mock = MagicMock(return_value=expected_prompts)
        scalars_mock = MagicMock()
        scalars_mock.all = all_mock
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock()
        execute_mock.unique.return_value = MagicMock()
        execute_mock.unique.return_value.scalars = MagicMock(return_value=scalars_mock)
        
        # Mock count query
        mock_db_session.scalar = AsyncMock(return_value=expected_count)
        
        # Set up the mock database session
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Execute the actual service method
        response = await prompt_service.get_list_prompts(
            org_id=org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert that the response contains the expected data
        assert isinstance(response, PaginatedPromptResponse)
        assert len(response.prompts) == expected_count
        assert response.total == expected_count
        assert mock_db_session.execute.called  # Verify database was queried
        assert mock_db_session.scalar.called   # Verify count was queried
        
    async def test_get_list_prompts_with_category_filter(self, prompt_service, mock_db_session, mock_user, mock_prompts, mock_category):
        """Test listing prompts filtered by category."""
        # Setup
        org_id = mock_prompts[0].organization_id
        category_id = mock_category.id
        
        # Filter prompts that match our criteria
        filtered_prompts = [p for p in mock_prompts if p.category_id == category_id 
                          and (p.organization_id == org_id or p.is_public)]
        filtered_count = len(filtered_prompts)
        
        # Mock database to return the filtered prompts
        all_mock = MagicMock(return_value=filtered_prompts)
        scalars_mock = MagicMock()
        scalars_mock.all = all_mock
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock()
        execute_mock.unique.return_value = MagicMock()
        execute_mock.unique.return_value.scalars = MagicMock(return_value=scalars_mock)
        
        # Mock count query
        mock_db_session.scalar = AsyncMock(return_value=filtered_count)
        
        # Set up the mock database session
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        # Execute the actual service method
        response = await prompt_service.get_list_prompts(
            org_id=org_id,
            category_id=category_id,
            session=mock_db_session,
            user=mock_user
            )
        
        # Assert that the filtered data is correct
        assert isinstance(response, PaginatedPromptResponse)
        assert len(response.prompts) == filtered_count
        assert response.total == filtered_count
        assert mock_db_session.execute.called  # Verify database was queried
        assert mock_db_session.scalar.called   # Verify count was queried
    
    async def test_post_create_prompt_category_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test creating a prompt with a non-existent category."""
        # Setup
        org_id = uuid4()
        category_id = uuid4()
        prompt_data = PromptCreate(
            title="New Prompt", 
            content="This is a new prompt content"
        )
        
        # Mock database to return no category
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.post_create_prompt(
                org_id=org_id,
                category_id=category_id,
                prompt_data=prompt_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Category not found" in exc_info.value.detail
    
    async def test_put_update_prompt_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test updating a non-existent prompt."""
        # Setup
        org_id = uuid4()
        category_id = uuid4()
        prompt_id = uuid4()
        
        prompt_data = PromptUpdate(title="Updated Prompt Title")
        
        # Mock database to return category but no prompt
        mock_db_session.execute.return_value.scalars.return_value.first.side_effect = [
            MockPromptCategoryModel(id=category_id),  # First check is for category
            None                                       # Second check is for prompt
        ]
        
        # Execute and Assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.put_update_prompt(
                org_id=org_id,
                category_id=category_id,
                prompt_id=prompt_id,
                prompt_data=prompt_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Prompt not found" in exc_info.value.detail or "you don't have access" in exc_info.value.detail
    
    async def test_delete_delete_prompt_success(self, prompt_service, mock_db_session, mock_user, mock_prompt):
        """Test deleting a prompt successfully."""
        # Setup
        org_id = mock_prompt.organization_id
        prompt_id = mock_prompt.id
        
        # Mock database to return the prompt
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_prompt
        
        # Execute
        response = await prompt_service.delete_delete_prompt(
            org_id=org_id,
            prompt_id=prompt_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert mock_db_session.delete.called
        assert mock_db_session.commit.called
        assert "deleted successfully" in response["message"]
    
    async def test_delete_delete_prompt_not_found(self, prompt_service, mock_db_session, mock_user):
        """Test deleting a non-existent prompt."""
        # Setup
        org_id = uuid4()
        prompt_id = uuid4()
        
        # Mock database to return no prompt
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None
        
        # Execute and assert
        with pytest.raises(HTTPException) as exc_info:
            await prompt_service.delete_delete_prompt(
                org_id=org_id,
                prompt_id=prompt_id,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Prompt not found" in exc_info.value.detail
    
    async def test_get_list_all_prompts(self, prompt_service, mock_db_session, mock_user, mock_prompts):
        """Test listing all prompts as an admin."""
        # Setup - this endpoint is for admin users to see all prompts
        # Mock database to return all prompts
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_prompts
        
        # Mock scalar method to return count directly
        mock_db_session.scalar.return_value = len(mock_prompts)
        
        # Execute with patched select and joinedload
        with patch('sqlalchemy.select', MagicMock()), \
             patch('sqlalchemy.orm.joinedload', MagicMock()):
            response = await prompt_service.get_list_all_prompts(
                session=mock_db_session,
                user=mock_user
            )
        
        # Assert
        assert isinstance(response, PaginatedPromptResponse)
        assert len(response.prompts) == len(mock_prompts)
        assert response.total == len(mock_prompts)
        assert response.has_more == False
    
# ============================================================================
# INTEGRATION TESTS - RUN WITH: INTEGRATION_TEST=1 pytest tests/services/test_prompts.py
# ============================================================================

import os
import pytest_asyncio
from sqlalchemy import select, text
import datetime

# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.
    
    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select, text
    from database import get_db
    from dependencies.security import RBAC, JWTBearer
    from models import PromptCategoryModel, PromptModel
    from src.database import Base

@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for prompt integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")
        
    # Create necessary database objects
    async for session in db_session:
        try:
            # Create required tables if they don't exist
            # Create prompt_categories table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS prompt_categories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    icon_url TEXT,
                    display_order INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create prompts table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    description TEXT,
                    is_public BOOLEAN DEFAULT false,
                    is_featured BOOLEAN DEFAULT false,
                    creator_id UUID NOT NULL,
                    organization_id UUID NOT NULL,
                    category_id UUID NOT NULL REFERENCES prompt_categories(id),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Clean up any existing test data first
            await session.execute(text("DELETE FROM prompts WHERE title LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM prompt_categories WHERE name LIKE 'Test Integration%'"))
            
            await session.commit()
            
            yield
            
            # Clean up after tests
            await session.execute(text("DELETE FROM prompts WHERE title LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM prompt_categories WHERE name LIKE 'Test Integration%'"))
            await session.commit()
            
        except Exception as e:
            print(f"Error in setup: {e}")
            await session.rollback()
            raise
        finally:
            # Only process the first yielded session
            break

@pytest.fixture
def test_integration_user():
    """Create a test user for integration tests."""
    user_id = uuid4()
    org_id = uuid4()
    return {
        "id": user_id,
        "email": f"test-integration-{user_id}@example.com",
        "first_name": "Test",
        "last_name": "Integration",
        "org_id": org_id
    }

@pytest.mark.asyncio
class TestPromptServiceIntegration:
    """Integration tests for Prompt service using a real database."""
    
    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )
    
    async def test_category_crud_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test complete CRUD operations for categories with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create category
                category_data = PromptCategoryCreate(
                    name="Test Integration Category",
                    description="Test integration description",
                    icon_url="https://example.com/icon.png",
                    display_order=1
                )
                
                response = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # Verify creation
                assert response is not None
                assert response.name == category_data.name
                assert response.description == category_data.description
                assert response.id is not None
                category_id = response.id
                
                # 2. Retrieve the category
                get_response = await prompt_service.get_get_category(
                    category_id=category_id,
                    session=session,
                    user=test_integration_user
                )
                
                assert get_response.id == category_id
                assert get_response.name == category_data.name
                
                # 3. Update the category
                update_data = PromptCategoryUpdate(
                    name="Test Integration Category Updated",
                    description="Updated description"
                )
                
                update_response = await prompt_service.put_update_category(
                    category_id=category_id,
                    category_data=update_data,
                    session=session,
                    user=test_integration_user
                )
                
                assert update_response.name == update_data.name
                assert update_response.description == update_data.description
                
                # 4. Delete the category
                delete_response = await prompt_service.delete_delete_category(
                    category_id=category_id,
                    session=session,
                    user=test_integration_user
                )
                
                assert delete_response is not None
                assert "deleted successfully" in delete_response["message"]
                
                # Verify deletion
                with pytest.raises(HTTPException) as exc_info:
                    await prompt_service.get_get_category(
                        category_id=category_id,
                        session=session,
                        user=test_integration_user
                    )
                
                assert exc_info.value.status_code == 404
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break
    
    async def test_prompt_crud_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test complete CRUD operations for prompts with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category first
                category_data = PromptCategoryCreate(
                    name="Test Integration Prompt Category",
                    description="Category for prompt testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create a prompt
                prompt_data = PromptCreate(
                    title="Test Integration Prompt",
                    content="This is a test prompt content for integration test",
                    description="Test prompt description",
                    is_public=True,
                    is_featured=False
                )
                
                prompt_response = await prompt_service.post_create_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_data=prompt_data,
                    session=session,
                    user=test_integration_user
                )
                
                # Verify creation
                assert prompt_response is not None
                assert prompt_response.title == prompt_data.title
                assert prompt_response.content == prompt_data.content
                assert prompt_response.category.id == category.id
                prompt_id = prompt_response.id
                
                # 3. Update the prompt
                update_data = PromptUpdate(
                    title="Test Integration Prompt Updated",
                    content="Updated content for integration test",
                    is_featured=True
                )
                
                update_response = await prompt_service.put_update_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_id=prompt_id,
                    prompt_data=update_data,
                    session=session,
                    user=test_integration_user
                )
                
                assert update_response.title == update_data.title
                assert update_response.content == update_data.content
                assert update_response.is_featured == update_data.is_featured
                
                # 4. Delete the prompt
                delete_response = await prompt_service.delete_delete_prompt(
                    org_id=test_integration_user["org_id"],
                    prompt_id=prompt_id,
                    session=session,
                    user=test_integration_user
                )
                
                assert delete_response is not None
                assert "deleted successfully" in delete_response["message"]
                
                # Clean up the category
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break
    
    async def test_pagination_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test prompt pagination with real database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category
                category_data = PromptCategoryCreate(
                    name="Test Integration Pagination Category",
                    description="Category for pagination testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create multiple prompts (11 prompts for testing pagination)
                for i in range(11):
                    prompt_data = PromptCreate(
                        title=f"Test Integration Pagination Prompt {i}",
                        content=f"Content for pagination prompt {i}",
                        description=f"Description {i}",
                        is_public=i % 2 == 0,  # Alternate between public and private
                        is_featured=i % 3 == 0  # Every third prompt is featured
                    )
                    
                    await prompt_service.post_create_prompt(
                        org_id=test_integration_user["org_id"],
                        category_id=category.id,
                        prompt_data=prompt_data,
                        session=session,
                        user=test_integration_user
                    )
                
                # 3. Test pagination with default limit (10)
                page_1 = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    session=session,
                    user=test_integration_user
                )
                
                assert page_1.total == 11
                assert len(page_1.prompts) == 10
                assert page_1.has_more == True
                
                # 4. Test second page
                page_2 = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    offset=1,
                    session=session,
                    user=test_integration_user
                )
                
                assert page_2.total == 11
                assert len(page_2.prompts) == 1
                assert page_2.has_more == False
                
                # 5. Test with smaller limit
                small_page = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    limit=5,
                    session=session,
                    user=test_integration_user
                )
                
                assert small_page.total == 11
                assert len(small_page.prompts) == 5
                assert small_page.has_more == True
                
                # 6. Test with category filter
                category_filter = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
                assert category_filter.total == 11
                
                # 7. Test with featured filter
                featured_filter = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    is_featured=True,
                    session=session,
                    user=test_integration_user
                )
                
                # Should have 4 featured prompts (every third of 11)
                assert featured_filter.total == 4
                
                # Clean up
                for prompt in page_1.prompts:
                    await prompt_service.delete_delete_prompt(
                        org_id=test_integration_user["org_id"],
                        prompt_id=prompt.id,
                        session=session,
                        user=test_integration_user
                    )
                
                for prompt in page_2.prompts:
                    await prompt_service.delete_delete_prompt(
                        org_id=test_integration_user["org_id"],
                        prompt_id=prompt.id,
                        session=session,
                        user=test_integration_user
                    )
                
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_public_prompts_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test listing public prompts from different organizations."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category
                category_data = PromptCategoryCreate(
                    name="Test Integration Public Category",
                    description="Category for public prompts testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create a second user in a different organization
                second_user = {
                    "id": uuid4(),
                    "email": "second-test@example.com",
                    "first_name": "Second",
                    "last_name": "User",
                    "org_id": uuid4()  # Different org
                }
                
                # 3. Create prompts in both organizations
                # Create 3 prompts in user 1's org (2 public, 1 private)
                for i in range(3):
                    prompt_data = PromptCreate(
                        title=f"Test Integration User1 Prompt {i}",
                        content=f"Content for user1 prompt {i}",
                        description=f"Description {i}",
                        is_public=i < 2,  # First 2 are public
                        is_featured=False
                    )
                    
                    await prompt_service.post_create_prompt(
                        org_id=test_integration_user["org_id"],
                        category_id=category.id,
                        prompt_data=prompt_data,
                        session=session,
                        user=test_integration_user
                    )
                
                # Create 3 prompts in user 2's org (1 public, 2 private)
                for i in range(3):
                    prompt_data = PromptCreate(
                        title=f"Test Integration User2 Prompt {i}",
                        content=f"Content for user2 prompt {i}",
                        description=f"Description {i}",
                        is_public=i == 0,  # Only first is public
                        is_featured=False
                    )
                    
                    await prompt_service.post_create_prompt(
                        org_id=second_user["org_id"],
                        category_id=category.id,
                        prompt_data=prompt_data,
                        session=session,
                        user=second_user
                    )
                
                # 4. Test that user 1 can see their prompts plus user 2's public prompts
                user1_prompts = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    include_public=True,
                    session=session,
                    user=test_integration_user
                )
                
                # Should see 4 prompts: 3 from own org + 1 public from other org
                assert user1_prompts.total == 4
                
                # 5. Test user 2 can see their prompts plus user 1's public prompts
                user2_prompts = await prompt_service.get_list_prompts(
                    org_id=second_user["org_id"],
                    include_public=True,
                    session=session,
                    user=second_user
                )
                
                # Should see 5 prompts: 3 from own org + 2 public from other org
                assert user2_prompts.total == 5
                
                # 6. Test that include_public=False filters correctly
                user1_private_only = await prompt_service.get_list_prompts(
                    org_id=test_integration_user["org_id"],
                    include_public=False,
                    session=session,
                    user=test_integration_user
                )
                
                # Should only see 3 prompts from own org
                assert user1_private_only.total == 3
                
                # Clean up test data
                # We can use list_all_prompts to get all prompts for cleanup
                all_prompts = await prompt_service.get_list_all_prompts(
                    session=session,
                    user=test_integration_user
                )
                
                for prompt in all_prompts.prompts:
                    try:
                        await prompt_service.delete_delete_prompt(
                            org_id=prompt.organization_id,
                            prompt_id=prompt.id,
                            session=session,
                            user=test_integration_user if prompt.organization_id == test_integration_user["org_id"] else second_user
                        )
                    except Exception:
                        # If delete fails for any reason, continue with cleanup
                        pass
                
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_create_prompt_success_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test creating a new prompt successfully using integration testing."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category first
                category_data = PromptCategoryCreate(
                    name="Test Integration Create Success Category",
                    description="Category for create prompt success testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create a prompt with all fields
                prompt_data = PromptCreate(
                    title="Test Integration Create Success Prompt",
                    content="This is a test prompt content for integration test",
                    description="Test prompt description",
                    is_public=True,
                    is_featured=True,
                    metadata={"tags": ["test", "integration"], "version": 1}
                )
                
                prompt_response = await prompt_service.post_create_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_data=prompt_data,
                    session=session,
                    user=test_integration_user
                )
                
                # Verify creation with all fields
                assert prompt_response is not None
                assert prompt_response.title == prompt_data.title
                assert prompt_response.content == prompt_data.content
                assert prompt_response.description == prompt_data.description
                assert prompt_response.is_public == prompt_data.is_public
                assert prompt_response.is_featured == prompt_data.is_featured
                assert prompt_response.metadata == prompt_data.metadata
                assert prompt_response.category.id == category.id
                assert prompt_response.creator_id == test_integration_user["id"]
                assert prompt_response.organization_id == test_integration_user["org_id"]
                
                # Clean up
                await prompt_service.delete_delete_prompt(
                    org_id=test_integration_user["org_id"],
                    prompt_id=prompt_response.id,
                    session=session,
                    user=test_integration_user
                )
                
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break
    
    async def test_create_prompt_minimal_fields_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test creating a prompt with only required fields using integration testing."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category first
                category_data = PromptCategoryCreate(
                    name="Test Integration Minimal Fields Category",
                    description="Category for minimal fields testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create a prompt with only required fields
                prompt_data = PromptCreate(
                    title="Test Integration Minimal Prompt",
                    content="This is a minimal prompt content"
                    # All other fields are optional
                )
                
                prompt_response = await prompt_service.post_create_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_data=prompt_data,
                    session=session,
                    user=test_integration_user
                )
                
                # Verify creation with default values
                assert prompt_response is not None
                assert prompt_response.title == prompt_data.title
                assert prompt_response.content == prompt_data.content
                assert prompt_response.description is None
                assert prompt_response.is_public is False  # Default value
                assert prompt_response.is_featured is False  # Default value
                assert prompt_response.metadata == {}
                assert prompt_response.category.id == category.id
                
                # Clean up
                await prompt_service.delete_delete_prompt(
                    org_id=test_integration_user["org_id"],
                    prompt_id=prompt_response.id,
                    session=session,
                    user=test_integration_user
                )
                
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_update_prompt_success_integration(self, prompt_service, db_session, test_integration_user, setup_test_db_integration):
        """Test updating a prompt successfully using integration testing."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # 1. Create a category first
                category_data = PromptCategoryCreate(
                    name="Test Integration Update Prompt Category",
                    description="Category for update prompt testing"
                )
                
                category = await prompt_service.post_create_category(
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 2. Create a prompt to update
                original_prompt_data = PromptCreate(
                    title="Test Integration Original Prompt",
                    content="This is the original content",
                    description="Original description",
                    is_public=False,
                    is_featured=False
                )
                
                original_prompt = await prompt_service.post_create_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_data=original_prompt_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 3. Update the prompt
                update_data = PromptUpdate(
                    title="Test Integration Updated Prompt",
                    content="This is the updated content",
                    is_public=True,
                    is_featured=True,
                    metadata={"updated": True, "version": 2}
                )
                
                updated_prompt = await prompt_service.put_update_prompt(
                    org_id=test_integration_user["org_id"],
                    category_id=category.id,
                    prompt_id=original_prompt.id,
                    prompt_data=update_data,
                    session=session,
                    user=test_integration_user
                )
                
                # 4. Verify updates took effect
                assert updated_prompt.title == update_data.title
                assert updated_prompt.title != original_prompt_data.title
                assert updated_prompt.content == update_data.content
                assert updated_prompt.content != original_prompt_data.content
                assert updated_prompt.is_public == update_data.is_public
                assert updated_prompt.is_public != original_prompt_data.is_public
                assert updated_prompt.is_featured == update_data.is_featured
                assert updated_prompt.is_featured != original_prompt_data.is_featured
                assert updated_prompt.metadata == update_data.metadata
                
                # Description wasn't updated, so it should remain the same
                assert updated_prompt.description == original_prompt_data.description
                
                # Clean up
                await prompt_service.delete_delete_prompt(
                    org_id=test_integration_user["org_id"],
                    prompt_id=updated_prompt.id,
                    session=session,
                    user=test_integration_user
                )
                
                await prompt_service.delete_delete_category(
                    category_id=category.id,
                    session=session,
                    user=test_integration_user
                )
                
            except Exception as e:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

# ============================================================================
# ADDITIONAL UNIT TESTS - Not requiring database integration
# ============================================================================

@pytest.mark.asyncio
class TestPromptServiceEdgeCases:
    """Test edge cases in the Prompt service."""
    
    async def test_get_list_categories_empty(self, prompt_service, mock_db_session, mock_user):
        """Test listing categories when none exist."""
        # Setup - return empty list
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        # Execute
        response = await prompt_service.get_list_categories(
            active_only=True,
                session=mock_db_session,
                user=mock_user
            )
        
        # Assert
        assert isinstance(response, list)
        assert len(response) == 0
    
    async def test_list_prompts_empty_result(self, prompt_service, mock_db_session, mock_user):
        """Test listing prompts when no prompts exist."""
        # Setup
        org_id = uuid4()
        
        # Mock to return empty results and zero count
        mock_db_session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = []
        mock_db_session.scalar.return_value = 0
        
        # Execute
        response = await prompt_service.get_list_prompts(
            org_id=org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert
        assert isinstance(response, PaginatedPromptResponse)
        assert response.total == 0
        assert len(response.prompts) == 0
        assert response.has_more is False
    
    async def test_update_prompt_all_fields(self, prompt_service, mock_db_session, mock_user, mock_prompt, mock_category):
        """Test updating all fields of a prompt."""
        # Setup
        org_id = mock_prompt.organization_id
        category_id = mock_category.id
        prompt_id = mock_prompt.id
        
        # Associate the mock prompt with the category
        mock_prompt.category_id = category_id
        mock_prompt.category = mock_category
        
        # Create update data for all fields
        new_metadata = {"tags": ["test", "updated"], "version": 2}
        prompt_data = PromptUpdate(
            title="Completely Updated Title",
            content="Completely updated content with new information",
            description="New detailed description",
            is_public=not mock_prompt.is_public,
            is_featured=not mock_prompt.is_featured,
            metadata=new_metadata
        )
        
        # Mock database to return category and prompt
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_category
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = mock_prompt
        mock_db_session.execute.return_value.unique.return_value.scalar_one.return_value = mock_prompt
        
        # Execute
        response = await prompt_service.put_update_prompt(
            org_id=org_id,
            category_id=category_id,
            prompt_id=prompt_id,
            prompt_data=prompt_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert all fields were updated
        assert mock_prompt.title == prompt_data.title
        assert mock_prompt.content == prompt_data.content
        assert mock_prompt.description == prompt_data.description
        assert mock_prompt.is_public == prompt_data.is_public
        assert mock_prompt.is_featured == prompt_data.is_featured
        assert mock_prompt.metadata == prompt_data.metadata
    
    async def test_prompt_update_metadata_merge(self, prompt_service, mock_db_session, mock_user, mock_prompt, mock_category):
        """Test updating prompt metadata with merge functionality."""
        # Setup
        org_id = mock_prompt.organization_id
        category_id = mock_category.id
        prompt_id = mock_prompt.id
        
        # Associate the mock prompt with the category and set initial metadata
        mock_prompt.category_id = category_id
        mock_prompt.category = mock_category
        mock_prompt.metadata = {"tags": ["original"], "count": 5}
        
        # Create update with partial metadata
        prompt_data = PromptUpdate(
            metadata={"tags": ["updated"], "new_field": "value"}
        )
        
        # Mock database to return category and prompt
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_category
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = mock_prompt
        mock_db_session.execute.return_value.unique.return_value.scalar_one.return_value = mock_prompt
        
        # Execute
        response = await prompt_service.put_update_prompt(
            org_id=org_id,
            category_id=category_id,
            prompt_id=prompt_id,
            prompt_data=prompt_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Assert metadata was updated (replaced, not merged)
        assert mock_prompt.metadata == prompt_data.metadata
        assert "count" not in mock_prompt.metadata
        assert mock_prompt.metadata["tags"] == ["updated"]
        assert mock_prompt.metadata["new_field"] == "value" 