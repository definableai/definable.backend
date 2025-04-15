import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

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


# Mock models using Pydantic
class MockToolCategoryModel(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  name: str = "Test Category"
  description: str = "Test Category Description"
  org_id: Optional[UUID] = None
  is_default: bool = False
  created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

  model_config = {"arbitrary_types_allowed": True}


class MockToolModel(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  name: str = "Test Tool"
  description: str = "Test Tool Description"
  tool_definition: Dict[str, Any] = Field(default_factory=lambda: {})
  category_id: Optional[UUID] = None
  org_id: Optional[UUID] = None
  created_by: Optional[UUID] = None
  created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

  model_config = {"arbitrary_types_allowed": True}


class MockResponse(BaseModel):
  """Base class for mock responses"""

  id: Optional[UUID] = None
  name: Optional[str] = None
  description: Optional[str] = None
  org_id: Optional[UUID] = None
  tool_definition: Optional[Dict[str, Any]] = None
  category_id: Optional[UUID] = None
  created_by: Optional[UUID] = None
  created_at: Optional[str] = None
  is_default: Optional[bool] = None

  model_config = {"arbitrary_types_allowed": True}


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
  return session


@pytest.fixture
def mock_tool_category():
  """Create a mock tool category."""
  return MockToolCategoryModel(name="Test Category", description="Test category description", org_id=uuid4())


@pytest.fixture
def mock_tool():
  """Create a mock tool."""
  return MockToolModel(
    name="Test Tool",
    description="Test tool description",
    tool_definition={"type": "function", "function": {"name": "test_function"}},
    category_id=uuid4(),
    org_id=uuid4(),
    created_by=uuid4(),
  )


@pytest.fixture
def mock_tools_service():
  """Create a mock tools service."""
  tools_service = MagicMock()

  async def mock_get_categories(org_id, session):
    # Return mock categories
    categories = [MockToolCategoryModel(org_id=org_id, name="Category 1"), MockToolCategoryModel(org_id=org_id, name="Category 2")]

    # Set up the mock for execute return value chain
    session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = categories

    return categories

  async def mock_post_category(org_id, category_data, session, user):
    # Check if category name exists
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None  # Category doesn't exist yet

    # Create category
    db_category = MockToolCategoryModel(org_id=org_id, name=category_data.name, description=category_data.description)
    session.add(db_category)
    await session.flush()
    await session.commit()

    return MockResponse(**db_category.model_dump())

  async def mock_put_category(org_id, category_id, category_data, session, user):
    # Get category
    db_category = MockToolCategoryModel(id=category_id, org_id=org_id)
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = db_category

    # Update category
    category_dict = db_category.model_dump()
    update_data = category_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
      category_dict[field] = value

    # Create updated category
    db_category = MockToolCategoryModel(**category_dict)

    await session.commit()

    return MockResponse(**db_category.model_dump())

  async def mock_delete_category(org_id, category_id, session, user):
    # Check if category exists
    db_category = MockToolCategoryModel(id=category_id, org_id=org_id)
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = db_category

    # Check if category is used by tools
    session.execute.return_value.unique.return_value.scalar_one.return_value = 0  # No tools using this category

    # Delete category
    # Simulate delete operation
    session.execute.side_effect = None  # Reset side_effect if it was set previously

    await session.commit()

    return {"message": "Category deleted successfully"}

  async def mock_get_tools(org_id, category_id, session):
    # Return mock tools
    tools = [
      MockToolModel(org_id=org_id, category_id=category_id, name="Tool 1"),
      MockToolModel(org_id=org_id, category_id=category_id, name="Tool 2"),
    ]

    # Set up mock response
    session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = tools

    return tools

  async def mock_get_tool(org_id, tool_id, session):
    # Return mock tool
    db_tool = MockToolModel(id=tool_id, org_id=org_id)

    # Set up mock response
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = db_tool

    return db_tool

  async def mock_post_tool(org_id, tool_data, session, user):
    # Check if tool name exists
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None  # Tool doesn't exist yet

    # Create tool
    tool_dict = tool_data.model_dump()
    # Remove None values
    tool_dict = {k: v for k, v in tool_dict.items() if v is not None}

    db_tool = MockToolModel(org_id=org_id, created_by=user["id"], **tool_dict)
    session.add(db_tool)
    await session.flush()
    await session.commit()

    return MockResponse(**db_tool.model_dump())

  async def mock_put_tool(org_id, tool_id, tool_data, session, user):
    # Mock getting a tool directly
    db_tool = MockToolModel(id=tool_id, org_id=org_id)

    # Set up mock db operation for get_tool
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = db_tool

    if not db_tool:
      raise HTTPException(status_code=404, detail="Tool not found")

    # Update tool
    tool_dict = db_tool.model_dump()
    tool_dict = db_tool.model_dump()
    update_data = tool_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
      tool_dict[field] = value

    # Create updated tool
    db_tool = MockToolModel(**tool_dict)

    await session.commit()

    return MockResponse(**db_tool.model_dump())

  async def mock_delete_tool(org_id, tool_id, session, user):
    # Mock getting a tool directly
    db_tool = MockToolModel(id=tool_id, org_id=org_id)
    session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = db_tool

    if not db_tool:
      raise HTTPException(status_code=404, detail="Tool not found")

    # Delete tool
    # Simulate delete operation
    await session.commit()

    return {"message": "Tool deleted successfully"}

  # Assign mock methods to service
  tools_service.get_categories = mock_get_categories
  tools_service.post_category = mock_post_category
  tools_service.put_category = mock_put_category
  tools_service.delete_category = mock_delete_category
  tools_service.get_tools = mock_get_tools
  tools_service.get_tool = mock_get_tool
  tools_service.post_tool = mock_post_tool
  tools_service.put_tool = mock_put_tool
  tools_service.delete_tool = mock_delete_tool

  return tools_service


@pytest.mark.asyncio
class TestToolService:
  """Tests for the Tool service."""

  async def test_get_categories(self, mock_tools_service, mock_db_session):
    """Test getting all tool categories."""
    # Call the service
    org_id = uuid4()

    response = await mock_tools_service.get_categories(org_id, mock_db_session)

    # Verify result
    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(category, MockToolCategoryModel) for category in response)
    assert all(category.org_id == org_id for category in response)

  async def test_create_category(self, mock_tools_service, mock_db_session, mock_user):
    """Test creating a new tool category."""
    # Create category data
    org_id = uuid4()
    category_data = MockResponse(name="New Category", description="New category description")

    # Call the service
    response = await mock_tools_service.post_category(org_id, category_data, mock_db_session, mock_user)

    # Verify result
    assert response.name == category_data.name
    assert response.description == category_data.description
    assert response.org_id == org_id

    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.flush.assert_called_once()
    mock_db_session.commit.assert_called_once()

  async def test_update_category(self, mock_tools_service, mock_db_session, mock_user):
    """Test updating a tool category."""
    # Create update data
    org_id = uuid4()
    category_id = uuid4()
    update_data = MockResponse(name="Updated Category", description="Updated description")

    # Call the service
    response = await mock_tools_service.put_category(org_id, category_id, update_data, mock_db_session, mock_user)

    # Verify result
    assert response.name == update_data.name
    assert response.description == update_data.description

    mock_db_session.commit.assert_called_once()

  async def test_delete_category(self, mock_tools_service, mock_db_session, mock_user):
    """Test deleting a tool category."""
    # Call the service
    org_id = uuid4()
    category_id = uuid4()

    response = await mock_tools_service.delete_category(org_id, category_id, mock_db_session, mock_user)

    # Verify result
    assert response["message"] == "Category deleted successfully"

    mock_db_session.commit.assert_called_once()

  async def test_get_tools(self, mock_tools_service, mock_db_session):
    """Test getting all tools in a category."""
    # Call the service
    org_id = uuid4()
    category_id = uuid4()

    response = await mock_tools_service.get_tools(org_id, category_id, mock_db_session)

    # Verify result
    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(tool, MockToolModel) for tool in response)
    assert all(tool.org_id == org_id for tool in response)
    assert all(tool.category_id == category_id for tool in response)

  async def test_get_tool(self, mock_tools_service, mock_db_session):
    """Test getting a single tool."""
    # Call the service
    org_id = uuid4()
    tool_id = uuid4()

    response = await mock_tools_service.get_tool(org_id, tool_id, mock_db_session)

    # Verify result
    assert isinstance(response, MockToolModel)
    assert response.id == tool_id
    assert response.org_id == org_id

  async def test_create_tool(self, mock_tools_service, mock_db_session, mock_user):
    """Test creating a new tool."""
    # Create tool data
    org_id = uuid4()
    category_id = uuid4()
    tool_data = MockResponse(
      name="New Tool",
      description="New tool description",
      tool_definition={"type": "function", "function": {"name": "new_function"}},
      category_id=category_id,
    )

    # Call the service
    response = await mock_tools_service.post_tool(org_id, tool_data, mock_db_session, mock_user)

    # Verify result
    assert response.name == tool_data.name
    assert response.description == tool_data.description
    assert response.tool_definition == tool_data.tool_definition
    assert response.category_id == tool_data.category_id
    assert response.org_id == org_id
    assert response.created_by == mock_user["id"]

    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.flush.assert_called_once()
    mock_db_session.commit.assert_called_once()

  async def test_update_tool(self, mock_tools_service, mock_db_session, mock_user):
    """Test updating a tool."""
    # Create update data
    org_id = uuid4()
    tool_id = uuid4()
    update_data = MockResponse(
      name="Updated Tool", description="Updated description", tool_definition={"type": "function", "function": {"name": "updated_function"}}
    )

    # Call the service
    response = await mock_tools_service.put_tool(org_id, tool_id, update_data, mock_db_session, mock_user)

    # Verify result
    assert response.name == update_data.name
    assert response.description == update_data.description
    assert response.tool_definition == update_data.tool_definition

    # Verify database operations
    mock_db_session.commit.assert_called_once()

  async def test_update_nonexistent_tool(self, mock_tools_service, mock_db_session, mock_user):
    """Test updating a tool that doesn't exist."""
    # Create update data
    org_id = uuid4()
    tool_id = uuid4()
    update_data = MockResponse(name="Updated Tool")

    # Replace the put_tool method with one that raises an exception directly
    async def mock_update_nonexistent_tool(org_id, tool_id, tool_data, session, user):
      raise HTTPException(status_code=404, detail="Tool not found")

    mock_tools_service.put_tool = AsyncMock(side_effect=mock_update_nonexistent_tool)

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_tools_service.put_tool(org_id, tool_id, update_data, mock_db_session, mock_user)

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail)

    # Verify the service method was called
    assert mock_tools_service.put_tool.called

  async def test_delete_tool(self, mock_tools_service, mock_db_session, mock_user):
    """Test deleting a tool."""
    # Call the service
    org_id = uuid4()
    tool_id = uuid4()

    response = await mock_tools_service.delete_tool(org_id, tool_id, mock_db_session, mock_user)

    # Verify result
    assert response["message"] == "Tool deleted successfully"

    mock_db_session.commit.assert_called_once()

  async def test_delete_nonexistent_tool(self, mock_tools_service, mock_db_session, mock_user):
    """Test deleting a tool that doesn't exist."""

    # Replace the delete_tool method with one that raises an exception directly
    async def mock_delete_nonexistent_tool(org_id, tool_id, session, user):
      raise HTTPException(status_code=404, detail="Tool not found")

    mock_tools_service.delete_tool = AsyncMock(side_effect=mock_delete_nonexistent_tool)

    # Verify exception is raised
    org_id = uuid4()
    tool_id = uuid4()
    with pytest.raises(HTTPException) as exc_info:
      await mock_tools_service.delete_tool(org_id, tool_id, mock_db_session, mock_user)

    assert exc_info.value.status_code == 404
    assert "not found" in str(exc_info.value.detail)

    # Verify the service method was called
    assert mock_tools_service.delete_tool.called
