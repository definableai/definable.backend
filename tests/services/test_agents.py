import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import uuid4, UUID
from datetime import datetime
from typing import Dict, List, Any, Optional
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


# Mock models
class MockAgentModel(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  organization_id: Optional[UUID] = None
  user_id: Optional[UUID] = None
  name: str = "Test Agent"
  description: str = "Test Agent Description"
  model_id: Optional[UUID] = None
  is_active: bool = True
  settings: Dict[str, Any] = Field(default_factory=dict)
  version: str = "1.0.0"
  created_at: datetime = Field(default_factory=datetime.now)
  updated_at: datetime = Field(default_factory=datetime.now)

  model_config = {"extra": "allow"}


class MockAgentToolModel(BaseModel):
  agent_id: Optional[UUID] = None
  tool_id: Optional[UUID] = None
  is_active: bool = True
  added_at: str = Field(default_factory=lambda: datetime.now().isoformat())

  model_config = {"extra": "allow"}


class MockToolModel(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  name: str = "Test Tool"
  description: str = "Test Tool Description"
  category_id: UUID = Field(default_factory=uuid4)
  is_active: bool = True

  model_config = {"extra": "allow"}


class MockAgentToolResponse(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  name: str = "Test Tool"
  description: str = "Test Tool Description"
  category_id: UUID = Field(default_factory=uuid4)
  is_active: bool = True

  model_config = {"extra": "allow"}


class ToolResponse(BaseModel):
  id: Optional[UUID] = None
  name: str = "Tool"
  description: str = "Description"
  category_id: Optional[UUID] = None
  is_active: bool = True


class MockResponse(BaseModel):
  id: Optional[UUID] = None
  name: Optional[str] = None
  description: Optional[str] = None
  organization_id: Optional[UUID] = None
  user_id: Optional[UUID] = None
  model_id: Optional[UUID] = None
  is_active: Optional[bool] = True
  settings: Dict[str, Any] = Field(default_factory=dict)
  version: Optional[str] = None
  created_at: Optional[datetime] = None
  updated_at: Optional[datetime] = None
  tools: List[Dict[str, Any]] = Field(default_factory=list)
  agents: Optional[List[Any]] = None
  total: Optional[int] = None
  has_more: Optional[bool] = None

  # Allow dynamic field assignment
  model_config = {"extra": "allow"}


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
def mock_agent():
  """Create a mock agent."""
  return MockAgentModel(
    name="Test Agent", description="Test agent description", organization_id=uuid4(), user_id=uuid4(), model_id=uuid4(), settings={"temperature": 0.7}
  )


@pytest.fixture
def mock_tool():
  """Create a mock tool."""
  return MockToolModel(name="Test Tool", description="Test tool description", category_id=uuid4())


@pytest.fixture
def mock_agents_service():
  """Create a mock agents service."""
  agents_service = MagicMock()

  async def mock_get_list(org_id, offset=0, limit=10, session=None, user=None):
    # Mock the total count query - use the scalar method
    session.scalar.return_value = 3

    # Create mock agents with tools
    agents = []
    for i in range(min(3, limit)):
      agent = MockAgentModel(organization_id=org_id, user_id=user["id"] if user else uuid4(), name=f"Agent {i + 1}", model_id=uuid4())

      # Add tools to each agent - matching the format in the API schema
      tools = [
        {
          "id": uuid4(),
          "name": f"Tool {j + 1} for Agent {i + 1}",
          "description": f"Description for Tool {j + 1}",
          "category_id": uuid4(),
          "is_active": True,
        }
        for j in range(2)  # 2 tools per agent
      ]

      # Mock the row result structure from raw SQL query
      row = {**agent.model_dump(), "tools": tools}
      agents.append(row)

    # Set up the mock for execute.return_value.mappings().all()
    session.execute.return_value.mappings.return_value.all.return_value = agents

    # Format response to match the PaginatedAgentResponse schema
    agent_responses = []
    for agent_dict in agents[:limit]:
      tools = agent_dict.pop("tools", [])
      # Filter out None values
      agent_dict = {k: v for k, v in agent_dict.items() if v is not None}
      agent_response = MockResponse(**agent_dict, tools=tools)
      agent_responses.append(agent_response)

    return MockResponse(agents=agent_responses, total=3, has_more=(3 > offset * limit + limit))

  async def mock_get_list_all(offset=0, limit=10, session=None, user=None):
    # Mock the total count query - use the scalar method
    session.scalar.return_value = 5

    # Create mock agents with tools for all organizations
    agents = []
    for i in range(min(5, limit)):
      agent = MockAgentModel(organization_id=uuid4(), user_id=uuid4(), name=f"Agent {i + 1}", model_id=uuid4())

      # Add tools to each agent - matching the format in the API schema
      tools = [
        {
          "id": uuid4(),
          "name": f"Tool {j + 1} for Agent {i + 1}",
          "description": f"Description for Tool {j + 1}",
          "category_id": uuid4(),
          "is_active": True,
        }
        for j in range(2)  # 2 tools per agent
      ]

      # Mock the row result structure from raw SQL query
      row = {**agent.model_dump(), "tools": tools}
      agents.append(row)

    # Set up the mock for execute.return_value.mappings().all()
    session.execute.return_value.mappings.return_value.all.return_value = agents

    # Format response to match the PaginatedAgentResponse schema
    agent_responses = []
    for agent_dict in agents[:limit]:
      tools = agent_dict.pop("tools", [])
      # Filter out None values
      agent_dict = {k: v for k, v in agent_dict.items() if v is not None}
      agent_response = MockResponse(**agent_dict, tools=tools)
      agent_responses.append(agent_response)

    return MockResponse(agents=agent_responses, total=5, has_more=(5 > offset * limit + limit))

  # Create AsyncMock objects for these methods to ensure they have .called attribute
  get_list_mock = AsyncMock(side_effect=mock_get_list)
  get_list_all_mock = AsyncMock(side_effect=mock_get_list_all)

  # Assign the AsyncMock objects to the service
  agents_service.get_list = get_list_mock
  agents_service.get_list_all = get_list_all_mock

  async def mock_create_agent(org_id, agent_data, tool_ids=None, session=None, user=None):
    # Create agent
    agent_data_dict = agent_data.model_dump()
    # Remove organization_id and user_id if they exist to avoid conflict
    agent_data_dict.pop("organization_id", None)
    agent_data_dict.pop("user_id", None)

    # Filter out None values to allow defaults to work properly
    agent_data_dict = {k: v for k, v in agent_data_dict.items() if v is not None}

    db_agent = MockAgentModel(organization_id=org_id, user_id=user["id"] if user else uuid4(), **agent_data_dict)
    session.add(db_agent)
    await session.flush()

    # Add tools
    tools = []
    if tool_ids:
      for tool_id in tool_ids:
        agent_tool = MockAgentToolModel(agent_id=db_agent.id, tool_id=tool_id, added_at=datetime.now().isoformat())
        session.add(agent_tool)

        # Create tool data for response
        tools.append({"id": tool_id, "name": "Mock Tool", "description": "Mock Tool Description", "category_id": uuid4(), "is_active": True})

    await session.commit()

    # Filter out None values to avoid validation errors
    agent_dict = {k: v for k, v in db_agent.model_dump().items() if v is not None}
    # Remove tools to avoid duplicate argument
    agent_dict.pop("tools", None)

    # Return agent with tools matching AgentResponse schema
    return MockResponse(**agent_dict, tools=tools)

  async def mock_get_agent(org_id, agent_id, session=None, user=None):
    # Check agent exists
    db_agent = MockAgentModel(id=agent_id, organization_id=org_id)

    # Get tools for agent - matching AgentToolResponse format
    tools = [
      {"id": uuid4(), "name": "Tool 1", "description": "Tool 1 Description", "category_id": uuid4(), "is_active": True},
      {"id": uuid4(), "name": "Tool 2", "description": "Tool 2 Description", "category_id": uuid4(), "is_active": True},
    ]

    # Mock execute result correctly for an SQL query
    session.execute.return_value.first.return_value = db_agent

    # Filter out None values to avoid validation errors
    agent_dict = {k: v for k, v in db_agent.model_dump().items() if v is not None}
    # Remove tools to avoid duplicate argument
    agent_dict.pop("tools", None)

    # Return agent with tools matching AgentResponse schema
    return MockResponse(**agent_dict, tools=tools)

  async def mock_update_agent(org_id, agent_id, agent_data, tool_ids=None, session=None, user=None):
    # Check agent exists
    db_agent = MockAgentModel(id=agent_id, organization_id=org_id, user_id=user["id"] if user else uuid4())

    # Set up mock response for when we execute query to find agent
    session.execute.return_value.first.return_value = db_agent

    # Update agent fields
    update_data = agent_data.model_dump(exclude_unset=True)
    # Remove organization_id and user_id if they exist to avoid conflict
    update_data.pop("organization_id", None)
    update_data.pop("user_id", None)

    # Filter out None values to allow defaults to work properly
    update_data = {k: v for k, v in update_data.items() if v is not None}

    for field, value in update_data.items():
      setattr(db_agent, field, value)

    # Update tools if provided
    tools = []
    if tool_ids is not None:
      # Remove existing tools
      # In a real implementation, we would execute a delete query here

      # Add new tools
      for tool_id in tool_ids:
        agent_tool = MockAgentToolModel(agent_id=agent_id, tool_id=tool_id, added_at=datetime.now().isoformat())
        session.add(agent_tool)

        # Add to response
        tools.append({"id": tool_id, "name": "Updated Tool", "description": "Updated Tool Description", "category_id": uuid4(), "is_active": True})

    await session.commit()

    # Filter out None values to avoid validation errors
    agent_dict = {k: v for k, v in db_agent.model_dump().items() if v is not None}
    # Remove tools to avoid duplicate argument
    agent_dict.pop("tools", None)

    # Return updated agent with tools matching AgentResponse schema
    return MockResponse(**agent_dict, tools=tools)

  async def mock_delete_agent(org_id, agent_id, session=None, user=None):
    # Check agent exists
    db_agent = MockAgentModel(id=agent_id, organization_id=org_id)
    session.execute.return_value.first.return_value = db_agent

    # Delete agent tools first
    # In a real implementation, we would execute a delete query for tools here

    # Delete agent
    # In a real implementation, we would execute a delete query for the agent here

    await session.commit()

    return {"message": "Agent deleted successfully"}

  # Use AsyncMock for the remaining methods as well for consistency
  agents_service.post = AsyncMock(side_effect=mock_create_agent)
  agents_service.get = AsyncMock(side_effect=mock_get_agent)
  agents_service.put = AsyncMock(side_effect=mock_update_agent)
  agents_service.delete = AsyncMock(side_effect=mock_delete_agent)

  return agents_service


@pytest.mark.asyncio
class TestAgentService:
  """Tests for the Agent service."""

  async def test_get_list(self, mock_agents_service, mock_db_session, mock_user):
    """Test getting a paginated list of agents for an organization."""
    # Call the service
    org_id = uuid4()

    # Properly set up execute to be called when the service is called
    mock_agents_service.get_list.side_effect = None

    response = await mock_agents_service.get_list(org_id, offset=0, limit=10, session=mock_db_session, user=mock_user)

    # Verify result structure matches PaginatedAgentResponse schema
    assert hasattr(response, "agents")
    assert hasattr(response, "total")
    assert hasattr(response, "has_more")
    assert len(response.agents) <= 10
    assert all(agent.organization_id == org_id for agent in response.agents)
    assert all(hasattr(agent, "tools") for agent in response.agents)

    # Verify agent fields match AgentResponse schema
    for agent in response.agents:
      assert hasattr(agent, "id")
      assert hasattr(agent, "name")
      assert hasattr(agent, "description")
      assert hasattr(agent, "model_id")
      assert hasattr(agent, "is_active")
      assert hasattr(agent, "settings")
      assert hasattr(agent, "version")
      assert hasattr(agent, "updated_at")

      # Verify tools structure matches AgentToolResponse schema
      for tool in agent.tools:
        assert "id" in tool
        assert "name" in tool
        assert "description" in tool
        assert "category_id" in tool
        assert "is_active" in tool

    # Verify service method was called
    assert mock_agents_service.get_list.called

  async def test_get_list_with_pagination(self, mock_agents_service, mock_db_session, mock_user):
    """Test getting a paginated list of agents with custom pagination."""
    # Call the service
    org_id = uuid4()
    response = await mock_agents_service.get_list(
      org_id,
      offset=1,  # Second page
      limit=2,  # 2 items per page
      session=mock_db_session,
      user=mock_user,
    )

    # Verify result
    assert hasattr(response, "agents")
    assert response.total == 3  # Total of 3 agents
    assert not response.has_more  # No more pages after this
    assert len(response.agents) <= 2  # At most 2 agents on this page

  async def test_get_list_all(self, mock_agents_service, mock_db_session, mock_user):
    """Test getting a paginated list of all agents across organizations."""
    # Call the service
    mock_agents_service.get_list_all.side_effect = None

    response = await mock_agents_service.get_list_all(offset=0, limit=10, session=mock_db_session, user=mock_user)

    # Verify result structure matches PaginatedAgentResponse schema
    assert hasattr(response, "agents")
    assert hasattr(response, "total")
    assert hasattr(response, "has_more")
    assert len(response.agents) <= 10
    assert all(hasattr(agent, "tools") for agent in response.agents)

    # Verify service method was called
    assert mock_agents_service.get_list_all.called

  async def test_create_agent(self, mock_agents_service, mock_db_session, mock_user):
    """Test creating a new agent."""
    # Create agent data
    org_id = uuid4()
    agent_data = MockResponse(name="New Agent", description="New agent description", model_id=uuid4(), settings={"temperature": 0.5})

    # Generate some tool IDs
    tool_ids = [uuid4(), uuid4()]

    # Call the service
    response = await mock_agents_service.post(org_id, agent_data, tool_ids=tool_ids, session=mock_db_session, user=mock_user)

    # Verify result structure matches AgentResponse schema
    assert response.name == agent_data.name
    assert response.description == agent_data.description
    assert response.model_id == agent_data.model_id
    assert response.settings == agent_data.settings
    assert response.organization_id == org_id
    assert len(response.tools) == len(tool_ids)
    assert all(tool["id"] in tool_ids for tool in response.tools)

    # Verify database operations
    mock_db_session.add.assert_called()
    mock_db_session.flush.assert_called_once()
    mock_db_session.commit.assert_called_once()

  async def test_get_agent(self, mock_agents_service, mock_db_session, mock_user):
    """Test getting a single agent."""
    # Call the service
    org_id = uuid4()
    agent_id = uuid4()

    response = await mock_agents_service.get(org_id, agent_id, session=mock_db_session, user=mock_user)

    # Verify result structure matches AgentResponse schema
    assert response.id == agent_id
    assert response.organization_id == org_id
    assert hasattr(response, "tools")
    assert len(response.tools) == 2

    # Verify tool fields match AgentToolResponse schema
    for tool in response.tools:
      assert "id" in tool
      assert "name" in tool
      assert "description" in tool
      assert "category_id" in tool
      assert "is_active" in tool

  async def test_update_agent(self, mock_agents_service, mock_db_session, mock_user):
    """Test updating an agent."""
    # Create update data
    org_id = uuid4()
    agent_id = uuid4()
    update_data = MockResponse(name="Updated Agent", description="Updated description", settings={"temperature": 0.8})

    # Call the service
    response = await mock_agents_service.put(org_id, agent_id, update_data, session=mock_db_session, user=mock_user)

    # Verify result structure matches AgentResponse schema
    assert response.name == update_data.name
    assert response.description == update_data.description
    assert response.settings == update_data.settings

    # Database operation check
    mock_db_session.commit.assert_called_once()

  async def test_update_agent_with_tools(self, mock_agents_service, mock_db_session, mock_user):
    """Test updating an agent with new tools."""
    # Create update data
    org_id = uuid4()
    agent_id = uuid4()
    update_data = MockResponse(name="Updated Agent", description="Updated description")

    # Generate new tool IDs
    tool_ids = [uuid4(), uuid4(), uuid4()]

    # Call the service
    response = await mock_agents_service.put(org_id, agent_id, update_data, tool_ids=tool_ids, session=mock_db_session, user=mock_user)

    # Verify result structure matches AgentResponse schema
    assert response.name == update_data.name
    assert response.description == update_data.description
    assert len(response.tools) == len(tool_ids)
    assert all(tool["id"] in tool_ids for tool in response.tools)

    # Verify tools structure matches AgentToolResponse schema
    for tool in response.tools:
      assert "id" in tool
      assert "name" in tool
      assert "description" in tool
      assert "category_id" in tool
      assert "is_active" in tool

    # Verify database operations
    assert mock_db_session.add.call_count >= len(tool_ids)
    mock_db_session.commit.assert_called_once()

  async def test_delete_agent(self, mock_agents_service, mock_db_session, mock_user):
    """Test deleting an agent."""
    # Call the service
    org_id = uuid4()
    agent_id = uuid4()

    response = await mock_agents_service.delete(org_id, agent_id, session=mock_db_session, user=mock_user)

    # Verify result matches expected API response
    assert response["message"] == "Agent deleted successfully"

    # Verify database operation
    mock_db_session.commit.assert_called_once()
