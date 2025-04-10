from datetime import datetime
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.agent_factory.generator import AgentGenerator
from models import AgentModel, AgentToolModel, LLMModel, ToolModel
from services.__base.acquire import Acquire

from .schema import AgentCreate, AgentResponse, PaginatedAgentResponse


class AgentService:
  """Agent service."""

  http_exposed = ["get=list", "get=list_all", "post=create", "post=update"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def get_list(
    self,
    org_id: UUID,
    offset: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "read")),
  ) -> PaginatedAgentResponse:
    """Get paginated list of agents for an organization."""
    # Base query for total count
    count_query = select(func.count(AgentModel.id)).where(AgentModel.organization_id == org_id)
    total = await session.scalar(count_query)

    # Main query with joins for tools
    query = text("""
        SELECT
            a.*,
            COALESCE(json_agg(
                json_build_object(
                    'id', t.id,
                    'name', t.name,
                    'description', t.description,
                    'category_id', t.category_id,
                    'is_active', t.is_active
                )
            ) FILTER (WHERE t.id IS NOT NULL), '[]') as tools
        FROM agents a
        LEFT JOIN agent_tools at ON a.id = at.agent_id
        LEFT JOIN tools t ON at.tool_id = t.id
        WHERE a.organization_id = :org_id
        GROUP BY a.id
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await session.execute(query, {"org_id": str(org_id), "limit": limit + 1, "offset": offset * limit})
    rows = result.mappings().all()

    # Process results
    agents = []
    for row in rows[:limit]:
      agent_dict = dict(row)
      tools = agent_dict.pop("tools", [])
      agent = AgentResponse(**agent_dict, tools=tools if tools != [None] else [])
      agents.append(agent)

    has_more = len(rows) > limit

    return PaginatedAgentResponse(agents=agents, total=total or 0, has_more=has_more)

  # TODO: Remove this endpoint, we can do it with the get_list endpoint.
  async def get_list_all(
    self,
    offset: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),  # Require admin access
  ) -> PaginatedAgentResponse:
    """Get paginated list of all agents across organizations."""
    # Base query for total count
    count_query = select(func.count(AgentModel.id))
    total = await session.scalar(count_query)

    # Main query with joins for tools
    query = text("""
        SELECT
            a.*,
            COALESCE(json_agg(
                json_build_object(
                    'id', t.id,
                    'name', t.name,
                    'description', t.description,
                    'category_id', t.category_id,
                    'is_active', t.is_active
                )
            ) FILTER (WHERE t.id IS NOT NULL), '[]') as tools
        FROM agents a
        LEFT JOIN agent_tools at ON a.id = at.agent_id
        LEFT JOIN tools t ON at.tool_id = t.id
        GROUP BY a.id
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await session.execute(query, {"limit": limit + 1, "offset": offset * limit})
    rows = result.mappings().all()

    # Process results
    agents = []
    for row in rows[:limit]:
      agent_dict = dict(row)
      tools = agent_dict.pop("tools", [])
      agent = AgentResponse(**agent_dict, tools=tools if tools != [None] else [])
      agents.append(agent)

    has_more = len(rows) > limit

    return PaginatedAgentResponse(agents=agents, total=total or 0, has_more=has_more)

  async def post_create(
    self,
    org_id: UUID,
    agent_data: AgentCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "write")),
  ) -> JSONResponse:
    """Create a new agent and generate its code."""
    self.logger.info("Starting agent creation", org_id=str(org_id), agent_name=agent_data.name, user_id=str(user["id"]))

    # Verify the model exists
    model_query = select(LLMModel).where(LLMModel.id == agent_data.model_id)
    result = await session.execute(model_query)
    model = result.scalar_one_or_none()

    if not model:
      self.logger.error("Model not found", model_id=str(agent_data.model_id))
      raise HTTPException(status_code=404, detail="Model not found")

    # Extract model details to pass to generator
    model_details = {
      "id": str(model.id),
      "name": model.name,  # Actual model name like "gpt-4o"
      "provider": model.provider,  # Provider like "openai"
      "version": model.version,  # Version info
    }

    self.logger.debug("Model verified", model_id=str(agent_data.model_id), model_name=model.name)

    # Check if agent with same name and version exists in this organization
    existing_query = select(AgentModel).where(
      AgentModel.name == agent_data.name, AgentModel.version == agent_data.version, AgentModel.organization_id == org_id
    )
    existing_result = await session.execute(existing_query)
    existing_agent = existing_result.scalar_one_or_none()

    if existing_agent:
      self.logger.error("Agent with this name and version already exists", name=agent_data.name, version=agent_data.version, org_id=str(org_id))
      raise HTTPException(
        status_code=409, detail=f"Agent '{agent_data.name}' with version '{agent_data.version}' already exists in your organization"
      )

    # Process tools with configs
    tools = []
    tool_ids = []  # Still need this for the agent-tool associations

    if agent_data.tool_configs:
      self.logger.info("Fetching tool configurations", tool_count=len(agent_data.tool_configs))

      # Extract tool IDs for database query
      query_tool_ids = [tool_config.tool_id for tool_config in agent_data.tool_configs]
      tool_query = select(ToolModel).where(ToolModel.id.in_(query_tool_ids))
      tool_result = await session.execute(tool_query)
      db_tools = tool_result.scalars().all()

      # Map of tool_id to user-provided config
      config_map = {str(tc.tool_id): tc for tc in agent_data.tool_configs}

      # Verify we found all requested tools
      if len(db_tools) != len(agent_data.tool_configs):
        self.logger.error("One or more tools not found")
        raise HTTPException(status_code=400, detail="One or more tools not found")

      # Prepare tool configurations with user-provided API keys
      for tool in db_tools:
        tool_id = str(tool.id)
        user_config = config_map[tool_id]
        tool_ids.append(tool.id)  # Store for association

        # Prepare the tool config with the API key/secret from user input
        tool_config = {
          "info": {
            "name": tool.name,
            "version": tool.version,
            "description": tool.description,
          },
          "function_info": tool.settings.get("function_info", {}),
          "configuration": tool.configuration,
          "generated_code": tool.generated_code,
          "api_key": user_config.api_key,  # User-provided API key
          "api_secret": user_config.api_secret,  # User-provided API secret
          "custom_config": user_config.config,  # Any additional config
        }

        # Add any other required fields
        if "requirements" in tool.settings:
          tool_config["requirements"] = tool.settings["requirements"]

        if "deployment" in tool.settings:
          tool_config["deployment"] = tool.settings["deployment"]

        tools.append(tool_config)

    # Store agent configuration
    settings = {
      "provider": agent_data.provider,
      "system_prompt": agent_data.system_prompt,
      "instructions": agent_data.instructions,
      "expected_output": agent_data.expected_output,
      "memory_config": agent_data.memory_config,
      "knowledge_base": agent_data.knowledge_base,
      "version": agent_data.version,
    }

    self.logger.info(f"settings : {settings}")

    # Create agent in database but don't commit yet
    db_agent = AgentModel(
      organization_id=org_id,
      user_id=user["id"],
      name=agent_data.name,
      description=agent_data.description,
      model_id=agent_data.model_id,
      is_active=agent_data.is_active,
      settings=settings,
      version=agent_data.version,
    )

    session.add(db_agent)
    await session.flush()  # To get the ID, but not committed yet

    # Create agent-tool associations but don't commit
    agent_tools = []
    for tool_id in tool_ids:
      agent_tool = AgentToolModel(agent_id=db_agent.id, tool_id=tool_id, is_active=True, added_at=datetime.now())
      session.add(agent_tool)
      agent_tools.append(agent_tool)

    try:
      # Try generating the agent code, now with model_details
      agent_generator = AgentGenerator(logger=self.logger)
      generation_result = await agent_generator.generate_agent(
        agent_name=agent_data.name.lower().replace(" ", "_"),
        provider=agent_data.provider,
        model_details=model_details,  # Pass complete model info
        tools=tools,
        description=agent_data.description,
        system_prompt=agent_data.system_prompt,
        instructions=agent_data.instructions,
        expected_output=agent_data.expected_output,
        memory_config=agent_data.memory_config,
        knowledge_base=agent_data.knowledge_base,
        version=agent_data.version,
      )

      # If code generation succeeded, then commit to database
      db_agent.settings["deployment"] = {
        "status": "generated",
        "deployment_path": generation_result["deployment"]["deployment_path"],
        "version_path": generation_result["deployment"]["version_path"],
        "generated_at": datetime.now().isoformat(),
      }

      await session.commit()

      # Success response
      return JSONResponse(
        status_code=201,
        content={
          "message": "Agent created and code generated successfully",
          "id": str(db_agent.id),
          "deployment_info": generation_result["deployment"],
        },
      )

    except Exception as e:
      # If code generation failed, roll back the transaction
      await session.rollback()

      # Return error response
      return JSONResponse(status_code=400, content={"message": "Agent creation failed due to code generation error", "error": str(e)})

  async def post_update(
    self,
    org_id: UUID,
    agent_id: UUID,
    agent_data: AgentCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "write")),
  ) -> JSONResponse:
    """Update an existing agent if it has not been deployed yet."""
    self.logger.info("Updating agent", org_id=str(org_id), agent_id=str(agent_id), user_id=str(user["id"]))

    # Verify agent exists and belongs to this organization
    agent_query = select(AgentModel).where(AgentModel.id == agent_id, AgentModel.organization_id == org_id)
    result = await session.execute(agent_query)
    agent = result.scalar_one_or_none()

    if not agent:
      self.logger.error("Agent not found or not authorized", agent_id=str(agent_id), org_id=str(org_id))
      raise HTTPException(status_code=404, detail="Agent not found or not authorized")

    # Check if agent has been deployed
    if agent.settings.get("deployment", {}).get("status") == "generated":
      self.logger.error("Cannot update agent after deployment", agent_id=str(agent_id))
      return JSONResponse(status_code=400, content={"message": "Cannot update agent after deployment. Create a new version instead."})

    # Verify the model exists
    model_query = select(LLMModel).where(LLMModel.id == agent_data.model_id)
    result = await session.execute(model_query)
    model = result.scalar_one_or_none()

    if not model:
      self.logger.error("Model not found", model_id=str(agent_data.model_id))
      raise HTTPException(status_code=404, detail="Model not found")

    # Update agent properties
    agent.name = agent_data.name
    agent.description = agent_data.description
    agent.model_id = agent_data.model_id
    agent.is_active = agent_data.is_active
    agent.version = agent_data.version

    # Update settings
    agent.settings = {
      "provider": agent_data.provider,
      "system_prompt": agent_data.system_prompt,
      "instructions": agent_data.instructions,
      "expected_output": agent_data.expected_output,
      "memory_config": agent_data.memory_config,
      "knowledge_base": agent_data.knowledge_base,
      "version": agent_data.version,
    }

    # Update tool associations if needed
    if agent_data.tool_configs:
      # Remove existing tool associations
      delete_query = text("""
            DELETE FROM agent_tools
            WHERE agent_id = :agent_id
        """)
      await session.execute(delete_query, {"agent_id": str(agent_id)})

      # Add new tool associations
      for tool_config in agent_data.tool_configs:
        agent_tool = AgentToolModel(agent_id=agent_id, tool_id=tool_config.tool_id, is_active=True, added_at=datetime.now())
        session.add(agent_tool)

    # Commit changes
    await session.commit()

    return JSONResponse(status_code=200, content={"message": "Agent updated successfully", "id": str(agent.id)})
