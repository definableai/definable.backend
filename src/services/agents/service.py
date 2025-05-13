from http import HTTPStatus
from typing import List, Optional, Tuple
from fastapi.responses import StreamingResponse
from httpx import AsyncClient, HTTPStatusError, Timeout
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config.settings import settings
from database import get_db
from dependencies.security import RBAC, JWTBearer
from models import AgentModel
from models.agent_model import AgentCategoryModel, AgentToolModel
from models.llm_model import LLMModel
from models.tool_model import ToolModel
from services.__base.acquire import Acquire

from .schema import AgentCategoryResponse, AgentCreate, AgentResponse, AgentToolResponse, PaginatedAgentResponse


class AgentService:
  """Agent service."""

  http_exposed = [
    "post=create",
    "get=list",
    "get=search",
    "get=hello",
    "get=by_category",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def post_create(
    self,
    org_id: UUID,
    data: AgentCreate,
    category_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "write")),
  ) -> AgentResponse:
    """Create a new agent."""
    try:
      # Check if the model exists
      query = select(LLMModel).where(LLMModel.id == data.model_id)
      model = await session.scalar(query)

      if not model:
        raise HTTPException(
          status_code=HTTPStatus.NOT_FOUND,
          detail="Model not found",
        )

      if not model.is_active:
        raise HTTPException(
          status_code=HTTPStatus.BAD_REQUEST,
          detail="Model is not active",
        )

      # Extract data
      user_id = user["id"]
      org_id = org_id

      # Create the agent instance
      agent = AgentModel(
        name=data.name,
        description=data.description,
        model_id=data.model_id,
        is_active=data.is_active,
        version=data.version,
        settings=data.settings or {},
        organization_id=org_id,
        user_id=user_id,
        category_id=category_id,
      )

      # Add the agent to the session and commit
      session.add(agent)
      await session.commit()
      await session.refresh(agent)

      # Get the category
      if category_id:
        _, category = await self._get_agents_and_category(category_id=category_id, session=session)

      # Return the created agent
      response_data = AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        model_id=agent.model_id,
        is_active=agent.is_active,
        version=agent.version,
        settings=agent.settings,
        organization_id=agent.organization_id,
        updated_at=agent.updated_at,
        tools=[],  # No tools are added during creation
        category=category.name if category else None,
      )

      return response_data
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error creating agent: {e}")
      raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail="Error creating agent",
      )

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
    # Main query with joins for tools and includes properties
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
            ) FILTER (WHERE t.id IS NOT NULL), '[]') as tools,
            COALESCE(a.properties, '{}'::jsonb) as properties  -- Explicitly include properties
        FROM agents a
        LEFT JOIN agent_tools at ON a.id = at.agent_id
        LEFT JOIN tools t ON at.tool_id = t.id
        WHERE a.organization_id = :org_id
        GROUP BY a.id
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await session.execute(
      query,
      {
        "org_id": str(org_id),
        "limit": limit + 1,
        "offset": offset * limit,
      },
    )
    rows = result.mappings().all()

    # Process results
    agents = []
    for row in rows[:limit]:
      agent_dict = dict(row)
      tools = agent_dict.pop("tools", [])
      agent = AgentResponse(
        **agent_dict,
        tools=tools if tools != [None] else [],
        properties=agent_dict.pop("properties", {}),
      )
      agents.append(agent)

    has_more = len(rows) > limit

    return PaginatedAgentResponse(agents=agents, total=total or 0, has_more=has_more)

  # TODO: Remove this endpoint, we can do it with the get_list endpoint.
  # TODO: Reduce the number of queries to the database. Currently, we are fetching all agents and then fetching the tools for each agent.
  async def get_search(
    self,
    offset: int = 0,
    limit: int = 10,
    category_id: Optional[UUID] = None,
    search: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> PaginatedAgentResponse:
    """Get paginated list of all agents across organizations."""
    # Build filter conditions
    filters = []
    if category_id:
      filters.append(AgentModel.category_id == category_id)
    if search:
      filters.append(
        or_(
          AgentModel.name.ilike(f"%{search}%"),
          AgentModel.description.ilike(f"%{search}%"),
        )
      )

    # Base query for total count
    count_query = select(func.count(AgentModel.id)).where(*filters)
    total = await session.scalar(count_query)

    # Main query - first get just the agents with their categories
    query = (
      select(AgentModel)
      .options(selectinload(AgentModel.category))
      .where(*filters)
      .order_by(AgentModel.created_at.desc())
      .offset(offset * limit)
      .limit(limit + 1)
    )

    # Execute the query
    result = await session.execute(query)
    rows = result.scalars().all()

    # Now separately fetch tools for these agents (if any)
    agent_ids = [agent.id for agent in rows[:limit]]
    tools_query = (
      select(AgentToolModel, ToolModel).join(ToolModel, AgentToolModel.tool_id == ToolModel.id).where(AgentToolModel.agent_id.in_(agent_ids))
    )

    tools_result = await session.execute(tools_query)
    tools_by_agent: dict[UUID, list[dict]] = {}

    # Organize tools by agent_id
    for agent_tool, tool in tools_result:
      if agent_tool.agent_id not in tools_by_agent:
        tools_by_agent[agent_tool.agent_id] = []

      tools_by_agent[agent_tool.agent_id].append({
        "id": tool.id,
        "name": tool.name,
        "description": tool.description,
        "category_id": tool.category_id,
        "is_active": tool.is_active,
      })

    # Process results
    agents = []
    for agent in rows[:limit]:
      agent_tools = tools_by_agent.get(agent.id, [])

      agent_response = AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        model_id=agent.model_id,
        is_active=agent.is_active,
        settings=agent.settings,
        version=agent.version,
        organization_id=agent.organization_id,
        updated_at=agent.updated_at,
        tools=[AgentToolResponse(**tool) for tool in agent_tools],
        properties=agent.properties,
        category=agent.category.name if agent.category else None,
      )
      agents.append(agent_response)

    has_more = len(rows) > limit

    return PaginatedAgentResponse(agents=agents, total=total or 0, has_more=has_more)

  async def get_hello(self, agent_id: UUID, session: AsyncSession = Depends(get_db)):
    """Get introduction from the agent."""
    try:
      query = select(AgentModel).where(AgentModel.id == agent_id)
      agent = await session.scalar(query)
      if not agent:
        raise HTTPException(
          status_code=HTTPStatus.NOT_FOUND,
          detail="Agent not found",
        )

      if not agent.is_active:
        raise HTTPException(
          status_code=HTTPStatus.BAD_REQUEST,
          detail="Agent is not active",
        )

      async def generate():
        async with AsyncClient(timeout=Timeout(60.0)) as client:
          url = f"{settings.agent_base_url}/hello"
          async with client.stream("GET", url) as response:
            if response.status_code != 200:
              raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Error calling the agent",
              )

            async for chunk in response.aiter_text():
              yield chunk
              if "DONE" in chunk:
                break

      return StreamingResponse(generate(), media_type="text/plain")

    except HTTPStatusError as e:
      raise HTTPException(
        status_code=e.response.status_code,
        detail=f"Agent returned error: {e.response.text}",
      )
    except Exception:
      raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail="Error calling the agent",
      )

  async def get_by_category(
    self,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> List[AgentCategoryResponse]:
    """Get all agents grouped by category with counts."""
    # Get all active categories
    query = select(AgentCategoryModel).where(AgentCategoryModel.is_active.is_(True))
    categories = (await session.scalars(query)).all()

    # Get agents for each category
    result = []
    for category in categories:
      # Get agents for this category
      query = select(AgentModel).where(AgentModel.category_id == category.id, AgentModel.is_active.is_(True))
      agents = (await session.scalars(query)).all()

      result.append(
        AgentCategoryResponse(
          id=category.id,
          name=category.name,
          description=category.description,
          agent_count=len(agents)
        )
      )

    return result

  ### PRIVATE METHODS ###
  async def _get_agents_and_category(
    self,
    agent_ids: Optional[List[UUID]] = None,
    category_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    require_active_category: bool = True,
  ) -> Tuple[List[AgentModel], Optional[AgentCategoryModel]]:
    """
    Unified method to get agents and optionally their category.
    Returns tuple of (agents, category) where category is None if not requested.
    """
    try:
      # Validate inputs
      if not agent_ids and not category_id:
        raise HTTPException(
          status_code=HTTPStatus.BAD_REQUEST,
          detail="Either agent_ids or category_id must be provided",
        )

      # Get category if requested
      category = None
      if category_id:
        query = select(AgentCategoryModel).where(AgentCategoryModel.id == category_id)
        category = await session.scalar(query)

        if not category:
          raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Agent category not found",
          )

        if require_active_category and not category.is_active:
          raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Agent category is not active",
          )

      # Build agents query
      query = select(AgentModel).where(AgentModel.is_active.is_(True))

      if agent_ids:
        query = query.where(AgentModel.id.in_(agent_ids))
      if category_id:
        query = query.where(AgentModel.category_id == category_id)

      agents = list((await session.scalars(query)).all())

      if not agents:
        raise HTTPException(
          status_code=HTTPStatus.NOT_FOUND,
          detail="Agents not found",
        )

      return agents, category

    except HTTPException:
      raise
    except Exception:
      raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail="Error getting agents and category",
      )
