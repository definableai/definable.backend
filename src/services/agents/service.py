from http import HTTPStatus
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from models import AgentModel
from models.llm_model import LLMModel
from services.__base.acquire import Acquire

from .schema import AgentCreate, AgentResponse, PaginatedAgentResponse


class AgentService:
  """Agent service."""

  http_exposed = ["post=create", "get=list", "get=list_all"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def post_create(
    self,
    model_id: UUID,
    data: AgentCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "create")),
  ) -> JSONResponse:
    """Create a new agent."""
    try:
      # Check if the model exists
      query = select(LLMModel).where(LLMModel.id == model_id)
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
      org_id = user["org_id"]

      # Create the agent instance
      agent = AgentModel(
        name=data.name,
        description=data.description,
        model_id=model_id,
        is_active=data.is_active,
        version=data.version,
        settings=data.settings or {},
        organization_id=org_id,
        user_id=user_id,
      )

      # Add the agent to the session and commit
      session.add(agent)
      await session.commit()
      await session.refresh(agent)

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
      )
      return JSONResponse(
        content=response_data.model_dump(),
        status_code=HTTPStatus.CREATED,
      )
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
