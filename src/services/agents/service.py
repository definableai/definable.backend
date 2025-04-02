from uuid import UUID

from fastapi import Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from services.__base.acquire import Acquire

from .model import AgentModel
from .schema import AgentResponse, PaginatedAgentResponse


class AgentService:
  """Agent service."""

  http_exposed = ["get=list", "get=list_all"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.models = acquire.models

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
