from http import HTTPStatus
from typing import Union
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from models import AgentModel, AgentAnalyticsModel, MessageModel, UserModel
from services.__base.acquire import Acquire

from .schema import AgentAnalyticsSchema, AgentResponse, PaginatedAgentResponse


class AgentService:
  """Agent service."""

  http_exposed = ["get=list", "get=list_all", "post=analytics"]

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

  async def post_analytics(
    self,
    agent_id: UUID,
    session_id: UUID,
    data: AgentAnalyticsSchema,
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Post analytics data."""
    try:
      # Check if the agent exists
      await self._if_exists(AgentModel, agent_id, session, "Agent not found")

      # Check if the session exists
      await self._if_exists(MessageModel, session_id, session, "Session not found")

      # Check if the user exists
      await self._if_exists(UserModel, data.user_id, session, "User not found")

      analytics = AgentAnalyticsModel(
        agent_id=agent_id,
        session_id=session_id,
        user_id=data.user_id,
        org_id=data.org_id,
        memory=data.memory,
        agent_data=data.agent_data,
        session_data=data.session_data,
      )

      # Add the instance to the session and commit
      session.add(analytics)
      await session.commit()

      return JSONResponse(
        content={
          "message": "Analytics data posted successfully",
        },
        status_code=HTTPStatus.CREATED,
      )

    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error posting analytics data: {e}")
      raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail="Error posting analytics data",
      )

  ### PRIVATE METHODS ###

  async def _if_exists(
    self,
    model: Union[AgentModel, MessageModel, UserModel],
    record_id: UUID,
    session: AsyncSession,
    error_message: str = "Record not found",
  ) -> bool:
    """Generic function to check if a record exists in the database"""

    try:
      query: Select = select(model).where(model.id == record_id)
      record = await session.scalar(query)
      if not record:
        raise HTTPException(
          status_code=HTTPStatus.NOT_FOUND,
          detail=error_message,
        )
      return True
    except Exception as e:
      self.logger.error(f"Error checking if record exists: {e}")
      raise HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail="Error checking record",
      )

