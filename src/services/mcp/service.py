from http import HTTPStatus

import requests
from composio_client import Composio
from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC
from models.auth_model import UserModel
from models.mcp_model import MCPServerModel, MCPSessionModel
from models.mcp_tool_model import MCPToolModel
from services.__base.acquire import Acquire
from services.mcp.schema import (
  MCPConnectedAccountCreate,
  MCPConnectedAccountResponse,
  MCPInstanceCreate,
  MCPInstanceResponse,
  MCPServerCreate,
  MCPServerListResponse,
  MCPServerResponse,
  MCPServerWithToolsResponse,
  MCPSessionListResponse,
)


class MCPService:
  http_exposed = [
    "post=create_server",
    "post=create_instance",
    "post=connect_account",
    "get=list_servers",
    "get=server_with_tools",
    "get=user_sessions",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.logger = acquire.logger
    self.client = Composio(api_key=settings.composio_api_key)

  def _get_auth_config(self, toolkit: str) -> str:
    """Get auth config ID for a toolkit using environment variable naming convention.
    Looks for environment variables like:
    - GMAIL_AUTH_CONFIG for toolkit "gmail"
    - GITHUB_AUTH_CONFIG for toolkit "github"
    - TWITTER_AUTH_CONFIG for toolkit "twitter"
    """
    env_var_name = f"{toolkit.upper()}_AUTH_CONFIG"

    auth_config = getattr(settings, env_var_name.lower(), None)

    if not auth_config:
      raise HTTPException(
        status_code=HTTPStatus.BAD_REQUEST, detail=f"Auth config not found for toolkit '{toolkit}'. Please set {env_var_name} environment variable."
      )

    return auth_config

  async def post_create_server(
    self,
    data: MCPServerCreate,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPServerResponse:
    try:
      name = data.name
      toolkit = data.toolkit
      auth_config_id = self._get_auth_config(toolkit)

      toolkit_slug = toolkit
      result = await session.execute(MCPToolModel.__table__.select().where(MCPToolModel.toolkit == toolkit_slug))
      db_tools = result.fetchall()
      allowed_tools = [row.slug for row in db_tools]

      if not allowed_tools:
        api_url = f"https://backend.composio.dev/api/v3/tools?toolkit_slug={toolkit_slug}"
        headers = {"x-api-key": settings.composio_api_key}
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        items = response.json().get("items", [])
        allowed_tools = []
        for item in items:
          tool = MCPToolModel(
            name=item["name"],
            slug=item["slug"],
            description=item.get("description"),
            toolkit=toolkit_slug,
          )
          session.add(tool)
          allowed_tools.append(item["slug"])
        await session.commit()

      mcp = self.client.mcp.create(
        auth_config_ids=[auth_config_id],
        name=name,
        allowed_tools=allowed_tools,
      )
      db_server = MCPServerModel(
        id=mcp.id,
        name=mcp.name,
        mcp_url=mcp.mcp_url,
        toolkits=mcp.toolkits,
        allowed_tools=mcp.allowed_tools,
        server_instance_count=mcp.server_instance_count,
      )
      session.add(db_server)
      await session.commit()
      await session.refresh(db_server)
      return MCPServerResponse(**mcp.model_dump())
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error creating MCP server: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error creating MCP server")

  async def post_create_instance(
    self,
    data: MCPInstanceCreate,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPInstanceResponse:
    try:
      server_id = data.server_id

      # Get user email from database
      user_id = user.get("id")
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      db_user = user_result.scalar_one_or_none()

      if not db_user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

      user_email = db_user.email

      url = f"https://backend.composio.dev/api/v3/mcp/servers/{server_id}/instances"
      payload = {"user_id": user_email}
      headers = {"x-api-key": settings.composio_api_key, "Content-Type": "application/json"}
      response = requests.post(url, json=payload, headers=headers)
      response.raise_for_status()
      instance = response.json()
      db_instance = MCPSessionModel(
        id=instance["id"],
        instance_id=instance["instance_id"],
        mcp_server_id=instance["mcp_server_id"],
      )
      session.add(db_instance)
      await session.commit()
      await session.refresh(db_instance)
      return MCPInstanceResponse(**instance)
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error creating MCP instance: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error creating MCP instance")

  async def post_connect_account(
    self,
    data: MCPConnectedAccountCreate,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPConnectedAccountResponse:
    try:
      auth_config_id = self._get_auth_config(data.toolkit)
      # Get user email from database
      user_id = user.get("id")
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      db_user = user_result.scalar_one_or_none()

      if not db_user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

      user_email = db_user.email

      # Create a new connected account using Composio API directly
      response = requests.post(
        "https://backend.composio.dev/api/v3/connected_accounts",
        headers={"x-api-key": settings.composio_api_key},
        json={"auth_config": {"id": auth_config_id}, "connection": {"user_id": user_email}},
      )
      response.raise_for_status()
      result = response.json()
      return MCPConnectedAccountResponse(**result)
    except Exception as e:
      self.logger.error(f"Error creating connected account: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error creating connected account")

  async def get_list_servers(
    self,
    user: dict = Depends(RBAC("mcp", "read")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPServerListResponse:
    try:
      # Get all MCP servers from database
      server_query = select(MCPServerModel)
      result = await session.execute(server_query)
      servers = result.scalars().all()

      # Convert to response format
      server_responses = []
      for server in servers:
        server_responses.append(
          MCPServerResponse(
            id=str(server.id),
            name=server.name,
            allowed_tools=server.allowed_tools or [],
            mcp_url=server.mcp_url,
            toolkits=server.toolkits or [],
            updated_at=server.updated_at.isoformat() if server.updated_at else None,
            created_at=server.created_at.isoformat() if server.created_at else None,
            server_instance_count=server.server_instance_count,
          )
        )

      return MCPServerListResponse(servers=server_responses)
    except Exception as e:
      self.logger.error(f"Error listing MCP servers: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error listing MCP servers")

  async def get_server_with_tools(
    self,
    server_id: str,
    user: dict = Depends(RBAC("mcp", "read")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPServerWithToolsResponse:
    try:
      # Get MCP server from database
      server_query = select(MCPServerModel).where(MCPServerModel.id == server_id)
      result = await session.execute(server_query)
      server = result.scalar_one_or_none()

      if not server:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="MCP server not found")

      # Get tools for this server
      tools_query = select(MCPToolModel).where(MCPToolModel.toolkit.in_(server.toolkits or []))
      tools_result = await session.execute(tools_query)
      tools = tools_result.scalars().all()

      tools_list = []
      for tool in tools:
        tools_list.append({
          "name": tool.name,
          "slug": tool.slug,
          "description": tool.description,
          "toolkit": tool.toolkit,
        })

      return MCPServerWithToolsResponse(
        id=str(server.id),
        name=server.name,
        allowed_tools=server.allowed_tools or [],
        mcp_url=server.mcp_url,
        toolkits=server.toolkits or [],
        updated_at=server.updated_at.isoformat() if server.updated_at else None,
        created_at=server.created_at.isoformat() if server.created_at else None,
        server_instance_count=server.server_instance_count,
        tools=tools_list,
      )
    except Exception as e:
      self.logger.error(f"Error getting MCP server with tools: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error getting MCP server with tools")

  async def get_user_sessions(
    self,
    user: dict = Depends(RBAC("mcp", "read")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPSessionListResponse:
    try:
      # Get user email from database
      user_id = user.get("id")
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      db_user = user_result.scalar_one_or_none()
      if not db_user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

      user_email = db_user.email

      # Get MCP sessions for this user (where instance_id matches user_email)
      sessions_query = select(MCPSessionModel).where(MCPSessionModel.instance_id == user_email)
      result = await session.execute(sessions_query)
      db_sessions = result.scalars().all()

      # Convert to response format
      session_responses = []
      for db_session in db_sessions:
        session_responses.append(
          MCPInstanceResponse(
            id=str(db_session.id),
            instance_id=db_session.instance_id,
            mcp_server_id=str(db_session.mcp_server_id),
            created_at=db_session.created_at.isoformat() if db_session.created_at else None,
            updated_at=db_session.updated_at.isoformat() if db_session.updated_at else None,
          )
        )

      return MCPSessionListResponse(sessions=session_responses)
    except Exception as e:
      self.logger.error(f"Error getting user MCP sessions: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error getting user MCP sessions")

  @staticmethod
  def build_instance_url(base_url: str, instance_id: str):
    return f"{base_url}/mcp?user_id={instance_id}"
