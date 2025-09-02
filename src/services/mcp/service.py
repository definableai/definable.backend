import uuid
from http import HTTPStatus

import httpx
from fastapi import Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC
from models.auth_model import UserModel
from models.mcp_model import MCPServerModel, MCPSessionModel, MCPToolkitModel, MCPUserModel
from models.mcp_tool_model import MCPToolModel
from services.__base.acquire import Acquire
from services.mcp.schema import (
  MCPConnectedAccountCreate,
  MCPConnectedAccountResponse,
  MCPGenerateUrlRequest,
  MCPGenerateUrlResponse,
  MCPInstanceCreate,
  MCPInstanceResponse,
  MCPServerListResponse,
  MCPServerResponse,
  MCPServerWithToolsResponse,
  MCPSessionListResponse,
)


class MCPService:
  http_exposed = [
    "post=create_instance",
    "post=connect_account",
    "post=generate_url",
    "get=list_servers",
    "get=server_with_tools",
    "get=user_sessions",
    "get=callback",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.logger = acquire.logger

  async def _get_auth_config(self, server_id: str, session: AsyncSession) -> str:
    """Get auth config ID for an MCP server from database."""
    server_query = select(MCPServerModel).where(MCPServerModel.id == server_id)
    result = await session.execute(server_query)
    server = result.scalar_one_or_none()

    if server and server.auth_config_ids:
      return server.auth_config_ids[0]  # Return first auth config ID

    raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No auth config found for server: {server_id}")

  async def post_create_instance(
    self,
    data: MCPInstanceCreate,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPInstanceResponse:
    try:
      user_id = user.get("id")
      server_id = data.server_id

      # Find the MCP user connection
      mcp_user_query = select(MCPUserModel).where(MCPUserModel.connected_account_id == data.connected_account_id)
      result = await session.execute(mcp_user_query)
      mcp_user = result.scalar_one_or_none()

      if not mcp_user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No connection found. Please connect your account first.")

      if mcp_user.connection_status != "active":
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Account connection is not active. Please complete the connection first.")

      # Create instance using stored composio_user_id
      url = f"https://backend.composio.dev/api/v3/mcp/servers/{server_id}/instances"
      payload = {"user_id": mcp_user.composio_user_id}
      headers = {"x-api-key": settings.composio_api_key, "Content-Type": "application/json"}
      response = httpx.post(url, json=payload, headers=headers)
      response.raise_for_status()
      instance = response.json()

      # Store instance in sessions table
      db_instance = MCPSessionModel(
        id=instance["id"],
        instance_id=instance["instance_id"],
        mcp_server_id=instance["mcp_server_id"],
        user_id=user_id,
        org_id=user.get("org_id"),
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
      auth_config_id = await self._get_auth_config(data.server_id, session)
      user_id = user.get("id")

      # Generate random composio user ID

      composio_user_id = str(uuid.uuid4())

      # Prepare connection payload with backend callback URL
      callback_url = f"{settings.backend_url}/api/mcp/callback?mcp_server_id={data.server_id}&mcp_session_id={composio_user_id}"
      connection_payload = {"user_id": composio_user_id, "callback_url": callback_url}

      # Create connected account using Composio API
      response = httpx.post(
        "https://backend.composio.dev/api/v3/connected_accounts",
        headers={"x-api-key": settings.composio_api_key},
        json={"auth_config": {"id": auth_config_id}, "connection": connection_payload},
      )
      response.raise_for_status()
      result = response.json()

      # Store the connection attempt in our database
      mcp_user = MCPUserModel(
        id=uuid.uuid4(),
        user_id=user_id,
        server_id=data.server_id,
        composio_user_id=composio_user_id,
        connected_account_id=result["id"],
        connection_status="pending",
      )
      session.add(mcp_user)
      await session.commit()

      return MCPConnectedAccountResponse(**result)
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error creating connected account: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error creating connected account")

  async def post_generate_url(
    self,
    data: MCPGenerateUrlRequest,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPGenerateUrlResponse:
    try:
      mcp_user_query = select(MCPUserModel).where((MCPUserModel.connected_account_id == data.account_id) & (MCPUserModel.server_id == data.server_id))
      result = await session.execute(mcp_user_query)
      mcp_user = result.scalar_one_or_none()

      if not mcp_user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No connection attempt found. Please connect your account first.")

      generate_response = httpx.post(
        "https://backend.composio.dev/api/v3/mcp/servers/generate",
        headers={"x-api-key": settings.composio_api_key},
        json={"mcp_server_id": data.server_id, "user_ids": [mcp_user.composio_user_id]},
      )
      generate_response.raise_for_status()
      url_data = generate_response.json()

      if not url_data.get("user_ids_url"):
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to generate MCP URL")

      mcp_url = url_data["user_ids_url"][0]
      await session.commit()

      return MCPGenerateUrlResponse(mcp_url=mcp_url, status="success")

    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error generating MCP URL: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error generating MCP URL")

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

      toolkit_query = select(MCPToolkitModel.id).where(MCPToolkitModel.slug.in_(server.toolkits or []))
      toolkit_result = await session.execute(toolkit_query)
      toolkit_ids = [row[0] for row in toolkit_result.fetchall()]

      # Then get tools that belong to these toolkits
      tools_query = (
        select(MCPToolModel, MCPToolkitModel)
        .join(MCPToolkitModel, MCPToolModel.toolkit_id == MCPToolkitModel.id)
        .where(MCPToolModel.toolkit_id.in_(toolkit_ids))
      )
      tools_result = await session.execute(tools_query)
      tools_data = tools_result.fetchall()

      tools_list = []
      for tool, toolkit in tools_data:
        tools_list.append({
          "name": tool.name,
          "slug": tool.slug,
          "description": tool.description,
          "toolkit": toolkit.name,  # Use toolkit name from the joined table
        })

      return MCPServerWithToolsResponse(
        id=str(server.id),
        name=server.name,
        allowed_tools=server.allowed_tools or [],
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

  async def get_callback(
    self,
    mcp_server_id: str,
    mcp_session_id: str,
    session: AsyncSession = Depends(get_db),
  ) -> RedirectResponse:
    """Handle OAuth callback, update connection status to active, and redirect to frontend."""
    try:
      # Find the connection attempt for this server and session
      mcp_user_query = select(MCPUserModel).where((MCPUserModel.server_id == mcp_server_id) & (MCPUserModel.composio_user_id == mcp_session_id))
      result = await session.execute(mcp_user_query)
      mcp_user = result.scalar_one_or_none()
      frontend_redirect_url = f"{settings.frontend_url}/mcp/callback?mcp_server_id={mcp_server_id}&mcp_session_id={mcp_session_id}"

      if not mcp_user:
        self.logger.warning(f"No connection attempt found for server {mcp_server_id} and session {mcp_session_id}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Connection attempt not found")

      # Update connection status to active
      mcp_user.connection_status = "active"
      await session.commit()

      self.logger.info(f"Successfully activated connection for server {mcp_server_id}, session {mcp_session_id}")

      # Redirect to frontend with parameters
      return RedirectResponse(url=frontend_redirect_url)

    except HTTPException:
      raise
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Error handling callback: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error processing callback")

  @staticmethod
  def build_instance_url(base_url: str, instance_id: str):
    return f"{base_url}/mcp?user_id={instance_id}"
