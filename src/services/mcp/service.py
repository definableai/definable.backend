import uuid
from http import HTTPStatus
from typing import List, Optional

from fastapi import Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC
from libs.composio.v1 import composio
from models.auth_model import UserModel
from models.mcp_model import MCPServerModel, MCPSessionModel
from models.mcp_tool_model import MCPToolModel
from services.__base.acquire import Acquire
from services.mcp.schema import (
  MCPConnectedAccountCreate,
  MCPConnectedAccountResponse,
  MCPGenerateUrlRequest,
  MCPGenerateUrlResponse,
  MCPInstanceResponse,
  MCPServerListResponse,
  MCPServerResponse,
  MCPServerWithToolsResponse,
  MCPSessionListResponse,
  MCPSessionStatus,
)


class MCPService:
  http_exposed = [
    "post=connect_account",
    "post=generate_url",
    "get=list_servers",
    "get=server_with_tools",
    "get=user_sessions",
    "get=callback",
    "get=list_mcp_instances",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.logger = acquire.logger

  async def _get_auth_config(self, server_id: str, session: AsyncSession) -> str:
    """Get auth config ID for an MCP server from database."""
    server_query = select(MCPServerModel).where(MCPServerModel.id == server_id)
    result = await session.execute(server_query)
    server = result.scalar_one_or_none()

    if server and server.auth_config_id:
      return server.auth_config_id

    raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No auth config found for server: {server_id}")

  async def post_connect_account(
    self,
    data: MCPConnectedAccountCreate,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> MCPConnectedAccountResponse:
    try:
      auth_config_id = await self._get_auth_config(data.server_id, session)
      user_id = user.get("id")
      org_id = user.get("org_id")

      composio_user_id = str(uuid.uuid4())

      # Prepare connection payload with backend callback URL
      callback_url = f"{settings.composio_callback_url}?mcp_server_id={data.server_id}&mcp_session_id={composio_user_id}"

      # Create connected account using Composio API
      result = await composio.create_connected_account(
        auth_config_id=auth_config_id,
        user_id=composio_user_id,
        callback_url=callback_url,
      )

      if not result.success:
        raise Exception(f"Failed to create connected account: {result.errors}")

      # Store the connection attempt in our database with pending status
      mcp_session = MCPSessionModel(
        id=uuid.uuid4(),
        instance_id=composio_user_id,  # Same as composio_user_id
        mcp_server_id=data.server_id,
        user_id=user_id,
        org_id=org_id,
        connected_account_id=result.data["id"],
        status="pending",
      )
      session.add(mcp_session)
      await session.commit()

      return MCPConnectedAccountResponse(**result.data)
    except Exception as e:
      await session.rollback()
      import traceback

      self.logger.error(f"Error creating connected account: {e}")
      self.logger.error(f"Full traceback: {traceback.format_exc()}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error creating connected account: {str(e)}")

  async def post_generate_url(
    self,
    data: MCPGenerateUrlRequest,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
    # Check if instance creation was successful
  ) -> MCPGenerateUrlResponse:
    try:
      mcp_session_query = select(MCPSessionModel).where(
        (MCPSessionModel.connected_account_id == data.account_id) & (MCPSessionModel.mcp_server_id == data.server_id)
      )
      result = await session.execute(mcp_session_query)
      mcp_session = result.scalar_one_or_none()

      if not mcp_session:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No connection attempt found. Please connect your account first.")

      url_data = await composio.generate_mcp_url(
        server_id=data.server_id,
        user_ids=[mcp_session.instance_id],  # instance_id is same as composio_user_id
      )

      if not url_data.success:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to generate MCP URL: {url_data.errors}")

      if not url_data.data.get("user_ids_url"):
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to generate MCP URL")

      mcp_url = url_data.data["user_ids_url"][0]
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
            toolkit_name=server.toolkit_name,
            toolkit_slug=server.toolkit_slug,
            toolkit_logo=server.toolkit_logo,
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

      # Get tools that belong to this server directly
      tools_query = select(MCPToolModel).where(MCPToolModel.mcp_server_id == server_id)
      tools_result = await session.execute(tools_query)
      tools = tools_result.scalars().all()

      tools_list = []
      for tool in tools:
        tools_list.append({
          "name": tool.name,
          "slug": tool.slug,
          "description": tool.description,
          "toolkit": server.toolkit_name,  # Use toolkit name from server
        })

      return MCPServerWithToolsResponse(
        id=str(server.id),
        name=server.name,
        toolkit_name=server.toolkit_name,
        toolkit_slug=server.toolkit_slug,
        toolkit_logo=server.toolkit_logo,
        created_at=server.created_at.isoformat() if server.created_at else None,
        server_instance_count=server.server_instance_count,
        tools=tools_list,
      )
    except Exception as e:
      self.logger.error(f"Error getting MCP server with tools: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error getting MCP server with tools")

  async def get_list_mcp_instances(
    self,
    org_id: str,
    status: Optional[MCPSessionStatus] = None,
    user: dict = Depends(RBAC("mcp", "read")),
    session: AsyncSession = Depends(get_db),
  ) -> List[MCPInstanceResponse]:
    try:
      # Base query with org_id filter
      mcp_instances_query = select(MCPSessionModel).where(MCPSessionModel.org_id == org_id)

      # Add status filter only if status is provided
      if status is not None:
        mcp_instances_query = mcp_instances_query.where(MCPSessionModel.status == status.value)

      result = await session.execute(mcp_instances_query)
      mcp_instances = list(result.scalars().all())

      return [MCPInstanceResponse.model_validate(instance) for instance in mcp_instances]
    except Exception as e:
      self.logger.error(f"Error getting MCP instances: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error getting MCP instances")

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
    """Handle OAuth callback, update status to active and auto-create instance."""
    try:
      # Find the connection attempt for this server and session
      mcp_session_query = select(MCPSessionModel).where(
        (MCPSessionModel.mcp_server_id == mcp_server_id) & (MCPSessionModel.instance_id == mcp_session_id)
      )
      result = await session.execute(mcp_session_query)
      mcp_session = result.scalar_one_or_none()
      frontend_redirect_url = f"{settings.frontend_url}/mcp/callback?mcp_server_id={mcp_server_id}&mcp_session_id={mcp_session_id}"

      if not mcp_session:
        self.logger.warning(f"No connection attempt found for server {mcp_server_id} and session {mcp_session_id}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Connection attempt not found")

      # Update status to active and auto-create instance
      mcp_session.status = "active"

      # Automatically create MCP instance
      try:
        instance = await composio.create_mcp_instance(
          server_id=mcp_server_id,
          user_id=mcp_session_id,  # instance_id is same as composio_user_id
        )

        if not instance.success:
          raise Exception(f"Failed to create MCP instance: {instance.errors}")

        # Update the session with instance info from Composio
        # The instance_id should already be the same, but we can update the ID if needed
        if instance.data and instance.data.get("id"):
          mcp_session.id = instance.data["id"]

        await session.commit()

        self.logger.info(f"Successfully activated connection and created instance for server {mcp_server_id}, session {mcp_session_id}")

      except Exception as instance_error:
        self.logger.error(f"Failed to create instance after OAuth success: {instance_error}")
        # Still mark as active even if instance creation fails
        await session.commit()

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
