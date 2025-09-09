import json
import uuid
from typing import AsyncGenerator

import httpx
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from libs.composio.v1 import composio
from libs.mcp_playground.v1.playground import MCPPlaygroundFactory
from models.mcp_model import MCPSessionModel
from services.__base.acquire import Acquire
from services.mcp_playground.schema import MCPPlaygroundMessage


class MCPPlaygroundService:
  """MCP Playground service for testing MCP integrations."""

  http_exposed = ["post=chat"]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.logger = acquire.logger
    self.chunk_size = 10  # Buffer size for streaming
    self.playground_factory = MCPPlaygroundFactory()
    self.user_sessions: dict[str, dict[str, str]] = {}

  def _get_or_create_session_id(self, user_id: str, mcp_server_id: str) -> str:
    """Get existing session ID for user+mcp_server_id combo or create new one."""
    if user_id not in self.user_sessions:
      self.user_sessions[user_id] = {}

    if mcp_server_id not in self.user_sessions[user_id]:
      session_id = str(uuid.uuid4())
      self.user_sessions[user_id][mcp_server_id] = session_id
      self.logger.info(f"Created new session {session_id} for user {user_id} with MCP server {mcp_server_id}")
    else:
      session_id = self.user_sessions[user_id][mcp_server_id]
      self.logger.info(f"Using existing session {session_id} for user {user_id}")

    return session_id

  async def _get_mcp_url_for_user(self, mcp_server_id: str, user_id: str, session: AsyncSession) -> str:
    """Get MCP URL for a user's active session with the given server."""
    try:
      # Find the user's active MCP session for this server
      session_query = select(MCPSessionModel).where(
        (MCPSessionModel.mcp_server_id == mcp_server_id) & (MCPSessionModel.user_id == user_id) & (MCPSessionModel.status == "active")
      )
      result = await session.execute(session_query)
      mcp_session = result.scalar_one_or_none()

      if not mcp_session:
        raise HTTPException(
          status_code=400, detail=f"No active MCP session found for server {mcp_server_id}. Please connect to the MCP server first."
        )

      url_data = await composio.generate_mcp_url(
        server_id=str(mcp_server_id),
        user_ids=[mcp_session.instance_id],
      )

      if not url_data.success:
        raise HTTPException(status_code=500, detail=f"Failed to generate MCP URL: {url_data.errors}")

      if not url_data.data.get("user_ids_url"):
        raise HTTPException(status_code=500, detail="Failed to generate MCP URL")

      mcp_url = url_data.data["user_ids_url"][0]
      self.logger.info(f"Generated MCP URL for server {mcp_server_id} and user {user_id}")
      return mcp_url

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error getting MCP URL: {e}")
      raise HTTPException(status_code=500, detail="Error generating MCP URL")

  def _parse_model(self, model: str) -> tuple[str, str]:
    """Parse raw model string to determine provider and model version."""
    model = model.lower().strip()

    if model.startswith("gpt") or model.startswith("o1"):
      return "openai", model
    elif model.startswith("claude"):
      return "anthropic", model
    elif model.startswith("deepseek"):
      return "deepseek", model
    else:
      raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

  async def post_chat(
    self,
    message_data: MCPPlaygroundMessage,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> StreamingResponse:
    """
    Chat with MCP servers using auto-managed sessions.
    - If mcp_server_ids provided: starts new sessions or continues existing sessions for those servers
    - If no mcp_server_ids: continues current active session for user
    """
    try:
      user_id = str(user["id"])

      if message_data.mcp_server_ids:
        # Multiple MCP server IDs provided - get URLs for each server
        mcp_urls = []
        for mcp_server_id in message_data.mcp_server_ids:
          mcp_server_id_str = str(mcp_server_id)
          mcp_url = await self._get_mcp_url_for_user(mcp_server_id_str, user_id, session)
          mcp_urls.append(mcp_url)

        # Use the first server ID for session management
        session_id = self._get_or_create_session_id(user_id, str(message_data.mcp_server_ids[0]))
        self.logger.info(f"Using MCP servers: {message_data.mcp_server_ids} with session: {session_id}")

      else:
        if user_id not in self.user_sessions or not self.user_sessions[user_id]:
          raise HTTPException(status_code=400, detail="No active MCP session. Please provide mcp_server_ids to start a new session.")

        user_session_data = self.user_sessions[user_id]
        mcp_server_id = list(user_session_data.keys())[-1]
        session_id = user_session_data[mcp_server_id]

        mcp_url = await self._get_mcp_url_for_user(mcp_server_id, user_id, session)
        mcp_urls = [mcp_url]

        self.logger.info(f"Continuing session {session_id} with existing MCP server {mcp_server_id}")

      # Get model information
      if message_data.model:
        provider, llm = self._parse_model(message_data.model)
      else:
        provider = "openai"
        llm = "gpt-4o"

      async def generate_mcp_response() -> AsyncGenerator[str, None]:
        """Generate streaming response from MCP playground factory."""
        try:
          self.logger.debug(f"Starting chat with session {session_id}")

          buffer: list[str] = []

          async for chunk in self.playground_factory.chat(
            session_id=session_id,
            mcp_urls=mcp_urls,
            message=message_data.content,
            llm=llm,
            provider=provider,
            memory_size=50,
          ):
            if chunk and hasattr(chunk, "content") and chunk.content:
              buffer.append(chunk.content)

              # Send chunks when buffer reaches threshold
              if len(buffer) >= self.chunk_size:
                data = {"message": "".join(buffer)}
                yield f"data: {json.dumps(data)}\n\n"
                buffer = []

          if buffer:
            data = {"message": "".join(buffer)}
            yield f"data: {json.dumps(data)}\n\n"

          yield f"data: {json.dumps({'message': 'DONE'})}\n\n"

          self.logger.info(f"MCP playground conversation completed for session {session_id}")

        except httpx.ConnectError:
          error_msg = "Failed to connect to MCP server. Please check the URL and try again."
          self.logger.error(f"MCP connection error: {error_msg}")
          yield f"data: {json.dumps({'error': error_msg})}\n\n"

        except httpx.TimeoutException:
          error_msg = "MCP server connection timed out. The server may be unavailable."
          self.logger.error(f"MCP timeout error: {error_msg}")
          yield f"data: {json.dumps({'error': error_msg})}\n\n"

        except Exception as e:
          error_msg = f"Error during MCP interaction: {str(e)}"
          self.logger.error(f"MCP playground error: {error_msg}", exc_info=True)
          yield f"data: {json.dumps({'error': error_msg})}\n\n"

      return StreamingResponse(
        generate_mcp_response(),
        media_type="text/event-stream",
        headers={
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "*",
        },
      )

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Unexpected error in MCP playground: {e}", exc_info=True)
      raise HTTPException(status_code=500, detail="An unexpected error occurred in the MCP playground")
