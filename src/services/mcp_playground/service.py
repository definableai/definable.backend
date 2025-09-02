import json
import uuid
from typing import AsyncGenerator

import httpx
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC
from libs.mcp_playground.v1.playground import MCPPlaygroundFactory
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
    # Store user sessions: {user_id: {mcp_url: session_id}}
    self.user_sessions: dict[str, dict[str, str]] = {}

  def _get_or_create_session_id(self, user_id: str, mcp_url: str) -> str:
    """Get existing session ID for user+mcp_url combo or create new one."""
    if user_id not in self.user_sessions:
      self.user_sessions[user_id] = {}

    if mcp_url not in self.user_sessions[user_id]:
      # Create new session ID for this user+mcp_url combination
      session_id = str(uuid.uuid4())
      self.user_sessions[user_id][mcp_url] = session_id
      self.logger.info(f"Created new session {session_id} for user {user_id} with MCP URL")
    else:
      session_id = self.user_sessions[user_id][mcp_url]
      self.logger.info(f"Using existing session {session_id} for user {user_id}")

    return session_id

  async def post_chat(
    self,
    message_data: MCPPlaygroundMessage,
    user: dict = Depends(RBAC("mcp", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> StreamingResponse:
    """
    Chat with an MCP server using auto-managed sessions.
    - If mcp_url provided: starts new session or continues existing session for that URL
    - If no mcp_url: continues current active session for user
    """
    try:
      user_id = str(user["id"])

      # Determine which MCP URL and session to use
      if message_data.mcp_url:
        # New MCP URL provided - get or create session for this URL
        mcp_url = str(message_data.mcp_url)
        if not mcp_url.startswith(("http://", "https://")):
          raise HTTPException(status_code=400, detail="Invalid MCP URL format")

        session_id = self._get_or_create_session_id(user_id, mcp_url)
        self.logger.info(f"Using MCP URL: {mcp_url[:50]}... with session: {session_id}")

      else:
        # No MCP URL provided - continue with current active session
        if user_id not in self.user_sessions or not self.user_sessions[user_id]:
          raise HTTPException(status_code=400, detail="No active MCP session. Please provide mcp_url to start a new session.")

        # Get the most recently used session (last added)
        user_session_data = self.user_sessions[user_id]
        mcp_url = list(user_session_data.keys())[-1]  # Last used MCP URL
        session_id = user_session_data[mcp_url]
        self.logger.info(f"Continuing session {session_id} with existing MCP URL")

      async def generate_mcp_response() -> AsyncGenerator[str, None]:
        """Generate streaming response from MCP playground factory."""
        try:
          self.logger.debug(f"Starting chat with session {session_id}")

          # Use MCPPlaygroundFactory for streaming chat
          buffer: list[str] = []

          async for chunk in self.playground_factory.chat(
            session_id=session_id,
            mcp_url=mcp_url,
            message=message_data.content,
            openai_api_key=settings.openai_api_key,
            memory_size=50,
          ):
            if chunk and hasattr(chunk, "content") and chunk.content:
              buffer.append(chunk.content)

              # Send chunks when buffer reaches threshold
              if len(buffer) >= self.chunk_size:
                data = {"message": "".join(buffer)}
                yield f"data: {json.dumps(data)}\n\n"
                buffer = []

          # Send remaining buffer
          if buffer:
            data = {"message": "".join(buffer)}
            yield f"data: {json.dumps(data)}\n\n"

          # Signal completion
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
