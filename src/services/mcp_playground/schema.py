from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class MCPPlaygroundMessage(BaseModel):
  """MCP Playground message schema."""

  content: str
  mcp_server_id: Optional[UUID] = None  # Optional - if not provided, continues current session
  model_id: Optional[UUID] = None  # Optional - if not provided, uses default model


class MCPPlaygroundResponse(BaseModel):
  """MCP Playground response schema."""

  message: str
  status: str
  session_id: UUID
