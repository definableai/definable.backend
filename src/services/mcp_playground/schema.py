from typing import Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl


class MCPPlaygroundMessage(BaseModel):
  """MCP Playground message schema."""

  content: str
  mcp_url: Optional[HttpUrl] = None  # Optional - if not provided, continues current session


class MCPPlaygroundResponse(BaseModel):
  """MCP Playground response schema."""

  message: str
  status: str
  session_id: UUID
