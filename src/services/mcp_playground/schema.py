from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class MCPPlaygroundMessage(BaseModel):
  """MCP Playground message schema."""

  content: str
  mcp_server_ids: Optional[List[UUID]] = None
  model: Optional[str] = None


class MCPPlaygroundResponse(BaseModel):
  """MCP Playground response schema."""

  message: str
  status: str
  session_id: UUID
