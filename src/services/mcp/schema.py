from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MCPServerCreate(BaseModel):
  name: str
  toolkit: str
  allowed_tools: Optional[List[str]] = None


class MCPServerResponse(BaseModel):
  id: str
  name: str
  allowed_tools: Optional[List[str]] = None
  mcp_url: str
  toolkits: Optional[List[str]] = None
  updated_at: str
  created_at: str
  server_instance_count: int


class MCPInstanceCreate(BaseModel):
  server_id: str


class MCPInstanceResponse(BaseModel):
  id: str
  instance_id: str
  mcp_server_id: str
  created_at: str
  updated_at: str


class MCPConnectedAccountCreate(BaseModel):
  toolkit: str


class MCPConnectedAccountResponse(BaseModel):
  id: str
  status: str
  redirect_url: str
  connectionData: Dict[str, Any]


class MCPServerListResponse(BaseModel):
  servers: List[MCPServerResponse]


class MCPServerWithToolsResponse(BaseModel):
  id: str
  name: str
  allowed_tools: Optional[List[str]] = None
  mcp_url: str
  toolkits: Optional[List[str]] = None
  updated_at: str
  created_at: str
  server_instance_count: int
  tools: List[Dict[str, Any]]


class MCPSessionListResponse(BaseModel):
  sessions: List[MCPInstanceResponse]
