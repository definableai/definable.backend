import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class MCPServerResponse(BaseModel):
  id: str
  name: str
  toolkit_name: str
  toolkit_slug: str
  toolkit_logo: Optional[str] = None
  created_at: str
  server_instance_count: int


class MCPInstanceCreate(BaseModel):
  server_id: str
  connected_account_id: str


class MCPInstanceResponse(BaseModel):
  id: UUID
  status: str
  instance_id: str
  mcp_server_id: UUID
  created_at: datetime.datetime

  class Config:
    from_attributes = True


class MCPConnectedAccountCreate(BaseModel):
  server_id: str


class MCPConnectedAccountResponse(BaseModel):
  id: str
  status: str
  redirect_url: str
  connectionData: Dict[str, Any]


class MCPGenerateUrlRequest(BaseModel):
  server_id: str
  account_id: str


class MCPGenerateUrlResponse(BaseModel):
  mcp_url: str
  status: str


class MCPServerListResponse(BaseModel):
  servers: List[MCPServerResponse]


class MCPServerWithToolsResponse(BaseModel):
  id: str
  name: str
  toolkit_name: str
  toolkit_slug: str
  toolkit_logo: Optional[str] = None
  created_at: str
  server_instance_count: int
  tools: List[Dict[str, Any]]


class MCPSessionListResponse(BaseModel):
  sessions: List[MCPInstanceResponse]


class MCPSessionStatus(str, Enum):
  ACTIVE = "active"
  PENDING = "pending"
  INACTIVE = "inactive"
