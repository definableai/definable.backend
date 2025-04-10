from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentBase(BaseModel):
  """Base agent schema."""

  name: str = Field(..., min_length=1, max_length=255)
  description: Optional[str] = None
  model_id: UUID
  is_active: bool = True
  settings: dict = Field(default_factory=dict)


class ToolConfig(BaseModel):
  tool_id: UUID
  api_key: Optional[str] = None
  api_secret: Optional[str] = None
  config: Dict[str, Any] = Field(default_factory=dict)


class AgentCreate(AgentBase):
  """Schema for creating an agent."""

  name: str
  description: str
  model_id: UUID
  provider: str
  system_prompt: str
  instructions: str
  expected_output: Dict[str, Any]
  memory_config: Optional[Dict[str, Any]] = None
  knowledge_base: Optional[Dict[str, Any]] = None
  is_active: bool = True
  version: str = "v1"
  tool_configs: List[ToolConfig] = Field(default_factory=list)


class AgentUpdate(BaseModel):
  """Update agent schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=255)
  description: Optional[str] = None
  model_id: Optional[UUID] = None
  is_active: Optional[bool] = None
  settings: Optional[dict] = None


class AgentToolResponse(BaseModel):
  """Tool response schema for agent."""

  id: UUID
  name: str
  description: Optional[str]
  category_id: UUID
  is_active: bool

  class Config:
    from_attributes = True


class AgentResponse(AgentBase):
  """Agent response schema."""

  id: UUID
  version: str
  organization_id: UUID
  updated_at: datetime
  tools: List[AgentToolResponse]

  class Config:
    from_attributes = True


class PaginatedAgentResponse(BaseModel):
  """Paginated agent response schema."""

  agents: List[AgentResponse]
  total: int
  has_more: bool
