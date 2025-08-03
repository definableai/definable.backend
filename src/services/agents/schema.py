from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentBase(BaseModel):
  """Base agent schema."""

  name: str = Field(..., min_length=1, max_length=255)
  description: Optional[str] = None
  model_id: Optional[UUID] = None
  is_active: bool = True
  settings: dict = Field(default_factory=dict)


class AgentCreate(AgentBase):
  """Create agent schema."""

  name: str = Field(..., min_length=1, max_length=255)
  description: Optional[str] = None
  is_active: bool = True
  version: str = Field(default="v1")
  settings: Dict[str, Any] = Field(default_factory=dict)


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
  category: Optional[str] = None
  properties: dict = Field(default_factory=dict)

  class Config:
    from_attributes = True


class PaginatedAgentResponse(BaseModel):
  """Paginated agent response schema."""

  agents: List[AgentResponse]
  total: int
  has_more: bool


class AgentCategoryResponse(BaseModel):
  """Agent category response schema."""

  id: UUID
  name: str
  description: Optional[str]
  agent_count: int

  class Config:
    from_attributes = True
