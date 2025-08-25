from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class LLMBase(BaseModel):
  """Base LLM schema."""

  name: str
  provider: str
  version: str
  is_active: bool = True
  config: dict
  props: dict


class LLMCreate(LLMBase):
  """Create LLM schema."""

  pass


class LLMUpdate(BaseModel):
  """Update LLM schema."""

  name: Optional[str] = None
  provider: Optional[str] = None
  version: Optional[str] = None
  is_active: Optional[bool] = None
  config: Optional[dict] = None
  props: Optional[dict] = None


class LLMResponse(LLMBase):
  """LLM response schema."""

  id: UUID

  class Config:
    from_attributes = True
