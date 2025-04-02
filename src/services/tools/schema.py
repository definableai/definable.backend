from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from config.settings import settings


class ToolCategoryBase(BaseModel):
  """Base tool category schema."""

  name: str = Field(..., min_length=1, max_length=255)
  description: Optional[str] = None


class ToolCategoryCreate(ToolCategoryBase):
  """Create tool category schema."""

  pass


class ToolCategoryUpdate(BaseModel):
  """Update tool category schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=255)
  description: Optional[str] = None


class ToolCategoryResponse(ToolCategoryBase):
  """Tool category response schema."""

  id: UUID

  class Config:
    from_attributes = True


# First, let's create models for the input/config parameter structure
class ToolParameter(BaseModel):
  name: str
  type: str  # can be string, number, boolean, array, object
  description: str
  required: bool = False
  default: Optional[Union[str, int, float, bool, None]] = None


class ToolOutput(BaseModel):
  type: str
  description: str


class ToolFunctionInfo(BaseModel):
  name: str
  is_async: bool
  description: str
  code: str


class ToolDeployment(BaseModel):
  framework: str
  toolkit_class: bool
  standalone_function: bool


class ToolInfo(BaseModel):
  name: str
  description: str
  version: str


class ToolSettings(BaseModel):
  function_info: ToolFunctionInfo
  requirements: List[str]
  deployment: Optional[ToolDeployment] = None


class ToolBase(BaseModel):
  """Base tool schema."""

  name: str = Field(..., min_length=1, max_length=255)
  description: str = Field(..., min_length=1, max_length=255)
  category_id: UUID
  logo_url: Optional[str] = Field(None, max_length=255)
  is_active: bool = True
  version: str = Field(..., max_length=50)
  is_public: bool = False
  is_verified: bool = False
  inputs: List[ToolParameter]
  outputs: ToolOutput
  configuration: Optional[List[ToolParameter]] = None
  settings: ToolSettings


class ToolCreate(ToolBase):
  """Create tool schema."""


class ToolUpdate(BaseModel):
  """Update tool schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=255)
  description: Optional[str] = None
  category_id: Optional[UUID] = None
  logo_url: Optional[str] = Field(None, max_length=255)
  is_active: Optional[bool] = None
  version: Optional[str] = Field(None, max_length=50)
  is_public: Optional[bool] = None
  is_verified: Optional[bool] = None
  inputs: Optional[List[ToolParameter]] = None
  outputs: Optional[ToolOutput] = None
  configuration: Optional[List[ToolParameter]] = None
  settings: Optional[ToolSettings] = None


class ToolResponse(ToolBase):
  """Tool response schema."""

  id: UUID
  category: Optional[ToolCategoryResponse] = None
  organization_id: UUID
  user_id: UUID
  created_at: datetime
  updated_at: datetime

  class Config:
    from_attributes = True


class PaginatedToolResponse(BaseModel):
  """Paginated tool response schema."""

  tools: List[ToolResponse]
  total: int
  has_more: bool


class ToolConfigItem(BaseModel):
  name: str
  value: Optional[Union[str, int, float, bool, None]] = None


class ToolTestRequest(BaseModel):
  """Tool test request schema."""

  input_prompt: str
  model_name: Optional[str] = "gpt-4o-mini"
  api_key: Optional[str] = settings.openai_api_key
  config_items: List[ToolConfigItem]  # TODO: can i take config from db and then dynamically ask user
