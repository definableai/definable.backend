from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PromptCategoryBase(BaseModel):
  """Base prompt category schema."""

  name: str = Field(..., min_length=1, max_length=100)
  description: Optional[str] = None
  icon_url: Optional[str] = None
  display_order: int = Field(default=0)
  is_active: bool = Field(default=True)


class PromptCategoryCreate(PromptCategoryBase):
  """Create prompt category schema."""

  pass


class PromptCategoryUpdate(BaseModel):
  """Update prompt category schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=100)
  description: Optional[str] = None
  icon_url: Optional[str] = None
  display_order: Optional[int] = None
  is_active: Optional[bool] = None


class PromptCategoryResponse(PromptCategoryBase):
  """Prompt category response schema."""

  id: UUID
  count: int = Field(default=0, description="Number of prompts in the category")

  class Config:
    from_attributes = True


class PromptBase(BaseModel):
  """Base prompt schema."""

  title: str = Field(..., min_length=1, max_length=200)
  content: str = Field(..., min_length=1)
  description: Optional[str] = None
  is_public: bool = Field(default=False)
  is_featured: bool = Field(default=False)
  metadata: Optional[Dict] = None


class PromptCreate(PromptBase):
  """Create prompt schema."""

  pass


class PromptUpdate(BaseModel):
  """Update prompt schema."""

  title: Optional[str] = Field(None, min_length=1, max_length=200)
  content: Optional[str] = Field(None, min_length=1)
  description: Optional[str] = None
  is_public: Optional[bool] = None
  is_featured: Optional[bool] = None
  metadata: Optional[Dict] = None


class SortBy(str, Enum):
  RATING = "rating"
  RECENT = "created_at"
  ALPHABETICAL = "title"


class PromptResponse(PromptBase):
  """Prompt response schema."""

  id: UUID
  creator_id: UUID
  organization_id: UUID
  created_at: datetime
  category: PromptCategoryResponse
  rating: Optional[float] = Field(
    default=None,
    description="Average rating extracted from metadata (if available)",
    examples=[4.5],
  )

  class Config:
    from_attributes = True

  @classmethod
  def model_validate(cls, obj, **kwargs):
    # Extract rating from metadata if it exists
    if hasattr(obj, "metadata") and obj.metadata and "rating" in obj.metadata:
      obj.rating = obj.metadata["rating"]
    return super().model_validate(obj, **kwargs)


class PaginatedPromptResponse(BaseModel):
  """Paginated prompt response schema."""

  prompts: List[PromptResponse]
  total: int
  has_more: bool

  class Config:
    from_attributes = True
