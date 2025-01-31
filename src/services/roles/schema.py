from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class PermissionBase(BaseModel):
  """Base permission schema."""

  name: str = Field(..., min_length=1, max_length=100)
  description: Optional[str] = None
  resource: str
  action: str

  @field_validator("name", mode="before")
  def default_name(cls, v, values):
    """Generate default name if not provided."""
    if not v and "resource" in values and "action" in values:
      return f"{values['resource']}:{values['action']}"
    return v


class PermissionCreate(PermissionBase):
  """Create permission schema."""

  pass


class PermissionResponse(PermissionBase):
  """Permission response schema."""

  id: UUID
  name: str
  description: Optional[str]
  resource: str
  action: str

  class Config:
    from_attributes = True


class RoleBase(BaseModel):
  """Base role schema."""

  name: str = Field(..., min_length=1, max_length=100)
  description: Optional[str] = None
  hierarchy_level: int = Field(..., ge=0, le=100)


class RoleCreate(RoleBase):
  """Create role schema."""

  permission_ids: List[UUID] = Field(default=[], min_length=1)


class RoleUpdate(BaseModel):
  """Update role schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=100)
  description: Optional[str] = None
  hierarchy_level: Optional[int] = Field(None, ge=0, le=100)
  permission_ids: Optional[List[UUID]] = Field(default=[], min_length=1)


class RoleResponse(RoleBase):
  """Role response schema."""

  id: UUID
  organization_id: UUID
  is_system_role: bool
  created_at: datetime
  permissions: List[PermissionResponse]

  class Config:
    from_attributes = True


class DefaultRoleType(str, Enum):
  """Default role type enum."""

  OWNER = "OWNER"  # Level 100
  ADMIN = "ADMIN"  # Level 90
  DEVELOPER = "DEVELOPER"  # Level 50
