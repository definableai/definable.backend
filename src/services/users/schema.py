from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class OrganizationInfo(BaseModel):
  """Organization information schema."""

  id: UUID
  name: str
  slug: str
  role_name: str | None = None
  role_id: UUID | None = None

  class Config:
    from_attributes = True


class UserDetailResponse(BaseModel):
  """User detail response schema."""

  id: UUID
  email: EmailStr
  first_name: str
  last_name: str
  full_name: str
  organizations: list[OrganizationInfo]

  class Config:
    from_attributes = True


class UserListResponse(BaseModel):
  """User list response schema."""

  users: list[UserDetailResponse]
  total: int

  class Config:
    from_attributes = True

class InviteSignup(BaseModel):
  """Invite signup schema."""

  first_name: str
  last_name: str
  email: EmailStr
  role: str

  class Config:
    from_attributes = True

class StytchUser(BaseModel):
  """Stytch user schema."""

  email: EmailStr
  stytch_id: str
  first_name: Optional[str] = Field(None)
  last_name: Optional[str] = Field(None)
  metadata: Optional[dict] = Field(None)