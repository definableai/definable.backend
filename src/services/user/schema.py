from uuid import UUID

from pydantic import BaseModel, EmailStr


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
  is_active: bool
  organizations: list[OrganizationInfo]

  class Config:
    from_attributes = True


class UserListResponse(BaseModel):
  """User list response schema."""

  users: list[UserDetailResponse]
  total: int
  page: int
  page_size: int

  class Config:
    from_attributes = True
