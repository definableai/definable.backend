from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class StytchUser(BaseModel):
  """Stytch user schema."""

  email: EmailStr
  stytch_id: str
  first_name: Optional[str] = Field(None)
  last_name: Optional[str] = Field(None)
  is_active: bool = Field(default=False)
  password_id: Optional[str] = Field(None)
  metadata: Optional[dict] = Field(None)


class UserSignup(BaseModel):
  """User signup schema."""

  email: EmailStr
  first_name: str
  last_name: str
  password: str


class InviteSignup(BaseModel):
  """Invite signup schema."""

  first_name: str
  last_name: str
  email: EmailStr
  role: str


class UserLogin(BaseModel):
  """User login schema."""

  email: EmailStr
  password: str


class UserResponse(BaseModel):
  """User response schema."""

  id: UUID
  email: EmailStr
  message: str

  class Config:
    from_attributes = True


class TokenResponse(BaseModel):
  """Token response schema."""

  access_token: str
  token_type: str = "bearer"


class TokenData(BaseModel):
  """Token data schema."""

  user_id: UUID
