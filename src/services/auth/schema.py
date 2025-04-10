from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserSignup(BaseModel):
  """User signup schema."""

  email: EmailStr
  first_name: str = Field(..., min_length=1, max_length=255)
  last_name: str = Field(..., min_length=1, max_length=255)
  password: str = Field(..., min_length=8)


class InviteSignup(BaseModel):
  """Invite signup schema."""

  first_name: str = Field(..., min_length=1, max_length=255)
  last_name: str = Field(..., min_length=1, max_length=255)
  password: str = Field(..., min_length=8)
  invite_token: str


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


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr


class PasswordResetToken(BaseModel):
    """Password reset token schema."""
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
