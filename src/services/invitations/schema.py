from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from models import InvitationStatus


class InvitationBase(BaseModel):
  """Base invitation schema."""

  role_id: UUID
  invitee_email: EmailStr = Field(..., min_length=1, max_length=255)
  expiry_time: Optional[datetime] = None


class InvitationCreate(InvitationBase):
  """Schema for creating a new invitation."""

  pass


class InvitationUpdate(BaseModel):
  """Schema for updating an invitation."""

  status: Optional[InvitationStatus] = None
  expiry_time: Optional[datetime] = None

  class Config:
    arbitrary_types_allowed = True


class InvitationResponse(BaseModel):
  """Schema for invitation response."""

  id: UUID
  organization_id: UUID
  role_id: UUID
  invitee_email: EmailStr
  invited_by: UUID
  status: InvitationStatus
  expiry_time: datetime
  created_at: datetime
  updated_at: Optional[datetime] = None

  class Config:
    from_attributes = True


class InvitationResendRequest(BaseModel):
  """Schema for resending an invitation."""

  invitation_id: UUID


class InvitationActionRequest(BaseModel):
  """Schema for accepting or rejecting an invitation."""

  email: EmailStr


class InviteeSignupRequest(BaseModel):
  """Schema for invitee signup."""

  invite_token: str
  password: str = Field(..., min_length=8, max_length=64)
  first_name: str = Field(..., min_length=1, max_length=50)
  last_name: str = Field(..., min_length=1, max_length=50)


class InvitationListParams(BaseModel):
  """Query parameters for listing invitations."""

  page: int = Field(default=1, ge=1)
  size: int = Field(default=10, ge=1, le=100)
  status: Optional[InvitationStatus] = None
  organization_id: Optional[UUID] = None


class InvitationListResponse(BaseModel):
  """Schema for listing invitations."""

  items: List[InvitationResponse]
  total: int
  page: int
  size: int

  class Config:
    from_attributes = True
