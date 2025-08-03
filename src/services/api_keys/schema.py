from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CreateAPIKeyRequest(BaseModel):
  """Request model for creating an API key."""

  name: Optional[str] = Field(None, description="Human-readable name for the API key", max_length=100)
  agent_id: Optional[UUID] = Field(None, description="Optional agent ID to link this key to")
  org_id: UUID = Field(..., description="Organization ID for the API key")
  permissions: dict = Field(default_factory=dict, description="Permissions object for the API key")
  expires_at: Optional[datetime] = Field(None, description="Optional expiration date for the API key")


class APIKeyResponse(BaseModel):
  """Response model for API key operations."""

  id: UUID
  user_id: UUID
  agent_id: Optional[UUID] = None
  name: Optional[str] = None
  permissions: dict
  is_active: bool
  expires_at: Optional[datetime] = None
  last_used_at: Optional[datetime] = None
  created_at: datetime
  updated_at: datetime

  class Config:
    from_attributes = True


class APIKeyWithTokenResponse(APIKeyResponse):
  """Response model for API key creation that includes the plaintext token."""

  api_key: str = Field(description="The plaintext API key - only returned once during creation")


class APIKeyListResponse(BaseModel):
  """Response model for listing API keys."""

  api_keys: list[APIKeyResponse]
  total: int


class UpdateAPIKeyRequest(BaseModel):
  """Request model for updating an API key."""

  name: Optional[str] = Field(None, description="Human-readable name for the API key", max_length=100)
  permissions: Optional[dict] = Field(None, description="Updated permissions object")
  is_active: Optional[bool] = Field(None, description="Active status of the API key")
  expires_at: Optional[datetime] = Field(None, description="Updated expiration date")


class VerifyAPIKeyRequest(BaseModel):
  """Request model for verifying an API key."""

  api_key: str = Field(description="The API key to verify")


class VerifyAPIKeyResponse(BaseModel):
  """Response model for API key verification."""

  valid: bool
  user_id: Optional[UUID] = None
  agent_id: Optional[UUID] = None
  permissions: Optional[dict] = None
  expires_at: Optional[datetime] = None
  last_used_at: Optional[datetime] = None
  message: Optional[str] = None


class AuthTokenResponse(BaseModel):
  """Response model for auth token operations."""

  auth_token: str = Field(description="The generated auth token")
  expires_at: Optional[datetime] = None
  user_id: UUID


class RefreshTokenRequest(BaseModel):
  """Request model for refreshing an auth token."""

  current_token: str = Field(description="The current auth token to refresh")


class RevokeAPIKeyRequest(BaseModel):
  """Request model for revoking an API key."""

  api_key_id: UUID = Field(description="The ID of the API key to revoke")


class RevokeTokenRequest(BaseModel):
  """Request model for revoking tokens."""

  token_type: str = Field(description="Type of token to revoke: 'auth' or 'api'")
  token_id: Optional[UUID] = Field(None, description="API key ID for API token revocation")
