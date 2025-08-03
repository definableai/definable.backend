import hashlib
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import JWTBearer
from models import APIKeyModel
from services.__base.acquire import Acquire

# Add import
from utils.auth_util import create_access_token

from .schema import (
  APIKeyListResponse,
  APIKeyResponse,
  APIKeyWithTokenResponse,
  CreateAPIKeyRequest,
  RevokeAPIKeyRequest,
  UpdateAPIKeyRequest,
)


class APIKeyService:
  """API Key management service."""

  http_exposed = ["post=create", "get=list", "put=update", "delete=revoke"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger

  async def post_create(
    self,
    request: CreateAPIKeyRequest,
    current_user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> APIKeyWithTokenResponse:
    """Create a new API key for the authenticated user."""
    try:
      self.logger.info(f"Creating API key for user {current_user['id']}")
      user_id = UUID(current_user["id"])

      self.logger.info(f"User ID: {user_id}")
      self.logger.info(f"Request: {request}")
      self.logger.info(f"Org ID: {request.org_id}")
      self.logger.info(f"Agent ID: {request.agent_id}")
      self.logger.info(f"Name: {request.name}")
      self.logger.info(f"Permissions: {request.permissions}")
      self.logger.info(f"Expires at: {request.expires_at}")
      # Generate the API key
      api_key = self._generate_api_key(user_id, request.org_id)
      api_key_hash = self._hash_api_key(api_key)

      self.logger.info(f"API key: {api_key}")
      self.logger.info(f"API key hash: {api_key_hash}")

      # Set default expiration if not provided
      expires_at = request.expires_at
      if expires_at is None and self.settings.API_KEY_DEFAULT_EXPIRY_DAYS:
        expires_at = datetime.utcnow() + timedelta(days=self.settings.API_KEY_DEFAULT_EXPIRY_DAYS)

      # Convert timezone-aware datetime to naive UTC datetime for database storage
      if expires_at and expires_at.tzinfo is not None:
        expires_at = expires_at.replace(tzinfo=None)

      self.logger.info(f"Final expires_at (naive): {expires_at}")

      # Create the API key model
      api_key_model = APIKeyModel(
        user_id=user_id,
        agent_id=request.agent_id,
        token_type="api",  # Always "api" for manually created keys
        api_key_token=api_key,  # Store plain JWT
        api_key_hash=api_key_hash,  # Store hash for indexing
        name=request.name,
        permissions=request.permissions,
        expires_at=expires_at,
      )

      self.logger.info(f"API key model: {api_key_model}")

      session.add(api_key_model)
      await session.commit()
      await session.refresh(api_key_model)
      self.logger.info(f"API key model refreshed: {api_key_model}")

      self.logger.info(f"Created API key for user {user_id}", api_key_id=str(api_key_model.id))

      # Return the response with the plain text API key (only time it's shown)
      self.logger.info(f"Returning API key with token response: {APIKeyWithTokenResponse(**api_key_model.__dict__, api_key=api_key)}")
      return APIKeyWithTokenResponse(**api_key_model.__dict__, api_key=api_key)

    except Exception as e:
      self.logger.error(f"Error creating API key: {str(e)}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create API key")

  async def get_list(
    self,
    current_user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
    offset: int = 0,
    limit: int = 50,
  ) -> APIKeyListResponse:
    """List all API keys for the authenticated user."""
    try:
      user_id = UUID(current_user["id"])

      # Query for user's API keys
      query = select(APIKeyModel).where(APIKeyModel.user_id == user_id).order_by(APIKeyModel.created_at.desc()).offset(offset).limit(limit)

      result = await session.execute(query)
      api_keys = result.scalars().all()

      # Count total keys
      count_query = select(APIKeyModel).where(APIKeyModel.user_id == user_id)
      count_result = await session.execute(count_query)
      total = len(count_result.scalars().all())

      return APIKeyListResponse(api_keys=[APIKeyResponse(**api_key.__dict__) for api_key in api_keys], total=total)

    except Exception as e:
      self.logger.error(f"Error listing API keys: {str(e)}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list API keys")

  async def put_update(
    self,
    api_key_id: UUID,
    request: UpdateAPIKeyRequest,
    current_user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> APIKeyResponse:
    """Update an existing API key."""
    try:
      user_id = UUID(current_user["id"])

      # Get the API key
      api_key = await session.get(APIKeyModel, api_key_id)
      if not api_key or api_key.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

      # Update fields
      if request.name is not None:
        api_key.name = request.name
      if request.permissions is not None:
        api_key.permissions = request.permissions
      if request.is_active is not None:
        api_key.is_active = request.is_active
      if request.expires_at is not None:
        api_key.expires_at = request.expires_at

      api_key.updated_at = datetime.utcnow()

      await session.commit()
      await session.refresh(api_key)

      self.logger.info(f"Updated API key {api_key_id} for user {user_id}")

      return APIKeyResponse(**api_key.__dict__)

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error updating API key: {str(e)}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update API key")

  async def delete_revoke(
    self,
    request: RevokeAPIKeyRequest,
    current_user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Revoke (deactivate) an API key."""
    try:
      user_id = UUID(current_user["id"])

      # Get the API key
      api_key = await session.get(APIKeyModel, request.api_key_id)
      if not api_key or api_key.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

      # Deactivate the key
      api_key.is_active = False
      api_key.updated_at = datetime.utcnow()

      await session.commit()

      self.logger.info(f"Revoked API key {request.api_key_id} for user {user_id}")

      return JSONResponse(content={"message": "API key revoked successfully"}, status_code=status.HTTP_200_OK)

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error revoking API key: {str(e)}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to revoke API key")

  ### PRIVATE METHODS ###

  # Update _generate_api_key method:
  def _generate_api_key(self, user_id: UUID, org_id: UUID) -> str:
    """Generate JWT API key with user_id and org_id."""
    self.logger.info(f"Generating API key for user {user_id} and org {org_id}")
    payload = {"user_id": str(user_id), "org_id": str(org_id), "type": "api_key"}
    self.logger.info(f"Payload: {payload}")
    return create_access_token(payload, timedelta(days=365))

  # Remove parse_token function entirely

  def _hash_api_key(self, api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

  def parse_token(self, token: str) -> dict:
    """Parse user_id and org_id from token."""
    parts = token.split("_")
    if len(parts) >= 5:
      return {"user_id": parts[2], "org_id": parts[4]}
    return {}
