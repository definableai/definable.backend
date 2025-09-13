import hashlib
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPBearer
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from libs.stytch.v1 import stytch_base
from models import APIKeyModel, OrganizationMemberModel, OrganizationModel, PermissionModel, RoleModel, RolePermissionModel, UserModel
from utils.auth_util import verify_jwt_token


class JWTBearer(HTTPBearer):
  def __init__(self, auto_error: bool = True):
    super().__init__(auto_error=auto_error)

  async def __call__(
    self,
    request: Request = None,  # type: ignore
    websocket: WebSocket = None,  # type: ignore
  ) -> Any:
    if request:
      credentials = await super().__call__(request)
      if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(status_code=403, detail="Invalid authorization")
      try:
        response = await stytch_base.authenticate_user_with_jkws(credentials.credentials)
        if response.success:
          return {"stytch_user_id": response.data["sub"]}
        else:
          raise HTTPException(status_code=403, detail=str(response.errors[0]["message"]))
      except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    elif websocket:
      token = websocket.query_params.get("token")
      if not token:
        raise HTTPException(status_code=403, detail="Invalid authorization")
      try:
        response = await stytch_base.authenticate_user_with_jkws(token)
        if response.success:
          return {"stytch_user_id": response.data["sub"]}
        else:
          raise HTTPException(status_code=403, detail=str(response.errors[0]["message"]))
      except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    else:
      raise HTTPException(status_code=403, detail="Invalid authorization")


class APIKeyAuth:
  """API Key authentication for webhook endpoints."""

  async def __call__(
    self,
    request: Request,
    session: AsyncSession = Depends(get_db),
  ) -> dict:
    """Authenticate using API key from x-api-key header."""
    try:
      # Get API key from header
      api_key = request.headers.get("x-api-key")
      if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Include 'x-api-key' header.")

      # Hash the API key for database lookup
      api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

      # Look up API key in database
      api_key_query = select(APIKeyModel).where(
        and_(
          APIKeyModel.api_key_hash == api_key_hash,
          APIKeyModel.is_active,
        )
      )
      result = await session.execute(api_key_query)
      api_key_model = result.scalar_one_or_none()

      if not api_key_model:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

      # Check if API key has expired
      if api_key_model.expires_at and api_key_model.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="API key has expired")

      # Verify the JWT token to get user_id and org_id
      try:
        token_payload = verify_jwt_token(api_key_model.api_key_token)
      except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key token: {str(e)}")

      user_id = UUID(token_payload.get("user_id"))
      org_id = UUID(token_payload.get("org_id"))

      # Check if user is owner of the organization
      member_query = (
        select(OrganizationMemberModel, RoleModel)
        .join(RoleModel, OrganizationMemberModel.role_id == RoleModel.id)
        .where(and_(OrganizationMemberModel.user_id == user_id, OrganizationMemberModel.organization_id == org_id))
      )
      member_result = await session.execute(member_query)
      member_data = member_result.first()

      if not member_data:
        raise HTTPException(status_code=403, detail="User is not a member of this organization")

      member, role = member_data
      # Check organization member status
      if member.status == "deleted":
        raise HTTPException(status_code=403, detail="User has been removed from this organization")
      if member.status == "suspended":
        raise HTTPException(status_code=403, detail="User access has been suspended in this organization")
      if member.status == "invited":
        raise HTTPException(status_code=403, detail="User invitation is still pending")
      if member.status != "active":
        raise HTTPException(status_code=403, detail=f"User is not active in this organization (status: {member.status})")

      # Check if user has owner role
      if role.name.lower() != "owner":
        raise HTTPException(status_code=403, detail="API key can only be used by organization owners")

      # Update last used timestamp
      api_key_model.last_used_at = datetime.utcnow()
      session.add(api_key_model)
      await session.commit()

      # Return authentication context
      return {
        "api_key_id": str(api_key_model.id),
        "user_id": token_payload.get("user_id"),
        "org_id": token_payload.get("org_id"),
        "agent_id": str(api_key_model.agent_id) if api_key_model.agent_id else None,
        "permissions": api_key_model.permissions,
        "auth_type": "api_key",
        "role": role.name,
        "role_level": role.hierarchy_level,
      }

    except HTTPException:
      raise
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"API key authentication failed: {str(e)}")


class InternalAuth:
  """Internal service authentication for background tasks and internal endpoints."""

  async def __call__(
    self,
    request: Request,
  ) -> dict:
    """Authenticate using internal token from x-internal-token header."""
    try:
      # Get internal token from header
      internal_token = request.headers.get("x-internal-token")
      if not internal_token:
        raise HTTPException(status_code=401, detail="Missing internal token. Include 'x-internal-token' header.")

      # Verify token matches configured internal token
      if internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")

      # Return authentication context for internal services
      return {
        "auth_type": "internal",
        "service": "internal",
        "authenticated": True,
      }

    except HTTPException:
      raise
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Internal authentication failed: {str(e)}")


class RBAC:
  """RBAC with wildcard support."""

  def __init__(self, required_resource: str, required_action: str):
    self.required_resource = required_resource
    self.required_action = required_action

  def _check_wildcard_match(self, permission_value: str, required_value: str) -> bool:
    """Check if permission matches required value with wildcard support."""
    if permission_value == "*":
      return True
    if "*" not in permission_value:
      return permission_value == required_value

    # Handle pattern matching with wildcards
    pattern_parts = permission_value.split("*")
    value = required_value

    # Check prefix
    if pattern_parts[0] and not required_value.startswith(pattern_parts[0]):
      return False

    # Check suffix
    if pattern_parts[-1] and not required_value.endswith(pattern_parts[-1]):
      return False

    # Check middle parts
    for part in pattern_parts[1:-1]:
      if part not in required_value:
        return False
      # Move past the matched part for next check
      value = value[value.find(part) + len(part) :]

    return True

  async def __call__(
    self,
    request: Request = None,  # type: ignore
    websocket: WebSocket = None,  # type: ignore
    token_payload: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> dict:
    try:
      stytch_user_id = token_payload.get("stytch_user_id")
      if request:
        org_id = request.query_params.get("org_id")
      elif websocket:
        org_id = websocket.query_params.get("org_id")
      else:
        raise HTTPException(status_code=403, detail="Invalid org id")

      # Complex query to get user, organization, role, and permissions data
      user_permissions_query = (
        select(
          UserModel.id.label("user_id"),
          UserModel.email,
          OrganizationModel.id.label("organization_id"),
          OrganizationModel.name.label("organization_name"),
          RoleModel.id.label("role_id"),
          RoleModel.name.label("role_name"),
          RoleModel.hierarchy_level,
          OrganizationMemberModel.status,
          PermissionModel.id.label("permission_id"),
          PermissionModel.name.label("permission_name"),
          PermissionModel.resource,
          PermissionModel.action,
        )
        .select_from(UserModel)
        .join(OrganizationMemberModel, UserModel.id == OrganizationMemberModel.user_id)
        .join(OrganizationModel, OrganizationMemberModel.organization_id == OrganizationModel.id)
        .join(RoleModel, OrganizationMemberModel.role_id == RoleModel.id)
        .join(RolePermissionModel, RoleModel.id == RolePermissionModel.role_id)
        .join(PermissionModel, RolePermissionModel.permission_id == PermissionModel.id)
        .where(and_(UserModel.stytch_id == stytch_user_id, OrganizationModel.id == org_id))
      )

      result = await session.execute(user_permissions_query)
      user_data = result.all()

      if not user_data:
        raise HTTPException(status_code=403, detail="User is not a member of this organization")

      # Get the first row to check member status and get user/org info
      first_row = user_data[0]

      # Check organization member status with specific error messages
      if first_row.status == "deleted":
        raise HTTPException(status_code=403, detail="Access denied: User has been removed from this organization")
      if first_row.status == "suspended":
        raise HTTPException(status_code=403, detail="Access denied: User access has been suspended in this organization")
      if first_row.status == "invited":
        raise HTTPException(status_code=403, detail="Access denied: User invitation is still pending acceptance")
      if first_row.status != "active":
        raise HTTPException(status_code=403, detail=f"Access denied: User is not active in this organization (status: {first_row.status})")

      # Check permissions with wildcard support
      has_permission = False
      prem_list = []

      for row in user_data:
        resource_match = self._check_wildcard_match(row.resource, self.required_resource)
        action_match = self._check_wildcard_match(row.action, self.required_action)
        if row.resource != "*":
          prem_list.append(f"{row.resource}_{row.action}")
        if resource_match and action_match:
          has_permission = True
          break

      if not has_permission:
        raise HTTPException(status_code=403, detail=f"Access denied. Required: {self.required_resource}:{self.required_action}")

      # Add role info to token payload
      token_payload["org_id"] = org_id
      token_payload["required_permission"] = f"{self.required_resource}_{self.required_action}"
      token_payload["role"] = first_row.role_name
      token_payload["role_level"] = first_row.hierarchy_level
      token_payload["role_id"] = first_row.role_id
      token_payload["permissions"] = prem_list
      return token_payload
    except Exception as e:
      raise HTTPException(status_code=403, detail=str(e))
