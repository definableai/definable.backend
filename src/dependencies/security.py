from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import HTTPBearer
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import logger
from database import get_db
from libs.stytch.v1 import stytch_base
from models import OrganizationMemberModel, PermissionModel, RoleModel, RolePermissionModel


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
        response = await stytch_base.authenticate_user(credentials.credentials)
        if response.success and response.data.status_code == 200:
          user = response.model_dump()["data"]["user"]
          logger.info(f"Got user data: {user}")
          return {"id": user["trusted_metadata"]["external_user_id"], "user_data": user}
        else:
          raise HTTPException(status_code=403, detail="Invalid authorization")
      except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    elif websocket:
      token = websocket.query_params.get("token")
      if not token:
        raise HTTPException(status_code=403, detail="Invalid authorization")
      try:
        response = await stytch_base.authenticate_user(token)
        if response.success and response.data.status_code == 200:
          user = response.model_dump()["data"]["user"]
          return {"id": user["trusted_metadata"]["external_user_id"], "user_data": user}
        else:
          raise HTTPException(status_code=403, detail="Invalid authorization")
      except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
    else:
      raise HTTPException(status_code=403, detail="Invalid authorization")


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
      user_id = UUID(token_payload.get("id"))
      if request:
        org_id = request.query_params.get("org_id")
      elif websocket:
        org_id = websocket.query_params.get("org_id")
      else:
        raise HTTPException(status_code=403, detail="Invalid org id")

      # Get user's role in organization
      member_query = select(OrganizationMemberModel).where(
        and_(
          OrganizationMemberModel.user_id == user_id, OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active"
        )
      )
      members = await session.execute(member_query)
      member = members.unique().scalar_one_or_none()

      if not member:
        raise HTTPException(status_code=403, detail="User not active in organization")

      # Get role permissions
      role_perms_query = (
        select(RolePermissionModel, PermissionModel)
        .join(PermissionModel, RolePermissionModel.permission_id == PermissionModel.id)
        .where(RolePermissionModel.role_id == member.role_id)
      )
      role_perms = await session.execute(role_perms_query)
      permissions = role_perms.unique().all()
      prem_list = []
      # Check permissions with wildcard support
      has_permission = False
      for _, permission in permissions:
        resource_match = self._check_wildcard_match(permission.resource, self.required_resource)
        action_match = self._check_wildcard_match(permission.action, self.required_action)
        if permission.resource != "*":
          prem_list.append(f"{permission.resource}_{permission.action}")
        if resource_match and action_match:
          has_permission = True
          break

      if not has_permission:
        raise HTTPException(status_code=403, detail=f"Access denied. Required: {self.required_resource}:{self.required_action}")

      # Get role details
      role_query = select(RoleModel).where(RoleModel.id == member.role_id)
      roles = await session.execute(role_query)
      role = roles.unique().scalar_one_or_none()
      if not role:
        raise HTTPException(status_code=403, detail="Role not found")

      # Add role info to token payload
      token_payload["org_id"] = org_id
      token_payload["required_permission"] = f"{self.required_resource}_{self.required_action}"
      token_payload["role"] = role.name
      token_payload["role_level"] = role.hierarchy_level
      token_payload["role_id"] = role.id
      token_payload["permissions"] = prem_list
      return token_payload
    except Exception as e:
      raise HTTPException(status_code=403, detail=str(e))
