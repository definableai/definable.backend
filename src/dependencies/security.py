from typing import Optional
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from services.org.model import OrganizationMemberModel
from services.roles.model import PermissionModel, RoleModel, RolePermissionModel


class JWTBearer(HTTPBearer):
  def __init__(self, auto_error: bool = True):
    super().__init__(auto_error=auto_error)

  async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
    credentials = await super().__call__(request)
    if not credentials or credentials.scheme != "Bearer":
      raise HTTPException(status_code=403, detail="Invalid authorization")

    try:
      payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
      return payload
    except jwt.InvalidTokenError:
      raise HTTPException(status_code=403, detail="Invalid or expired token")


class RBAC:
  def __init__(self, required_resource: str, required_action: str):
    self.required_resource = required_resource
    self.required_action = required_action

  async def __call__(self, request: Request, token_payload: dict = Depends(JWTBearer()), session: AsyncSession = Depends(get_db)) -> dict:
    print(request.query_params)
    user_id = UUID(token_payload.get("id"))
    org_id = request.query_params.get("org_id")
    print(user_id, org_id)
    # Get user's role in organization
    member_query = select(OrganizationMemberModel).where(
      and_(OrganizationMemberModel.user_id == user_id, OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active")
    )
    members = await session.execute(member_query)
    member = members.unique().scalar_one_or_none()

    if not member:
      raise HTTPException(status_code=403, detail="User not active in organization")

    # Get role permissions
    role_perms_query = (
      select(RolePermissionModel, PermissionModel)
      .join(PermissionModel, RolePermissionModel.permission_id == PermissionModel.id)
      .where(
        and_(
          RolePermissionModel.role_id == member.role_id,
          PermissionModel.resource == self.required_resource,
          PermissionModel.action == self.required_action,
        )
      )
    )
    role_perms = await session.execute(role_perms_query)
    role_perm = role_perms.unique().scalar_one_or_none()

    if not role_perm:
      raise HTTPException(status_code=403, detail=f"Access denied. Required: {self.required_resource}:{self.required_action}")

    role_query = select(RoleModel).where(RoleModel.id == member.role_id)
    roles = await session.execute(role_query)
    role = roles.unique().scalar_one_or_none()

    if not role:
      raise HTTPException(status_code=403, detail="Role not found")

    token_payload["role"] = role.name
    token_payload["role_level"] = role.hierarchy_level
    token_payload["role_id"] = role.id
    return token_payload
