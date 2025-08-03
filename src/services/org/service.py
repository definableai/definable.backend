from datetime import timedelta
from typing import List
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import JWTBearer
from models import APIKeyModel, OrganizationMemberModel, OrganizationModel, RoleModel
from services.__base.acquire import Acquire
from services.roles.service import RoleService
from utils.auth_util import create_access_token

from .schema import OrganizationResponse


class OrganizationService:
  """Organization service."""

  http_exposed = ["post=create_org", "post=add_member", "get=list"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def post_create_org(
    self,
    name: str,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> OrganizationResponse:
    """Create default organization for new user."""
    # Create organization
    slug = f"{name}-{str(uuid4())[:8]}"
    base_slug = slug
    counter = 1

    # Ensure unique slug
    while True:
      query = select(OrganizationModel).where(OrganizationModel.slug == slug)
      result = await session.execute(query)
      if not result.scalar_one_or_none():
        break
      slug = f"{base_slug}-{counter}"
      counter += 1

    org = OrganizationModel(
      name=name,
      slug=slug,
      settings={},
    )
    session.add(org)
    await session.flush()

    # only owner can create an organization, so set a default role owner
    query = select(RoleModel).where(RoleModel.name == "owner")
    result = await session.execute(query)
    owner_role = result.unique().scalar_one_or_none()
    if not owner_role:
      raise HTTPException(status_code=500, detail="Default OWNER role not found")

    # now add organization member
    member = OrganizationMemberModel(
      organization_id=org.id,
      user_id=user.get("id"),
      role_id=owner_role.id,
      status="active",
    )
    session.add(member)
    await session.commit()
    return org

  async def post_add_member(
    self,
    organization_id: UUID,
    user_id: UUID,
    role_id: UUID,
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Add member to organization."""
    # Get role
    role = await RoleService._get_role(role_id=role_id, organization_id=organization_id, session=session)
    if not role:
      raise HTTPException(
        status_code=404,
        detail=f"Role {role_id} not found",
      )

    # Create member
    member = OrganizationMemberModel(
      organization_id=organization_id,
      user_id=user_id,
      role_id=role.id,
      status="active",
    )
    session.add(member)
    return JSONResponse(status_code=201, content={"message": "Member added successfully"})

  async def get_list(
    self,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> List[OrganizationResponse]:
    """Get all organizations that user belongs to."""

    # Check if user has auth-type API key, generate if missing
    user_id = UUID(user.get("id"))

    # Check for existing auth token in api_keys table
    auth_key_query = select(APIKeyModel).where(APIKeyModel.user_id == user_id, APIKeyModel.token_type == "auth", APIKeyModel.is_active)
    auth_key_result = await session.execute(auth_key_query)
    existing_auth_key = auth_key_result.scalar_one_or_none()

    self.logger.info(f"User {user_id} has existing auth key: {existing_auth_key is not None}")

    if not existing_auth_key:
      # Get user's primary organization
      member_query = (
        select(OrganizationMemberModel).where(OrganizationMemberModel.user_id == user_id, OrganizationMemberModel.status == "active").limit(1)
      )
      member_result = await session.execute(member_query)
      member = member_result.scalar_one_or_none()

      if member:
        await self._create_auth_api_key(user_id, member.organization_id, session)
        await session.commit()
        self.logger.info(f"Created auth API key for user {user_id}")

    query = (
      select(OrganizationModel)
      .outerjoin(OrganizationMemberModel, OrganizationModel.id == OrganizationMemberModel.organization_id)
      .where(OrganizationMemberModel.user_id == user.get("id"))
    )
    # self.logger.debug(f"Getting organizations for user {user.get('id')}")
    logger.debug(f"Getting organizations for user {user.get('id')}")
    result = await session.execute(query)
    orgs = result.unique().scalars().all()

    # Convert SQLAlchemy models to Pydantic models
    return [OrganizationResponse.model_validate(org) for org in orgs]

  async def _create_auth_api_key(self, user_id: UUID, org_id: UUID, session: AsyncSession) -> str:
    """Create auth-type API key for existing user."""
    import hashlib
    from datetime import datetime

    # Generate JWT token
    payload = {"user_id": str(user_id), "org_id": str(org_id), "type": "auth_token"}
    auth_token = create_access_token(payload, timedelta(days=365))

    # Create API key entry with token_type="auth"
    api_key_model = APIKeyModel(
      user_id=user_id,
      agent_id=None,  # No specific agent for auth tokens
      token_type="auth",
      name="Default Auth Token",
      api_key_token=auth_token,
      api_key_hash=hashlib.sha256(auth_token.encode()).hexdigest(),
      permissions={"*": True},  # Full permissions for auth tokens
      expires_at=datetime.utcnow() + timedelta(days=365),
      is_active=True,
    )

    session.add(api_key_model)
    await session.flush()
    return auth_token
