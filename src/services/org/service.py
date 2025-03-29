from typing import List
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import JWTBearer
from models import OrganizationMemberModel, OrganizationModel, RoleModel
from services.__base.acquire import Acquire
from services.roles.service import RoleService
from loguru import logger

from .schema import OrganizationResponse


class OrganizationService:
    """Organization service."""

    http_exposed = ["post=create_org", "post=add_member", "get=list"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire

    async def post_create_org(
        self,
        name: str,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(JWTBearer()),
    ) -> OrganizationResponse:
        """Create default organization for new user."""
        # Create organization
        logger.info(f"Creating organization for user {user.get('id')}")
        slug = f"{name}-{str(uuid4())[:8]}"
        base_slug = slug
        counter = 1

        # Ensure unique slug
        while True:
            query = select(OrganizationModel).where(OrganizationModel.slug == slug)
            logger.info(f"Checking if slug {slug} is unique")
            result = await session.execute(query)
            if not result.scalar_one_or_none():
                logger.info(f"Slug {slug} is unique")
                break
            slug = f"{base_slug}-{counter}"
            counter += 1

        org = OrganizationModel(
            name=name,
            slug=slug,
            settings={},
        )
        logger.info(f"Adding organization {org} to session")
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
        logger.info(f"Adding member {member} to session")
        session.add(member)
        logger.info(f"Committing session")
        try:
            await session.commit()
            logger.info(f"Committed session")
        except Exception as e:
            logger.error(f"Error committing session: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create organization: {str(e)}"
            )
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
        role = await RoleService._get_role(
            role_id=role_id, organization_id=organization_id, session=session
        )
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
        return JSONResponse(
            status_code=201, content={"message": "Member added successfully"}
        )

    async def get_list(
        self,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(JWTBearer()),
    ) -> List[OrganizationResponse]:
        """Get all organizations that user belongs to."""
        query = (
            select(OrganizationModel)
            .outerjoin(
                OrganizationMemberModel,
                OrganizationModel.id == OrganizationMemberModel.organization_id,
            )
            .where(OrganizationMemberModel.user_id == user.get("id"))
        )
        result = await session.execute(query)
        orgs = result.unique().scalars().all()

        # Convert SQLAlchemy models to Pydantic models
        return [OrganizationResponse.model_validate(org) for org in orgs]
