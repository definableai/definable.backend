from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import JWTBearer
from models import OrganizationMemberModel, OrganizationModel, RoleModel, UserModel
from services.__base.acquire import Acquire

from .schema import OrganizationInfo, UserDetailResponse


class UserService:
  """User service."""

  http_exposed = ["get=me", "get=details"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger

  async def get_me(
    self,
    current_user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> UserDetailResponse:
    """
    Get current user details including organization information.
    """
    try:
      self.logger.info(f"Getting user details for current user {current_user['id']}")
      return await self._get_user_details(current_user["id"], session)
    except Exception as e:
      self.logger.error(f"Error in get_me: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve user details",
      )

  async def get_details(
    self,
    user_id: UUID,
    org_id: UUID,
    user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> UserDetailResponse:
    """
    Get details for a specific user by ID.

    The logged-in user must be in the same organization as the target user.
    Access is restricted by role hierarchy:
    - Owners can view anyone
    - Admins can view developers but not owners
    - Developers can only view themselves

    Args:
        user_id: ID of the user to view
        org_id: Organization ID context
        user: JWT token payload of the logged-in user
        session: Database session

    Returns:
        UserDetailResponse: User details with organization information

    Raises:
        HTTPException: If user is not found or the logged-in user doesn't have permission
    """
    try:
      self.logger.info(f"User {user['id']} requesting details for user {user_id} in org {org_id}")

      # First, verify that the logged-in user belongs to the specified organization
      logged_in_user_id = UUID(user["id"])

      # Check if the logged-in user is a member of the specified organization
      logged_in_member_query = select(OrganizationMemberModel).where(
        OrganizationMemberModel.user_id == logged_in_user_id,
        OrganizationMemberModel.organization_id == org_id,
        OrganizationMemberModel.status == "active",
      )
      logged_in_result = await session.execute(logged_in_member_query)
      logged_in_member = logged_in_result.scalar_one_or_none()

      if not logged_in_member:
        self.logger.warning(f"User {logged_in_user_id} is not a member of organization {org_id}")
        raise HTTPException(
          status_code=status.HTTP_403_FORBIDDEN,
          detail="You don't have access to this organization",
        )

      # Get the logged-in user's role
      logged_in_role = await session.get(RoleModel, logged_in_member.role_id) if logged_in_member.role_id else None
      if not logged_in_role:
        self.logger.warning(f"User {logged_in_user_id} has no role in organization {org_id}")
        raise HTTPException(
          status_code=status.HTTP_403_FORBIDDEN,
          detail="You don't have a role in this organization",
        )

      # If user is checking their own details, proceed
      if logged_in_user_id == user_id:
        self.logger.info(f"User {logged_in_user_id} is viewing their own details")
        return await self._get_user_details(user_id, session)

      # Check if target user belongs to the same organization
      target_member_query = select(OrganizationMemberModel).where(
        OrganizationMemberModel.user_id == user_id,
        OrganizationMemberModel.organization_id == org_id,
        OrganizationMemberModel.status == "active",
      )
      target_member_result = await session.execute(target_member_query)
      target_member = target_member_result.scalar_one_or_none()

      if not target_member:
        self.logger.warning(f"Target user {user_id} is not a member of organization {org_id}")
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail="User not found in this organization",
        )

      # Get the target user's role
      target_role = await session.get(RoleModel, target_member.role_id) if target_member.role_id else None
      if not target_role:
        self.logger.warning(f"Target user {user_id} has no role in organization {org_id}")
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail="User has no role in this organization",
        )

      # Check role-based permissions
      # Owners (level 100) can view anyone
      # Admins (level 90) can view developers (level 50) but not owners
      # Developers (level 50) can view only themselves (already checked above)

      if logged_in_role.name.upper() == "OWNER":
        # Owners can view anyone
        self.logger.info(f"Owner {logged_in_user_id} viewing user {user_id}")
        return await self._get_user_details(user_id, session)
      elif logged_in_role.name.upper() == "ADMIN" and target_role.name.upper() != "OWNER":
        # Admins can view non-owners
        self.logger.info(f"Admin {logged_in_user_id} viewing non-owner user {user_id}")
        return await self._get_user_details(user_id, session)
      else:
        # Default deny
        self.logger.warning(f"User {logged_in_user_id} (role: {logged_in_role.name}) not allowed to view {user_id} (role: {target_role.name})")
        raise HTTPException(
          status_code=status.HTTP_403_FORBIDDEN,
          detail="You don't have permission to view this user",
        )

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error in get_details: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve user details",
      )

  async def _get_user_details(self, user_id: UUID, session: AsyncSession) -> UserDetailResponse:
    """
    Helper method to get user details with organization information.
    """
    try:
      # First, get the user
      self.logger.debug(f"Fetching user with ID: {user_id}")
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      user = user_result.scalar_one_or_none()

      if not user:
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail="User not found",
        )

      # Load organization memberships directly without joins
      self.logger.debug(f"Fetching organization memberships for user: {user_id}")
      member_query = select(OrganizationMemberModel).where(
        OrganizationMemberModel.user_id == user_id,
        OrganizationMemberModel.status == "active",
      )
      member_result = await session.execute(member_query)
      members = member_result.scalars().all()

      if not members:
        self.logger.info(f"No organizations found for user: {user_id}")
        # Return user details without organizations rather than a 404
        return UserDetailResponse(
          id=user.id,
          email=user.email,
          first_name=user.first_name,
          last_name=user.last_name,
          full_name=user.full_name,
          is_active=user.is_active,
          organizations=[],
        )

      # Process each membership one at a time
      organizations = []
      for member in members:
        try:
          # Fetch organization details
          org_id = member.organization_id
          self.logger.debug(f"Fetching organization details for org ID: {org_id}")
          org = await session.get(OrganizationModel, org_id)

          if not org:
            self.logger.warning(f"Organization not found for ID: {org_id}")
            continue

          # Fetch role if present
          role = None
          if member.role_id:
            self.logger.debug(f"Fetching role details for role ID: {member.role_id}")
            role = await session.get(RoleModel, member.role_id)

          # Create organization info object
          org_info = OrganizationInfo(
            id=org.id,
            name=org.name,
            slug=org.slug,
            role_id=role.id if role else None,
            role_name=role.name if role else None,
          )
          organizations.append(org_info)
        except Exception as e:
          self.logger.error(f"Error processing organization {member.organization_id}: {str(e)}", exc_info=True)
          # Continue with other organizations

      # Return user details with organizations
      self.logger.info(f"Returning user details with {len(organizations)} organizations")
      return UserDetailResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=user.full_name,
        is_active=user.is_active,
        organizations=organizations,
      )
    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error in _get_user_details: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve user details",
      )
