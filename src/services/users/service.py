from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.stytch.v1 import stytch_base
from models import OrganizationMemberModel, OrganizationModel, RoleModel, UserModel
from services.__base.acquire import Acquire

from .schema import InviteSignup, OrganizationInfo, StytchUser, UserDetailResponse, UserListResponse


class UserService:
  """User service."""

  http_exposed = ["get=me", "get=list", "post=invite"]

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

  async def get_list(
    self,
    org_id: UUID,
    offset: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("users", "read")),
  ) -> UserListResponse:
    """
    Get a paginated list of all users in an organization.

    Args:
        org_id: Organization ID
        offset: Pagination offset (page number, 0-based)
        limit: Number of items per page
        session: Database session
        user: Current user with appropriate permissions

    Returns:
        Paginated list of users with organization details
    """
    try:
      self.logger.info(f"Getting list of users for organization {org_id}")
      self.logger.debug(f"Pagination parameters: offset={offset}, limit={limit}")

      # Count total users in the organization
      count_query = (
        select(func.count(UserModel.id))
        .join(OrganizationMemberModel, UserModel.id == OrganizationMemberModel.user_id)
        .where(OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active")
      )
      total = await session.scalar(count_query) or 0
      self.logger.debug(f"Total users in organization {org_id}: {total}")

      # Get user IDs with pagination
      user_query = (
        select(UserModel.id)
        .join(OrganizationMemberModel, UserModel.id == OrganizationMemberModel.user_id)
        .where(OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active")
        .order_by(UserModel.created_at.desc())
        .offset(offset * limit)
        .limit(limit + 1)  # Get one extra to check if there are more
      )
      self.logger.debug(f"Executing user query with offset={offset * limit}, limit={limit + 1}")

      result = await session.execute(user_query)
      user_ids = list(result.scalars().all())
      self.logger.debug(f"Retrieved {len(user_ids)} user IDs from database")

      # Check if there are more users
      has_more = len(user_ids) > limit
      self.logger.debug(f"Has more users: {has_more}")

      user_ids = user_ids[:limit]  # Remove the extra item

      # Get detailed user information for each user
      self.logger.debug(f"Fetching detailed information for {len(user_ids)} users")
      user_details = []
      for user_id in user_ids:
        try:
          self.logger.debug(f"Fetching details for user: {user_id}")
          user_detail = await self._get_user_details(user_id, session)
          user_details.append(user_detail)
        except Exception as e:
          self.logger.warning(f"Failed to get details for user {user_id}: {str(e)}")
          # Continue with other users instead of failing the entire request

      self.logger.info(f"Successfully retrieved {len(user_details)} users for organization {org_id}")
      return UserListResponse(users=user_details, total=total)

    except Exception as e:
      self.logger.error(f"Error in get_list: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve user list",
      )

  async def post_invite(
    self,
    user_data: InviteSignup,
    org_id: UUID,
    token_payload: dict = Depends(RBAC("users", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Post signup invite."""
    email = user_data.email
    try:
      self.logger.info(f"Processing invite request for email: {email}")
      self.logger.debug(f"Invite data: first_name={user_data.first_name}, last_name={user_data.last_name}")
      self.logger.debug("User does not exist, creating new user and sending invitation")

      # User does not exist, create new user
      user = await stytch_base.invite_user(email=email, first_name=user_data.first_name, last_name=user_data.last_name)
      self.logger.info(f"Stytch user: {user}")
      self.logger.info("Stytch invitation sent, creating invitation record")

      self.logger.debug(f"Creating invitation record for org_id={org_id}, role_id={user_data.role}")
      self.logger.debug("Invitation record created successfully")

      self.logger.info(f"User invitation process completed successfully for {email}")
      return JSONResponse(
        content={
          "message": "User invited successfully",
        },
        status_code=status.HTTP_200_OK,
      )

    except Exception as e:
      self.logger.error(f"Error inviting user: {str(e)}", email=email, exc_info=True)
      from traceback import print_exc

      print_exc()
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

  ### PRIVATE METHODS ###
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

  async def _setup_user(self, user_data: StytchUser, org_id: UUID, role: str, session: AsyncSession) -> UserModel:
    """Setup a user."""
    user = await session.execute(select(UserModel).where(UserModel.stytch_id == user_data.stytch_id))
    results = user.unique().scalar_one_or_none()
    if results:
      self.logger.info("User already exists", user_id=str(results.id))
      await self._setup_organization(results.id, org_id, role, session)
    else:
      try:
        self.logger.info("Setting up user in the organization", org_id=str(org_id))
        user = UserModel(
          stytch_id=user_data.stytch_id,
          email=user_data.email,
          first_name=user_data.first_name,
          last_name=user_data.last_name,
        )
        session.add(user)
        await session.flush()
        user_id = user.id  # type: ignore
        await self._setup_organization(user_id, org_id, role, session)
        await session.commit()
        await session.refresh(user)
        self.logger.info(f"User created: {user}")
        return user
      except Exception as e:
        self.logger.error(f"Error in _setup_user: {str(e)}", exc_info=True)
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail="Failed to setup user",
        )

  async def _setup_organization(self, user_id: UUID, org_id: UUID, role: str, session: AsyncSession) -> None:
    """Setup organization."""
    # Get organization from database
    org = await session.get(OrganizationModel, org_id)
    if not org:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Organization not found",
      )

    query = select(RoleModel).where(RoleModel.name == role)
    result = await session.execute(query)
    role_model = result.unique().scalar_one_or_none()

    # Check if role exists
    if not role_model:
      self.logger.error("Role not found", role=role)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Role not found",
      )

    # Create organization member entry
    member = OrganizationMemberModel(
      organization_id=org.id,
      user_id=user_id,
      role_id=role_model.id,  # Self-invited as creator
      status="active",  # Directly active as creator
    )
    session.add(member)
    await session.flush()

    self.logger.info(
      "Organization setup complete",
      user_id=str(user_id),
      org_id=str(org_id),
    )
