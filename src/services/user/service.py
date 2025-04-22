import json
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.stytch.v1 import stytch_base
from models import InvitationModel, InvitationStatus, OrganizationMemberModel, OrganizationModel, RoleModel, UserModel
from services.__base.acquire import Acquire

from .schema import InviteSignup, OrganizationInfo, UserDetailResponse


class UserService:
  """User service."""

  http_exposed = ["get=me", "get=list", "post=invite", "put=invite"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger
    self.schemas = acquire.schemas

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
    user: dict = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """
    Get a list of all users in an organization.

    Only organization owners can access this endpoint.

    Args:
        org_id: Organization ID
        user: JWT token payload of the logged-in user
        session: Database session

    Returns:
        List[UserListResponse]: List of users in the organization

    Raises:
        HTTPException: If user doesn't have permission or other errors occur
    """
    try:
      self.logger.info(f"User {user['id']} requesting list of all members in org {org_id}")

      # Verify that the logged-in user belongs to the specified organization
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

      # Check if user is an OWNER
      if logged_in_role.name.upper() != "OWNER":
        self.logger.warning(f"User {logged_in_user_id} with role {logged_in_role.name} attempted to list all members")
        raise HTTPException(
          status_code=status.HTTP_403_FORBIDDEN,
          detail="Only organization owners can view all members",
        )

      # Get all active members of the organization
      members_query = (
        select(OrganizationMemberModel, UserModel, RoleModel)
        .join(UserModel, OrganizationMemberModel.user_id == UserModel.id)
        .join(RoleModel, OrganizationMemberModel.role_id == RoleModel.id)
        .where(OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active")
      )

      result = await session.execute(members_query)
      members_data = result.all()

      # Transform the data into the response format
      users_list = []
      for member, user_model, role in members_data:
        users_list.append({
          "id": str(user_model.id),
          "email": user_model.email,
          "first_name": user_model.first_name,
          "last_name": user_model.last_name,
          "role": {"id": str(role.id), "name": role.name},
          "created_at": user_model.created_at.isoformat() if user_model.created_at else None,
        })

      self.logger.info(f"Returning list of {len(users_list)} members for organization {org_id}")
      return JSONResponse(content=users_list, status_code=status.HTTP_200_OK)

    except Exception as e:
      self.logger.error(f"Error in get_list: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to retrieve users list",
      )

  async def post_invite(
    self,
    user_data: InviteSignup,
    token_payload: dict = Depends(RBAC("users", "invite")),
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Post signup invite."""
    try:
      # Check if user exists
      email = user_data.email
      query = select(UserModel).where(UserModel.email == email)
      result = await session.execute(query)
      user = result.scalars().first()

      if user:
        # Get user from Stytch
        stytch_user = await stytch_base.get_user(str(user.id))
        if not stytch_user.success:  # If user does not exist in Stytch, invite them
          invite_response = await stytch_base.invite_user(
            email=email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
          )

        if not invite_response.success:
          self.logger.error("Failed to invite user through Stytch", email=email, error=invite_response.errors)
          raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send invitation")
      else:
        # User does not exist, create new user
        user = await stytch_base.invite_user(email=email, first_name=user_data.first_name, last_name=user_data.last_name)

        invitation = InvitationModel(
          organization_id=UUID(token_payload.get("org_id")),
          role_id=user_data.role,
          invitee_email=email,
          invited_by=UUID(token_payload.get("user_id")),
          status=InvitationStatus.PENDING,
        )
        session.add(invitation)
        await session.commit()

      return JSONResponse(
        content={
          "message": "User invited successfully",
        },
        status_code=status.HTTP_200_OK,
      )

    except Exception as e:
      self.logger.error("Error inviting user", email=email, error=str(e))
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

  async def put_invite(
    self,
    request: Request,
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Update invitation status."""
    try:
      content = await request.body()
      data = json.loads(content.decode("utf-8"))
      invitation_id = data.get("trusted_metadata").get("invitation_id")
      invitation_status = data.get("trusted_metadata").get("invitation_status")

      query = update(InvitationModel).where(InvitationModel.id == invitation_id).values(status=invitation_status)
      await session.execute(query)
      await session.commit()

      return JSONResponse(
        content={
          "message": "Invitation updated successfully",
        },
        status_code=status.HTTP_200_OK,
      )
    except Exception as e:
      self.logger.error("Error updating invitation status", invitation_id=invitation_id, error=str(e))
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
