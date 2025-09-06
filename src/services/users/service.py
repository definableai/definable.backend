from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.stytch.v1 import stytch_base
from models import InvitationModel, OrganizationMemberModel, OrganizationModel, RoleModel, UserModel
from models.invitations_model import InvitationStatus
from services.__base.acquire import Acquire

from .schema import InviteResponse, InviteSignup, OrganizationInfo, StytchUser, UserDetailResponse, UserListResponse


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
    Get a hybrid list of active users and pending invitations for an organization.

    Args:
        org_id: Organization ID
        offset: Pagination offset (page number, 0-based)
        limit: Number of items per page
        session: Database session
        user: Current user with appropriate permissions

    Returns:
        Paginated list combining active users and pending invitations
    """
    try:
      self.logger.info(f"Getting list of users for organization {org_id}")
      self.logger.debug(f"Pagination parameters: offset={offset}, limit={limit}")

      # Get all items (users + invitations) to properly handle sorting and pagination
      all_items = []

      # 1. Get active users
      active_users = await self._get_active_users(org_id, session)
      for user_detail in active_users:
        all_items.append(user_detail)

      # 2. Get pending invitations
      pending_invitations = await self._get_pending_invitations(org_id, session)
      for invitation_detail in pending_invitations:
        all_items.append(invitation_detail)

      # 3. Sort by status (active first) and then by creation date (newest first)
      all_items.sort(
        key=lambda x: (
          0 if x.status == "active" else 1,  # Active users first
          x.invited_at or (str(x.id) if x.id else ""),  # Then by date (newest first)
        ),
        reverse=True,
      )

      total = len(all_items)
      self.logger.debug(f"Total items (users + invitations): {total}")

      # 4. Apply pagination
      start_idx = offset * limit
      end_idx = start_idx + limit
      paginated_items = all_items[start_idx:end_idx]

      self.logger.info(f"Successfully retrieved {len(paginated_items)} items for organization {org_id}")
      return UserListResponse(users=paginated_items, total=total)

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
  ) -> InviteResponse:
    """Post signup invite."""
    email = user_data.email
    try:
      self.logger.info(f"Processing invite request for email: {email}")
      self.logger.debug(f"Invite data: first_name={user_data.first_name}, last_name={user_data.last_name}")
      self.logger.debug("User does not exist, creating new user and sending invitation")

      # 1. Validate organization exists
      org = await session.get(OrganizationModel, org_id)
      if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

      # 2. Get role details by role ID
      role_query = select(RoleModel).where(RoleModel.id == user_data.role)
      role_result = await session.execute(role_query)
      role = role_result.unique().scalar_one_or_none()
      if not role:
        self.logger.error(f"Role '{user_data.role}' not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")

      # 3. Create user entry immediately (will be activated when invitation is accepted)
      invited_user = UserModel(
        email=email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
      )
      session.add(invited_user)
      await session.flush()  # Get the user ID
      self.logger.info(f"Created invited user: {invited_user.id}")

      # 4. Create invitation record with proper role ID
      invitation = InvitationModel(
        invitee_email=email,
        organization_id=org_id,
        role_id=role.id,
        expiry_time=datetime.now(timezone.utc) + timedelta(days=7),
      )
      session.add(invitation)
      await session.flush()  # Get the invitation ID

      # 5. Create organization member entry with "invited" status
      org_member = OrganizationMemberModel(
        organization_id=org_id,
        user_id=invited_user.id,
        role_id=role.id,
        invited_by=token_payload["id"],  # Track who invited this user
        invite_id=invitation.id,  # Link to the invitation record
        status="invited",  # Mark as invited until acceptance
      )
      session.add(org_member)

      await session.commit()
      await session.refresh(invitation)
      await session.refresh(invited_user)
      await session.refresh(org_member)
      self.logger.info(f"Invitation record created: {invitation}")
      self.logger.info(f"Organization member created with invited status: {org_member.id}")

      # 6. Send Stytch invitation with user ID and invitation flag
      user = await stytch_base.invite_user_for_organization(
        email=email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        external_user_id=str(invited_user.id),  # Send the newly created user ID
      )
      self.logger.info(f"Stytch user with organization context: {user}")
      self.logger.info("Stytch invitation sent with organization context")

      self.logger.debug(f"Creating invitation record for org_id={org_id}, role_id={role.id}")
      self.logger.debug("Invitation record created successfully")

      self.logger.info(f"User invitation process completed successfully for {email}")

      return InviteResponse(
        id=str(invitation.id),
        email=email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        full_name=f"{user_data.first_name} {user_data.last_name}",
        organizations=OrganizationInfo(
          id=org_id,
          name=org.name,
          slug=org.slug,
          role_name=role.name,
          role_id=role.id,
        ),
      )

    except Exception as e:
      self.logger.error(f"Error inviting user: {str(e)}", email=email, exc_info=True)
      from traceback import print_exc

      print_exc()
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

  async def delete(
    self,
    user_id: UUID,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    current_user: dict = Depends(RBAC("users", "delete")),
  ) -> dict:
    """Delete a user from organization (soft delete via organization member status)."""
    try:
      self.logger.info(f"Attempting to remove user {user_id} from organization {org_id}")

      # Get the user to verify they exist
      user_to_delete = await session.get(UserModel, user_id)
      if not user_to_delete:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

      # Find the organization member record
      member_query = select(OrganizationMemberModel).where(
        and_(OrganizationMemberModel.user_id == user_id, OrganizationMemberModel.organization_id == org_id)
      )
      member_result = await session.execute(member_query)
      org_member = member_result.scalar_one_or_none()

      if not org_member:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User is not a member of this organization")

      # Check if user is already deleted from this organization
      if org_member.status == "deleted":
        self.logger.warning(f"User {user_id} is already deleted from organization {org_id}")
        return {"message": "User is already deleted from this organization", "user_id": str(user_id), "organization_id": str(org_id)}

      # Soft delete: mark organization member as deleted and track who deleted them
      org_member.status = "deleted"
      org_member.deleted_by = current_user["id"]

      await session.commit()
      await session.refresh(org_member)

      self.logger.info(f"User {user_id} removed from organization {org_id} successfully")
      return {
        "message": "User removed from organization successfully",
        "user_id": str(user_id),
        "organization_id": str(org_id),
        "deleted_by": str(current_user["id"]),
        "deleted_at": datetime.now(timezone.utc).isoformat(),
      }

    except Exception as e:
      self.logger.error(f"Error removing user {user_id} from organization {org_id}: {str(e)}", exc_info=True)
      await session.rollback()
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to remove user: {str(e)}")

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

  async def _get_active_users(self, org_id: UUID, session: AsyncSession) -> list[UserDetailResponse]:
    """Get all active users in the organization."""
    try:
      # Get user IDs for active members
      user_query = (
        select(UserModel.id)
        .join(OrganizationMemberModel, UserModel.id == OrganizationMemberModel.user_id)
        .where(OrganizationMemberModel.organization_id == org_id, OrganizationMemberModel.status == "active")
        .order_by(UserModel.created_at.desc())
      )

      result = await session.execute(user_query)
      user_ids = list(result.scalars().all())

      # Get detailed user information for each user
      user_details = []
      for user_id in user_ids:
        try:
          user_detail = await self._get_user_details(user_id, session)
          # Set status and invitation-related fields for active users
          user_detail.status = "active"
          user_detail.invite_id = None
          user_detail.invited_at = None
          user_details.append(user_detail)
        except Exception as e:
          self.logger.warning(f"Failed to get details for user {user_id}: {str(e)}")
          continue

      return user_details
    except Exception as e:
      self.logger.error(f"Error getting active users: {str(e)}")
      return []

  async def _get_pending_invitations(self, org_id: UUID, session: AsyncSession) -> list[UserDetailResponse]:
    """Get all pending invitations for the organization."""
    try:
      # Query pending invitations with role information
      invitation_query = (
        select(InvitationModel, RoleModel, OrganizationModel)
        .join(RoleModel, InvitationModel.role_id == RoleModel.id)
        .join(OrganizationModel, InvitationModel.organization_id == OrganizationModel.id)
        .where(InvitationModel.organization_id == org_id, InvitationModel.status.in_([InvitationStatus.PENDING]))
        .order_by(InvitationModel.created_at.desc())
      )

      result = await session.execute(invitation_query)
      invitations = result.unique().all()

      invitation_details = []
      for invitation, role, org in invitations:
        try:
          # Determine status based on expiry
          status = "pending"
          if invitation.expiry_time < datetime.now(timezone.utc):
            status = "expired"

          # Since we don't have first_name/last_name in invitation, use placeholder or extract from email
          email_parts = invitation.invitee_email.split("@")[0].split(".")
          first_name = email_parts[0].title() if email_parts else "Invited"
          last_name = email_parts[1].title() if len(email_parts) > 1 else "User"

          # Create UserDetailResponse for invitation
          invitation_detail = UserDetailResponse(
            id=None,  # No user ID for pending invitations
            email=invitation.invitee_email,
            first_name=first_name,
            last_name=last_name,
            full_name=f"{first_name} {last_name}",
            status=status,
            invite_id=str(invitation.id),
            invited_at=invitation.created_at.isoformat() if invitation.created_at else None,
            organizations=[
              OrganizationInfo(
                id=org.id,
                name=org.name,
                slug=org.slug,
                role_name=role.name,
                role_id=role.id,
              )
            ],
          )
          invitation_details.append(invitation_detail)
        except Exception as e:
          self.logger.warning(f"Failed to process invitation {invitation.id}: {str(e)}")
          continue

      return invitation_details
    except Exception as e:
      self.logger.error(f"Error getting pending invitations: {str(e)}")
      return []
