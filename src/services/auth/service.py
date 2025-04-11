from datetime import timedelta
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
  InvitationModel,
  InvitationStatus,
  OrganizationMemberModel,
  OrganizationModel,
  RoleModel,
  UserModel,
)
from services.__base.acquire import Acquire

from .schema import InviteSignup, TokenResponse, UserLogin, UserResponse, UserSignup
from uuid import UUID
from dependencies.security import InviteTokenBearer


class AuthService:
  """Authentication service."""

  http_exposed = ["post=signup", "post=login", "get=me", "post=signup_invite"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger

  async def post_signup(self, user_data: UserSignup, session: AsyncSession = Depends(get_db)) -> UserResponse:
    """Sign up a new user."""
    try:
      self.logger.info(f"Starting user signup process for email: {user_data.email}")

      # Check if user exists
      query = select(UserModel).where(UserModel.email == user_data.email)
      result = await session.execute(query)
      if result.unique().scalar_one_or_none():
        self.logger.error(f"Email already registered: {user_data.email}")
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Email already registered",
        )

      # Create user
      self.logger.debug("Creating new user", email=user_data.email)
      user_dict = user_data.model_dump(exclude={"confirm_password"})
      user_dict["password"] = self.utils.get_password_hash(user_dict["password"])
      db_user = UserModel(**user_dict)
      session.add(db_user)
      await session.flush()

      # Set up default organization with owner role
      await self._setup_default_organization(db_user, session)
      await session.commit()
      await session.refresh(db_user)

      self.logger.info("User signup completed successfully", user_id=str(db_user.id))
      return UserResponse(id=db_user.id, email=db_user.email, message="User created successfully")

    except HTTPException:
      # Re-raise known exceptions
      raise
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Signup failed: {str(e)}", exc_info=True)
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to create user account",
      )

  async def post_signup_invite(
    self,
    user_data: InviteSignup,
    token_payload: dict = Depends(InviteTokenBearer()),
    session: AsyncSession = Depends(get_db),
  ) -> UserResponse:
    try:
      # Get invitation
      query = select(InvitationModel).where(
        and_(
          InvitationModel.id == UUID(token_payload["invitation_id"]),
          InvitationModel.invitee_email == token_payload["email"],
        )
      )
      result = await session.execute(query)
      invitation = result.unique().scalar_one_or_none()

      if not invitation:
        self.logger.error(f"Invalid invitation token: {token_payload}")
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Invalid invitation",
        )

      self.logger.info(f"Starting invite signup process for email: {invitation.invitee_email}")

      # Check if invitation is pending
      if invitation.status != int(InvitationStatus.PENDING):
        self.logger.error(f"Invitation is not pending: {invitation.id}")
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail=f"Invitation is not pending (current status: {InvitationStatus(invitation.status).name})",
        )

      # Check if user exists
      user_query = select(UserModel).where(UserModel.email == invitation.invitee_email)
      user_result = await session.execute(user_query)
      if user_result.unique().scalar_one_or_none():
        self.logger.error(f"Email already registered: {invitation.invitee_email}")
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Email already registered",
        )

      # Create user
      self.logger.debug("Creating new user", email=invitation.invitee_email)
      user_dict = user_data.model_dump()
      user_dict["email"] = invitation.invitee_email
      user_dict["password"] = self.utils.get_password_hash(user_dict["password"])
      db_user = UserModel(**user_dict)
      session.add(db_user)
      await session.flush()

      # Add user to organization with invited role
      try:
        self.logger.debug("Adding user to organization", user_id=str(db_user.id))
        member = OrganizationMemberModel(
          organization_id=invitation.organization_id,
          user_id=db_user.id,
          role_id=invitation.role_id,
          status="active",  # Directly active as they're invited
        )
        session.add(member)

        # Update invitation status
        invitation.status = int(InvitationStatus.ACCEPTED)
        session.add(invitation)

        await session.commit()
        await session.refresh(db_user)

        self.logger.info(
          "Invite signup completed successfully",
          user_id=str(db_user.id),
          org_id=str(invitation.organization_id),
        )
        return UserResponse(
          id=db_user.id,
          email=db_user.email,
          message="User created and added to organization successfully",
        )

      except Exception as e:
        await session.rollback()
        self.logger.error(f"Failed to add user to organization: {str(e)}")
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"Failed to add user to organization: {str(e)}",
        )

    except Exception as e:
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(e),
      )

  async def post_login(self, form_data: UserLogin, session: AsyncSession = Depends(get_db)) -> TokenResponse:
    """Login user."""
    self.logger.info("Login attempt", email=form_data.email)
    # Get user
    query = select(UserModel).where(UserModel.email == form_data.email, UserModel.is_active)
    result = await session.execute(query)
    user = result.scalar_one_or_none()

    if not user or not self.utils.verify_password(form_data.password, user.password):
      self.logger.warning("Failed login attempt", email=form_data.email)
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
      )

    # Create access token
    self.logger.debug("Creating access token", user_id=str(user.id))
    access_token = self.utils.create_access_token(
      data={"id": str(user.id)},
      expires_delta=timedelta(minutes=self.settings.jwt_expire_minutes),
    )

    self.logger.info("Login successful", user_id=str(user.id))
    return TokenResponse(access_token=access_token)

  async def get_me(
    self,
    current_user: dict,
  ) -> UserResponse:
    """Get current user."""
    return UserResponse.model_validate(current_user)

  ### PRIVATE METHODS ###

  async def _setup_default_organization(self, user: UserModel, session: AsyncSession) -> OrganizationModel:
    """
    Create default organization and set up user as owner.

    Args:
        user: The user model instance
        session: The database session

    Returns:
        The created organization
    """
    # Create organization name and slug
    self.logger.debug("Setting up default organization", user_id=str(user.id))
    org_name = f"default_{str(uuid4())[:8]}"
    self.logger.debug("Creating organization", name=org_name)

    # Create organization
    org = OrganizationModel(
      name="Default Org",
      slug=org_name,
      settings={},  # Default empty settings
      is_active=True,
    )
    session.add(org)
    await session.flush()
    # assign a default role to the user
    # Get OWNER role from default roles
    query = select(RoleModel).where(RoleModel.name == "owner")
    result = await session.execute(query)
    owner_role = result.unique().scalar_one_or_none()

    if not owner_role:
      self.logger.error("Default owner role not found")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Default owner role not found",
      )

    # Create organization member
    self.logger.debug("Assigning owner role", user_id=str(user.id), org_id=str(org.id))

    # Create organization member entry
    member = OrganizationMemberModel(
      organization_id=org.id,
      user_id=user.id,
      role_id=owner_role.id,  # Self-invited as creator
      status="active",  # Directly active as creator
    )
    session.add(member)
    await session.flush()

    self.logger.info(
      "Default organization setup complete",
      user_id=str(user.id),
      org_id=str(org.id),
    )

    return org