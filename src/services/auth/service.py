import json
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from libs.stytch.v1 import stytch_base
from models import (
  OrganizationMemberModel,
  OrganizationModel,
  RoleModel,
  UserModel,
)
from models.invitations_model import InvitationModel, InvitationStatus
from services.__base.acquire import Acquire
from utils import verify_svix_signature

from .schema import InviteSignup, StytchUser


class AuthService:
  """Authentication service."""

  http_exposed = ["post=signup_invite", "put=invitation_update"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger

  async def post(self, request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Post request."""
    self.logger.info("Received Stytch webhook", headers=list(request.headers.keys()))
    signature = request.headers["svix-signature"]
    svix_id = request.headers["svix-id"]
    svix_timestamp = request.headers["svix-timestamp"]
    body = await request.body()

    self.logger.debug("Verifying webhook signature", svix_id=svix_id, svix_timestamp=svix_timestamp)
    status = verify_svix_signature(svix_id, svix_timestamp, body.decode("utf-8"), signature)
    if not status:
      self.logger.error("Invalid webhook signature", svix_id=svix_id)
      raise HTTPException(status_code=400, detail="Invalid signature")
    data = json.loads(body.decode("utf-8"))
    self.logger.debug("Webhook payload received", action=data.get("action"))

    status = verify_svix_signature(svix_id, svix_timestamp, body.decode("utf-8"), signature)
    if not status:
      raise HTTPException(status_code=400, detail="Invalid signature")
    data = json.loads(body.decode("utf-8"))

    user = data["user"]
    if data["action"] == "CREATE":
      if len(user["emails"]) == 0:
        self.logger.warning("No email found in user data", user_id=user.get("user_id"))
        return JSONResponse(content={"message": "No email found"})

      self.logger.info("Creating new user from webhook", stytch_id=user.get("user_id"), email=user["emails"][0]["email"])
      db_user = await self._create_new_user(
        StytchUser(
          email=user["emails"][0]["email"],
          stytch_id=user["user_id"],
          first_name=user["name"]["first_name"],
          last_name=user["name"]["last_name"],
          is_active=user["emails"][0]["verified"],
          password_id=user["password"]["password_id"],
          metadata=data,
        ),
        db,
      )
      if db_user:
        self.logger.info("Updating Stytch user with external ID", stytch_id=user["user_id"], external_id=str(db_user.id))
        await stytch_base.update_user(
          user["user_id"],
          str(db_user.id),
        )
      self.logger.info("User creation process completed", stytch_id=user["user_id"])
      return JSONResponse(content={"message": "User created successfully"})

    elif data["action"] == "UPDATE":
      self.logger.info("Processing user update webhook", stytch_id=user.get("user_id"))

      # Find user by stytch_id
      user_query = select(UserModel).where(UserModel.stytch_id == user["user_id"])
      result = await db.execute(user_query)
      db_user = result.scalars().first()

      if not db_user:
        self.logger.warning("User not found for update", stytch_id=user.get("user_id"))
        return JSONResponse(content={"message": "User not found"}, status_code=404)

      # Update user information
      if "name" in user:
        if "first_name" in user["name"]:
          self.logger.info("Updating user first name", stytch_id=user["user_id"], old_name=db_user.first_name, new_name=user["name"]["first_name"])
          db_user.first_name = user["name"]["first_name"]

        if "last_name" in user["name"]:
          self.logger.info("Updating user last name", stytch_id=user["user_id"], old_name=db_user.last_name, new_name=user["name"]["last_name"])
          db_user.last_name = user["name"]["last_name"]

      # Update password_id if included in the payload
      if "password" in user and user["password"] and len(user["password"]) > 0:
        password_id = user["password"]["password_id"]
        if password_id:
          self.logger.info("Updating user password_id", stytch_id=user["user_id"], password_id=password_id)
          db_user.password_id = password_id

      # Update metadata if needed
      db_user._metadata = data

      await db.commit()
      self.logger.info("User update completed", stytch_id=user["user_id"])
      return JSONResponse(content={"message": "User updated successfully"})

    else:
      self.logger.warning("Unhandled webhook action", action=data.get("action"))
      return JSONResponse(content={"message": "Unhandled action"})

  async def post_signup_invite(
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

  async def put_invitation_update(
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

  async def _create_new_user(self, user_data: StytchUser, session: AsyncSession) -> UserModel | None:
    """Create a new user."""
    user = await session.execute(select(UserModel).where(UserModel.stytch_id == user_data.stytch_id))
    results = user.unique().scalar_one_or_none()
    if results:
      return None
    else:
      user = UserModel(
        email=user_data.email,
        stytch_id=user_data.stytch_id,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        is_active=user_data.is_active,
        password_id=user_data.password_id,
        _metadata=user_data.metadata,
      )
      session.add(user)
      await session.flush()
      await self._setup_default_organization(user, session)
      await session.commit()
      await session.refresh(user)
      return user

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
