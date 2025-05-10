import json
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
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
from services.__base.acquire import Acquire
from utils import verify_svix_signature

from .schema import InviteSignup, StytchUser, TestLogin, TestResponse, TestSignup


class AuthService:
  """Authentication service."""

  http_exposed = ["post=signup_invite", "post=test_signup", "post=test_login"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger

  async def post(self, request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Post request."""
    signature = request.headers["svix-signature"]
    svix_id = request.headers["svix-id"]
    svix_timestamp = request.headers["svix-timestamp"]
    body = await request.body()

    status = verify_svix_signature(svix_id, svix_timestamp, body.decode("utf-8"), signature)
    if not status:
      raise HTTPException(status_code=400, detail="Invalid signature")

    data = json.loads(body.decode("utf-8"))
    self.logger.debug(f"Received event: {data}")

    if data["action"] == "CREATE":
      user = data["user"]

      if user["untrusted_metadata"].get("temp"):
        return JSONResponse(content={"message": "User created from temp"})

      if len(user["emails"]) == 0:
        return JSONResponse(content={"message": "No email found"})

      db_user = await self._create_new_user(
        StytchUser(
          email=user["emails"][0]["email"],
          stytch_id=user["user_id"],
          first_name=user["name"]["first_name"],
          last_name=user["name"]["last_name"],
          metadata=data,
        ),
        db,
      )
      if db_user:
        await stytch_base.update_user(
          user["user_id"],
          str(db_user.id),
        )
      return JSONResponse(content={"message": "User created successfully"})
    else:
      return JSONResponse(content={"message": "Invalid action"})

  async def post_signup_invite(
    self,
    user_data: InviteSignup,
    token_payload: dict = Depends(RBAC("kb", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> JSONResponse:
    """Post signup invite."""
    print(token_payload)
    return JSONResponse(content={"status": "success"})

  async def post_test_signup(self, test_signup: TestSignup, db: AsyncSession = Depends(get_db)) -> TestResponse:
    """Post test signup."""
    create_user_response = await stytch_base.create_user_with_password(
      test_signup.first_name, test_signup.last_name, test_signup.email, test_signup.password
    )
    if create_user_response.success is False:
      raise HTTPException(status_code=500, detail=create_user_response.model_dump_json())

    db_user = await self._create_new_user(
      StytchUser(
        email=create_user_response.data.user.emails[0].email,
        stytch_id=create_user_response.data.user_id,
        first_name=create_user_response.data.user.name.first_name,
        last_name=create_user_response.data.user.name.last_name,
        metadata={},
      ),
      db,
    )
    if db_user:
      print(db_user.id, db_user.email, db_user.stytch_id)
      t = await stytch_base.update_user(
        create_user_response.data.user_id,
        str(db_user.id),
      )
      print(t.data.model_dump_json())
      return TestResponse(
        user_id=db_user.id,
        email=create_user_response.data.user.emails[0].email,
        stytch_token=create_user_response.data.session_token,
        stytch_user_id=create_user_response.data.user_id,
      )
    else:
      raise HTTPException(status_code=500, detail="Failed to create user in database")

  async def post_test_login(self, test_login: TestLogin, db: AsyncSession = Depends(get_db)) -> TestResponse:
    """Post test login."""
    authenticate_user_response = await stytch_base.authenticate_user_with_password(test_login.email, test_login.password)
    if authenticate_user_response.success is False:
      raise HTTPException(status_code=500, detail=authenticate_user_response.model_dump_json())

    # query usermodel with stytch_id
    user = await db.execute(select(UserModel).where(UserModel.stytch_id == authenticate_user_response.data.user.user_id))
    db_user = user.unique().scalar_one_or_none()
    if not db_user:
      raise HTTPException(status_code=500, detail="User not found in database")

    return TestResponse(
      user_id=db_user.id,
      email=db_user.email,
      stytch_token=authenticate_user_response.data.session_token,
      stytch_user_id=authenticate_user_response.data.user.user_id,
    )

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
