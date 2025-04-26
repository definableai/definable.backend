import asyncio
from typing import Dict

from stytch import Client
from stytch.consumer.models.magic_links_email import InviteResponse
from stytch.consumer.models.sessions import AuthenticateResponse
from stytch.consumer.models.users import CreateResponse, DeleteResponse, GetResponse, Name, UpdateResponse

from common.logger import logger
from config.settings import settings
from libs.response import LibResponse


class StytchBase:
  """Stytch base class."""

  def __init__(self):
    """Initialize stytch client."""
    self.client = Client(
      project_id=settings.stytch_project_id,
      secret=settings.stytch_secret,
      environment=settings.stytch_environment,
    )
    self.sessions: Dict[str, AuthenticateResponse] = {}

  async def create_user(
    self, email: str, first_name: str | None = None, last_name: str | None = None
  ) -> LibResponse[CreateResponse] | LibResponse[None]:
    """Create user."""
    try:
      response = await self.client.users.create_async(
        email, Name(first_name=first_name, last_name=last_name), untrusted_metadata={"message": "created from postman"}
      )
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def get_user(self, user_id: str) -> LibResponse[GetResponse] | LibResponse[None]:
    """Get user."""
    try:
      response = await self.client.users.get_async(user_id)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def get_sessions(self, user_id: str) -> LibResponse[GetResponse] | LibResponse[None]:
    """Get sessions."""
    try:
      response = await self.client.sessions.get_async(user_id)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def invite_user(
    self, email: str, first_name: str | None = None, last_name: str | None = None
  ) -> LibResponse[InviteResponse] | LibResponse[None]:
    """Invite user."""
    try:
      name = Name(first_name=first_name, last_name=last_name)
      response = await self.client.magic_links.email.invite_async(
        email,
        name=name,
      )
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def update_user(self, user_id: str, external_id: str) -> LibResponse[UpdateResponse] | LibResponse[None]:
    """Update user."""
    try:
      response = await self.client.users.update_async(user_id, trusted_metadata={"external_user_id": external_id})
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def authenticate_user(self, session_token: str) -> LibResponse[AuthenticateResponse] | LibResponse[None]:
    """Authenticate user."""
    # maintain object scoped sessions registry
    # WARNING: this is not thread safe
    try:
      # if session_token in self.sessions:  # check time as well
      #   return LibResponse.success_response(self.sessions[session_token])
      # else:
      response = await self.client.sessions.authenticate_async(session_token=session_token)
      if response.status_code == 200:
        self.sessions[session_token] = response
        return LibResponse.success_response(response)
      else:
        return LibResponse.error_response([{"message": "Invalid session token"}])
    except Exception as e:
      if session_token in self.sessions:
        del self.sessions[session_token]
      return LibResponse.error_response([{"message": str(e)}])

  async def create_invite(self, user_id: str) -> LibResponse[CreateResponse] | LibResponse[None]:
    """Create invite."""
    try:
      response = await self.client.magic_links.create_async(user_id)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def authenticate_magic_token(self, token: str) -> LibResponse[AuthenticateResponse] | LibResponse[None]:
    """Authenticate magic token."""
    try:
      response = await self.client.magic_links.authenticate_async(token)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def delete_user(self, user_id: str) -> LibResponse[DeleteResponse] | LibResponse[None]:
    """Delete user."""
    try:
      response = await self.client.users.delete_async(user_id)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def create_user_with_password(
    self, first_name: str, last_name: str, email: str, password: str
  ) -> LibResponse[CreateResponse] | LibResponse[None]:
    """Create user with password."""
    try:
      response = await self.client.passwords.create_async(
        email,
        password,
        name=Name(first_name=first_name, last_name=last_name),
        session_duration_minutes=1440,
        untrusted_metadata={"message": "created from postman"},
      )
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])

  async def authenticate_user_with_password(self, email: str, password: str) -> LibResponse[AuthenticateResponse] | LibResponse[None]:
    """Authenticate user with password."""
    try:
      response = await self.client.passwords.authenticate_async(email, password, session_duration_minutes=1440)
      return LibResponse.success_response(response)
    except Exception as e:
      return LibResponse.error_response([{"message": str(e)}])


if __name__ == "__main__":

  async def main():
    stytch_base = StytchBase()
    # create user with password
    # create_user_with_password_response = await stytch_base.create_user_with_password(
    #   "Anandesh", "Sharma", "anandesh.sharma@zyeta.io", "Anandesh123@@563"
    # )
    # logger.info(create_user_with_password_response.model_dump_json())

    # login user with password
    login_user_with_password_response = await stytch_base.authenticate_user_with_password("Audrey.Zemlak77@hotmail.com", "Rock23987423@")
    logger.info(login_user_with_password_response.model_dump_json())
    # print(await stytch_base.get_sessions("user-test-32157863-a7a7-4c32-812c-154bb2360ae5"))
    # print(await stytch_base.invite_user("anandeshsharma5@zyeta.io", "Anandesh", "Sharma"))
    # print(await stytch_base.authenticate_user("RbYYgfNxHDqQA9YJaRWRVoVfF2iY08bemVdYtaIOhvOa"))
    # print(await stytch_base.update_user("user-test-9865995b-2dea-419a-b990-ec1fbb3e542c", "123"))

  asyncio.run(main())
