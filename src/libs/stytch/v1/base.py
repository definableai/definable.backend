import asyncio
from typing import Dict

from stytch import Client
from stytch.consumer.models.magic_links_email import InviteResponse
from stytch.consumer.models.sessions import AuthenticateResponse
from stytch.consumer.models.users import GetResponse, Name, UpdateResponse

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
      response = await self.client.users.update_async(user_id, external_id=external_id)
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


if __name__ == "__main__":

  async def main():
    stytch_base = StytchBase()
    print(await stytch_base.get_sessions("user-test-32157863-a7a7-4c32-812c-154bb2360ae5"))
    # print(await stytch_base.invite_user("anandeshsharma5@zyeta.io", "Anandesh", "Sharma"))
    # print(await stytch_base.authenticate_user("RbYYgfNxHDqQA9YJaRWRVoVfF2iY08bemVdYtaIOhvOa"))
    # print(await stytch_base.update_user("user-test-9865995b-2dea-419a-b990-ec1fbb3e542c", "123"))

  asyncio.run(main())
