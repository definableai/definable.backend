from typing import List

import httpx

from config.settings import settings


class Composio:
  """Client for interacting with Composio API."""

  def __init__(self):
    """Initialize Composio client."""
    self.api_key = settings.composio_api_key
    self.base_url = "https://backend.composio.dev/api/v3"
    self.headers = {
      "x-api-key": self.api_key,
      "Content-Type": "application/json",
    }

  async def create_connected_account(
    self,
    auth_config_id: str,
    user_id: str,
    callback_url: str,
  ) -> dict:
    """
    Create a connected account for a user.

    Args:
        auth_config_id: The auth configuration ID
        user_id: The user ID for the connection
        callback_url: Callback URL for OAuth flow

    Returns:
        dict: Connected account response
    """
    payload = {
      "auth_config": {"id": auth_config_id},
      "connection": {
        "user_id": user_id,
        "callback_url": callback_url,
      },
    }

    async with httpx.AsyncClient() as client:
      response = await client.post(
        f"{self.base_url}/connected_accounts",
        headers=self.headers,
        json=payload,
      )
      response.raise_for_status()
      return response.json()

  async def create_mcp_instance(
    self,
    server_id: str,
    user_id: str,
  ) -> dict:
    """
    Create an MCP server instance for a user.

    Args:
        server_id: The MCP server ID
        user_id: The user ID (composio_user_id)

    Returns:
        dict: Instance creation response
    """
    payload = {"user_id": user_id}

    async with httpx.AsyncClient() as client:
      response = await client.post(
        f"{self.base_url}/mcp/servers/{server_id}/instances",
        headers=self.headers,
        json=payload,
      )
      response.raise_for_status()
      return response.json()

  async def generate_mcp_url(
    self,
    server_id: str,
    user_ids: List[str],
  ) -> dict:
    """
    Generate MCP URL for given server and users.

    Args:
        server_id: The MCP server ID
        user_ids: List of user IDs

    Returns:
        dict: URL generation response
    """
    payload = {
      "mcp_server_id": server_id,
      "user_ids": user_ids,
    }

    async with httpx.AsyncClient() as client:
      response = await client.post(
        f"{self.base_url}/mcp/servers/generate",
        headers=self.headers,
        json=payload,
      )
      response.raise_for_status()
      return response.json()

  async def list_mcp_instances(self, server_id: str) -> dict:
    """
    List MCP instances for a server.

    Args:
        server_id: The MCP server ID

    Returns:
        dict: List of instances
    """
    async with httpx.AsyncClient() as client:
      response = await client.get(
        f"{self.base_url}/mcp/servers/{server_id}/instances",
        headers=self.headers,
      )
      response.raise_for_status()
      return response.json()


# Singleton instance
composio = Composio()
