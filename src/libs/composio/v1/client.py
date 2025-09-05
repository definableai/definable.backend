from typing import Any, Dict, List

import httpx

from config.settings import settings
from libs.response import LibResponse


class Composio:
  """Client for interacting with Composio API."""

  def __init__(self):
    """Initialize Composio client."""
    self.api_key = settings.composio_api_key
    self.base_url = settings.composio_base_url
    self.headers = {
      "x-api-key": self.api_key,
      "Content-Type": "application/json",
    }

  async def create_connected_account(
    self,
    auth_config_id: str,
    user_id: str,
    callback_url: str,
  ) -> LibResponse[Dict[str, Any]]:
    """
    Create a connected account for a user.

    Args:
        auth_config_id: The auth configuration ID
        user_id: The user ID for the connection
        callback_url: Callback URL for OAuth flow

    Returns:
        LibResponse[Dict[str, Any]]: Connected account response
    """
    payload = {
      "auth_config": {"id": auth_config_id},
      "connection": {
        "user_id": user_id,
        "callback_url": callback_url,
      },
    }

    try:
      async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10)) as client:
        response = await client.post(
          f"{self.base_url}/connected_accounts",
          headers=self.headers,
          json=payload,
        )
        response.raise_for_status()
        return LibResponse.success_response(data=response.json())
    except httpx.HTTPStatusError as e:
      return LibResponse.error_response(errors=[{"message": str(e)}])

  async def create_mcp_instance(
    self,
    server_id: str,
    user_id: str,
  ) -> LibResponse[Dict[str, Any]]:
    """
    Create an MCP server instance for a user.

    Args:
        server_id: The MCP server ID
        user_id: The user ID (composio_user_id)

    Returns:
        LibResponse[Dict[str, Any]]: Instance creation response
    """
    payload = {"user_id": user_id}

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{self.base_url}/mcp/servers/{server_id}/instances",
          headers=self.headers,
          json=payload,
        )
        response.raise_for_status()
        return LibResponse.success_response(data=response.json())
    except httpx.HTTPStatusError as e:
      return LibResponse.error_response(errors=[{"message": str(e)}])

  async def generate_mcp_url(
    self,
    server_id: str,
    user_ids: List[str],
  ) -> LibResponse[Dict[str, Any]]:
    """
    Generate MCP URL for given server and users.

    Args:
        server_id: The MCP server ID
        user_ids: List of user IDs

    Returns:
        LibResponse[Dict[str, Any]]: URL generation response
    """
    payload = {
      "mcp_server_id": server_id,
      "user_ids": user_ids,
    }

    try:
      async with httpx.AsyncClient() as client:
        response = await client.post(
          f"{self.base_url}/mcp/servers/generate",
          headers=self.headers,
          json=payload,
        )
        response.raise_for_status()
        return LibResponse.success_response(data=response.json())
    except httpx.HTTPStatusError as e:
      return LibResponse.error_response(errors=[{"message": str(e)}])

  async def list_mcp_instances(self, server_id: str) -> LibResponse[Dict[str, Any]]:
    """
    List MCP instances for a server.

    Args:
        server_id: The MCP server ID

    Returns:
        LibResponse[Dict[str, Any]]: List of instances
    """
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          f"{self.base_url}/mcp/servers/{server_id}/instances",
          headers=self.headers,
        )
        response.raise_for_status()
        return LibResponse.success_response(data=response.json())
    except httpx.HTTPStatusError as e:
      return LibResponse.error_response(errors=[{"message": str(e)}])


# Singleton instance
composio = Composio()
