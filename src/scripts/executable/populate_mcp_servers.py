#!/usr/bin/env python3
"""
Populates MCP servers from Composio into the database.
"""

import json
import os
import random
import sys

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from common.logger import log as logger
from config.settings import settings
from scripts.core.base_script import BaseScript


class PopulateMCPServersScript(BaseScript):
  """Script to populate MCP servers from Composio."""

  TOOLKITS = {
    "gmail": "oauth2",
    "github": "oauth2"
  }

  def __init__(self):
    super().__init__("populate_mcp_servers")

  async def create_auth_config(self, client: httpx.AsyncClient, toolkit_slug: str, auth_scheme: str) -> str:
    """Create auth config for a toolkit and return the auth_config_id."""
    try:
      if auth_scheme.lower() == "oauth2":
        # For OAuth2, use composio managed auth
        auth_config_payload = {
          "toolkit": {
            "slug": toolkit_slug
          },
          "auth_config": {
            "type": "use_composio_managed_auth"
          }
        }
      else:
        # For custom auth schemes (API_KEY, BEARER_TOKEN, etc.)
        auth_config_payload = {
          "toolkit": {
            "slug": toolkit_slug
          },
          "auth_config": {
            "type": "use_custom_auth",
            "authScheme": auth_scheme.upper()
          }
        }

      auth_response = await client.post(
        "https://backend.composio.dev/api/v3/auth_configs",
        headers={"x-api-key": settings.composio_api_key},
        json=auth_config_payload
      )

      if auth_response.status_code == 201:
        auth_data = auth_response.json()
        auth_config_id = auth_data.get("auth_config", {}).get("id")
        logger.info(f"Created auth config for {toolkit_slug} with {auth_scheme}: {auth_config_id}")
        return auth_config_id
      else:
        logger.warning(f"Failed to create auth config for {toolkit_slug}: {auth_response.status_code}")
        logger.warning(f"Response: {auth_response.text}")
        return ""

    except Exception as e:
      logger.error(f"Error creating auth config for {toolkit_slug}: {e}")
      return ""

  async def fetch_non_deprecated_tools(self, client: httpx.AsyncClient, toolkit_slug: str) -> list:
    """Fetch all non-deprecated tools for a toolkit."""
    tool_slugs = []
    cursor = None

    while True:
      try:
        url = f"https://backend.composio.dev/api/v3/tools?toolkit_slug={toolkit_slug}"
        if cursor:
          url += f"&cursor={cursor}"

        response = await client.get(
          url,
          headers={"x-api-key": settings.composio_api_key}
        )

        if response.status_code != 200:
          logger.error(f"Error fetching tools for {toolkit_slug}: {response.status_code}")
          logger.error(response.text)
          break

        data = response.json()
        items = data.get("items", [])

        # Filter out deprecated tools dynamically
        for item in items:
          # Check if tool is deprecated using the is_deprecated field
          is_deprecated = item.get("deprecated", {}).get("is_deprecated", False)

          if not is_deprecated:
            tool_slugs.append(item["slug"])
          else:
            logger.info(f"Skipping deprecated tool: {item['slug']}")

        cursor = data.get("next_cursor")
        if not cursor:
          break

      except Exception as e:
        logger.error(f"Error fetching tools for {toolkit_slug}: {e}")
        break

    logger.info(f"Found {len(tool_slugs)} non-deprecated tools for {toolkit_slug}")
    return tool_slugs

  async def create_mcp_server(self, client: httpx.AsyncClient, toolkit_slug: str, auth_config_id: str, tool_slugs: list) -> dict:
    """Create MCP server via Composio API."""
    try:
      random_number = random.randint(1, 9999)
      server_name = f"definable-{toolkit_slug}-{random_number}"

      server_payload = {
        "name": server_name,
        "auth_config_ids": [auth_config_id],
        "allowed_tools": tool_slugs
      }

      response = await client.post(
        "https://backend.composio.dev/api/v3/mcp/servers",
        headers={"x-api-key": settings.composio_api_key},
        json=server_payload
      )

      if response.status_code == 201:
        server_data = response.json()
        logger.info(f"Created MCP server for {toolkit_slug}: {server_data.get('id')}")
        return server_data
      else:
        logger.error(f"Failed to create MCP server for {toolkit_slug}: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return {}

    except Exception as e:
      logger.error(f"Error creating MCP server for {toolkit_slug}: {e}")
      return {}

  async def get_toolkit_info(self, client: httpx.AsyncClient, toolkit_slug: str) -> dict:
    """Get toolkit information for database storage."""
    try:
      toolkit_response = await client.get(
        f"https://backend.composio.dev/api/v3/toolkits/{toolkit_slug}",
        headers={"x-api-key": settings.composio_api_key},
      )

      if toolkit_response.status_code == 200:
        return toolkit_response.json()
      else:
        logger.warning(f"Failed to fetch toolkit info for {toolkit_slug}: {toolkit_response.status_code}")
        return {}

    except Exception as e:
      logger.error(f"Error fetching toolkit info for {toolkit_slug}: {e}")
      return {}

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting MCP servers population...")
    logger.info(f"Processing {len(self.TOOLKITS)} toolkits: {', '.join(self.TOOLKITS.keys())}")

    servers_created = 0
    tools_created = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
      for toolkit_slug, auth_scheme in self.TOOLKITS.items():
        try:
          logger.info(f"Processing toolkit: {toolkit_slug} with auth scheme: {auth_scheme}")

          # Check if server already exists
          existing_server = await db.execute(
            text("SELECT id FROM mcp_servers WHERE toolkit_slug = :toolkit_slug"),
            {"toolkit_slug": toolkit_slug}
          )
          if existing_server.first():
            logger.info(f"Server for toolkit {toolkit_slug} already exists, skipping...")
            continue

          # Step 1: Create auth config
          auth_config_id = await self.create_auth_config(client, toolkit_slug, auth_scheme)
          if not auth_config_id:
            logger.warning(f"Skipping {toolkit_slug} - failed to create auth config")
            continue

          # Step 2: Fetch non-deprecated tools
          tool_slugs = await self.fetch_non_deprecated_tools(client, toolkit_slug)
          if not tool_slugs:
            logger.warning(f"Skipping {toolkit_slug} - no tools found")
            continue

          # Step 3: Create MCP server via API
          server_data = await self.create_mcp_server(client, toolkit_slug, auth_config_id, tool_slugs)
          if not server_data:
            logger.warning(f"Skipping {toolkit_slug} - failed to create MCP server")
            continue

          # Step 4: Get toolkit info for database
          toolkit_info = await self.get_toolkit_info(client, toolkit_slug)

          # Step 5: Store server in database
          composio_server_id = server_data.get("id")
          if not composio_server_id:
            logger.warning(f"No server ID returned from Composio for {toolkit_slug}, skipping database storage")
            continue

          await db.execute(
            text("""
              INSERT INTO mcp_servers (
                id, name, auth_config_id, toolkit_name, toolkit_slug, toolkit_logo,
                auth_scheme, expected_input_fields, server_instance_count
              )
              VALUES (
                :id, :name, :auth_config_id, :toolkit_name, :toolkit_slug, :toolkit_logo,
                :auth_scheme, :expected_input_fields, :server_instance_count
              )
            """),
            {
              "id": composio_server_id,
              "name": server_data.get("name"),
              "auth_config_id": auth_config_id,
              "toolkit_name": toolkit_info.get("name", toolkit_slug.title()),
              "toolkit_slug": toolkit_slug,
              "toolkit_logo": toolkit_info.get("meta", {}).get("logo"),
              "auth_scheme": auth_scheme,
              "expected_input_fields": json.dumps(toolkit_info.get("expected_input_fields", [])),
              "server_instance_count": server_data.get("server_instance_count", 0),
            },
          )

          # Use the Composio server ID for tools
          server_id = composio_server_id
          servers_created += 1

          # Step 6: Store tools in database
          for tool_slug in tool_slugs:
            try:
              # Get tool details
              tool_response = await client.get(
                f"https://backend.composio.dev/api/v3/tools/{tool_slug}",
                headers={"x-api-key": settings.composio_api_key},
              )

              if tool_response.status_code == 200:
                tool_data = tool_response.json()

                # Check if tool already exists for this server
                existing_tool = await db.execute(
                  text("SELECT id FROM mcp_tools WHERE slug = :slug AND mcp_server_id = :mcp_server_id"),
                  {"slug": tool_slug, "mcp_server_id": server_id},
                )

                if not existing_tool.first():
                  await db.execute(
                    text("""
                      INSERT INTO mcp_tools (mcp_server_id, name, slug, description)
                      VALUES (:mcp_server_id, :name, :slug, :description)
                    """),
                    {
                      "mcp_server_id": server_id,
                      "name": tool_data.get("name", tool_slug),
                      "slug": tool_slug,
                      "description": tool_data.get("description", ""),
                    },
                  )
                  tools_created += 1

            except Exception as tool_error:
              logger.warning(f"Failed to process tool {tool_slug}: {tool_error}")
              continue

          logger.info(f"Successfully processed toolkit {toolkit_slug}")

        except Exception as toolkit_error:
          logger.error(f"Failed to process toolkit {toolkit_slug}: {toolkit_error}")
          continue

    # Commit all changes
    await db.commit()

    logger.info(f"Successfully processed {servers_created} servers and {tools_created} tools")

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback the script execution by deleting created MCP data."""
    logger.info("Rolling back MCP servers population...")

    # Delete in reverse order due to foreign key constraints
    # Delete tools first
    tools_result = await db.execute(text("DELETE FROM mcp_tools"))
    tools_deleted = getattr(tools_result, "rowcount", 0)

    # Delete servers
    servers_result = await db.execute(text("DELETE FROM mcp_servers"))
    servers_deleted = getattr(servers_result, "rowcount", 0)

    await db.commit()

    logger.info(f"Deleted {tools_deleted} tools and {servers_deleted} servers during rollback")

  async def verify(self, db: AsyncSession) -> bool:
    """Verify script execution was successful."""
    # Check if any servers were created using raw SQL
    result = await db.execute(text("SELECT COUNT(*) FROM mcp_servers"))
    server_count = result.scalar()

    if not server_count:
      logger.error("No MCP servers found after execution")
      return False

    logger.info(f"Verification passed: Found {server_count} MCP servers")
    return True


def main():
  """Entry point for backward compatibility."""
  script = PopulateMCPServersScript()
  script.main()


if __name__ == "__main__":
  script = PopulateMCPServersScript()
  script.run_cli()
