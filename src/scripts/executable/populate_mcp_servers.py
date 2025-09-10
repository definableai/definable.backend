#!/usr/bin/env python3
"""
Populates MCP servers from Composio into the database.
"""

import json
import os
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

  def __init__(self):
    super().__init__("populate_mcp_servers")

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting MCP servers population...")

    # Get MCP servers from Composio API
    async with httpx.AsyncClient() as client:
      response = await client.get(
        "https://backend.composio.dev/api/v3/mcp/servers",
        headers={"x-api-key": settings.composio_api_key},
      )
      response.raise_for_status()
      mcp_data = response.json()

    logger.info(f"Found {len(mcp_data.get('items', []))} MCP servers to process")

    servers_created = 0
    toolkits_created = 0
    tools_created = 0

    for mcp_item in mcp_data.get("items", []):
      try:
        logger.info(f"Processing MCP server: {mcp_item.get('name')}")

        # Extract server data
        name = mcp_item.get("name")
        toolkits = mcp_item.get("toolkits", [])
        auth_config_ids = mcp_item.get("auth_config_ids", [])
        allowed_tools = mcp_item.get("allowed_tools", [])
        server_instance_count = mcp_item.get("server_instance_count", 0)

        auth_scheme = None
        expected_input_fields = []

        if auth_config_ids:
          auth_config_id = auth_config_ids[0]
          try:
            async with httpx.AsyncClient() as auth_client:
              auth_response = await auth_client.get(
                f"https://backend.composio.dev/api/v3/auth_configs/{auth_config_id}",
                headers={"x-api-key": settings.composio_api_key},
              )

              if auth_response.status_code == 200:
                auth_data = auth_response.json()
                auth_scheme = auth_data.get("auth_scheme")
                expected_input_fields = auth_data.get("expected_input_fields", [])
              else:
                logger.warning(f"Failed to fetch auth config {auth_config_id}: {auth_response.status_code}")
          except Exception as auth_error:
            logger.warning(f"Error fetching auth config {auth_config_id}: {auth_error}")

        # Check if server already exists using raw SQL
        existing_server = await db.execute(text("SELECT id FROM mcp_servers WHERE name = :name"), {"name": name})
        if existing_server.first():
          logger.info(f"Server {name} already exists, skipping...")
          continue

        # Create MCP server using raw SQL
        await db.execute(
          text("""
            INSERT INTO mcp_servers (name, toolkits, auth_config_ids, auth_scheme, expected_input_fields, allowed_tools, server_instance_count)
            VALUES (:name, :toolkits, :auth_config_ids, :auth_scheme, :expected_input_fields, :allowed_tools, :server_instance_count)
          """),
          {
            "name": name,
            "toolkits": json.dumps(toolkits),
            "auth_config_ids": json.dumps(auth_config_ids),
            "auth_scheme": auth_scheme,
            "expected_input_fields": json.dumps(expected_input_fields),
            "allowed_tools": json.dumps(allowed_tools),
            "server_instance_count": server_instance_count,
          },
        )
        servers_created += 1

        # Process toolkits for this server
        for toolkit_slug in toolkits:
          try:
            # Get toolkit details from Composio
            async with httpx.AsyncClient() as toolkit_client:
              toolkit_response = await toolkit_client.get(
                f"https://backend.composio.dev/api/v3/toolkits/{toolkit_slug}",
                headers={"x-api-key": settings.composio_api_key},
              )

              if toolkit_response.status_code == 200:
                toolkit_data = toolkit_response.json()

                # Check if toolkit already exists using raw SQL
                existing_toolkit = await db.execute(text("SELECT id FROM mcp_toolkits WHERE slug = :slug"), {"slug": toolkit_slug})
                existing_toolkit_row = existing_toolkit.first()

                if not existing_toolkit_row:
                  # Create toolkit using raw SQL
                  toolkit_result = await db.execute(
                    text("""
                      INSERT INTO mcp_toolkits (name, slug, logo)
                      VALUES (:name, :slug, :logo)
                      RETURNING id
                    """),
                    {
                      "name": toolkit_data.get("name", toolkit_slug),
                      "slug": toolkit_slug,
                      "logo": toolkit_data.get("meta", {}).get("logo"),
                    },
                  )
                  toolkit_id = toolkit_result.scalar_one()
                  toolkits_created += 1
                else:
                  toolkit_id = existing_toolkit_row[0]

                try:
                  cursor = None
                  total_toolkit_tools = 0
                  page_num = 1

                  while True:
                    async with httpx.AsyncClient() as tools_client:
                      url = f"https://backend.composio.dev/api/v3/tools?toolkit_slug={toolkit_slug}"
                      if cursor:
                        url += f"&cursor={cursor}"

                      tools_response = await tools_client.get(
                        url,
                        headers={"x-api-key": settings.composio_api_key},
                      )

                      if tools_response.status_code == 200:
                        tools_data = tools_response.json()
                        current_page_tools = 0

                        for tool_data in tools_data.get("items", []):
                          try:
                            # Check if tool already exists using raw SQL
                            existing_tool = await db.execute(
                              text("SELECT id FROM mcp_tools WHERE slug = :slug AND toolkit_id = :toolkit_id"),
                              {"slug": tool_data.get("slug"), "toolkit_id": toolkit_id},
                            )

                            if not existing_tool.first():
                              # Create tool using raw SQL
                              await db.execute(
                                text("""
                                  INSERT INTO mcp_tools (toolkit_id, name, slug, description)
                                  VALUES (:toolkit_id, :name, :slug, :description)
                                """),
                                {
                                  "toolkit_id": toolkit_id,
                                  "name": tool_data.get("name"),
                                  "slug": tool_data.get("slug"),
                                  "description": tool_data.get("description", ""),
                                },
                              )
                              tools_created += 1
                              current_page_tools += 1

                          except Exception:
                            continue

                        total_toolkit_tools += current_page_tools

                        # Check for next cursor
                        next_cursor = tools_data.get("next_cursor")
                        if not next_cursor:
                          break
                        cursor = next_cursor
                        page_num += 1

                      else:
                        break

                except Exception:
                  pass

              else:
                logger.warning(f"Failed to fetch toolkit {toolkit_slug}: {toolkit_response.status_code}")

          except Exception:
            continue

      except Exception as server_error:
        logger.error(f"Failed to process server {mcp_item.get('name', 'unknown')}: {server_error}")
        continue

    # Commit all changes
    await db.commit()

    logger.info(f"Successfully processed {servers_created} servers, {toolkits_created} toolkits, and {tools_created} tools")

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback the script execution by deleting created MCP data."""
    logger.info("Rolling back MCP servers population...")

    # Delete in reverse order due to foreign key constraints
    # Delete tools first
    from sqlalchemy import text

    tools_result = await db.execute(text("DELETE FROM mcp_tools"))
    tools_deleted = getattr(tools_result, "rowcount", 0)

    # Delete toolkits
    toolkits_result = await db.execute(text("DELETE FROM mcp_toolkits"))
    toolkits_deleted = getattr(toolkits_result, "rowcount", 0)

    # Delete servers
    servers_result = await db.execute(text("DELETE FROM mcp_servers"))
    servers_deleted = getattr(servers_result, "rowcount", 0)

    await db.commit()

    logger.info(f"Deleted {tools_deleted} tools, {toolkits_deleted} toolkits, and {servers_deleted} servers during rollback")

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
