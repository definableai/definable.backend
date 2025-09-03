#!/usr/bin/env python3
"""
Populates MCP servers from Composio into the database.
"""

import os
import sys
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from common.logger import log as logger
from config.settings import settings
from models.mcp_model import MCPServerModel, MCPToolkitModel
from models.mcp_tool_model import MCPToolModel
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
        server_id = mcp_item.get("id")
        name = mcp_item.get("name")
        toolkits = mcp_item.get("toolkits", [])
        auth_config_ids = mcp_item.get("auth_config_ids", [])
        auth_scheme = mcp_item.get("auth_scheme")
        expected_input_fields = mcp_item.get("expected_input_fields", [])
        allowed_tools = mcp_item.get("allowed_tools", [])
        server_instance_count = mcp_item.get("server_instance_count", 0)

        # Check if server already exists
        existing_server = await db.execute(select(MCPServerModel).where(MCPServerModel.id == server_id))
        if existing_server.scalar_one_or_none():
          logger.info(f"Server {name} already exists, skipping...")
          continue

        # Create MCP server
        server = MCPServerModel(
          id=server_id,
          name=name,
          toolkits=toolkits,
          auth_config_ids=auth_config_ids,
          auth_scheme=auth_scheme,
          expected_input_fields=expected_input_fields,
          allowed_tools=allowed_tools,
          server_instance_count=server_instance_count,
        )
        db.add(server)
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

                # Check if toolkit already exists
                existing_toolkit = await db.execute(select(MCPToolkitModel).where(MCPToolkitModel.slug == toolkit_slug))

                if not existing_toolkit.scalar_one_or_none():
                  # Create toolkit
                  toolkit = MCPToolkitModel(
                    id=uuid.uuid4(),
                    name=toolkit_data.get("name", toolkit_slug),
                    slug=toolkit_slug,
                    logo=toolkit_data.get("logo"),
                  )
                  db.add(toolkit)
                  await db.flush()  # Get the ID
                  toolkits_created += 1

                  # Process tools for this toolkit
                  for tool_data in toolkit_data.get("tools", []):
                    try:
                      # Check if tool already exists
                      existing_tool = await db.execute(
                        select(MCPToolModel).where(MCPToolModel.name == tool_data.get("name"), MCPToolModel.toolkit_id == toolkit.id)
                      )

                      if not existing_tool.scalar_one_or_none():
                        # Create tool
                        tool = MCPToolModel(
                          id=uuid.uuid4(),
                          toolkit_id=toolkit.id,
                          name=tool_data.get("name"),
                          slug=tool_data.get("name", "").lower().replace(" ", "_"),
                          description=tool_data.get("description", ""),
                        )
                        db.add(tool)
                        tools_created += 1

                    except Exception as tool_error:
                      logger.warning(f"Failed to process tool {tool_data.get('name', 'unknown')}: {tool_error}")
                      continue

              else:
                logger.warning(f"Failed to fetch toolkit {toolkit_slug}: {toolkit_response.status_code}")

          except Exception as toolkit_error:
            logger.warning(f"Failed to process toolkit {toolkit_slug}: {toolkit_error}")
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
    # Check if any servers were created
    result = await db.execute(select(MCPServerModel))
    servers = result.scalars().all()

    if not servers:
      logger.error("No MCP servers found after execution")
      return False

    logger.info(f"Verification passed: Found {len(servers)} MCP servers")
    return True


def main():
  """Entry point for backward compatibility."""
  script = PopulateMCPServersScript()
  script.main()


if __name__ == "__main__":
  script = PopulateMCPServersScript()
  script.run_cli()
