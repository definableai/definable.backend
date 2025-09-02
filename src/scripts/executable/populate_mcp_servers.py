#!/usr/bin/env python3
"""
Populates MCP servers from Composio into the database.
Uses the script_run_tracker table to track execution.
Enhanced with click commands for rerun and rollback functionality.
"""

import asyncio
import os
import sys
import uuid
from typing import Optional

import click
import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from config.settings import settings
from database.postgres import async_session
from models.mcp_model import MCPServerModel, MCPToolkitModel
from models.mcp_tool_model import MCPToolModel

SCRIPT_NAME = "populate_mcp_servers"


async def check_script_executed(db: AsyncSession, script_name: str) -> bool:
  """Checks if this script has already been executed successfully."""
  result = await db.execute(
    text("SELECT status FROM script_run_tracker WHERE script_name = :script_name ORDER BY updated_at DESC LIMIT 1"),
    {"script_name": script_name},
  )
  row = result.fetchone()
  return row[0] == "success" if row else False


async def log_script_execution(db: AsyncSession, script_name: str, status: str, error_message: Optional[str] = None):
  """Logs the script execution with status."""
  try:
    # Check if entry exists
    result = await db.execute(text("SELECT 1 FROM script_run_tracker WHERE script_name = :script_name"), {"script_name": script_name})
    exists = result.scalar()

    if exists:
      # Update existing record
      await db.execute(
        text("""
                    UPDATE script_run_tracker
                    SET status = :status, error_message = :error_message, updated_at = NOW()
                    WHERE script_name = :script_name
                    """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    else:
      # Insert new record
      await db.execute(
        text("""
                    INSERT INTO script_run_tracker (script_name, status, error_message, executed_at, updated_at)
                    VALUES (:script_name, :status, :error_message, NOW(), NOW())
                    """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    await db.commit()
  except Exception as e:
    logger.error(f"Failed to log script execution: {e}")


async def get_auth_config_details(auth_config_id: str) -> dict:
  """Get auth config details from Composio API."""
  try:
    response = httpx.get(
      f"https://backend.composio.dev/api/v3/auth_configs/{auth_config_id}",
      headers={"x-api-key": settings.composio_api_key},
    )
    response.raise_for_status()
    return response.json()
  except Exception as e:
    logger.error(f"Failed to get auth config details for {auth_config_id}: {e}")
    return {}


async def populate_mcp_servers(db: AsyncSession, force: bool = False) -> bool:
  """Populate MCP servers from Composio."""
  try:
    # Check if script has already been executed
    if not force and await check_script_executed(db, SCRIPT_NAME):
      logger.info(f"Script {SCRIPT_NAME} has already been executed successfully. Use --force to rerun.")
      return True

    logger.info("Starting MCP servers population...")

    # Get MCP servers from Composio API
    response = httpx.get(
      "https://backend.composio.dev/api/v3/mcp/servers",
      headers={"x-api-key": settings.composio_api_key},
    )
    response.raise_for_status()
    mcp_data = response.json()

    logger.info(f"Found {len(mcp_data.get('items', []))} MCP servers to process")

    created_servers = []
    created_toolkits = []
    created_tools = []

    for mcp_item in mcp_data.get("items", []):
      try:
        logger.info(f"Processing MCP server: {mcp_item.get('name')}")

        # Get auth config details for the first auth config
        auth_scheme = None
        expected_input_fields = []
        auth_config_ids = mcp_item.get("auth_config_ids", [])
        if auth_config_ids:
          auth_config_details = await get_auth_config_details(auth_config_ids[0])
          auth_scheme = auth_config_details.get("auth_scheme")
          expected_input_fields = auth_config_details.get("expected_input_fields", [])

        # Create MCP server (removed mcp_url)
        server = MCPServerModel(
          id=mcp_item.get("id"),
          name=mcp_item.get("name"),
          toolkits=mcp_item.get("toolkits", []),
          auth_config_ids=auth_config_ids,
          auth_scheme=auth_scheme,
          expected_input_fields=expected_input_fields,
          allowed_tools=mcp_item.get("allowed_tools", []),
          server_instance_count=mcp_item.get("server_instance_count", 0),
        )
        db.add(server)
        created_servers.append(mcp_item.get("id"))

        # Process toolkits
        for toolkit_name in mcp_item.get("toolkits", []):
          # Check if toolkit already exists
          result = await db.execute(select(MCPToolkitModel).where(MCPToolkitModel.slug == toolkit_name))
          existing_toolkit = result.scalar_one_or_none()

          if not existing_toolkit:
            toolkit = MCPToolkitModel(
              id=uuid.uuid4(),
              name=toolkit_name,
              slug=toolkit_name,
              logo=mcp_item.get("toolkit_icons", {}).get(toolkit_name),
            )
            db.add(toolkit)
            created_toolkits.append(toolkit_name)

            # Get toolkit ID for tool creation
            await db.flush()  # Flush to get the ID
            toolkit_id = toolkit.id
          else:
            toolkit_id = existing_toolkit.id

          # Process tools for this toolkit
          toolkit_tools = [tool for tool in mcp_item.get("allowed_tools", []) if toolkit_name in tool.lower()]
          for tool_slug in toolkit_tools:
            # Check if tool already exists
            result = await db.execute(select(MCPToolModel).where((MCPToolModel.slug == tool_slug) & (MCPToolModel.toolkit_id == toolkit_id)))
            existing_tool = result.scalar_one_or_none()

            if not existing_tool:
              tool = MCPToolModel(
                name=tool_slug.replace("_", " ").title(),
                slug=tool_slug,
                description=f"Tool for {toolkit_name}: {tool_slug}",
                toolkit_id=toolkit_id,
              )
              db.add(tool)
              created_tools.append(tool_slug)

        await db.commit()
        logger.info(f"Successfully processed MCP server: {mcp_item.get('name')}")

      except Exception as e:
        logger.error(f"Failed to process MCP server {mcp_item.get('name')}: {e}")
        await db.rollback()
        continue

    # Log successful execution
    await log_script_execution(
      db, SCRIPT_NAME, "success", f"Created {len(created_servers)} servers, {len(created_toolkits)} toolkits, {len(created_tools)} tools"
    )

    logger.info("✅ MCP servers population completed successfully!")
    logger.info(f"   - Servers created: {len(created_servers)}")
    logger.info(f"   - Toolkits created: {len(created_toolkits)}")
    logger.info(f"   - Tools created: {len(created_tools)}")

    return True

  except Exception as e:
    logger.error(f"❌ MCP servers population failed: {e}")
    await log_script_execution(db, SCRIPT_NAME, "failed", str(e))
    await db.rollback()
    return False


async def rollback_mcp_servers(db: AsyncSession) -> bool:
  """Rollback MCP servers population by deleting created entries."""
  try:
    logger.info("Starting MCP servers rollback...")

    # Delete all MCP data in reverse order
    await db.execute(text("DELETE FROM mcp_tools"))
    await db.execute(text("DELETE FROM mcp_toolkits"))
    await db.execute(text("DELETE FROM mcp_servers"))

    await db.commit()

    # Log rollback execution
    await log_script_execution(db, SCRIPT_NAME, "rolled_back", "Deleted all MCP servers, toolkits, and tools")

    logger.info("✅ MCP servers rollback completed successfully!")
    return True

  except Exception as e:
    logger.error(f"❌ MCP servers rollback failed: {e}")
    await db.rollback()
    return False


@click.group()
def cli():
  """MCP Servers Population Script."""
  pass


@cli.command()
@click.option("--force", is_flag=True, help="Force rerun even if script has already been executed")
def run(force):
  """Run the MCP servers population script."""

  async def _run():
    async with async_session() as db:
      success = await populate_mcp_servers(db, force=force)
      sys.exit(0 if success else 1)

  asyncio.run(_run())


@cli.command()
def rollback():
  """Rollback the MCP servers population."""

  async def _rollback():
    async with async_session() as db:
      success = await rollback_mcp_servers(db)
      sys.exit(0 if success else 1)

  asyncio.run(_rollback())


if __name__ == "__main__":
  cli()
