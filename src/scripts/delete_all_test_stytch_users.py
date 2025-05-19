#!/usr/bin/env python3
"""
Script to delete all test users from Stytch.
This script should only be run in the test environment.
"""

import asyncio
import os
import sys
from typing import List, Optional

import stytch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from stytch.consumer.models.users import User

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from config.settings import settings
from database.postgres import async_session


class StytchUserCleaner:
  """Class to clean up Stytch users in test environments."""

  def __init__(self):
    """Initialize the Stytch client."""
    self.client = stytch.Client(
      project_id=settings.stytch_project_id,
      secret=settings.stytch_secret,
      environment=settings.stytch_environment,
    )
    self.is_test_environment = self._check_environment()

  def _check_environment(self) -> bool:
    """
    Check if we're in a test environment.
    Returns True only if we're in 'test' or explicitly defined test environments.
    """
    environment = settings.stytch_environment.lower()
    test_environments = ["test", "sandbox", "development", "dev"]
    is_test = environment in test_environments or "test" in environment

    logger.info(f"Current environment: {environment}")
    if is_test:
      logger.info("Test environment detected. Script will proceed.")
    else:
      logger.warning("PRODUCTION ENVIRONMENT DETECTED! Script will not delete users.")

    return is_test

  async def search_users(self, limit: int = 100) -> Optional[List[User]]:
    """
    Search for users in Stytch.

    Args:
        limit: Maximum number of users to retrieve

    Returns:
        List of Stytch users if successful, None otherwise
    """
    if not self.is_test_environment:
      logger.error("Cannot search users in production environment")
      return None

    try:
      # Search for all users
      response = await self.client.users.search_async(
        limit=limit,
      )

      if response.status_code != 200:
        logger.error(f"Failed to search users: {response}")
        return None

      logger.info(f"Found {len(response.results)} users")
      return response.results
    except Exception as e:
      logger.error(f"Error searching users: {str(e)}")
      return None

  async def delete_user(self, user_id: str) -> bool:
    """
    Delete a user from Stytch.

    Args:
        user_id: The Stytch user ID

    Returns:
        True if successful, False otherwise
    """
    if not self.is_test_environment:
      logger.error(f"Cannot delete user {user_id} in production environment")
      return False

    try:
      response = await self.client.users.delete_async(user_id=user_id)

      if response.status_code != 200:
        logger.error(f"Failed to delete user {user_id}: {response}")
        return False

      logger.info(f"Successfully deleted user {user_id}")
      return True
    except Exception as e:
      logger.error(f"Error deleting user {user_id}: {str(e)}")
      return False

  async def delete_all_users(self) -> None:
    """Delete all users in the test environment."""
    if not self.is_test_environment:
      logger.error("Cannot delete users in production environment")
      return

    users = await self.search_users()
    if not users:
      logger.info("No users found to delete")
      return

    delete_count = 0
    fail_count = 0

    for user in users:
      success = await self.delete_user(user.user_id)
      if success:
        delete_count += 1
      else:
        fail_count += 1

    logger.info(f"Successfully deleted {delete_count} users")
    if fail_count > 0:
      logger.warning(f"Failed to delete {fail_count} users")


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
                    SET status = :status,
                        error_message = :error_message,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE script_name = :script_name
                """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    else:
      # Insert new record
      await db.execute(
        text("""
                    INSERT INTO script_run_tracker (script_name, status, error_message)
                    VALUES (:script_name, :status, :error_message)
                """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    await db.commit()
    logger.info(f"Logged execution of script '{script_name}' with status: {status}")
  except Exception as e:
    await db.rollback()
    logger.error(f"Failed to log script execution: {str(e)}")
    raise


async def main():
  """Main function to run the script."""
  script_name = "delete_all_test_stytch_users"
  logger.info(f"Starting {script_name} script...")

  async with async_session() as db:
    try:
      # Check if script has already been executed successfully
      if await check_script_executed(db, script_name):
        logger.info(f"Script '{script_name}' has already been executed successfully. Skipping.")
        return

      # Log script execution as pending
      await log_script_execution(db, script_name, "pending")

      # Check environment and confirm safety
      if settings.stytch_environment.lower() not in ["test", "sandbox", "development", "dev"]:
        user_input = input(
          f"WARNING: You're running this in the '{settings.stytch_environment}' environment. "
          f"This will DELETE ALL USERS. Are you absolutely sure? (type 'YES' to proceed): "
        )
        if user_input != "YES":
          logger.info("Operation cancelled by user")
          return

      cleaner = StytchUserCleaner()
      await cleaner.delete_all_users()

      # Log successful script execution
      await log_script_execution(db, script_name, "success")
      logger.info(f"Script '{script_name}' completed successfully.")

    except Exception as e:
      error_message = str(e)
      logger.error(f"Error executing script: {error_message}")

      # Log failed execution
      await log_script_execution(db, script_name, "failed", error_message)
      raise


if __name__ == "__main__":
  asyncio.run(main())
