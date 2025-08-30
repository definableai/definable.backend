#!/usr/bin/env python3
"""
Script to delete all test users from Stytch.
This script should only be run in the test environment.
"""

import os
import sys
from typing import List, Optional

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import stytch
from sqlalchemy.ext.asyncio import AsyncSession
from stytch.consumer.models.users import User

from scripts.base_script import BaseScript
from common.logger import log as logger
from config.settings import settings


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


class DeleteStytchUsersScript(BaseScript):
  """Script to delete all test users from Stytch."""

  def __init__(self):
    super().__init__("delete_all_test_stytch_users")

  async def execute(self, db: AsyncSession) -> None:
    """Delete all test users from Stytch."""
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


def main():
  """Entry point for backward compatibility."""
  script = DeleteStytchUsersScript()
  script.main()


if __name__ == "__main__":
  script = DeleteStytchUsersScript()
  script.run_cli()
