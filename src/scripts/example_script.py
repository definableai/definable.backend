#!/usr/bin/env python3
"""
Example script demonstrating how to use the BaseScript architecture.
This template shows the minimal implementation needed for a new script.
"""

import os
import sys
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from scripts.base_script import BaseScript


class ExampleScript(BaseScript):
  """
  Example script that demonstrates the BaseScript usage.
  Replace this with your actual script logic.
  """

  def __init__(self):
    # Initialize with unique script name
    super().__init__("example_script")

  async def execute(self, db: AsyncSession) -> None:
    """
    Main script execution logic.
    Implement your script's core functionality here.
    """
    logger.info("Starting example script execution...")

    # Example: Add your script logic here
    # This could be database operations, external API calls, file processing, etc.

    # Example database operation:
    # result = await db.execute(text("SELECT COUNT(*) FROM some_table"))
    # count = result.scalar()
    # logger.info(f"Found {count} records")

    logger.info("Example script execution completed.")

  async def rollback(self, db: AsyncSession) -> None:
    """
    Rollback logic for the script.
    Implement this to undo changes made by the execute method.
    """
    logger.info("Starting example script rollback...")

    # Example: Add your rollback logic here
    # This could be deleting records, reverting changes, etc.

    logger.info("Example script rollback completed.")

  async def verify(self, db: AsyncSession) -> bool:
    """
    Verify script execution was successful.
    Return True if everything is as expected, False otherwise.
    """
    logger.info("Verifying example script execution...")

    # Example: Add your verification logic here
    # This could be checking record counts, validating data, etc.

    # Example verification:
    # result = await db.execute(text("SELECT COUNT(*) FROM some_table WHERE condition = true"))
    # count = result.scalar()
    # return count > 0

    return True  # Always passes for this example


def main():
  """Entry point for backward compatibility."""
  script = ExampleScript()
  script.main()


if __name__ == "__main__":
  # Create and run the script
  script = ExampleScript()
  script.run_cli()
