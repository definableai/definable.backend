#!/usr/bin/env python3
"""
Base script interface for standardizing script development.
Provides a simple, generic architecture for script execution, tracking, and management.
"""

import asyncio
import os
import platform
import sys
import time
from abc import ABC, abstractmethod
from typing import Optional

import click
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from database.postgres import async_session


class BaseScript(ABC):
  """
  Base class for all scripts providing common functionality.

  Features provided:
  - Script execution tracking
  - Rollback capabilities
  - Status checking
  - Error handling and logging
  - CLI interface with Click
  - Force rerun support
  """

  def __init__(self, script_name: str):
    """Initialize the script with a unique name."""
    self.script_name = script_name

  @abstractmethod
  async def execute(self, db: AsyncSession) -> None:
    """
    Main script execution logic.
    Implement your script's core functionality here.

    Args:
        db: Database session for performing operations
    """
    pass

  async def rollback(self, db: AsyncSession) -> None:
    """
    Rollback logic for the script.
    Override this method to provide rollback functionality.

    Args:
        db: Database session for performing operations
    """
    logger.warning(f"No rollback implementation provided for script '{self.script_name}'")

  async def verify(self, db: AsyncSession) -> bool:
    """
    Verify script execution was successful.
    Override this method to provide custom verification logic.

    Args:
        db: Database session for performing operations

    Returns:
        True if verification passes, False otherwise
    """
    return True

  async def check_script_executed(self, db: AsyncSession) -> bool:
    """Check if this script has already been executed successfully."""
    result = await db.execute(
      text("SELECT status FROM script_run_tracker WHERE script_name = :script_name ORDER BY updated_at DESC LIMIT 1"),
      {"script_name": self.script_name},
    )
    row = result.fetchone()
    return row[0] == "success" if row else False

  async def log_script_execution(self, db: AsyncSession, status: str, error_message: Optional[str] = None):
    """Log the script execution with status."""
    try:
      # Check if entry exists
      result = await db.execute(text("SELECT 1 FROM script_run_tracker WHERE script_name = :script_name"), {"script_name": self.script_name})
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
          {"script_name": self.script_name, "status": status, "error_message": error_message},
        )
      else:
        # Insert new record
        await db.execute(
          text("""
                        INSERT INTO script_run_tracker (script_name, status, error_message)
                        VALUES (:script_name, :status, :error_message)
                    """),
          {"script_name": self.script_name, "status": status, "error_message": error_message},
        )
      await db.commit()
      logger.info(f"Logged execution of script '{self.script_name}' with status: {status}")
    except Exception as e:
      await db.rollback()
      logger.error(f"Failed to log script execution: {str(e)}")
      raise

  async def run_script(self, force_rerun: bool = False):
    """Main function to execute the script with tracking and error handling."""
    logger.info(f"Starting {self.script_name} script...")

    async with async_session() as db:
      try:
        # Check if script has already been executed successfully
        already_executed = await self.check_script_executed(db)

        if not force_rerun and already_executed:
          logger.info(f"Script '{self.script_name}' has already been executed successfully. Use --force to rerun.")
          return

        # Log script execution as pending
        await self.log_script_execution(db, "pending")

        # Execute the main script logic
        await self.execute(db)

        # Verify execution was successful
        verification_passed = await self.verify(db)

        if verification_passed:
          # Log successful script execution
          await self.log_script_execution(db, "success")
          logger.info(f"Script '{self.script_name}' completed successfully.")
        else:
          raise Exception("Script verification failed")

      except Exception as e:
        error_message = str(e)
        logger.error(f"Error executing script: {error_message}")

        # Log failed execution
        await self.log_script_execution(db, "failed", error_message)
        raise

  async def rollback_script(self):
    """Rollback the script execution."""
    logger.info(f"Starting rollback for {self.script_name} script...")

    async with async_session() as db:
      try:
        # Check if script was executed successfully
        if not await self.check_script_executed(db):
          logger.warning(f"Script '{self.script_name}' was not executed successfully. Nothing to rollback.")
          return

        # Execute rollback logic
        await self.rollback(db)

        # Log rollback execution
        await self.log_script_execution(db, "rolled_back")
        logger.info(f"Script '{self.script_name}' rolled back successfully.")

      except Exception as e:
        error_message = str(e)
        logger.error(f"Error during rollback: {error_message}")

        # Log failed rollback
        await self.log_script_execution(db, "failed", f"Rollback failed: {error_message}")
        raise

  async def check_status(self):
    """Check the execution status of the script."""
    async with async_session() as db:
      try:
        result = await db.execute(
          text("""
                        SELECT status, error_message, updated_at
                        FROM script_run_tracker
                        WHERE script_name = :script_name
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """),
          {"script_name": self.script_name},
        )
        row = result.fetchone()

        if not row:
          click.echo(f"Script '{self.script_name}' has never been executed.")
          return

        status, error_msg, updated_at = row
        click.echo(f"Script: {self.script_name}")
        click.echo(f"Status: {status}")
        click.echo(f"Last Updated: {updated_at}")

        if error_msg and status == "failed":
          click.echo(f"Error: {error_msg}")

      except Exception as e:
        click.echo(f"Error checking status: {e}")

  def create_cli(self):
    """Create Click CLI interface for the script."""

    @click.group()
    def cli():
      f"""{self.script_name} script with management capabilities."""
      pass

    @cli.command()
    @click.option("--force", is_flag=True, help="Force rerun even if script was already executed successfully")
    def run(force):
      """Run the script."""
      try:
        asyncio.run(self.run_script(force_rerun=force))

        # Give connections time to close properly on Windows
        if platform.system() == "Windows":
          time.sleep(1)

      except KeyboardInterrupt:
        logger.info("Script interrupted by user")
      except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

    @cli.command()
    def rollback():
      """Rollback the script to previous state."""
      try:
        asyncio.run(self.rollback_script())

        # Give connections time to close properly on Windows
        if platform.system() == "Windows":
          time.sleep(1)

      except KeyboardInterrupt:
        logger.info("Rollback interrupted by user")
      except Exception as e:
        logger.error(f"Rollback failed: {e}")
        sys.exit(1)

    @cli.command()
    def status():
      """Check the execution status of the script."""
      try:
        asyncio.run(self.check_status())
      except Exception as e:
        click.echo(f"Failed to check status: {e}")
        sys.exit(1)

    return cli

  def main(self):
    """Entry point for backward compatibility - runs the script directly."""
    asyncio.run(self.run_script())

  def run_cli(self):
    """
    Run the CLI interface.
    Call this in your script's __main__ block.
    """
    if len(sys.argv) == 1:
      # No arguments, run the script directly for backward compatibility
      try:
        self.main()

        # Give connections time to close properly on Windows
        if platform.system() == "Windows":
          time.sleep(1)

      except KeyboardInterrupt:
        logger.info("Script interrupted by user")
      except Exception as e:
        logger.error(f"Script failed: {e}")
    else:
      # Arguments provided, use click CLI
      cli = self.create_cli()
      cli()
