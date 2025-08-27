#!/usr/bin/env python3
"""
Ensures LLM charge rows exist and are up-to-date in the llm_charges table.
Tracks execution in script_run_tracker with rerun, rollback, and status commands.
"""

import asyncio
import os
import platform
import sys
import time
from typing import List, Dict, Optional
from uuid import uuid4

import click
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from database.postgres import async_session
from models import ChargeModel


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
    result = await db.execute(text("SELECT 1 FROM script_run_tracker WHERE script_name = :script_name"), {"script_name": script_name})
    exists = result.scalar()

    if exists:
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
    logger.error(f"Failed to log script execution: {e}")
    raise


async def check_charges_table_exists(db: AsyncSession):
  """Verify that the charges table exists as defined by ChargeModel."""
  try:
    table_name = ChargeModel.__tablename__
    result = await db.execute(
      text("SELECT 1 FROM information_schema.tables WHERE table_name = :table_name"),
      {"table_name": table_name},
    )
    if not result.scalar():
      raise Exception(f"{table_name} table does not exist")
    logger.info(f"{table_name} table exists")
  except Exception as e:
    logger.error(f"Error checking charges table: {e}")
    raise


def get_model_charge() -> List[Dict[str, str | int]]:
  """Returns charge rows derived from migration + new additions."""
  return [
    # ------------------------
    # OpenAI LLM Models
    # ------------------------
    {
      "name": "gpt-4.1",
      "amount": 6,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "GPT-4.1 model usage charge",
    },
    {
      "name": "gpt-4o",
      "amount": 3,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "GPT-4o model usage charge",
    },
    {
      "name": "gpt-4o-mini",
      "amount": 2,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "GPT-4o Mini model usage charge",
    },
    {
      "name": "gpt-3.5-turbo",
      "amount": 1,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "GPT-3.5 Turbo model usage charge",
    },
    # ------------------------
    # Anthropic LLM Models
    # ------------------------
    {
      "name": "claude-3.7-sonnet",
      "amount": 6,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "Claude 3.7 Sonnet model usage charge",
    },
    {
      "name": "claude-3.5-sonnet",
      "amount": 4,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "Claude 3.5 Sonnet model usage charge",
    },
    {
      "name": "claude-3-haiku",
      "amount": 1,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "Claude 3 Haiku model usage charge",
    },
    # ------------------------
    # DeepSeek LLM Models
    # ------------------------
    {
      "name": "deepseek-chat",
      "amount": 2,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "DeepSeek Chat model usage charge",
    },
    {
      "name": "deepseek-reasoner",
      "amount": 4,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "DeepSeek Reason model usage charge",
    },
    # ------------------------
    # Other OpenAI O-Series Models
    # ------------------------
    {
      "name": "o4-mini",
      "amount": 2,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "O4 Mini model usage charge",
    },
    {
      "name": "o1-preview",
      "amount": 7,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "O1 Preview model usage charge",
    },
    {
      "name": "o1",
      "amount": 4,
      "service": "llm",
      "action": "generate",
      "unit": "credit",
      "measure": "token",
      "description": "O1 model usage charge",
    },
    # ------------------------
    # Knowledge Base Services
    # ------------------------
    {
      "name": "o1-small-text-indexing",
      "amount": 3,
      "service": "kb",
      "action": "index",
      "unit": "credit",
      "measure": "token",
      "description": "Text indexing with OpenAI Ada embedding model",
    },
    {
      "name": "o1-small-text-retrieval",
      "amount": 3,
      "service": "kb",
      "action": "retrieval",
      "unit": "credit",
      "measure": "token",
      "description": "Text retrieval with OpenAI Ada embedding model",
    },
    {
      "name": "pdf-extraction",
      "amount": 5,
      "service": "kb",
      "action": "extract",
      "unit": "credit",
      "measure": "page",
      "description": "PDF text extraction per page",
    },
    {
      "name": "excel-ext",
      "amount": 4,
      "service": "kb",
      "action": "extraction",
      "unit": "credit",
      "measure": "sheet",
      "description": "Extract data from Excel spreadsheets",
    },
  ]


async def update_llm_charges(db: AsyncSession, charges_data: List[Dict[str, str | int]]):
  """Update or insert charges data using the schema from ChargeModel."""
  table_name = ChargeModel.__tablename__
  try:
    for charge in charges_data:
      name = charge["name"]

      result = await db.execute(
        text(f"SELECT 1 FROM {table_name} WHERE name = :name"),
        {"name": name},
      )
      exists = result.scalar()

      if exists:
        await db.execute(
          text(f"""
            UPDATE {table_name}
            SET amount = :amount, unit = :unit, measure = :measure,
                service = :service, action = :action, description = :description,
                updated_at = CURRENT_TIMESTAMP
            WHERE name = :name
          """),
          charge,
        )
        logger.info(f"Updated charge for {name}")
      else:
        # Generate and bind a new id for insert to avoid NULL ids when DB default is missing
        charge_with_id = {**charge, "id": uuid4()}
        await db.execute(
          text(f"""
            INSERT INTO {table_name}
            (id, name, amount, unit, measure, service, action, description, created_at, updated_at)
            VALUES
            (:id, :name, :amount, :unit, :measure, :service, :action, :description, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
          """),
          charge_with_id,
        )
        logger.info(f"Inserted new charge for {name}")

    await db.commit()
  except Exception as e:
    await db.rollback()
    logger.error(f"Error updating charges: {e}")
    raise


async def run_script(force_rerun: bool = False):
  """Main function to execute the script."""
  script_name = "ensure_charges_props"
  logger.info(f"Starting {script_name} script...")

  async with async_session() as db:
    try:
      already_executed = await check_script_executed(db, script_name)
      if not force_rerun and already_executed:
        logger.info(f"Script '{script_name}' already executed successfully. Use --force to rerun.")
        return

      await log_script_execution(db, script_name, "pending")
      await check_charges_table_exists(db)

      charges_data = get_model_charge()
      await update_llm_charges(db, charges_data)

      await log_script_execution(db, script_name, "success")
      logger.info(f"Script '{script_name}' completed successfully.")
    except Exception as e:
      await log_script_execution(db, script_name, "failed", str(e))
      raise


async def rollback_script():
  """Rollback inserted charges by deleting them."""
  script_name = "ensure_charges_props"
  logger.info(f"Starting rollback for {script_name}...")

  async with async_session() as db:
    try:
      if not await check_script_executed(db, script_name):
        logger.warning(f"Script '{script_name}' not executed successfully. Nothing to rollback.")
        return

      charges = get_model_charge()
      table_name = ChargeModel.__tablename__

      deleted_count = 0
      for charge in charges:
        result = await db.execute(
          text(f"DELETE FROM {table_name} WHERE name = :name"),
          {"name": charge["name"]},
        )
        if result.rowcount > 0:
          deleted_count += result.rowcount
          logger.info(f"Deleted charge {charge['name']}")

      await db.commit()
      logger.info(f"Deleted {deleted_count} charges during rollback")
      await log_script_execution(db, script_name, "rolled_back")
    except Exception as e:
      await log_script_execution(db, script_name, "failed", f"Rollback failed: {e}")
      raise


@click.group()
def cli():
  """Ensure Charges script with rerun and rollback support."""
  pass


@cli.command()
@click.option("--force", is_flag=True, help="Force rerun even if already executed successfully")
def run(force):
  try:
    asyncio.run(run_script(force_rerun=force))
    if platform.system() == "Windows":
      time.sleep(1)
  except Exception as e:
    logger.error(f"Run failed: {e}")
    sys.exit(1)


@cli.command()
def rollback():
  try:
    asyncio.run(rollback_script())
    if platform.system() == "Windows":
      time.sleep(1)
  except Exception as e:
    logger.error(f"Rollback failed: {e}")
    sys.exit(1)


@cli.command()
def status():
  """Check execution status from script_run_tracker."""
  script_name = "ensure_charges_props"

  async def check_status():
    async with async_session() as db:
      result = await db.execute(
        text("""
          SELECT status, error_message, updated_at
          FROM script_run_tracker
          WHERE script_name = :script_name
          ORDER BY updated_at DESC
          LIMIT 1
        """),
        {"script_name": script_name},
      )
      row = result.fetchone()
      if not row:
        click.echo(f"Script '{script_name}' has never been executed.")
        return

      status, error_msg, updated_at = row
      click.echo(f"Script: {script_name}")
      click.echo(f"Status: {status}")
      click.echo(f"Last Updated: {updated_at}")
      if error_msg and status == "failed":
        click.echo(f"Error: {error_msg}")

  asyncio.run(check_status())


def main():
  asyncio.run(run_script())


if __name__ == "__main__":
  if len(sys.argv) == 1:
    try:
      main()
      if platform.system() == "Windows":
        time.sleep(1)
    except Exception as e:
      logger.error(f"Script failed: {e}")
  else:
    cli()
