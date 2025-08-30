#!/usr/bin/env python3
"""
Script to synchronize USD billing plans with Stripe plans.
First attempts to match existing Stripe plans by name, amount, and currency.
If no match is found, creates new plans in Stripe.
Uses the script_run_tracker table to track execution.
Enhanced with click commands for rerun and rollback functionality.
"""

import asyncio
import os
import platform
import sys
import time
from typing import List, Optional, Tuple

import click
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from database.postgres import async_session
from libs.payments.stripe.v1.engine import engine as stripe_engine


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


async def fetch_billing_plans(db: AsyncSession) -> List[Tuple[str, str, float, int, str]]:
  """Fetch all USD billing plans from the database."""
  result = await db.execute(
    text("""
            SELECT id, name, amount, credits, currency
            FROM billing_plans
            WHERE currency = 'USD' AND is_active = true
            ORDER BY amount ASC
        """)
  )
  plans = result.fetchall()
  logger.info(f"Found {len(plans)} USD billing plans in database")
  return [tuple(row) for row in plans]


async def check_stripe_plan_exists(db: AsyncSession, billing_plan_id: str) -> Optional[str]:
  """Check if a Stripe plan ID exists for a given billing plan."""
  result = await db.execute(
    text("SELECT plan_id FROM billing_plans WHERE id = :billing_plan_id"),
    {"billing_plan_id": billing_plan_id},
  )
  row = result.fetchone()
  return row[0] if row and row[0] else None


async def find_matching_stripe_plan(plan_name: str, amount: float, currency: str) -> Optional[str]:
  """Find existing Stripe plan that matches database plan details."""
  try:
    logger.info(f"Searching for existing Stripe plan matching: {plan_name}, {amount} {currency}")

    # Fetch all plans from Stripe
    response = stripe_engine.fetch_all_plans(count=100)

    if not response.is_successful():
      logger.error(f"Failed to fetch Stripe plans: {response.errors}")
      return None

    stripe_plans = response.data
    amount_in_cents = int(amount * 100)  # Convert to cents for comparison

    # Search for matching plan by name, amount, and currency
    for plan in stripe_plans:
      stripe_name = plan.get("name", "")
      stripe_amount = plan.get("amount", 0)
      stripe_currency = plan.get("currency", "")

      # Match by name, amount, and currency
      if stripe_name == plan_name and stripe_amount == amount_in_cents and stripe_currency.upper() == currency.upper():
        plan_id = plan.get("id")
        logger.info(f"Found matching Stripe plan: {plan_id} for '{plan_name}'")
        return plan_id

    logger.info(f"No matching Stripe plan found for '{plan_name}' with amount {amount} {currency}")
    return None

  except Exception as e:
    logger.error(f"Error searching for matching Stripe plan: {str(e)}")
    return None


async def update_billing_plan_with_stripe_id(db: AsyncSession, billing_plan_id: str, stripe_plan_id: str):
  """Update billing plan with Stripe plan ID."""
  await db.execute(
    text("""
            UPDATE billing_plans
            SET plan_id = :stripe_plan_id,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :billing_plan_id
        """),
    {"billing_plan_id": billing_plan_id, "stripe_plan_id": stripe_plan_id},
  )
  await db.commit()
  logger.info(f"Updated billing plan {billing_plan_id} with Stripe plan ID: {stripe_plan_id}")


async def create_stripe_plans(db: AsyncSession):
  """Create or match Stripe plans for all USD billing plans."""
  try:
    billing_plans = await fetch_billing_plans(db)

    if not billing_plans:
      logger.info("No USD billing plans found. Nothing to process.")
      return

    created_plans = 0
    matched_plans = 0
    skipped_plans = 0

    logger.info(f"Processing {len(billing_plans)} USD billing plans...")

    for plan_id, plan_name, amount, plan_credits, currency in billing_plans:
      # Skip if plan is not USD currency (extra safety check)
      if currency != "USD":
        logger.warning(f"Skipping non-USD plan: {plan_name} ({currency})")
        skipped_plans += 1
        continue

      # Check if database already has plan_id
      existing_stripe_plan_id = await check_stripe_plan_exists(db, plan_id)

      if existing_stripe_plan_id:
        logger.info(f"Database already has Stripe plan ID for '{plan_name}': {existing_stripe_plan_id}")
        skipped_plans += 1
        continue

      # First, try to find matching plan in Stripe
      logger.info(f"Checking for existing Stripe plan matching '{plan_name}': USD {amount}")
      matching_plan_id = await find_matching_stripe_plan(plan_name, float(amount), currency)

      if matching_plan_id:
        # Found matching plan, update database with the plan_id
        logger.info(f"Found matching Stripe plan: {matching_plan_id} for '{plan_name}'")
        await update_billing_plan_with_stripe_id(db, plan_id, matching_plan_id)
        matched_plans += 1
      else:
        # No matching plan found, create new one
        amount_in_cents = int(amount * 100)
        description = f"{plan_name} plan - {plan_credits} credits for USD {amount}"

        logger.info(f"Creating new Stripe plan for '{plan_name}': USD {amount} ({amount_in_cents} cents)")

        # Create plan using Stripe engine
        response = stripe_engine.create_plan(
          name=plan_name,
          amount=amount_in_cents,
          currency=currency.lower(),
          period="month",  # Default to monthly billing
          interval=1,
          description=description,
        )

        if response.is_successful() and response.data:
          stripe_plan_id = response.data.get("id")
          logger.info(f"Successfully created Stripe plan: {stripe_plan_id}")

          # Update billing plan with Stripe plan ID
          await update_billing_plan_with_stripe_id(db, plan_id, stripe_plan_id)
          created_plans += 1

        else:
          error_details = response.errors[0] if response.errors else {"message": "Unknown error"}
          logger.error(f"Failed to create Stripe plan for '{plan_name}': {error_details.get('message')}")
          raise Exception(f"Stripe plan creation failed: {error_details.get('message')}")

    logger.info(f"Plan processing summary: {created_plans} created, {matched_plans} matched, {skipped_plans} skipped")

  except Exception as e:
    logger.error(f"Error processing Stripe plans: {str(e)}")
    raise


async def verify_plans_created(db: AsyncSession) -> bool:
  """Verify that all USD billing plans have associated Stripe plan IDs."""
  result = await db.execute(
    text("""
            SELECT COUNT(*) as total_plans,
                   COUNT(plan_id) as plans_with_stripe_id
            FROM billing_plans
            WHERE currency = 'USD' AND is_active = true
        """)
  )
  row = result.fetchone()

  if row:
    total_plans, plans_with_stripe_id = row
    logger.info(f"Verification: {plans_with_stripe_id}/{total_plans} USD billing plans have Stripe plan IDs")

    if total_plans == plans_with_stripe_id:
      logger.info("All USD billing plans have been synchronized with Stripe")
      return True
    else:
      missing_count = total_plans - plans_with_stripe_id
      logger.warning(f"{missing_count} billing plans are missing Stripe plan IDs")
      return False

  return False


async def get_created_stripe_plan_ids(db: AsyncSession) -> List[str]:
  """Get all Stripe plan IDs that were created by this script."""
  result = await db.execute(
    text("""
            SELECT plan_id
            FROM billing_plans
            WHERE currency = 'USD' AND plan_id IS NOT NULL AND is_active = true
        """)
  )
  plan_ids = [row[0] for row in result.fetchall() if row[0]]
  logger.info(f"Found {len(plan_ids)} Stripe plan IDs in database")
  return plan_ids


async def deactivate_stripe_plans(plan_ids: List[str]) -> int:
  """Deactivate Stripe plans using the Stripe engine."""
  deactivated_count = 0

  for plan_id in plan_ids:
    try:
      logger.info(f"Deactivating Stripe plan: {plan_id}")
      response = stripe_engine.deactivate_plan(plan_id)

      if response.is_successful():
        logger.info(f"Successfully deactivated Stripe plan: {plan_id}")
        deactivated_count += 1
      else:
        error_details = response.errors[0] if response.errors else {"message": "Unknown error"}
        logger.error(f"Failed to deactivate Stripe plan {plan_id}: {error_details.get('message')}")

    except Exception as e:
      logger.error(f"Error deactivating Stripe plan {plan_id}: {str(e)}")

  return deactivated_count


async def run_script(force_rerun: bool = False):
  """Main function to execute the script."""
  script_name = "create_stripe_plans"
  logger.info(f"Starting {script_name} script...")

  async with async_session() as db:
    try:
      # Check if script has already been executed successfully
      already_executed = await check_script_executed(db, script_name)

      if not force_rerun and already_executed:
        logger.info(f"Script '{script_name}' has already been executed successfully. Use --force to rerun.")
        return

      # Log script execution as pending
      await log_script_execution(db, script_name, "pending")

      # Create Stripe plans for all USD billing plans
      await create_stripe_plans(db)

      # Verify plans were created successfully
      verification_passed = await verify_plans_created(db)

      if verification_passed:
        # Log successful script execution
        await log_script_execution(db, script_name, "success")
        logger.info(f"Script '{script_name}' completed successfully.")
      else:
        raise Exception("Plan verification failed - not all billing plans have Stripe plan IDs")

    except Exception as e:
      error_message = str(e)
      logger.error(f"Error executing script: {error_message}")

      # Log failed execution
      await log_script_execution(db, script_name, "failed", error_message)
      raise


async def rollback_script():
  """Rollback the script execution by deactivating Stripe plans and clearing plan IDs."""
  script_name = "create_stripe_plans"
  logger.info(f"Starting rollback for {script_name} script...")

  async with async_session() as db:
    try:
      # Check if script was executed successfully
      if not await check_script_executed(db, script_name):
        logger.warning(f"Script '{script_name}' was not executed successfully. Nothing to rollback.")
        return

      # Get all Stripe plan IDs that need to be deactivated
      stripe_plan_ids = await get_created_stripe_plan_ids(db)

      if stripe_plan_ids:
        logger.info(f"Deactivating {len(stripe_plan_ids)} Stripe plans...")
        deactivated_count = await deactivate_stripe_plans(stripe_plan_ids)
        logger.info(f"Successfully deactivated {deactivated_count}/{len(stripe_plan_ids)} Stripe plans")

      # Clear Stripe plan IDs from billing plans
      result = await db.execute(
        text("""
                    UPDATE billing_plans
                    SET plan_id = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE currency = 'USD' AND plan_id IS NOT NULL
                """)
      )

      cleared_count = result.rowcount
      await db.commit()
      logger.info(f"Cleared Stripe plan IDs from {cleared_count} billing plans")

      # Log rollback execution
      await log_script_execution(db, script_name, "rolled_back")
      logger.info(f"Script '{script_name}' rolled back successfully.")

    except Exception as e:
      error_message = str(e)
      logger.error(f"Error during rollback: {error_message}")

      # Log failed rollback
      await log_script_execution(db, script_name, "failed", f"Rollback failed: {error_message}")
      raise


@click.group()
def cli():
  """Create Stripe Plans script with rerun capabilities."""
  pass


@cli.command()
@click.option("--force", is_flag=True, help="Force rerun even if script was already executed successfully")
def run(force):
  """Synchronize USD billing plans with Stripe.

  First tries to match existing Stripe plans by name, amount, and currency.
  If matches are found, links them to database plans.
  If no matches are found, creates new plans in Stripe.
  """
  try:
    asyncio.run(run_script(force_rerun=force))

    # Give connections time to close properly
    if platform.system() == "Windows":
      time.sleep(1)

  except KeyboardInterrupt:
    logger.info("Script interrupted by user")
  except Exception as e:
    logger.error(f"Script failed: {e}")
    sys.exit(1)


@cli.command()
def rollback():
  """Rollback the create Stripe plans script to previous state.

  This will deactivate all created Stripe plans and clear plan IDs from the database.
  Note: Stripe plans cannot be deleted, only deactivated.
  """
  try:
    asyncio.run(rollback_script())

    # Give connections time to close properly
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
  script_name = "create_stripe_plans"

  async def check_status():
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

        # Show verification status
        verify_result = await db.execute(
          text("""
                        SELECT COUNT(*) as total_plans,
                               COUNT(plan_id) as plans_with_stripe_id
                        FROM billing_plans
                        WHERE currency = 'USD' AND is_active = true
                    """)
        )
        verify_row = verify_result.fetchone()
        if verify_row:
          total, with_stripe = verify_row
          click.echo(f"Current Status: {with_stripe}/{total} USD plans have Stripe plan IDs")

      except Exception as e:
        click.echo(f"Error checking status: {e}")

  try:
    asyncio.run(check_status())
  except Exception as e:
    click.echo(f"Failed to check status: {e}")
    sys.exit(1)


def main():
  """Entry point for backward compatibility - synchronizes USD billing plans with Stripe."""
  asyncio.run(run_script())


if __name__ == "__main__":
  # Check if any arguments were provided
  if len(sys.argv) == 1:
    # No arguments, run the script directly for backward compatibility
    try:
      main()

      # Give connections time to close properly
      if platform.system() == "Windows":
        time.sleep(1)

    except KeyboardInterrupt:
      logger.info("Script interrupted by user")
    except Exception as e:
      logger.error(f"Script failed: {e}")
  else:
    # Arguments provided, use click CLI
    cli()
