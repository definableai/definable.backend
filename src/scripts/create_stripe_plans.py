#!/usr/bin/env python3
"""
Script to synchronize USD billing plans with Stripe plans.
First attempts to match existing Stripe plans by name, amount, and currency.
If no match is found, creates new plans in Stripe.
"""

import os
import sys
from typing import List, Optional, Tuple

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scripts.base_script import BaseScript
from common.logger import log as logger
from libs.payments.stripe.v1.engine import engine as stripe_engine


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


class CreateStripePlansScript(BaseScript):
  """Script to synchronize USD billing plans with Stripe plans."""

  def __init__(self):
    super().__init__("create_stripe_plans")

  async def execute(self, db: AsyncSession) -> None:
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

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback the script execution by deactivating Stripe plans and clearing plan IDs."""
    logger.info("Rolling back Stripe plans...")

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

    cleared_count = getattr(result, "rowcount", 0)
    await db.commit()
    logger.info(f"Cleared Stripe plan IDs from {cleared_count} billing plans")

  async def verify(self, db: AsyncSession) -> bool:
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


def main():
  """Entry point for backward compatibility - synchronizes USD billing plans with Stripe."""
  script = CreateStripePlansScript()
  script.main()


if __name__ == "__main__":
  script = CreateStripePlansScript()
  script.run_cli()
