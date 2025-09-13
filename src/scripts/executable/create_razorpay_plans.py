#!/usr/bin/env python3
"""
Script to synchronize INR billing plans with Razorpay plans.
First attempts to match existing Razorpay plans by name, amount, and currency.
If no match is found, creates new plans in Razorpay.
"""

import os
import sys
from typing import List, Optional, Tuple

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import log as logger
from libs.payments.razorpay.v1.engine import engine as razorpay_engine
from scripts.core.base_script import BaseScript


async def fetch_billing_plans(db: AsyncSession) -> List[Tuple[str, str, float, int, str, str]]:
  """Fetch all INR billing plans from the database."""
  result = await db.execute(
    text("""
            SELECT id, name, amount, credits, currency, cycle
            FROM billing_plans
            WHERE currency = 'INR' AND is_active = true
            ORDER BY cycle ASC, amount ASC
        """)
  )
  plans = result.fetchall()
  logger.info(f"Found {len(plans)} INR billing plans in database")

  # Log breakdown of free vs paid plans and cycle distribution
  free_count = sum(1 for plan in plans if plan[2] <= 0)
  paid_count = len(plans) - free_count
  monthly_count = sum(1 for plan in plans if plan[5] == "monthly")
  yearly_count = sum(1 for plan in plans if plan[5] == "yearly")

  logger.info(f"Plan breakdown: {paid_count} paid plans, {free_count} free plans")
  logger.info(f"Cycle breakdown: {monthly_count} monthly plans, {yearly_count} yearly plans")

  return [tuple(row) for row in plans]


async def check_razorpay_plan_exists(db: AsyncSession, billing_plan_id: str) -> Optional[str]:
  """Check if a Razorpay plan ID exists for a given billing plan."""
  result = await db.execute(
    text("SELECT plan_id FROM billing_plans WHERE id = :billing_plan_id"),
    {"billing_plan_id": billing_plan_id},
  )
  row = result.fetchone()
  return row[0] if row and row[0] else None


async def find_matching_razorpay_plan(plan_name: str, amount: float, currency: str) -> Optional[str]:
  """Find existing Razorpay plan that matches database plan details."""
  try:
    logger.info(f"Searching for existing Razorpay plan matching: {plan_name}, {amount} {currency}")

    # Fetch all plans from Razorpay
    response = razorpay_engine.fetch_all_plans(count=100)

    if not response.is_successful():
      logger.error(f"Failed to fetch Razorpay plans: {response.errors}")
      return None

    razorpay_plans = response.data
    amount_in_paise = int(amount * 100)  # Convert to paise for comparison

    # Search for matching plan by name, amount, and currency
    for plan in razorpay_plans:
      plan_item = plan.get("item", {})
      razorpay_name = plan_item.get("name", "")
      razorpay_amount = plan_item.get("amount", 0)
      razorpay_currency = plan_item.get("currency", "")

      # Match by name, amount, and currency
      if razorpay_name == plan_name and razorpay_amount == amount_in_paise and razorpay_currency == currency:
        plan_id = plan.get("id")
        logger.info(f"Found matching Razorpay plan: {plan_id} for '{plan_name}'")
        return plan_id

    logger.info(f"No matching Razorpay plan found for '{plan_name}' with amount {amount} {currency}")
    return None

  except Exception as e:
    logger.error(f"Error searching for matching Razorpay plan: {str(e)}")
    return None


async def update_billing_plan_with_razorpay_id(db: AsyncSession, billing_plan_id: str, razorpay_plan_id: str):
  """Update billing plan with Razorpay plan ID."""
  await db.execute(
    text("""
            UPDATE billing_plans
            SET plan_id = :razorpay_plan_id,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :billing_plan_id
        """),
    {"billing_plan_id": billing_plan_id, "razorpay_plan_id": razorpay_plan_id},
  )
  await db.commit()
  logger.info(f"Updated billing plan {billing_plan_id} with Razorpay plan ID: {razorpay_plan_id}")


class CreateRazorpayPlansScript(BaseScript):
  """Script to synchronize INR billing plans with Razorpay plans."""

  def __init__(self):
    super().__init__("create_razorpay_plans")

  async def execute(self, db: AsyncSession) -> None:
    """Create or match Razorpay plans for all INR billing plans."""
    try:
      billing_plans = await fetch_billing_plans(db)

      if not billing_plans:
        logger.info("No INR billing plans found. Nothing to process.")
        return

      created_plans = 0
      matched_plans = 0
      skipped_plans = 0

      logger.info(f"Processing {len(billing_plans)} INR billing plans...")

      for plan_id, plan_name, amount, plan_credits, currency, cycle in billing_plans:
        # Skip if plan is not INR currency (extra safety check)
        if currency != "INR":
          logger.warning(f"Skipping non-INR plan: {plan_name} ({currency})")
          skipped_plans += 1
          continue

        # Skip free plans (amount = 0) as Razorpay doesn't support them
        if amount <= 0:
          logger.info(f"Skipping free plan '{plan_name}' {cycle} (amount: ₹{amount}) - Razorpay requires minimum ₹1")
          skipped_plans += 1
          continue

        # Check if database already has plan_id
        existing_razorpay_plan_id = await check_razorpay_plan_exists(db, plan_id)

        if existing_razorpay_plan_id:
          logger.info(f"Database already has Razorpay plan ID for '{plan_name}': {existing_razorpay_plan_id}")
          skipped_plans += 1
          continue

        # First, try to find matching plan in Razorpay
        logger.info(f"Checking for existing Razorpay plan matching '{plan_name}' {cycle}: INR {amount}")
        # Create a unique name combining plan name and cycle for Razorpay
        razorpay_plan_name = f"{plan_name}_{cycle}" if cycle == "yearly" else plan_name
        matching_plan_id = await find_matching_razorpay_plan(razorpay_plan_name, float(amount), currency)

        if matching_plan_id:
          # Found matching plan, update database with the plan_id
          logger.info(f"Found matching Razorpay plan: {matching_plan_id} for '{plan_name}'")
          await update_billing_plan_with_razorpay_id(db, plan_id, matching_plan_id)
          matched_plans += 1
        else:
          # No matching plan found, create new one
          amount_in_paise = int(amount * 100)
          cycle_description = "monthly" if cycle == "monthly" else "annual"
          description = f"{plan_name} {cycle_description} plan - {plan_credits} credits for INR {amount}"

          logger.info(f"Creating new Razorpay plan for '{plan_name}' {cycle}: INR {amount} ({amount_in_paise} paise)")

          # Determine billing period based on cycle
          period = "monthly" if cycle == "monthly" else "yearly"
          # Create plan using Razorpay engine
          response = razorpay_engine.create_plan(
            name=razorpay_plan_name,
            amount=amount_in_paise,
            currency=currency,
            period=period,
            interval=1,
            description=description,
          )

          if response.is_successful() and response.data:
            razorpay_plan_id = response.data.get("id")
            logger.info(f"Successfully created Razorpay plan: {razorpay_plan_id}")

            # Update billing plan with Razorpay plan ID
            await update_billing_plan_with_razorpay_id(db, plan_id, razorpay_plan_id)
            created_plans += 1

          else:
            error_details = response.errors[0] if response.errors else {"message": "Unknown error"}
            logger.error(f"Failed to create Razorpay plan for '{plan_name}': {error_details.get('message')}")
            raise Exception(f"Razorpay plan creation failed: {error_details.get('message')}")

      logger.info("Plan processing summary:")
      logger.info(f"  - Created: {created_plans} new Razorpay plans")
      logger.info(f"  - Matched: {matched_plans} existing Razorpay plans")
      logger.info(f"  - Skipped: {skipped_plans} plans (free plans or already processed)")

      if created_plans + matched_plans == 0:
        logger.warning("No Razorpay plans were created or matched. This may be expected if all plans are free.")

    except Exception as e:
      logger.error(f"Error processing Razorpay plans: {str(e)}")
      raise

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback the script execution by clearing Razorpay plan IDs."""
    logger.info("Rolling back Razorpay plans...")

    # Clear Razorpay plan IDs from billing plans
    result = await db.execute(
      text("""
                UPDATE billing_plans
                SET plan_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE currency = 'INR' AND plan_id IS NOT NULL
            """)
    )

    cleared_count = getattr(result, "rowcount", 0)
    await db.commit()
    logger.info(f"Cleared Razorpay plan IDs from {cleared_count} billing plans")

  async def verify(self, db: AsyncSession) -> bool:
    """Verify that all paid INR billing plans have associated Razorpay plan IDs."""
    result = await db.execute(
      text("""
            SELECT COUNT(*) as total_plans,
                   COUNT(plan_id) as plans_with_razorpay_id,
                   COUNT(CASE WHEN amount > 0 THEN 1 END) as paid_plans,
                   COUNT(CASE WHEN amount > 0 AND plan_id IS NOT NULL THEN 1 END) as paid_plans_with_razorpay_id
            FROM billing_plans
            WHERE currency = 'INR' AND is_active = true
        """)
    )
    row = result.fetchone()

    if row:
      total_plans, plans_with_razorpay_id, paid_plans, paid_plans_with_razorpay_id = row
      free_plans = total_plans - paid_plans

      logger.info("Verification results:")
      logger.info(f"  - Total INR plans: {total_plans}")
      logger.info(f"  - Paid plans: {paid_plans}")
      logger.info(f"  - Free plans: {free_plans}")
      logger.info(f"  - Paid plans with Razorpay IDs: {paid_plans_with_razorpay_id}/{paid_plans}")

      if paid_plans == paid_plans_with_razorpay_id:
        logger.info("✅ All paid INR billing plans have been synchronized with Razorpay")
        logger.info(f"ℹ️  Free plans ({free_plans}) don't require Razorpay plan IDs")
        return True
      else:
        missing_count = paid_plans - paid_plans_with_razorpay_id
        logger.warning(f"❌ {missing_count} paid billing plans are missing Razorpay plan IDs")
        return False

    return False


def main():
  """Entry point for backward compatibility - synchronizes INR billing plans with Razorpay."""
  script = CreateRazorpayPlansScript()
  script.main()


if __name__ == "__main__":
  script = CreateRazorpayPlansScript()
  script.run_cli()
