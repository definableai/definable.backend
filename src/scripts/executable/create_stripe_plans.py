#!/usr/bin/env python3
"""
Script to synchronize USD billing plans with Stripe plans.
First attempts to match existing Stripe plans by name, amount, currency, and cycle.
If no match is found, creates new plans in Stripe.
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
from libs.payments.stripe.v1.engine import engine as stripe_engine
from scripts.core.base_script import BaseScript


async def fetch_billing_plans(db: AsyncSession) -> List[Tuple[str, str, float, int, str, str]]:
  """Fetch all USD billing plans from the database with cycle information."""
  result = await db.execute(
    text("""
            SELECT id, name, amount, credits, currency, cycle
            FROM billing_plans
            WHERE currency = 'USD' AND is_active = true
            ORDER BY cycle ASC, amount ASC
        """)
  )
  plans = result.fetchall()
  logger.info(f"Found {len(plans)} USD billing plans in database")

  # Log breakdown of free vs paid plans and cycle distribution
  free_count = sum(1 for plan in plans if plan[2] <= 0)
  paid_count = len(plans) - free_count
  monthly_count = sum(1 for plan in plans if plan[5] == "MONTHLY")
  yearly_count = sum(1 for plan in plans if plan[5] == "YEARLY")

  logger.info(f"Plan breakdown: {paid_count} paid plans, {free_count} free plans")
  logger.info(f"Cycle breakdown: {monthly_count} monthly plans, {yearly_count} yearly plans")

  return [tuple(row) for row in plans]


async def check_stripe_plan_exists(db: AsyncSession, billing_plan_id: str) -> Optional[str]:
  """Check if a Stripe plan ID exists for a given billing plan."""
  result = await db.execute(
    text("SELECT plan_id FROM billing_plans WHERE id = :billing_plan_id"),
    {"billing_plan_id": billing_plan_id},
  )
  row = result.fetchone()
  return row[0] if row and row[0] else None


async def find_matching_stripe_plan(plan_name: str, amount: float, currency: str, cycle: str) -> Optional[str]:
  """Find existing Stripe plan that matches database plan details including cycle."""
  try:
    # Create cycle-aware plan name for matching
    stripe_plan_name = f"{plan_name}_{cycle.lower()}" if cycle == "YEARLY" else plan_name
    logger.info(f"Searching for existing Stripe plan matching: {stripe_plan_name}, {amount} {currency}, {cycle.lower()} billing")

    # Fetch all plans from Stripe
    response = stripe_engine.fetch_all_plans(count=100)

    if not response.is_successful():
      logger.error(f"Failed to fetch Stripe plans: {response.errors}")
      return None

    stripe_plans = response.data
    amount_in_cents = int(amount * 100)  # Convert to cents for comparison

    # Search for matching plan by name, amount, currency, and billing cycle
    for plan in stripe_plans:
      stripe_name = plan.get("name", "")
      stripe_amount = plan.get("amount", 0)
      stripe_currency = plan.get("currency", "")
      recurring_interval = plan.get("recurring", {}).get("interval", "")

      # Expected billing interval based on cycle
      expected_interval = "month" if cycle == "MONTHLY" else "year"

      if (
        stripe_name == stripe_plan_name
        and stripe_amount == amount_in_cents
        and stripe_currency.upper() == currency.upper()
        and recurring_interval == expected_interval
      ):
        plan_id = plan.get("id")
        logger.info(f"Found matching Stripe plan: {plan_id} for '{stripe_plan_name}' ({cycle.lower()})")
        return plan_id

    logger.info(f"No matching Stripe plan found for '{stripe_plan_name}' with amount {amount} {currency} ({cycle.lower()} billing)")
    return None

  except Exception as e:
    logger.error(f"Error searching for matching Stripe plan: {str(e)}")
    return None


async def create_stripe_plan(plan_name: str, plan_description: str, amount: float, _credits: int, currency: str, cycle: str) -> Optional[str]:
  """Create a new plan in Stripe and return its ID (cycle-aware)."""
  try:
    # Create cycle-aware plan name and billing period
    stripe_plan_name = f"{plan_name}_{cycle.lower()}" if cycle == "YEARLY" else plan_name
    period = "month" if cycle == "MONTHLY" else "year"

    # Update description for yearly plans
    description = plan_description or f"{_credits} credits for ${amount}"
    if cycle == "YEARLY":
      description += " (Yearly billing with discount)"

    logger.info(f"Creating new Stripe plan: {stripe_plan_name} - ${amount} {currency} ({_credits} credits, {period}ly billing)")

    # Convert amount to cents (Stripe expects integers)
    amount_in_cents = int(round(amount * 100))

    response = stripe_engine.create_plan(
      name=stripe_plan_name,
      amount=amount_in_cents,
      currency=currency.lower(),
      period=period,
      interval=1,
      description=description,
    )

    if response.is_successful():
      plan_id = response.data["id"]
      logger.info(f"Successfully created Stripe plan: {plan_id} for {cycle.lower()} billing")
      return plan_id
    else:
      logger.error(f"Failed to create Stripe plan '{stripe_plan_name}': {response.errors}")
      return None

  except Exception as e:
    logger.error(f"Error creating Stripe plan '{stripe_plan_name}': {str(e)}")
    return None


async def update_billing_plan_with_stripe_id(db: AsyncSession, billing_plan_id: str, stripe_plan_id: str, plan_name: str, cycle: str):
  """Update billing plan with Stripe plan ID (cycle-aware)."""
  try:
    await db.execute(
      text("UPDATE billing_plans SET plan_id = :stripe_plan_id WHERE id = :billing_plan_id"),
      {"stripe_plan_id": stripe_plan_id, "billing_plan_id": billing_plan_id},
    )
    logger.info(f"Updated billing plan {billing_plan_id} ({plan_name} {cycle.lower()}) with Stripe plan ID: {stripe_plan_id}")

  except Exception as e:
    logger.error(f"Failed to update billing plan {billing_plan_id} ({plan_name} {cycle.lower()}) with Stripe ID: {str(e)}")
    raise


class CreateStripePlansScript(BaseScript):
  """Script to sync USD billing plans with Stripe."""

  def __init__(self):
    super().__init__("create_stripe_plans")
    self.description = "Synchronize USD billing plans with Stripe"

  async def execute(self, session: AsyncSession, *args, **kwargs):
    """Execute the script to sync plans with Stripe."""
    logger.info("Starting Stripe plans synchronization for USD billing plans...")

    # Counters
    already_synced = 0
    matched_existing = 0
    created_new = 0
    failed = 0

    try:
      # Fetch all USD billing plans from database
      billing_plans = await fetch_billing_plans(session)

      if not billing_plans:
        logger.warning("No USD billing plans found in database")
        return

      logger.info(f"Processing {len(billing_plans)} USD billing plans...")

      for billing_plan_id, plan_name, amount, _credits, currency, cycle in billing_plans:
        logger.info(f"Processing plan: {plan_name} ({cycle.lower()}) - ${amount} - {_credits} credits")

        # Check if billing plan already has a Stripe plan ID
        existing_stripe_id = await check_stripe_plan_exists(session, billing_plan_id)

        if existing_stripe_id:
          logger.info(f"Billing plan '{plan_name}' ({cycle.lower()}) already has Stripe ID: {existing_stripe_id}")
          already_synced += 1
          continue

        # Try to find matching plan in Stripe (cycle-aware)
        matching_plan_id = await find_matching_stripe_plan(plan_name, amount, currency, cycle)

        if matching_plan_id:
          logger.info(f"Found matching Stripe plan for '{plan_name}' ({cycle.lower()}): {matching_plan_id}")
          await update_billing_plan_with_stripe_id(session, billing_plan_id, matching_plan_id, plan_name, cycle)
          matched_existing += 1

        else:
          logger.info(f"No matching plan found. Creating new Stripe plan for '{plan_name}' ({cycle.lower()})...")

          new_stripe_id = await create_stripe_plan(
            plan_name=plan_name,
            plan_description=f"{_credits} credits for ${amount}",
            amount=amount,
            _credits=_credits,
            currency=currency,
            cycle=cycle,
          )

          if new_stripe_id:
            await update_billing_plan_with_stripe_id(session, billing_plan_id, new_stripe_id, plan_name, cycle)
            created_new += 1
            logger.info(f"Successfully created and linked new Stripe plan for '{plan_name}' ({cycle.lower()}): {new_stripe_id}")
          else:
            failed += 1
            logger.error(f"Failed to create Stripe plan for '{plan_name}' ({cycle.lower()})")

      # Summary
      logger.info("Stripe plans synchronization completed!")
      logger.info(f"Results: {already_synced} already synced, {matched_existing} matched existing, {created_new} created new, {failed} failed")

      # Verify results
      logger.info("Verification: Checking all USD plans for Stripe synchronization...")
      verification_plans = await fetch_billing_plans(session)
      total_plans = len(verification_plans)
      synced_plans = 0
      monthly_plans = sum(1 for p in verification_plans if p[5] == "MONTHLY")
      yearly_plans = sum(1 for p in verification_plans if p[5] == "YEARLY")

      for billing_plan_id, plan_name, amount, _credits, currency, cycle in verification_plans:
        stripe_id = await check_stripe_plan_exists(session, billing_plan_id)
        if stripe_id:
          synced_plans += 1
          logger.info(f"  ✓ {plan_name} ({cycle.lower()}): ${amount} → {_credits} credits [Stripe ID: {stripe_id}]")
        else:
          logger.info(f"  ✗ {plan_name} ({cycle.lower()}): ${amount} → {_credits} credits [No Stripe ID]")

      logger.info(f"Final result: {synced_plans}/{total_plans} USD plans synchronized with Stripe")
      logger.info(f"Plan distribution: {monthly_plans} monthly, {yearly_plans} yearly")

    except Exception as e:
      logger.error(f"Error processing Stripe plans: {str(e)}")
      raise


def main():
  """Main entry point."""
  script = CreateStripePlansScript()
  script.run_cli()


if __name__ == "__main__":
  main()
