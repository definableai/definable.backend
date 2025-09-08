from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from models import BillingPlanModel
from services.__base.acquire import Acquire

from .schema import BillingCycleType, BillingPlanCreateSchema, BillingPlanResponseSchema, BillingPlanUpdateSchema, CurrencyType


class BillingPlansService:
  """Service for managing billing plans."""

  http_exposed = [
    "get=list",
    "patch=plan",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def get(
    self,
    plan_id: UUID,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> BillingPlanResponseSchema:
    """Get a billing plan by ID."""
    plan = await session.get(BillingPlanModel, plan_id)
    if not plan:
      raise HTTPException(status_code=404, detail="Billing plan not found")
    return BillingPlanResponseSchema.from_plan(plan)

  async def get_list(
    self,
    org_id: UUID,
    is_active: Optional[bool] = None,
    currency: Optional[CurrencyType] = None,
    cycle: Optional[BillingCycleType] = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get all billing plans with optional filtering."""
    query = select(BillingPlanModel)

    # Apply filters
    conditions = []
    if is_active is not None:
      conditions.append(BillingPlanModel.is_active == is_active)
    if currency:
      conditions.append(BillingPlanModel.currency == currency)
    if cycle:
      conditions.append(BillingPlanModel.cycle == cycle)

    if conditions:
      query = query.where(and_(*conditions))

    # Apply sorting: currency, cycle, amount
    query = query.order_by(BillingPlanModel.currency, BillingPlanModel.cycle, BillingPlanModel.amount)

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Execute query
    result = await session.execute(query)
    plans = result.scalars().all()

    # Count total results for pagination
    count_query = select(BillingPlanModel)
    if conditions:
      count_query = count_query.where(and_(*conditions))
    count_result = await session.execute(count_query)
    total_count = len(count_result.scalars().all())

    # Group results by currency and cycle for better organization
    grouped_plans: dict = {}
    for plan in plans:
      currency_key = plan.currency
      cycle_key = plan.cycle
      if currency_key not in grouped_plans:
        grouped_plans[currency_key] = {}
      if cycle_key not in grouped_plans[currency_key]:
        grouped_plans[currency_key][cycle_key] = []
      grouped_plans[currency_key][cycle_key].append(BillingPlanResponseSchema.from_plan(plan))

    return {
      "items": [BillingPlanResponseSchema.from_plan(plan) for plan in plans],
      "grouped": grouped_plans,
      "pagination": {
        "total": total_count,
        "offset": offset,
        "limit": limit,
      },
    }

  async def post_create(
    self,
    plan_data: BillingPlanCreateSchema,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> BillingPlanResponseSchema:
    """Create a new billing plan."""
    # Check if a plan with this name already exists
    existing_query = select(BillingPlanModel).where(
      BillingPlanModel.name == plan_data.name,
      BillingPlanModel.currency == plan_data.currency,
    )
    existing_result = await session.execute(existing_query)
    existing_plan = existing_result.scalars().first()

    if existing_plan:
      raise HTTPException(status_code=400, detail=f"Billing plan with name '{plan_data.name}' already exists")

    # Create new plan with UUID
    new_plan = BillingPlanModel(
      id=uuid4(),  # Explicitly set a UUID
      **plan_data.model_dump(),
    )
    session.add(new_plan)
    await session.commit()
    await session.refresh(new_plan)

    self.logger.info(f"Created new billing plan: {new_plan.name}")
    return BillingPlanResponseSchema.from_plan(new_plan)

  async def patch_plan(
    self,
    plan_id: UUID,
    plan_data: BillingPlanUpdateSchema,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> BillingPlanResponseSchema:
    """Update an existing billing plan."""
    # Get existing plan
    plan = await session.get(BillingPlanModel, plan_id)
    if not plan:
      raise HTTPException(status_code=404, detail="Billing plan not found")

    # If name, currency, or cycle is being changed, check for duplicates
    new_name = plan_data.name if plan_data.name is not None else plan.name
    new_currency = plan_data.currency if plan_data.currency is not None else plan.currency
    new_cycle = plan_data.cycle if plan_data.cycle is not None else plan.cycle

    # Only check for duplicates if the combination is actually changing
    if (new_name, new_currency, new_cycle) != (plan.name, plan.currency, plan.cycle):
      existing_query = select(BillingPlanModel).where(
        BillingPlanModel.name == new_name,
        BillingPlanModel.currency == new_currency,
        BillingPlanModel.cycle == new_cycle,
        BillingPlanModel.id != plan_id,  # Exclude current plan
      )
      existing_result = await session.execute(existing_query)
      existing_plan = existing_result.scalars().first()

      if existing_plan:
        raise HTTPException(status_code=400, detail=f"Billing plan with name '{new_name}' ({new_cycle} {new_currency}) already exists")

    # Update plan with provided data
    update_data = plan_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
      setattr(plan, key, value)

    await session.commit()
    await session.refresh(plan)

    self.logger.info(f"Updated billing plan: {plan.name} ({plan.cycle} {plan.currency}) - ID: {plan_id}")
    return BillingPlanResponseSchema.from_plan(plan)

  async def delete(
    self,
    org_id: UUID,
    plan_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, str]:
    """Delete a billing plan."""
    self.logger.info(f"Attempting to delete plan with ID: {plan_id}")

    # Query the plan directly to debug
    query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
    result = await session.execute(query)
    plans = result.scalars().all()
    self.logger.info(f"Found {len(plans)} plans matching ID: {plan_id}")

    plan = await session.get(BillingPlanModel, plan_id)
    if not plan:
      self.logger.warning(f"Plan with ID {plan_id} not found")
      raise HTTPException(status_code=404, detail="Billing plan not found")

    # Store plan name for logging
    plan_name = plan.name

    # Delete the plan
    await session.delete(plan)
    await session.commit()

    self.logger.info(f"Deleted billing plan: {plan_name} ({plan.cycle} {plan.currency}) - ID: {plan_id}")
    return {"message": f"Billing plan '{plan_name}' ({plan.cycle} {plan.currency}) deleted successfully"}
