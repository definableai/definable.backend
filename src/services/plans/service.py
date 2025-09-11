from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from models import BillingPlanModel, PlanFeatureCategoryModel, PlanFeatureLimitModel, PlanFeatureModel
from services.__base.acquire import Acquire

from .schema import BillingCycleType, BillingPlanCreateSchema, BillingPlanResponseSchema, BillingPlanUpdateSchema, CurrencyType


class BillingPlansService:
  """Service for managing billing plans."""

  http_exposed = [
    "get=get",
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
    include_features: bool = True,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> BillingPlanResponseSchema:
    """Get a billing plan by ID with features using separate queries."""
    try:
      # Get the plan
      plan_query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
      result = await session.execute(plan_query)
      plan = result.scalar_one_or_none()

      if not plan:
        raise HTTPException(status_code=404, detail="Billing plan not found")

      # Fetch features separately using helper method
      manual_features = {}
      if include_features:
        manual_features = await self._fetch_plan_features([plan_id], session)

    except HTTPException:
      raise
    except Exception as e:
      # Handle enum conversion errors - fallback to simple query
      if "is not among the defined enum values" in str(e):
        self.logger.warning(f"Enum conversion error for plan {plan_id}: {str(e)}")
        query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
        result = await session.execute(query)
        plan = result.scalar_one_or_none()

        if not plan:
          raise HTTPException(status_code=404, detail="Billing plan not found")

        # Initialize empty manual features for consistency
        manual_features = {str(plan.id): []}
        include_features = False
        self.logger.info("Disabled feature loading for single plan due to enum conversion issues")
      else:
        raise

    return BillingPlanResponseSchema.from_plan(plan, include_features, manual_features)

  async def get_list(
    self,
    org_id: UUID,
    is_active: Optional[bool] = None,
    currency: Optional[CurrencyType] = None,
    cycle: Optional[BillingCycleType] = None,
    limit: int = 100,
    offset: int = 0,
    include_features: bool = True,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get all billing plans with optional filtering and features using separate queries.

    Args:
      org_id: Organization ID (required by RBAC)
      is_active: Filter by active status (True/False). If None, returns all plans.
      currency: Filter by currency (USD/INR). If None, returns all currencies.
      cycle: Filter by billing cycle (MONTHLY/YEARLY). If None, returns BOTH monthly and yearly plans.
      limit: Maximum number of results to return (default: 100)
      offset: Number of results to skip for pagination (default: 0)
      include_features: Whether to include feature details (default: True)

    Returns:
      Dict with 'items' (list of plans), 'grouped' (plans grouped by currency/cycle), and 'pagination' info.
    """
    # Build base query for plans only - we'll fetch features separately for better performance
    query = select(BillingPlanModel)

    # Apply filters
    conditions = []
    filter_descriptions = []

    if is_active is not None:
      conditions.append(BillingPlanModel.is_active == is_active)
      filter_descriptions.append(f"is_active={is_active}")

    if currency:
      conditions.append(BillingPlanModel.currency == currency)
      filter_descriptions.append(f"currency={currency}")

    # Cycle filter - if None, include both MONTHLY and YEARLY
    if cycle is not None:
      conditions.append(BillingPlanModel.cycle == cycle)
      filter_descriptions.append(f"cycle={cycle}")
      self.logger.info(f"Filtering plans by cycle: {cycle}")
    else:
      self.logger.info("No cycle filter applied - returning both MONTHLY and YEARLY plans")

    if conditions:
      query = query.where(and_(*conditions))
      self.logger.info(f"Applied filters: {', '.join(filter_descriptions)}")
    else:
      self.logger.info("No filters applied - returning all active plans")

    # Apply sorting: currency, cycle, amount
    query = query.order_by(BillingPlanModel.currency, BillingPlanModel.cycle, BillingPlanModel.amount)

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Execute query with error handling for enum issues
    try:
      # Get all plans
      result = await session.execute(query)
      plans = list(result.scalars().all())

      # Fetch features separately using helper method
      plans_features_dict = {}
      if include_features and plans:
        plan_ids = [plan.id for plan in plans]
        plans_features_dict = await self._fetch_plan_features(plan_ids, session)
      else:
        # Initialize empty manual features for consistency
        plans_features_dict = {str(plan.id): [] for plan in plans}

    except Exception as e:
      # Handle enum conversion errors
      if "is not among the defined enum values" in str(e):
        self.logger.warning(f"Enum conversion error detected: {str(e)}")
        # Retry without feature loading to avoid enum issues
        query_no_features = select(BillingPlanModel)
        if conditions:
          query_no_features = query_no_features.where(and_(*conditions))
        query_no_features = query_no_features.order_by(BillingPlanModel.currency, BillingPlanModel.cycle, BillingPlanModel.amount)
        query_no_features = query_no_features.offset(offset).limit(limit)

        result = await session.execute(query_no_features)
        plans = list(result.scalars().all())
        # Initialize empty manual features for consistency
        plans_features_dict = {str(plan.id): [] for plan in plans}
        include_features = False  # Disable features due to enum issues
        self.logger.info("Disabled feature loading due to enum conversion issues")
      else:
        raise

    # Count total results for pagination
    count_query = select(BillingPlanModel)
    if conditions:
      count_query = count_query.where(and_(*conditions))
    count_result = await session.execute(count_query)
    total_count = len(count_result.scalars().all())

    # Log results summary
    monthly_count = sum(1 for p in plans if p.cycle == "MONTHLY")
    yearly_count = sum(1 for p in plans if p.cycle == "YEARLY")
    self.logger.info(f"Retrieved {len(plans)} plans (Monthly: {monthly_count}, Yearly: {yearly_count}) out of {total_count} total")

    # Group results by currency and cycle for better organization
    grouped_plans: dict = {}
    for plan in plans:
      currency_key = plan.currency
      cycle_key = plan.cycle
      if currency_key not in grouped_plans:
        grouped_plans[currency_key] = {}
      if cycle_key not in grouped_plans[currency_key]:
        grouped_plans[currency_key][cycle_key] = []
      grouped_plans[currency_key][cycle_key].append(BillingPlanResponseSchema.from_plan(plan, include_features, plans_features_dict))

    return {
      "items": [BillingPlanResponseSchema.from_plan(plan, include_features, plans_features_dict) for plan in plans],
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
    return BillingPlanResponseSchema.from_plan(new_plan, include_features=False, manual_features={})

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
    return BillingPlanResponseSchema.from_plan(plan, include_features=False, manual_features={})

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

### PRIVATE METHODS ###

  async def _fetch_plan_features(self, plan_ids: list[UUID], session: AsyncSession) -> dict[str, list]:
      """Fetch plan features separately and return as a dict mapped by plan_id.

      Features are determined by plan tier (name) and currency, independent of billing cycle.
      Since the populate script only creates features for MONTHLY plans, we use the monthly
      plan as the canonical source for feature definitions.
      """
      if not plan_ids:
        return {}

      # Get the plans to determine their names and currencies
      plans_query = select(BillingPlanModel).where(BillingPlanModel.id.in_(plan_ids))
      plans_result = await session.execute(plans_query)
      plans = plans_result.scalars().all()

      # Get all unique plan tiers (name + currency combinations)
      plan_tiers = set()
      plan_tier_mapping = {}  # original_plan_id -> (name, currency)

      for plan in plans:
        tier = (plan.name, plan.currency)
        plan_tiers.add(tier)
        plan_tier_mapping[str(plan.id)] = tier

      if not plan_tiers:
        return {str(plan_id): [] for plan_id in plan_ids}

      # Find the monthly plans that represent each tier (canonical feature source)
      tier_to_monthly_plan = {}
      for name, currency in plan_tiers:
        monthly_query = select(BillingPlanModel).where(
          BillingPlanModel.name == name, BillingPlanModel.currency == currency, BillingPlanModel.cycle == "MONTHLY"
        )
        monthly_result = await session.execute(monthly_query)
        monthly_plan = monthly_result.scalar_one_or_none()

        if monthly_plan:
          tier_to_monthly_plan[(name, currency)] = monthly_plan.id

      monthly_plan_ids = list(tier_to_monthly_plan.values())

      if not monthly_plan_ids:
        return {str(plan_id): [] for plan_id in plan_ids}

      # Fetch features from the canonical monthly plans
      features_query = (
        select(PlanFeatureLimitModel, PlanFeatureModel, PlanFeatureCategoryModel)
        .join(PlanFeatureModel, PlanFeatureLimitModel.feature_id == PlanFeatureModel.id)
        .join(PlanFeatureCategoryModel, PlanFeatureModel.category_id == PlanFeatureCategoryModel.id)
        .where(PlanFeatureLimitModel.billing_plan_id.in_(monthly_plan_ids))
        .order_by(PlanFeatureCategoryModel.sort_order, PlanFeatureModel.sort_order)
      )

      features_result = await session.execute(features_query)
      feature_rows = features_result.fetchall()

      # Group features by tier
      tier_features = {}
      for limit, feature, category in feature_rows:
        monthly_plan_id = limit.billing_plan_id

        # Find which tier this monthly plan represents
        for tier, plan_id in tier_to_monthly_plan.items():
          if plan_id == monthly_plan_id:
            if tier not in tier_features:
              tier_features[tier] = []

            # Manually attach the feature and category to the limit
            limit.feature = feature
            feature.category = category
            tier_features[tier].append(limit)
            break

      # Map tier features back to all original plans of that tier
      plans_features_dict: dict[str, list] = {}
      for plan_id in plan_ids:
        plan_id_str = str(plan_id)
        tier = plan_tier_mapping.get(plan_id_str)
        if tier and tier in tier_features:
          plans_features_dict[plan_id_str] = tier_features[tier]
        else:
          plans_features_dict[plan_id_str] = []

      return plans_features_dict