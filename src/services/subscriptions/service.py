# Standard library imports
import time
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

# Third-party imports
from fastapi import Depends, HTTPException
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Local application imports
from database import get_db
from dependencies.security import RBAC
from libs.payments.razorpay.v1 import razorpay_engine
from models import (
  BillingCycle,
  BillingPlanModel,
  PaymentProviderModel,
  SubscriptionModel,
  TransactionModel,
  TransactionStatus,
  UserModel,
)
from services.__base.acquire import Acquire

from .schema import SubscriptionDetailResponse, SubscriptionResponse


class SubscriptionsService:
  """Service for creating subscriptions."""

  http_exposed = [
    "post=create",
    "get=get",
    "get=list",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.logger = acquire.logger
    self.razorpay_engine = razorpay_engine

  async def post_create(
    self,
    plan_id: UUID,
    org_id: UUID,
    token: dict = Depends(RBAC("billing", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> SubscriptionResponse:
    """Create a new subscription for an organization."""

    # Check plan exists
    query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
    result = await session.execute(query)
    plan = result.scalar_one_or_none()

    if not plan:
      raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Billing plan not found")

    user_id = token["id"]
    if not user_id:
      raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="User ID is required")

    org_id = token["org_id"]
    if not org_id:
      raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Organization ID is required")

    # Check if plan uses INR currency (Razorpay)
    if plan.currency == "INR":
      response = await self._create_razorpay_subscription(plan, user_id, org_id, session)
      subscription_details = response["subscription_details"]
      default_expire_by = int(time.time()) + (24 * 60 * 60)  # 1 day from now

      return SubscriptionResponse(
        subscription_id=response["db_subscription_id"],  # Return database ID, not Razorpay ID
        subscription_url=response["subscription_url"],
        created_at=datetime.fromtimestamp(subscription_details.get("created_at", int(time.time()))),
        expire_by=datetime.fromtimestamp(subscription_details.get("expire_by", default_expire_by)),
      )

    # For non-INR currencies, return error for now
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Subscription creation for currency {plan.currency} is not supported yet")

  async def get(
    self,
    subscription_id: UUID,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> SubscriptionDetailResponse:
    """Get a subscription by ID for the specified organization."""
    self.logger.info(f"Retrieving subscription {subscription_id} for organization {org_id}")

    # Query subscription with organization validation and eager-load provider and plan
    query = (
      select(SubscriptionModel)
      .options(selectinload(SubscriptionModel.provider), selectinload(SubscriptionModel.plan))
      .where(SubscriptionModel.id == subscription_id, SubscriptionModel.organization_id == org_id)
    )

    result = await session.execute(query)
    subscription = result.scalar_one_or_none()

    if not subscription:
      self.logger.warning(f"Subscription {subscription_id} not found for organization {org_id}")
      raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Subscription not found or you don't have access to it")

    self.logger.info(f"Successfully retrieved subscription {subscription_id}")
    return SubscriptionDetailResponse.from_model(subscription)

  async def get_list(
    self,
    org_id: UUID,
    limit: int = 50,
    offset: int = 0,
    is_active: Optional[bool] = None,
    provider_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    cycle: Optional[BillingCycle] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get all subscriptions for the specified organization with advanced filtering and pagination.

    Args:
      org_id: Organization ID to filter subscriptions
      limit: Number of results to return (default: 50)
      offset: Number of results to skip (default: 0)
      is_active: Filter by active/inactive status
      provider_id: Filter by payment provider ID
      user_id: Filter by specific user ID
      start_at: Filter by creation date start (inclusive)
      end_at: Filter by creation date end (inclusive)
      cycle: Filter by billing cycle (MONTHLY/YEARLY)
      session: Database session
      user: Authenticated user (from RBAC)

    Returns:
      Dictionary containing subscriptions list and pagination info
    """
    self.logger.info(f"Retrieving subscriptions for organization {org_id} (limit={limit}, offset={offset})")

    # Build query with organization filter and eager-load provider and plan
    query = (
      select(SubscriptionModel)
      .options(selectinload(SubscriptionModel.provider), selectinload(SubscriptionModel.plan))
      .where(SubscriptionModel.organization_id == org_id)
    )

    # Build filter conditions list
    filters = []
    filter_descriptions = []

    # Apply is_active filter if specified
    if is_active is not None:
      filters.append(SubscriptionModel.is_active == is_active)
      filter_descriptions.append(f"is_active={is_active}")

    # Apply provider_id filter if specified
    if provider_id is not None:
      filters.append(SubscriptionModel.provider_id == provider_id)
      filter_descriptions.append(f"provider_id={provider_id}")

    # Apply user_id filter if specified
    if user_id is not None:
      filters.append(SubscriptionModel.user_id == user_id)
      filter_descriptions.append(f"user_id={user_id}")

    # Apply date range filters if specified
    if start_at is not None:
      filters.append(SubscriptionModel.created_at >= start_at)
      filter_descriptions.append(f"start_at>={start_at}")

    if end_at is not None:
      filters.append(SubscriptionModel.created_at <= end_at)
      filter_descriptions.append(f"end_at<={end_at}")

    # Apply cycle filter if specified (filter by billing cycle in plan directly)
    if cycle is not None:
      filters.append(BillingPlanModel.cycle == cycle.value)
      query = query.join(BillingPlanModel, SubscriptionModel.plan_id == BillingPlanModel.id)
      filter_descriptions.append(f"cycle={cycle.value}")

    # Apply all filters to query
    if filters:
      query = query.where(and_(*filters))
      self.logger.info(f"Applied filters: {', '.join(filter_descriptions)}")

    # Get total count for pagination
    count_result = await session.execute(select(func.count()).select_from(query.subquery()))
    total_count = count_result.scalar_one()

    # Apply pagination and ordering
    query = query.order_by(SubscriptionModel.created_at.desc()).limit(limit).offset(offset)

    # Execute query
    result = await session.execute(query)
    subscriptions = result.scalars().all()

    self.logger.info(f"Retrieved {len(subscriptions)} subscriptions (total: {total_count})")

    # Convert to response schemas
    subscription_responses = [SubscriptionDetailResponse.from_model(sub) for sub in subscriptions]

    return {
      "subscriptions": [sub.dict() for sub in subscription_responses],
      "pagination": {"total": total_count, "limit": limit, "offset": offset, "has_more": (offset + len(subscriptions)) < total_count},
    }

  ### Private methods ###

  async def _create_razorpay_subscription(
    self,
    plan: BillingPlanModel,
    user_id: UUID,
    org_id: UUID,
    session: AsyncSession,
  ) -> Dict[str, Any]:
    """Create a Razorpay subscription for INR currency plans."""

    try:
      # Get user details for customer creation
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      user = user_result.scalar_one_or_none()

      if not user:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

      # Ensure user has required fields
      user_email = getattr(user, "email", None)
      user_first_name = getattr(user, "first_name", "")
      user_last_name = getattr(user, "last_name", "")

      if not user_email:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="User email is required for subscription creation")

      self.logger.info(f"Creating Razorpay subscription for plan {plan.id} with currency INR")

      # Get Razorpay payment provider
      razorpay_provider = await PaymentProviderModel.get_by_name("razorpay", session)
      if not razorpay_provider:
        self.logger.error("Razorpay payment provider not found in database")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Payment provider not found")

      # Check if plan has razorpay plan_id, if not create one
      razorpay_plan_id = plan.plan_id

      if not razorpay_plan_id:
        # Create Razorpay plan if it doesn't exist - cycle aware
        period = "monthly" if plan.cycle == "MONTHLY" else "yearly"
        plan_name = f"{plan.name}_{plan.cycle.lower()}" if plan.cycle == "YEARLY" else plan.name

        razorpay_plan_data = {
          "name": plan_name,
          "amount": int(plan.amount * 100),  # Convert to paise
          "currency": plan.currency,
          "period": period,
          "description": f"{plan.description or plan.name} ({plan.cycle.lower()} billing)",
        }

        self.logger.info(f"Creating Razorpay plan with {period} billing cycle for plan: {plan.name}")

        plan_response = self.razorpay_engine.create_plan(**razorpay_plan_data)
        if not plan_response.success:
          self.logger.error(f"Failed to create Razorpay plan: {plan_response.errors}")
          raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription plan: {plan_response.errors}")

        razorpay_plan = plan_response.data
        razorpay_plan_id = razorpay_plan["id"]

        # Update the billing plan with the razorpay plan_id
        plan.plan_id = razorpay_plan_id
        await session.commit()
        await session.refresh(plan)

      # Calculate subscription timing based on billing cycle
      expire_by = int(time.time()) + (15 * 60)  # Expire in 15 minutes
      # Set billing count and expiry based on cycle
      if plan.is_yearly:
        total_count = 1  # 1 year for yearly plans
      else:
        total_count = 12  # 12 months for monthly plans

      self.logger.info(f"Creating Razorpay subscription with {plan.cycle.lower()} billing: total_count={total_count}")

      # Create subscription with cycle-aware parameters
      # Note: Setting start_at to null to allow immediate start after payment capture
      # Customer will be created automatically by Razorpay during payment process
      subscription_response = await self.razorpay_engine.create_subscription(
        plan_id=razorpay_plan_id,
        total_count=total_count,
        quantity=1,
        expire_by=expire_by,
        start_at=None,
        customer_notify=True,
        notes={
          "organization_id": str(org_id),
          "user_id": str(user_id),
          "plan_id": str(plan.id),
          "plan_name": plan.name,
          "billing_cycle": plan.cycle.lower(),
          "credits_per_cycle": str(plan.credits),
          "user_email": user_email,
          "user_name": f"{user_first_name} {user_last_name}".strip(),
        },
        notify_info={"notify_email": user_email},
      )

      if not subscription_response.success:
        self.logger.error(f"Failed to create Razorpay subscription: {subscription_response.errors}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {subscription_response.errors}")

      subscription = subscription_response.data

      self.logger.info(f"Successfully created Razorpay subscription: {subscription['id']}")

      # Get the Razorpay payment provider from database
      razorpay_provider = await PaymentProviderModel.get_by_name("razorpay", session)
      if not razorpay_provider:
        self.logger.warning("Razorpay payment provider not found in database")
        # Create it if it doesn't exist
        razorpay_provider = PaymentProviderModel(
          id=uuid4(),
          name="razorpay",
          is_active=True,
        )
        session.add(razorpay_provider)
        await session.commit()
        await session.refresh(razorpay_provider)

      # Create subscription record in database - initially inactive until activated webhook
      db_subscription = SubscriptionModel(
        id=uuid4(),
        organization_id=org_id,
        user_id=user_id,
        provider_id=razorpay_provider.id,
        plan_id=plan.id,  # Link to the billing plan
        subscription_id=subscription["id"],
        settings={
          "plan_id": razorpay_plan_id,
          "status": subscription.get("status"),
          "currency": plan.currency,
          "cycle": plan.cycle.lower(),
          "amount": plan.amount,
          "credits": plan.credits,
          "start_at": subscription.get("start_at"),  # Will be null initially, set after payment
          "end_at": subscription.get("end_at"),
          "total_count": subscription.get("total_count"),
          "remaining_count": subscription.get("remaining_count"),
          "short_url": subscription.get("short_url", ""),
          "user_email": user_email,
          "user_name": f"{user_first_name} {user_last_name}".strip(),
        },
        is_active=False,  # Will be activated via subscription.activated webhook
      )

      session.add(db_subscription)

      # Create transaction record for the subscription payment - marked as PENDING
      db_transaction = TransactionModel(
        id=uuid4(),
        user_id=user_id,
        organization_id=org_id,
        type="CREDIT",  # This will add credits when payment is successful
        status=TransactionStatus.PENDING,
        credits=plan.credits,
        amount=plan.amount,
        description=f"Subscription payment for {plan.name} ({plan.cycle.lower()})",
        payment_provider="razorpay",
        payment_metadata={
          "subscription_id": subscription["id"],
          "razorpay_plan_id": razorpay_plan_id,
          "subscription_amount": plan.amount,
          "subscription_currency": plan.currency,
          "subscription_credits": plan.credits,
          "db_subscription_id": str(db_subscription.id),
          "user_email": user_email,
          "user_name": f"{user_first_name} {user_last_name}".strip(),
        },
        transaction_metadata={
          "subscription_creation": True,
          "plan_id": str(plan.id),
          "plan_name": plan.name,
          "billing_cycle": plan.cycle.lower(),
        },
      )

      session.add(db_transaction)
      await session.commit()
      await session.refresh(db_subscription)
      await session.refresh(db_transaction)

      self.logger.info(f"Created subscription record in database: {db_subscription.id}")

      return {
        "success": True,
        "provider": "razorpay",
        "subscription_id": subscription["id"],
        "subscription_url": subscription.get("short_url", ""),
        "plan_id": razorpay_plan_id,
        "status": subscription.get("status"),
        "currency": plan.currency,
        "cycle": plan.cycle.lower(),
        "amount": plan.amount,
        "credits": plan.credits,
        "is_yearly": plan.is_yearly,
        "is_monthly": plan.is_monthly,
        "effective_amount": round(plan.effective_amount, 2),
        "monthly_equivalent": round(plan.monthly_equivalent, 2),
        "start_at": subscription.get("start_at"),  # Will be set after payment capture
        "end_at": subscription.get("end_at"),
        "total_count": subscription.get("total_count"),
        "remaining_count": subscription.get("remaining_count"),
        "subscription_details": subscription,
        "db_subscription_id": str(db_subscription.id),
      }

    except HTTPException:
      # Re-raise HTTP exceptions as-is
      raise
    except Exception as e:
      self.logger.error(f"Unexpected error creating Razorpay subscription: {type(e).__name__}: {str(e)}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {str(e)}")
