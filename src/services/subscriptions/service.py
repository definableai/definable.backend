# Standard library imports
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

# Third-party imports
from fastapi import Depends, HTTPException
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Local application imports
from database import get_db
from dependencies.security import RBAC
from libs.payments.razorpay.v1 import razorpay_engine
from models import (
  BillingCycle,
  BillingPlanModel,
  CustomerModel,
  PaymentProviderModel,
  SubscriptionModel,
  TransactionModel,
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
      import time

      response = await self._create_razorpay_subscription(plan, user_id, org_id, session)
      subscription_details = response["subscription_details"]
      default_expire_by = int(time.time()) + (60 * 60)  # 1 day from now

      return SubscriptionResponse(
        subscription_id=response["subscription_id"],
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

    # Query subscription with organization validation and eager-load provider
    query = (
      select(SubscriptionModel)
      .options(selectinload(SubscriptionModel.provider))
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

    # Build query with organization filter and eager-load provider
    query = select(SubscriptionModel).options(selectinload(SubscriptionModel.provider)).where(SubscriptionModel.organization_id == org_id)

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

    # Apply cycle filter if specified (filter by billing cycle in settings JSON)
    if cycle is not None:
      filters.append(func.lower(SubscriptionModel.settings.op("->>")("cycle")) == cycle.value.lower())
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

      # Get or create Razorpay customer and store in customer table
      customer_id = await self._get_or_create_razorpay_customer_with_storage(
        user_id, user_email, f"{user_first_name} {user_last_name}".strip(), session
      )

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
      import time

      start_at = int(time.time()) + 300  # Start 5 minutes from now

      # Set billing count and expiry based on cycle
      if plan.is_yearly:
        total_count = 1  # 1 year for yearly plans
        expire_by = int(time.time()) + (365 * 24 * 60 * 60)  # Expire in 1 year
      else:
        total_count = 12  # 12 months for monthly plans
        expire_by = int(time.time()) + (365 * 24 * 60 * 60)  # Still expire in 1 year

      self.logger.info(f"Creating Razorpay subscription with {plan.cycle.lower()} billing: total_count={total_count}")

      # Create subscription with cycle-aware parameters
      subscription_response = self.razorpay_engine.create_subscription(
        plan_id=razorpay_plan_id,
        customer_id=customer_id,
        total_count=total_count,
        quantity=1,
        start_at=start_at,
        expire_by=expire_by,
        customer_notify=True,
        notes={
          "organization_id": str(org_id),
          "user_id": str(user_id),
          "plan_id": str(plan.id),
          "plan_name": plan.name,
          "billing_cycle": plan.cycle.lower(),
          "credits_per_cycle": str(plan.credits),
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
        subscription_id=subscription["id"],
        settings={
          "plan_id": razorpay_plan_id,
          "customer_id": customer_id,
          "status": subscription.get("status"),
          "currency": plan.currency,
          "cycle": plan.cycle.lower(),
          "amount": plan.amount,
          "credits": plan.credits,
          "start_at": subscription.get("start_at"),
          "end_at": subscription.get("end_at"),
          "total_count": subscription.get("total_count"),
          "remaining_count": subscription.get("remaining_count"),
          "short_url": subscription.get("short_url", ""),
        },
        is_active=False,  # Will be activated via subscription.activated webhook
      )

      session.add(db_subscription)
      await session.commit()
      await session.refresh(db_subscription)

      self.logger.info(f"Created subscription record in database: {db_subscription.id}")

      return {
        "success": True,
        "provider": "razorpay",
        "subscription_id": subscription["id"],
        "subscription_url": subscription.get("short_url", ""),
        "plan_id": razorpay_plan_id,
        "customer_id": customer_id,
        "status": subscription.get("status"),
        "currency": plan.currency,
        "cycle": plan.cycle.lower(),
        "amount": plan.amount,
        "credits": plan.credits,
        "is_yearly": plan.is_yearly,
        "is_monthly": plan.is_monthly,
        "effective_amount": round(plan.effective_amount, 2),
        "monthly_equivalent": round(plan.monthly_equivalent, 2),
        "start_at": subscription.get("start_at"),
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

  async def _get_or_create_razorpay_customer_with_storage(self, user_id: UUID, email: str, name: str, session: Optional[AsyncSession] = None) -> str:
    """Get or create a Razorpay customer and store in customer table."""
    # Use provided session or fall back to self.session
    if not session:
      raise ValueError("No database session available")

    # Check if customer exists in our customer table for this user
    customer_query = select(CustomerModel).where(
      CustomerModel.user_id == user_id, CustomerModel.payment_provider == "razorpay", CustomerModel.is_active
    )
    customer_result = await session.execute(customer_query)
    existing_customer = customer_result.scalar_one_or_none()

    if existing_customer:
      self.logger.info(f"Found existing customer in database: {existing_customer.customer_id}")
      return existing_customer.customer_id

    # No customer found for this user, check Razorpay directly
    customer_response = self.razorpay_engine.fetch_customer_by_email(email)
    razorpay_customer_id = None

    if customer_response.success and customer_response.data:
      # Found existing customer in Razorpay
      razorpay_customer = customer_response.data
      razorpay_customer_id = razorpay_customer["id"]
      self.logger.info(f"Found existing Razorpay customer: {razorpay_customer_id}")

      # Check if this Razorpay customer is already in our database (possibly with different user)
      existing_razorpay_customer_query = select(CustomerModel).where(
        CustomerModel.customer_id == razorpay_customer_id, CustomerModel.payment_provider == "razorpay", CustomerModel.is_active
      )
      existing_razorpay_result = await session.execute(existing_razorpay_customer_query)
      existing_razorpay_customer = existing_razorpay_result.scalar_one_or_none()

      if existing_razorpay_customer:
        if existing_razorpay_customer.user_id == user_id:
          # Same user, return the existing customer
          self.logger.info(f"Razorpay customer {razorpay_customer_id} already linked to current user")
          return razorpay_customer_id
        else:
          # Different user, this should not happen but handle gracefully
          self.logger.warning(f"Razorpay customer {razorpay_customer_id} is linked to different user {existing_razorpay_customer.user_id}")
          # Return the existing customer ID anyway since it's the same email/customer
          return razorpay_customer_id
    else:
      # Create new customer in Razorpay
      notes = {"user_id": str(user_id)}
      create_response = self.razorpay_engine.create_customer(name=name, email=email, notes=notes)

      if not create_response.success:
        # Check if customer already exists error
        if create_response.errors and any("already exists" in str(error) for error in create_response.errors):
          # Try to fetch by email again
          self.logger.info(f"Customer already exists, trying to fetch by email: {email}")
          customer_response = self.razorpay_engine.fetch_customer_by_email(email)
          if customer_response.success and customer_response.data:
            razorpay_customer_id = customer_response.data["id"]

        if not razorpay_customer_id:
          error_msg = str(create_response.errors) if hasattr(create_response, "errors") else "Unknown error"
          raise Exception(f"Failed to create customer: {error_msg}")
      else:
        razorpay_customer = create_response.data
        razorpay_customer_id = razorpay_customer["id"]
        self.logger.info(f"Successfully created new Razorpay customer: {razorpay_customer_id}")

    # Only create database record if it doesn't exist yet
    check_existing_query = select(CustomerModel).where(
      CustomerModel.customer_id == razorpay_customer_id, CustomerModel.payment_provider == "razorpay", CustomerModel.is_active
    )
    check_result = await session.execute(check_existing_query)
    existing_record = check_result.scalar_one_or_none()

    if not existing_record:
      # Store customer in our database
      new_customer = CustomerModel(
        id=uuid4(),
        user_id=user_id,
        payment_provider="razorpay",
        customer_id=razorpay_customer_id,
        is_active=True,
        provider_metadata={"email": email, "name": name},
      )

      session.add(new_customer)
      await session.commit()
      await session.refresh(new_customer)

      self.logger.info(f"Stored customer {razorpay_customer_id} in database for user {user_id}")
    else:
      self.logger.info(f"Customer {razorpay_customer_id} already exists in database")

    return razorpay_customer_id

  async def _get_or_create_razorpay_customer_via_engine(self, user_id: UUID, email: str, name: str, session: Optional[AsyncSession] = None) -> str:
    """Get or create a Razorpay customer using the engine."""
    # Use provided session or fall back to self.session
    if not session:
      raise ValueError("No database session available")

    # Check existing customer in payment_metadata
    transaction_result = await session.execute(
      select(TransactionModel.payment_metadata)
      .where(TransactionModel.user_id == user_id)
      .where(TransactionModel.payment_provider == "razorpay")
      .where(text("payment_metadata->>'customer_id' IS NOT NULL"))
      .order_by(TransactionModel.created_at.desc())
    )

    result = transaction_result.first()
    customer_id = result[0].get("customer_id") if result and result[0] else None
    self.logger.debug("Customer ID lookup result", customer_id=customer_id)

    if customer_id:
      return customer_id

    # Try to find existing customer by email first
    customer_response = self.razorpay_engine.fetch_customer_by_email(email)
    if customer_response.success and customer_response.data:
      existing_customer = customer_response.data
      self.logger.info(f"Found existing customer: {existing_customer['id']}")
      return existing_customer["id"]

    # Create new customer
    notes = {"user_id": str(user_id)}
    customer_response = self.razorpay_engine.create_customer(name=name, email=email, notes=notes)

    if not customer_response.success:
      # Check if customer already exists error
      if customer_response.errors and any("already exists" in str(error) for error in customer_response.errors):
        # Try to fetch by email again
        self.logger.info(f"Customer already exists, trying to fetch by email: {email}")
        customer_response = self.razorpay_engine.fetch_customer_by_email(email)
        if customer_response.success and customer_response.data:
          return customer_response.data["id"]

      # If we still can't create or find, raise the error
      error_msg = str(customer_response.errors) if hasattr(customer_response, "errors") else "Unknown error"
      raise Exception(f"Failed to create customer: {error_msg}")

    customer = customer_response.data
    self.logger.info(f"Successfully created new customer: {customer['id']}")
    return customer["id"]
