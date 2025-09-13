# import asyncio
import json
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import stripe
from fastapi import Depends, HTTPException, Request
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC

# from dependencies.usage import Usage  # Import the Usage dependency at the top of the file
from libs.payments.razorpay.v1 import razorpay_engine
from models import (
  BillingPlanModel,
  CustomerModel,
  OrganizationMemberModel,
  PaymentProviderModel,
  ProcessingStatus,
  SubscriptionModel,
  TransactionLogModel,
  TransactionModel,
  TransactionStatus,
  TransactionType,
  UserModel,
  WalletModel,
)
from services.__base.acquire import Acquire

# from utils.charge import Charge
from .schema import (
  BillingPlanResponseSchema,
  CheckoutSessionCreateSchema,
  CreditCalculationResponseSchema,
  RazorpayWebhookEvent,
  TransactionWithInvoiceSchema,
  WalletResponseSchema,
)


class BillingService:
  """Billing service for managing credits, transactions, and billable functions."""

  http_exposed = [
    "get=wallet",
    "get=calculate_credits",
    "get=invoice",
    "get=transactions",
    "get=usage_history",
    "post=checkout",
    "post=checkout_cancel",
    "post=stripe_webhook",
    "post=create_subscription_with_plan_id",
    "post=razorpay_webhook",
    "post=verify_razorpay_payment",
  ]

  # Webhook event to status code mapping
  WEBHOOK_STATUS_MAPPING = {
    # Payment events
    RazorpayWebhookEvent.PAYMENT_AUTHORIZED: "DFP001",  # payment authenticated
    RazorpayWebhookEvent.PAYMENT_CAPTURED: "DFP002",  # payment captured
    RazorpayWebhookEvent.PAYMENT_FAILED: "DFP100",  # payment error
    # Order events
    RazorpayWebhookEvent.ORDER_PAID: "DFO001",  # order paid
    # Invoice events
    RazorpayWebhookEvent.INVOICE_PAID: "DFI001",  # invoice paid
    # Subscription events
    RazorpayWebhookEvent.SUBSCRIPTION_CHARGED: "DFS001",  # subscription charged
    RazorpayWebhookEvent.SUBSCRIPTION_ACTIVATED: "DFS002",  # subscription activated
    RazorpayWebhookEvent.SUBSCRIPTION_AUTHENTICATED: "DFS003",  # subscription authenticated
    RazorpayWebhookEvent.SUBSCRIPTION_PAUSED: "DFS004",  # subscription paused
    RazorpayWebhookEvent.SUBSCRIPTION_RESUMED: "DFS005",  # subscription resumed
    RazorpayWebhookEvent.SUBSCRIPTION_CANCELLED: "DFS006",  # subscription cancelled
    RazorpayWebhookEvent.SUBSCRIPTION_PENDING: "DFS200",  # subscription pending
  }

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.credits_per_usd = 1000  # 1 USD = 1000 credits
    stripe.api_key = settings.stripe_secret_key
    self.session: Optional[AsyncSession] = None
    self.request_id = str(uuid4())  # Add a unique ID per service instance
    self.logger = acquire.logger
    self.razorpay_engine = razorpay_engine  # Initialize Razorpay engine

  async def _create_transaction_log(
    self, event_id: Optional[str], webhook_event: RazorpayWebhookEvent, payload: Dict[str, Any], signature: Optional[str], session: AsyncSession
  ) -> TransactionLogModel:
    """Create a transaction log entry for webhook events."""

    # Get Razorpay payment provider
    razorpay_provider = await PaymentProviderModel.get_by_name("razorpay", session)
    if not razorpay_provider:
      self.logger.error("Razorpay payment provider not found in database")
      raise ValueError("Razorpay payment provider not found")

    # Check if transaction log already exists for this event and provider
    if event_id:
      existing_log = await TransactionLogModel.get_by_event_id(event_id, razorpay_provider.id, webhook_event.value, session)
      if existing_log:
        self.logger.info(f"Transaction log already exists for event {event_id}, returning existing log {existing_log.id}")
        return existing_log

    # Map webhook event to status code
    status_code = self.WEBHOOK_STATUS_MAPPING.get(webhook_event)

    # Extract entity information from payload
    entity_type = None
    entity_id = None
    customer_id = None
    amount = None
    currency = None

    # Extract entity details based on event type
    payload_data = payload.get("payload", {})

    if webhook_event.value.startswith("subscription."):
      entity_type = "subscription"
      subscription_entity = payload_data.get("subscription", {}).get("entity", {})
      entity_id = subscription_entity.get("id")
      customer_id = subscription_entity.get("customer_id")

      # Check for payment information in subscription events
      payment_entity = payload_data.get("payment", {}).get("entity", {})
      if payment_entity:
        amount = payment_entity.get("amount", 0) / 100 if payment_entity.get("amount") else None  # Convert paise to rupees
        currency = "INR"

    elif webhook_event.value.startswith("payment."):
      entity_type = "payment"
      payment_entity = payload_data.get("payment", {}).get("entity", {})
      entity_id = payment_entity.get("id")
      customer_id = payment_entity.get("customer_id")
      amount = payment_entity.get("amount", 0) / 100 if payment_entity.get("amount") else None  # Convert paise to rupees
      currency = payment_entity.get("currency", "INR")

    elif webhook_event.value.startswith("order."):
      entity_type = "order"
      order_entity = payload_data.get("order", {}).get("entity", {})
      entity_id = order_entity.get("id")
      amount = order_entity.get("amount", 0) / 100 if order_entity.get("amount") else None  # Convert paise to rupees
      currency = order_entity.get("currency", "INR")

    elif webhook_event.value.startswith("invoice."):
      entity_type = "invoice"
      invoice_entity = payload_data.get("invoice", {}).get("entity", {})
      entity_id = invoice_entity.get("id")
      customer_id = invoice_entity.get("customer_id")
      amount = invoice_entity.get("amount", 0) / 100 if invoice_entity.get("amount") else None  # Convert paise to rupees
      currency = invoice_entity.get("currency", "INR")

    # Extract metadata for processing
    extracted_metadata = {
      "event": payload.get("event"),
      "account_id": payload.get("account_id"),
      "created_at": payload.get("created_at"),
      "entity_type": entity_type,
      "entity_id": entity_id,
      "customer_id": customer_id,
    }

    # Add event-specific metadata
    if webhook_event == RazorpayWebhookEvent.SUBSCRIPTION_CHARGED:
      subscription_entity = payload_data.get("subscription", {}).get("entity", {})
      extracted_metadata.update({
        "plan_id": subscription_entity.get("plan_id"),
        "subscription_status": subscription_entity.get("status"),
        "current_start": subscription_entity.get("current_start"),
        "current_end": subscription_entity.get("current_end"),
        "paid_count": subscription_entity.get("paid_count"),
        "remaining_count": subscription_entity.get("remaining_count"),
      })

    # Create transaction log
    transaction_log = TransactionLogModel(
      event_id=event_id,
      provider_id=razorpay_provider.id,
      event_type=webhook_event.value,
      status_code=status_code,
      entity_type=entity_type,
      entity_id=entity_id,
      customer_id=customer_id,
      amount=amount,
      currency=currency,
      processing_status=ProcessingStatus.PENDING,
      payload=payload,
      extracted_metadata=extracted_metadata,
      signature=signature,
    )

    try:
      session.add(transaction_log)
      await session.commit()
      await session.refresh(transaction_log)

      self.logger.info(f"Created transaction log {transaction_log.id} for {webhook_event.value} event {event_id}")
      return transaction_log
    except Exception as e:
      # Handle constraint violations (e.g., duplicate entries) gracefully
      await session.rollback()

      # If it's a duplicate entry error, try to fetch the existing log
      if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
        if event_id:
          existing_log = await TransactionLogModel.get_by_event_id(event_id, razorpay_provider.id, webhook_event.value, session)
          if existing_log:
            self.logger.info(f"Duplicate transaction log detected, returning existing log {existing_log.id} for event {event_id}")
            return existing_log

      # Re-raise if we can't handle it
      self.logger.error(f"Failed to create transaction log for event {event_id}: {str(e)}")
      raise

  async def _update_transaction_log_status(
    self, transaction_log: TransactionLogModel, status: ProcessingStatus, error_message: Optional[str] = None, session: Optional[AsyncSession] = None
  ):
    """Update transaction log processing status."""
    transaction_log.processing_status = status
    transaction_log.processed_at = datetime.utcnow() if status == ProcessingStatus.PROCESSED else None

    if error_message:
      transaction_log.error_message = error_message
      transaction_log.retry_count += 1

    if session:
      await session.commit()

    self.logger.info(f"Updated transaction log {transaction_log.id} status to {status.value}")

  async def get_wallet(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> WalletResponseSchema:
    """Get user's credit balance with spent credits tracking."""
    user_id = UUID(user["id"])
    self.logger.debug("Starting wallet retrieval", user_id=str(user_id), org_id=str(org_id))
    self.session = session

    wallet = await self._get_or_create_wallet(org_id)

    # Calculate low balance indicator
    low_balance = False

    if wallet.balance <= 0:
      self.logger.debug("Balance is 0, resetting spent credits", user_id=str(user_id))
      low_balance = True
      wallet.credits_spent = 0
      wallet.last_reset_date = datetime.utcnow()
      await session.commit()
      await session.refresh(wallet)
    else:
      total_cycle_credits = wallet.balance + (wallet.credits_spent or 0)
      low_balance = wallet.balance / total_cycle_credits < 0.1 if total_cycle_credits > 0 else False

    self.logger.debug("Wallet retrieved successfully", balance=wallet.balance, spent=wallet.credits_spent or 0)

    return WalletResponseSchema(
      id=UUID(str(wallet.id)),
      balance=wallet.balance - (wallet.hold or 0),
      hold=wallet.hold,
      credits_spent=wallet.credits_spent or 0,
      low_balance=low_balance,
    )

  async def get_plans(
    self,
    org_id: UUID,
    region: str = "USD",
    cycle: Optional[str] = None,
    include_inactive: bool = False,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> List[BillingPlanResponseSchema]:
    """Get available billing plans for a specific region and cycle (Razorpay focus)."""
    query = select(BillingPlanModel).where(BillingPlanModel.currency == region)

    if not include_inactive:
      query = query.where(BillingPlanModel.is_active)

    # Add cycle filtering for Razorpay (INR) plans
    if cycle and region == "INR":
      # Convert lowercase input to uppercase to match database enum
      cycle_upper = cycle.upper() if cycle.lower() in ["monthly", "yearly"] else cycle
      query = query.where(BillingPlanModel.cycle == cycle_upper)
      self.logger.info(f"Filtering Razorpay plans by cycle: {cycle_upper}")

    # Order by cycle (monthly first), then by amount
    query = query.order_by(BillingPlanModel.cycle, BillingPlanModel.amount)

    result = await session.execute(query)
    plans = result.scalars().all()

    # Use the new from_model method to include computed fields
    plan_schemas = [BillingPlanResponseSchema.from_model(plan) for plan in plans]

    # Log for Razorpay plans specifically
    if region == "INR":
      monthly_count = sum(1 for p in plan_schemas if p.is_monthly)
      yearly_count = sum(1 for p in plan_schemas if p.is_yearly)
      self.logger.info(f"Retrieved {len(plan_schemas)} Razorpay plans: {monthly_count} monthly, {yearly_count} yearly")

    return plan_schemas

  async def get_calculate_credits(
    self,
    org_id: UUID,
    amount: float,
    currency: str = "USD",
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> CreditCalculationResponseSchema:
    """Calculate credits for a given amount with applicable discounts."""
    # Determine discount percentage based on amount

    # Define conversion rates for different currencies
    credits_per_unit = {
      "USD": 1000,  # 1 USD = 1000 credits
      "INR": 12,  # 1 INR ≈ 10.1 credits (based on ₹99 = 1000 credits)
    }

    # Get the conversion rate for the specified currency
    conversion_rate = credits_per_unit.get(currency, self.credits_per_usd)

    # Base credits using currency-specific conversion
    base_credits = int(float(amount) * conversion_rate)

    # Apply discount as bonus credits
    # bonus_credits = int(base_credits * discount_percentage / 100)
    total_credits = base_credits

    return CreditCalculationResponseSchema(
      amount=amount,
      credits=total_credits,
      currency=currency,
    )

  async def get_invoice(
    self,
    org_id: UUID,
    transaction_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, str]:
    """Get the invoice URL for a transaction."""
    user_id = UUID(user["id"])
    self.logger.debug(f"Getting invoice URL for transaction: {transaction_id}, user: {user_id}")

    # Get the transaction
    query = select(TransactionModel).where(
      TransactionModel.id == transaction_id,
      TransactionModel.user_id == user_id,
    )
    result = await session.execute(query)
    transaction = result.scalar_one_or_none()

    if not transaction:
      self.logger.warning(f"Transaction {transaction_id} not found for user {user_id}")
      raise HTTPException(
        status_code=404,
        detail="Transaction not found or you don't have permission to access it",
      )

    # Handle based on payment provider
    if transaction.payment_provider == "stripe":
      invoice_id = transaction.invoice_id
      if not invoice_id:
        self.logger.warning(f"No Stripe invoice ID found for transaction {transaction_id}")
        raise HTTPException(status_code=404, detail="No invoice available for this transaction")

      # Get the invoice from Stripe
      invoice = stripe.Invoice.retrieve(invoice_id)
      self.logger.debug(f"Retrieved Stripe invoice: {invoice.id} for transaction {transaction_id}")
      invoice_url = invoice.hosted_invoice_url or ""

    elif transaction.payment_provider == "razorpay":
      invoice_id = transaction.invoice_id
      if not invoice_id:
        self.logger.warning(f"No Razorpay invoice ID found for transaction {transaction_id}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No invoice available for this transaction")

      # Get the invoice from Razorpay using engine
      try:
        invoice_response = self.razorpay_engine.fetch_invoice(invoice_id)
        if not invoice_response.success:
          self.logger.error(f"Error fetching Razorpay invoice: {invoice_response.errors}")
          raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error retrieving invoice: {invoice_response.errors}")

        invoice = invoice_response.data
        invoice_url = invoice.get("short_url") or ""
        self.logger.debug(f"Retrieved Razorpay invoice URL: {invoice_url}")
      except Exception as e:
        self.logger.error(f"Error fetching Razorpay invoice: {str(e)}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error retrieving invoice: {str(e)}")
    else:
      self.logger.warning(f"Unsupported payment provider: {transaction.payment_provider}")
      raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invoices not supported for {transaction.payment_provider}")

    # Return the hosted invoice URL
    return {
      "invoice_url": invoice_url,
      "status": "success",
      "message": "Invoice retrieved successfully",
    }

  async def get_transactions(
    self,
    org_id: UUID,
    limit: int = 50,
    offset: int = 0,
    transaction_type: Optional[TransactionType] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get user's credit transactions history with filtering options."""
    user_id = UUID(user["id"])
    self.logger.debug(f"Fetching credit transactions for user {user_id} with limit={limit}, offset={offset}")

    # Start with base query - filter for CREDIT transactions
    query = select(TransactionModel).where(TransactionModel.user_id == user_id, TransactionModel.type == TransactionType.CREDIT)
    self.logger.debug(f"Base query: {query}")

    # Apply additional filters
    if date_from:
      self.logger.debug(f"Filtering by date from: {date_from}")
      query = query.where(TransactionModel.created_at >= date_from)

    if date_to:
      self.logger.debug(f"Filtering by date to: {date_to}")
      query = query.where(TransactionModel.created_at <= date_to)

    # Get total count for pagination
    self.logger.debug("Executing count query")
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.execute(count_query)
    total_count = total.scalar_one()
    self.logger.debug(f"Total matching credit transactions: {total_count}")

    # Apply pagination
    query = query.order_by(TransactionModel.created_at.desc()).limit(limit).offset(offset)
    self.logger.debug(f"Final query with pagination: {query}")

    # Execute query
    self.logger.debug("Executing main query")
    result = await session.execute(query)
    transactions = result.scalars().all()
    self.logger.debug(f"Retrieved {len(transactions)} credit transactions")

    try:
      # Use a list comprehension for elegance and efficiency
      transactions_response = [TransactionWithInvoiceSchema.from_transaction(tx).dict() for tx in transactions]
    except Exception as e:
      self.logger.error(f"Error processing transactions: {e}")
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Error processing transactions")

    self.logger.debug(f"Returning {len(transactions_response)} credit transactions")
    return {
      "transactions": transactions_response,
      "pagination": {"total": total_count, "limit": limit, "offset": offset},
    }

  async def get_usage_history(
    self,
    org_id: UUID,
    limit: int = 50,
    offset: int = 0,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    group_by: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get credit usage history for the user.

    Shows all DEBIT transactions, including those converted from HOLD status,
    but excludes Stripe payment transactions.
    """
    self.session = session
    user_id = UUID(user["id"])
    self.logger.debug(f"Starting usage history retrieval for user {user_id}")

    # Base query - using the view instead of raw transactions
    base_query_text = "SELECT * FROM transaction_usage_stats WHERE user_id = :user_id"
    params: Dict[str, Any] = {"user_id": str(user_id)}

    # Apply date filters
    if date_from:
      base_query_text += " AND created_at >= :date_from"
      params["date_from"] = date_from.isoformat()

    if date_to:
      base_query_text += " AND created_at <= :date_to"
      params["date_to"] = date_to.isoformat()

    # Get total count for pagination
    count_query_text = f"SELECT COUNT(*) FROM ({base_query_text}) AS filtered_usage"
    count_result = await session.execute(text(count_query_text), params)
    total_count = count_result.scalar_one()
    self.logger.debug(f"Total matching usage records: {total_count}")

    # Handle grouping
    if group_by:
      group_clause = ""
      if group_by == "day":
        group_clause = "usage_date"
      elif group_by == "month":
        group_clause = "usage_month"
      elif group_by == "service":
        group_clause = "service"
      elif group_by == "transaction_type":
        group_clause = "type"
      else:
        group_clause = "id"  # Default to no grouping

      # Build query for grouped data
      query_text = f"""
        SELECT
          {group_clause} as group_key,
          SUM(credits) as credits_used,
          COUNT(*) as transaction_count,
          SUM(cost_usd) as cost_usd
        FROM ({base_query_text}) AS filtered_usage
        GROUP BY {group_clause}
        ORDER BY {group_clause}
        LIMIT :limit OFFSET :offset
      """

      params["limit"] = limit
      params["offset"] = offset

      # Execute the grouped query
      result = await session.execute(text(query_text), params)
      grouped_results = result.mappings().all()
      self.logger.info(f"Retrieved {len(grouped_results)} grouped usage records")

      # Format the response
      usage_items = []
      total_credits_used = 0

      for item in grouped_results:
        formatted_item = {
          "group_key": str(item["group_key"]),
          "credits_used": int(item["credits_used"]),
          "transaction_count": int(item["transaction_count"]),
          "cost_usd": float(item["cost_usd"]),
        }

        # Add type-specific fields
        if group_by == "day" or group_by == "month":
          formatted_item["period"] = str(item["group_key"])
        elif group_by == "service":
          formatted_item["service"] = str(item["group_key"])

        usage_items.append(formatted_item)
        total_credits_used += int(item["credits_used"])

    else:
      # No grouping - return individual transactions
      query_text = f"""
        {base_query_text}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
      """

      params["limit"] = limit
      params["offset"] = offset

      result = await session.execute(text(query_text), params)
      transactions = result.mappings().all()
      self.logger.info(f"Retrieved {len(transactions)} individual usage records")

      # Format individual transactions
      usage_items = []
      total_credits_used = 0

      for tx in transactions:
        metadata = tx["transaction_metadata"] or {}
        charge_name = metadata.get("charge_name", "Unknown Operation")

        # Create a formatted object based on your desired response structure
        transaction_data = {
          "id": str(tx["id"]),
          "timestamp": tx["created_at"].isoformat(),
          "description": tx["description"] or f"Hold for {charge_name}",
          "charge_name": charge_name,
          "service": tx["service"],
          "credits_used": tx["credits"] or 0,
          "cost_usd": float(tx["cost_usd"]),
          "transaction_type": tx["type"],
          "status": tx["status"],
          "user": {"id": str(tx["user_id"]), "email": tx["user_email"], "name": f"{tx['user_first_name']} {tx['user_last_name']}".strip()},
          "action": tx["action"] or metadata.get("action", ""),
        }

        # Extract qty from metadata if available
        if metadata and isinstance(metadata, dict) and "qty" in metadata:
          transaction_data["qty"] = metadata["qty"]

        usage_items.append(transaction_data)
        total_credits_used += int(tx["credits"] or 0)

    # Return the formatted response
    return {
      "usage_history": usage_items,
      "total_credits_used": total_credits_used,
      "total_cost_usd": round(total_credits_used / self.credits_per_usd, 2),
      "pagination": {"total": total_count, "limit": limit, "offset": offset},
    }

  async def post_checkout(
    self,
    org_id: UUID,
    checkout_data: Union[CheckoutSessionCreateSchema, Dict[str, Any]],
    payment_provider: str = "stripe",
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, Any]:
    """Create checkout session for purchasing credits with region-specific payment providers."""
    self.logger.debug(f"Raw checkout data received: {checkout_data}")
    user_id = UUID(user["id"])
    self.logger.info(f"Creating checkout session for user {user_id}")
    self.session = session

    # Extract common checkout parameters
    amount, plan_id, customer_email, credit_amount = await self._extract_and_validate_checkout_params(checkout_data, org_id, user, session)

    # Branch based on payment provider
    if payment_provider.lower() == "razorpay":
      return await self._process_razorpay_checkout(user_id, org_id, amount, credit_amount, customer_email, session)
    else:
      return await self._process_stripe_checkout(user_id, org_id, amount, credit_amount, customer_email, session)

  async def _extract_and_validate_checkout_params(self, checkout_data, org_id, user, session) -> Tuple[float, Optional[UUID], Optional[str], int]:
    """Extract and validate checkout parameters."""
    # Handle nested checkout_data structure
    if isinstance(checkout_data, dict) and "checkout_data" in checkout_data:
      checkout_data = checkout_data["checkout_data"]

    # Extract parameters
    if isinstance(checkout_data, CheckoutSessionCreateSchema):
      plan_id = getattr(checkout_data, "plan_id", None)
      amount = checkout_data.amount
      customer_email = checkout_data.customer_email
    else:
      amount = checkout_data.get("amount")
      plan_id = checkout_data.get("plan_id")
      customer_email = checkout_data.get("customer_email")

    if amount is not None:
      amount = float(str(amount))

    # Calculate credits based on plan or amount
    credit_amount = None
    if plan_id:
      self.logger.debug(f"Plan-based purchase requested with plan_id: {plan_id}")
      plan_query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
      plan_result = await session.execute(plan_query)
      plan = plan_result.scalar_one_or_none()

      if not plan:
        self.logger.warning(f"Plan not found: {plan_id}")
        raise HTTPException(status_code=404, detail="Billing plan not found or inactive")

      amount = float(plan.amount)
      credit_amount = int(plan.credits)
      self.logger.info(f"Using plan: {plan.name}, amount: ${amount}, credits: {credit_amount}")
    elif amount:
      self.logger.debug(f"Custom amount purchase requested: ${amount}")
      calculation = await self.get_calculate_credits(org_id=org_id, amount=amount, session=session, user=user)
      credit_amount = int(calculation.credits)
      self.logger.info(f"Calculated credits for ${amount}: {credit_amount} credits")
    else:
      self.logger.error("Neither amount nor plan_id provided in checkout request")
      raise HTTPException(
        status_code=400,
        detail="Either amount or plan_id must be provided for checkout",
      )

    return amount, plan_id, customer_email, credit_amount

  async def _process_stripe_checkout(self, user_id, org_id, amount, credit_amount, customer_email, session) -> Dict[str, str]:
    """Process Stripe checkout."""
    success_url = settings.stripe_success_url
    cancel_url = settings.stripe_cancel_url

    # Get customer email if not provided
    if not customer_email:
      user_db = await self._get_user_by_id(user_id, session)
      customer_email = getattr(user_db, "email", None) if user_db else None

    if not customer_email:
      raise HTTPException(status_code=400, detail="Email is required for Stripe checkout")

    # Get or create Stripe customer
    customer = await self._get_or_create_stripe_customer(user_id, customer_email, session)
    unit_amount = int(round(amount * 100))

    # Create Stripe checkout session
    stripe_session = stripe.checkout.Session.create(
      customer=customer.id,
      payment_method_types=["card"],
      line_items=[
        {
          "price_data": {
            "currency": "usd",
            "product_data": {"name": f"Purchase {credit_amount} Credits"},
            "unit_amount": unit_amount,
          },
          "quantity": 1,
        }
      ],
      mode="payment",
      success_url=str(success_url) if success_url else "",
      cancel_url=str(cancel_url) if cancel_url else "",
      invoice_creation={"enabled": True},
      metadata={
        "user_id": str(user_id),
        "org_id": str(org_id),
        "credits": str(credit_amount),
      },
    )

    # Create transaction record
    transaction = TransactionModel(
      id=uuid4(),
      user_id=user_id,
      organization_id=org_id,
      type=TransactionType.CREDIT,
      status=TransactionStatus.PENDING,
      amount=amount,
      credits=credit_amount,
      payment_provider="stripe",
      description=f"Purchase of {credit_amount} credits",
      payment_metadata={
        "customer_id": customer.id,
        "session_id": stripe_session.id,
      },
      transaction_metadata={
        "org_id": str(org_id),
        "checkout_session_id": stripe_session.id,
      },
    )

    session.add(transaction)
    await session.commit()

    self.logger.info(f"Created PENDING Stripe transaction {transaction.id}")
    return {"checkout_url": stripe_session.url or "", "session_id": stripe_session.id or ""}

  async def _process_razorpay_checkout(self, user_id, org_id, amount, credit_amount, customer_email, session) -> Dict[str, Any]:
    """Process Razorpay checkout with direct invoice generation."""
    # Get user details
    user_db = await self._get_user_by_id(user_id, session)
    if not user_db:
      raise HTTPException(status_code=404, detail="User not found")

    # Get or create customer and store in customer table
    user_email = getattr(user_db, "email", "")
    user_first_name = getattr(user_db, "first_name", "")
    user_last_name = getattr(user_db, "last_name", "")

    if not user_email:
      raise HTTPException(status_code=400, detail="User email is required for checkout")

    customer_id = await self._get_or_create_razorpay_customer_with_storage(
      user_id, user_email, f"{user_first_name} {user_last_name}".strip(), session
    )

    # Convert USD to INR (Razorpay requires amount in paise)
    amount_inr = int(amount * 100)  # Example conversion rate
    callback_url = settings.stripe_success_url

    try:
      # Create a unique receipt ID
      receipt_id = f"rcpt_{str(uuid4())[:30]}"

      invoice_data = {
        "type": "invoice",
        "description": f"Purchase of {credit_amount} credits",
        "customer_id": customer_id,
        "line_items": [
          {
            "name": f"{credit_amount} Credits",
            "description": f"Purchase of {credit_amount} credits",
            "amount": amount_inr,
            "currency": "INR",
            "quantity": 1,
          }
        ],
        "currency": "INR",
        "receipt": receipt_id,
        "email_notify": False,
        "sms_notify": False,
        "notes": {"org_id": str(org_id), "user_id": str(user_id), "credits": str(credit_amount)},
        "callback_url": callback_url,
        "callback_method": "get",
      }

      invoice_response = self.razorpay_engine.create_invoice(invoice_data)
      if not invoice_response.success:
        raise Exception(f"Failed to create invoice: {invoice_response.errors}")

      invoice = invoice_response.data

      # Create transaction record
      transaction = TransactionModel(
        id=uuid4(),
        user_id=user_id,
        organization_id=org_id,
        type=TransactionType.CREDIT,
        status=TransactionStatus.PENDING,
        amount=amount,
        credits=credit_amount,
        payment_provider="razorpay",
        description=f"Purchase of {credit_amount} credits via Razorpay",
        payment_metadata={
          "customer_id": customer_id,
          "invoice_id": invoice.get("id"),
        },
        transaction_metadata={"receipt": receipt_id, "amount_inr": amount_inr},
      )

      session.add(transaction)
      await session.commit()

      self.logger.info(f"Created Razorpay invoice: {invoice.get('id')} with transaction: {transaction.id}")

      # Return invoice URL for redirect
      return {
        "checkout_url": invoice.get("short_url"),  # Return this URL for redirect
      }

    except Exception as e:
      self.logger.error(f"Error creating Razorpay invoice: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Failed to create invoice: {str(e)}")

  async def _get_user_by_id(self, user_id: UUID, session: AsyncSession) -> Optional[UserModel]:
    """Get user by ID."""
    user_query = select(UserModel).where(UserModel.id == user_id)
    user_result = await session.execute(user_query)
    return user_result.scalar_one_or_none()

  async def post_checkout_cancel(
    self,
    org_id: UUID,
    session_id: str,
    cancel_reason: Optional[str] = "User cancelled checkout",
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, Any]:
    """
    Handle user-initiated checkout cancellations.

    This endpoint should be called from the client when a user lands on the cancel URL.
    """
    self.session = session
    user_id = UUID(user["id"])
    self.logger.info(f"Processing user-cancelled checkout session: {session_id} for user {user_id}")

    # Find the pending transaction with matching session ID
    query = (
      select(TransactionModel)
      .where(
        and_(
          text("transaction_metadata->>'checkout_session_id' = :session_id"),
          TransactionModel.status == TransactionStatus.PENDING,
          TransactionModel.user_id == user_id,
        )
      )
      .params(session_id=session_id)
    )

    result = await session.execute(query)
    transaction = result.scalar_one_or_none()

    if not transaction:
      self.logger.warning(f"No matching transaction found for cancelled session: {session_id}")
      raise HTTPException(status_code=404, detail="Transaction not found or already processed")

    # Update the transaction status to CANCELLED
    transaction.status = TransactionStatus.CANCELLED

    # Add metadata about cancellation
    if not transaction.transaction_metadata:
      transaction.transaction_metadata = {}

    transaction.transaction_metadata.update({
      "cancelled_at": datetime.utcnow().isoformat(),
      "cancellation_reason": cancel_reason,
      "cancellation_type": "user_initiated",
    })

    # Try to cancel the Stripe session if it's still active
    try:
      stripe_session = stripe.checkout.Session.retrieve(session_id)
      if stripe_session.status == "open":
        # Only try to expire if it's still open
        stripe.checkout.Session.expire(session_id)
        self.logger.info(f"Expired Stripe checkout session: {session_id}")
    except stripe.StripeError as e:
      # Log but don't fail if we can't expire the session
      self.logger.warning(f"Could not expire Stripe session {session_id}: {str(e)}")

    await session.commit()
    self.logger.info(f"Marked transaction {transaction.id} as CANCELLED due to user cancellation")

    return {"status": "success", "message": "Checkout cancelled successfully", "transaction_id": str(transaction.id)}

  async def post_stripe_webhook(self, request: Request, session: AsyncSession = Depends(get_db)) -> None:
    """Handle Stripe webhook events."""
    self.session = session
    log_context = {"method": "post_stripe_webhook", "request_id": self.request_id}

    # Get the webhook payload and signature header
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    # Verify webhook signature
    try:
      event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except stripe.SignatureVerificationError:
      self.logger.error("Invalid webhook signature", exc_info=True, extra=log_context)
      raise HTTPException(status_code=400, detail="Invalid signature")

    log_context["event_id"] = event.id
    log_context["event_type"] = event.type

    self.logger.info("Processing Stripe webhook event", extra=log_context)

    # Handle checkout.session.completed event
    if event.type == "checkout.session.completed":
      session_obj = event.data.object
      session_id = session_obj.id
      self.logger.info(f"Processing completed checkout session: {session_id}")

      # Get metadata from the session
      metadata = session_obj.metadata
      user_id = metadata.get("user_id")
      credit_amount = metadata.get("credits")

      if not user_id or not credit_amount:
        self.logger.error("Missing user_id or credits in session metadata")
        raise HTTPException(status_code=400, detail="Missing required metadata")

      # Convert to proper types
      user_id = UUID(user_id)
      credit_amount = int(credit_amount)

      # Find the pending transaction with matching session ID
      query = (
        select(TransactionModel)
        .where(and_(text("transaction_metadata->>'checkout_session_id' = :session_id"), TransactionModel.status == TransactionStatus.PENDING))
        .params(session_id=session_id)
      )

      self.logger.debug(f"Looking for transaction with checkout_session_id: {session_id}")

      result = await session.execute(query)
      transaction = result.scalar_one_or_none()

      self.logger.debug(f"Found transaction: {transaction}")

      if transaction:
        # Update transaction status
        transaction.status = TransactionStatus.COMPLETED

        # Update payment metadata with payment details
        if not transaction.payment_metadata:
          transaction.payment_metadata = {}
        transaction.payment_metadata.update({
          "payment_intent_id": session_obj.payment_intent,
          "invoice_id": session_obj.invoice,
        })

        # Add credits to organization's wallet
        org_id = UUID(metadata.get("org_id"))
        await self._add_credits(user_id, org_id, credit_amount, f"Purchase of {credit_amount} credits", session)

        await session.commit()
        self.logger.info(f"Updated transaction {transaction.id} and added {credit_amount} credits to org {org_id}")
      else:
        self.logger.error(f"No matching transaction found for session: {session_id}")
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Handle checkout.session.expired event
    elif event.type == "checkout.session.expired":
      session_obj = event.data.object
      self.logger.info(f"Processing expired checkout session: {session_obj.id}")

      # Find the pending transaction with matching session ID
      query = (
        select(TransactionModel)
        .where(and_(text("transaction_metadata->>'checkout_session_id' = :session_id"), TransactionModel.status == TransactionStatus.PENDING))
        .params(session_id=session_obj.id)
      )

      result = await session.execute(query)
      transaction = result.scalar_one_or_none()

      if transaction:
        # Update the transaction status to CANCELLED
        transaction.status = TransactionStatus.CANCELLED

        # Add metadata about expiration
        if transaction.transaction_metadata is None:
          transaction.transaction_metadata = {}
        transaction.transaction_metadata.update({"expired_at": datetime.utcnow().isoformat(), "expiration_reason": "Checkout session expired"})

        await session.commit()
        self.logger.info(f"Marked transaction {transaction.id} as CANCELLED due to expired checkout session")

      else:
        self.logger.warning(f"No matching transaction found for expired session: {session_obj.id}")

    # Only log the debug message if session_id is defined
    if "session_id" in locals():
      self.logger.debug(f"Looking for transaction with checkout_session_id: {session_id}", query=str(query))

  async def _get_or_create_stripe_customer(
    self,
    user_id: UUID,
    email: Optional[str],
    session: AsyncSession,
  ) -> stripe.Customer:
    """Get or create a Stripe customer for the user."""
    self.logger.debug("Getting or creating Stripe customer", user_id=str(user_id), email=email)

    # Check if customer exists in payment_metadata
    transaction_result = await session.execute(
      select(TransactionModel.payment_metadata)
      .where(TransactionModel.user_id == user_id)
      .where(TransactionModel.payment_provider == "stripe")
      .where(text("payment_metadata->>'customer_id' IS NOT NULL"))
      .order_by(TransactionModel.created_at.desc())
    )

    result = transaction_result.first()
    customer_id = result[0].get("customer_id") if result and result[0] else None
    self.logger.debug("Customer ID lookup result", customer_id=customer_id)

    if customer_id:
      self.logger.debug(f"Found existing Stripe customer: {customer_id}")
      return stripe.Customer.retrieve(customer_id)

    self.logger.debug("No existing customer found, creating new one")

    # Create new customer
    if not email:
      raise HTTPException(status_code=400, detail="Email is required to create a customer")

    customer = stripe.Customer.create(email=email, metadata={"user_id": str(user_id)})
    self.logger.info(f"Created new Stripe customer: {customer.id}")
    return customer

  async def _get_or_create_wallet(self, org_id: UUID, session: Optional[AsyncSession] = None) -> WalletModel:
    """Get or create organization's wallet."""
    self.logger.debug(f"Getting or creating wallet for organization {org_id}")

    # Use provided session or fall back to self.session
    db_session = session or self.session
    if not db_session:
      raise ValueError("No database session available")

    # Then try to get existing wallet
    query = select(WalletModel).where(WalletModel.organization_id == org_id)
    result = await db_session.execute(query)
    wallet = result.scalar_one_or_none()

    if wallet:
      self.logger.debug(f"Found existing wallet for organization {org_id}: balance={wallet.balance}")
      return wallet

    self.logger.info(f"No wallet found for organization {org_id}, creating new entry")

    # Create new wallet entry with explicit UUID
    new_wallet = WalletModel(
      id=uuid4(),
      organization_id=org_id,
      balance=0,
      hold=0,
      credits_spent=0,
    )

    db_session.add(new_wallet)
    await db_session.commit()
    await db_session.refresh(new_wallet)
    self.logger.info(f"Created new wallet for organization {org_id}")
    return new_wallet

  async def _add_credits(
    self, user_id: UUID, org_id: UUID, amount: int, description: Optional[str] = None, session: Optional[AsyncSession] = None
  ) -> bool:
    """Add credits to organization's wallet while tracking the user who performed the action."""
    if amount <= 0:
      return True

    db_session = session or self.session
    if not db_session:
      raise ValueError("No database session available")

    wallet = await self._get_or_create_wallet(org_id, db_session)

    # Update wallet balance
    old_balance = wallet.balance
    wallet.balance += amount

    await db_session.commit()
    self.logger.info(f"Added {amount} credits to organization {org_id}, balance: {old_balance} -> {wallet.balance}")

    return True

  async def post_razorpay_webhook(self, request: Request, session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Handle Razorpay webhook events with comprehensive logging."""
    self.session = session

    # Get webhook payload
    payload = await request.json()
    signature = request.headers.get("x-razorpay-signature")
    event = payload.get("event")

    # Extract entity ID based on event type
    event_id = None
    if event and event.startswith("subscription."):
      # For subscription events, get the subscription ID
      event_id = payload.get("payload", {}).get("subscription", {}).get("entity", {}).get("id")
    elif event and event.startswith("payment."):
      # For payment events, get the payment ID
      event_id = payload.get("payload", {}).get("payment", {}).get("entity", {}).get("id")
    elif event and event.startswith("invoice."):
      # For invoice events, get the invoice ID
      event_id = payload.get("payload", {}).get("invoice", {}).get("entity", {}).get("id")
    elif event and event.startswith("order."):
      # For order events, get the order ID
      event_id = payload.get("payload", {}).get("order", {}).get("entity", {}).get("id")
    else:
      # Fallback to top-level entity if present
      event_id = payload.get("entity", {}).get("id") if isinstance(payload.get("entity"), dict) else None

    self.logger.info(f"Processing Razorpay webhook event: {event} (entity_id: {event_id})")

    # Verify webhook signature
    signature_verified = False
    try:
      if signature:
        payload_string = json.dumps(payload, separators=(",", ":"))
        signature_response = self.razorpay_engine.verify_webhook_signature(payload_string, signature)
        signature_verified = signature_response.success and signature_response.data
        if signature_verified:
          self.logger.info("Webhook signature verified successfully")
        else:
          self.logger.warning("Webhook signature verification failed")
      else:
        self.logger.warning("No signature provided with webhook")
    except Exception as e:
      self.logger.warning(f"Webhook signature verification error: {str(e)}")

    # Process different webhook events using the enum
    webhook_event = None
    transaction_log = None

    try:
      webhook_event = RazorpayWebhookEvent(event)
    except ValueError:
      self.logger.warning(f"Unknown webhook event: {event}")
      return {"status": "success", "message": f"Unknown event {event} received but ignored"}

    # Create transaction log entry for all webhook events
    try:
      transaction_log = await self._create_transaction_log(
        event_id=event_id, webhook_event=webhook_event, payload=payload, signature=signature, session=session
      )
    except Exception as e:
      self.logger.error(f"Failed to create transaction log: {str(e)}")
      transaction_log = None
      # Continue processing even if logging fails, but ensure session is still usable
      try:
        await session.rollback()
      except Exception as rollback_error:
        self.logger.error(f"Failed to rollback session after transaction log error: {str(rollback_error)}")

    # Route to appropriate handler based on event type
    result = {"status": "success", "message": "Webhook received"}
    processing_error = None

    try:
      if webhook_event == RazorpayWebhookEvent.INVOICE_PAID:
        result = await self._handle_invoice_paid(payload, session, transaction_log)

      elif webhook_event == RazorpayWebhookEvent.PAYMENT_CAPTURED:
        result = await self._handle_payment_captured(payload, session, transaction_log)

      elif webhook_event == RazorpayWebhookEvent.PAYMENT_FAILED:
        result = await self._handle_payment_failed(payload, session, transaction_log)

      elif webhook_event in [
        RazorpayWebhookEvent.SUBSCRIPTION_ACTIVATED,
        RazorpayWebhookEvent.SUBSCRIPTION_AUTHENTICATED,
        RazorpayWebhookEvent.SUBSCRIPTION_CHARGED,
        RazorpayWebhookEvent.SUBSCRIPTION_PAUSED,
        RazorpayWebhookEvent.SUBSCRIPTION_RESUMED,
        RazorpayWebhookEvent.SUBSCRIPTION_CANCELLED,
        RazorpayWebhookEvent.SUBSCRIPTION_PENDING,
      ]:
        result = await self._handle_subscription_event(webhook_event, payload, session, transaction_log)

      elif webhook_event in [
        RazorpayWebhookEvent.PAYMENT_AUTHORIZED,
        RazorpayWebhookEvent.ORDER_PAID,
      ]:
        # These events are logged but don't require specific processing
        self.logger.info(f"Webhook event {webhook_event.value} logged successfully")

      # Update transaction log status to processed
      if transaction_log:
        await self._update_transaction_log_status(transaction_log, ProcessingStatus.PROCESSED, session=session)

    except Exception as e:
      processing_error = str(e)
      self.logger.error(f"Error processing webhook {webhook_event.value}: {processing_error}")

      if transaction_log:
        await self._update_transaction_log_status(transaction_log, ProcessingStatus.FAILED, error_message=processing_error, session=session)

      result = {"status": "error", "message": f"Processing failed: {processing_error}"}

    return result

  async def _handle_invoice_paid(
    self, payload: Dict[str, Any], session: AsyncSession, transaction_log: Optional[TransactionLogModel] = None
  ) -> Dict[str, Any]:
    """Handle invoice.paid webhook events."""
    # Extract data from the webhook payload
    invoice_entity = payload.get("payload", {}).get("invoice", {}).get("entity", {})
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})

    invoice_id = invoice_entity.get("id")
    payment_id = payment_entity.get("id")

    if not invoice_id or not payment_id:
      self.logger.error("Missing invoice_id or payment_id in webhook payload")
      return {"status": "error", "message": "Missing required parameters"}

    self.logger.info(f"Processing invoice.paid webhook: invoice_id={invoice_id}, payment_id={payment_id}")

    # Find transaction by invoice_id using payment_metadata
    query = (
      select(TransactionModel)
      .where(text("payment_metadata->>'invoice_id' = :invoice_id"), TransactionModel.payment_provider == "razorpay")
      .params(invoice_id=invoice_id)
    )
    result = await session.execute(query)
    transaction = result.scalar_one_or_none()

    if not transaction:
      self.logger.error(f"No transaction found for invoice_id: {invoice_id}")
      return {"status": "error", "message": "Transaction not found"}

    # Only process if transaction is not already completed
    if transaction.status == TransactionStatus.COMPLETED:
      self.logger.info(f"Transaction {transaction.id} already completed, skipping")
      return {"status": "success", "message": "Transaction already processed"}

    self.logger.info(f"Found transaction {transaction.id} for invoice {invoice_id}")

    # Get amounts from webhook payload
    invoice_amount = invoice_entity.get("amount") or 0
    payment_amount = payment_entity.get("amount") or 0

    self.logger.info(f"Amounts - Invoice: {invoice_amount}, Payment: {payment_amount}")

    # Validate amounts match (in smallest currency unit)
    expected_amount = int(transaction.amount * 100) if transaction.amount else 0
    if payment_amount != expected_amount:
      self.logger.warning(f"Amount mismatch: expected {expected_amount}, got {payment_amount}")
      # Continue processing but log the discrepancy

    # Update transaction status to completed
    transaction.status = TransactionStatus.COMPLETED

    # Update payment_metadata with Razorpay payment details
    if not transaction.payment_metadata:
      transaction.payment_metadata = {}

    transaction.payment_metadata.update({
      "payment_id": payment_id,
      "invoice_id": invoice_id,
      "payment_amount": payment_amount,
      "invoice_amount": invoice_amount,
    })

    # Update transaction_metadata with processing info
    if not transaction.transaction_metadata:
      transaction.transaction_metadata = {}

    transaction.transaction_metadata.update({
      "payment_status": "captured",
      "webhook_event": "invoice.paid",
      "processed_at": datetime.utcnow().isoformat(),
      "payment_method": payment_entity.get("method"),
      "payment_captured": payment_entity.get("captured"),
    })

    # Add credits to the organization's wallet
    await self._add_credits(
      transaction.user_id,
      transaction.organization_id,
      transaction.credits,
      f"Purchase of {transaction.credits} credits via Razorpay (Invoice: {invoice_id})",
      session,
    )

    await session.commit()
    self.logger.info(f"Successfully processed Razorpay payment: {payment_id} for invoice: {invoice_id}")

    return {"status": "success", "message": "Payment processed successfully"}

  async def _handle_payment_captured(
    self, payload: Dict[str, Any], session: AsyncSession, transaction_log: Optional[TransactionLogModel] = None
  ) -> Dict[str, Any]:
    """Handle payment.captured webhook events."""
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    payment_id = payment_entity.get("id")
    invoice_id = payment_entity.get("invoice_id")
    subscription_id = payment_entity.get("subscription_id")

    if not payment_id:
      self.logger.error("Missing payment_id in payment.captured webhook payload")
      return {"status": "error", "message": "Missing payment_id"}

    self.logger.info(f"Processing payment.captured webhook: payment_id={payment_id}, invoice_id={invoice_id}, subscription_id={subscription_id}")

    transaction = None

    # First try to find transaction by subscription_id (for subscription payments)
    if subscription_id:
      self.logger.info(f"Looking for transaction by subscription_id: {subscription_id}")
      query = (
        select(TransactionModel)
        .where(
          text("payment_metadata->>'subscription_id' = :subscription_id"),
          TransactionModel.payment_provider == "razorpay",
          TransactionModel.status == TransactionStatus.PENDING,
        )
        .order_by(TransactionModel.created_at.desc())
      )
      result = await session.execute(query, {"subscription_id": subscription_id})
      transaction = result.scalar_one_or_none()

      if transaction:
        self.logger.info(f"Found pending subscription transaction: {transaction.id}")

    # If no subscription transaction found, try by invoice_id (for regular payments)
    if not transaction and invoice_id:
      self.logger.info(f"Looking for transaction by invoice_id: {invoice_id}")
      query = (
        select(TransactionModel)
        .where(
          text("payment_metadata->>'invoice_id' = :invoice_id"),
          TransactionModel.payment_provider == "razorpay",
          TransactionModel.status == TransactionStatus.PENDING,
        )
        .order_by(TransactionModel.created_at.desc())
      )
      result = await session.execute(query, {"invoice_id": invoice_id})
      transaction = result.scalar_one_or_none()

    if not transaction:
      self.logger.warning(f"No pending transaction found for payment_id: {payment_id}")
      # Check if there's already a completed transaction
      if subscription_id:
        completed_query = select(TransactionModel).where(
          text("payment_metadata->>'subscription_id' = :subscription_id"),
          TransactionModel.payment_provider == "razorpay",
          TransactionModel.status == TransactionStatus.COMPLETED,
        )
        completed_result = await session.execute(completed_query, {"subscription_id": subscription_id})
        completed_transaction = completed_result.scalar_one_or_none()

        if completed_transaction:
          self.logger.info(f"Subscription payment already processed: {completed_transaction.id}")
          return {"status": "success", "message": "Subscription payment already processed"}

      # If no transaction found at all, log and return success
      self.logger.info(f"No transaction found for payment_id: {payment_id}")
      return {"status": "success", "message": "Payment captured event received but no transaction found"}

    # Found pending transaction - mark it as completed
    self.logger.info(f"Processing payment.captured for transaction: {transaction.id}")

    # Update transaction status to COMPLETED
    transaction.status = TransactionStatus.COMPLETED

    # Update payment metadata with captured payment details
    if not transaction.payment_metadata:
      transaction.payment_metadata = {}

    transaction.payment_metadata.update({
      "payment_id": payment_id,
      "captured_at": datetime.utcnow().isoformat(),
      "payment_status": "captured",
    })

    # Update transaction metadata
    if not transaction.transaction_metadata:
      transaction.transaction_metadata = {}

    transaction.transaction_metadata.update({
      "payment_captured": True,
      "webhook_event": "payment.captured",
      "processed_at": datetime.utcnow().isoformat(),
    })

    # Add credits to the organization wallet
    await self._add_credits(
      transaction.user_id, transaction.organization_id, transaction.credits, f"Payment captured - {transaction.description}", session
    )

    # If this is a subscription transaction, also activate the subscription
    if subscription_id and "db_subscription_id" in transaction.payment_metadata:
      try:
        subscription_query = select(SubscriptionModel).where(SubscriptionModel.id == transaction.payment_metadata["db_subscription_id"])
        sub_result = await session.execute(subscription_query)
        db_subscription = sub_result.scalar_one_or_none()

        if db_subscription and not db_subscription.is_active:
          db_subscription.is_active = True
          self.logger.info(f"Activated subscription: {db_subscription.id}")

      except Exception as e:
        self.logger.error(f"Failed to activate subscription: {str(e)}")
        # Don't fail the webhook processing for subscription activation errors

    await session.commit()
    self.logger.info(f"Successfully processed payment.captured: {payment_id}, added {transaction.credits} credits")

    return {
      "status": "success",
      "message": "Payment captured and processed successfully",
      "data": {
        "transaction_id": str(transaction.id),
        "payment_id": payment_id,
        "credits_added": transaction.credits,
      },
    }

  async def _handle_payment_failed(
    self, payload: Dict[str, Any], session: AsyncSession, transaction_log: Optional[TransactionLogModel] = None
  ) -> Dict[str, Any]:
    """Handle payment.failed webhook events."""
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    payment_id = payment_entity.get("id")
    invoice_id = payment_entity.get("invoice_id")

    if not payment_id:
      self.logger.error("Missing payment_id in payment.failed webhook payload")
      return {"status": "error", "message": "Missing payment_id"}

    # Extract error details from the payment
    error_code = payment_entity.get("error_code")
    error_description = payment_entity.get("error_description")
    error_source = payment_entity.get("error_source")
    error_step = payment_entity.get("error_step")
    error_reason = payment_entity.get("error_reason")
    payment_status = payment_entity.get("status")

    self.logger.info(f"Processing payment.failed webhook: payment_id={payment_id}, invoice_id={invoice_id}, error={error_code}")

    # Try to find the transaction by invoice_id if available
    transaction = None
    if invoice_id:
      query = (
        select(TransactionModel)
        .where(text("payment_metadata->>'invoice_id' = :invoice_id"), TransactionModel.payment_provider == "razorpay")
        .params(invoice_id=invoice_id)
      )
      result = await session.execute(query)
      transaction = result.scalar_one_or_none()

    # If not found by invoice_id, try to find by customer_id and pending status (for recent transactions)
    if not transaction:
      customer_id = payment_entity.get("customer_id")
      if customer_id:
        # Look for recent pending transactions for this customer
        query = (
          select(TransactionModel)
          .where(
            text("payment_metadata->>'customer_id' = :customer_id"),
            TransactionModel.payment_provider == "razorpay",
            TransactionModel.status == TransactionStatus.PENDING,
          )
          .order_by(TransactionModel.created_at.desc())
          .limit(1)
          .params(customer_id=customer_id)
        )
        result = await session.execute(query)
        transaction = result.scalar_one_or_none()

    if transaction:
      self.logger.info(f"Found transaction {transaction.id} for failed payment {payment_id}")

      # Update transaction status to FAILED
      transaction.status = TransactionStatus.FAILED

      # Update payment metadata with error details
      if not transaction.payment_metadata:
        transaction.payment_metadata = {}

      transaction.payment_metadata.update({
        "payment_id": payment_id,
        "payment_status": payment_status,
        "failed_at": datetime.utcnow().isoformat(),
      })

      # Update transaction metadata with detailed error information
      if not transaction.transaction_metadata:
        transaction.transaction_metadata = {}

      transaction.transaction_metadata.update({
        "payment_failed": True,
        "error_code": error_code,
        "error_description": error_description,
        "error_source": error_source,
        "error_step": error_step,
        "error_reason": error_reason,
        "webhook_event": "payment.failed",
        "processed_at": datetime.utcnow().isoformat(),
      })

      # Update transaction log with user info if available
      if transaction_log and transaction:
        transaction_log.user_id = transaction.user_id
        transaction_log.organization_id = transaction.organization_id

      await session.commit()
      self.logger.info(f"Updated transaction {transaction.id} status to FAILED due to payment failure")

      return {
        "status": "success",
        "message": "Payment failure processed successfully",
        "data": {"transaction_id": str(transaction.id), "payment_id": payment_id, "error_code": error_code, "error_description": error_description},
      }
    else:
      self.logger.warning(f"No transaction found for failed payment: {payment_id}")
      return {"status": "success", "message": "Payment failed event received but no matching transaction found"}

  async def _handle_subscription_event(
    self, webhook_event: RazorpayWebhookEvent, payload: Dict[str, Any], session: AsyncSession, transaction_log: Optional[TransactionLogModel] = None
  ) -> Dict[str, Any]:
    """Handle subscription-related webhook events with proper credit processing."""
    subscription_entity = payload.get("payload", {}).get("subscription", {}).get("entity", {})
    subscription_id = subscription_entity.get("id")
    plan_id = subscription_entity.get("plan_id")
    customer_id = subscription_entity.get("customer_id")
    status = subscription_entity.get("status")

    self.logger.info(
      f"Processing subscription event {webhook_event.value}: subscription_id={subscription_id}, plan_id={plan_id}, customer_id={customer_id}, status={status}"  # noqa: E501
    )

    # Extract payment information if available
    payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    payment_id = payment_entity.get("id") if payment_entity else None
    payment_amount = payment_entity.get("amount") if payment_entity else None  # Amount in paise

    # Handle subscription.activated event - activate the subscription in our database
    if webhook_event == RazorpayWebhookEvent.SUBSCRIPTION_ACTIVATED:
      try:
        self.logger.info(f"Processing subscription activation for subscription_id: {subscription_id}")

        # Find subscription record in our database
        subscription_query = select(SubscriptionModel).where(SubscriptionModel.subscription_id == subscription_id)
        subscription_result = await session.execute(subscription_query)
        db_subscription = subscription_result.scalar_one_or_none()

        if not db_subscription:
          self.logger.error(f"Subscription record not found in database for subscription_id: {subscription_id}")
          return {"status": "error", "message": "Subscription record not found"}

        # Activate the subscription
        self.logger.info(f"Found subscription record: id={db_subscription.id}, current is_active={db_subscription.is_active}")

        if not db_subscription.is_active:
          self.logger.info(f"Activating subscription {subscription_id}...")
          db_subscription.is_active = True

          # Update subscription settings with latest status from webhook
          if db_subscription.settings:
            db_subscription.settings.update({"status": status, "activated_at": datetime.utcnow().isoformat(), "webhook_event": webhook_event.value})

          await session.commit()
          await session.refresh(db_subscription)

          # Update transaction log with subscription info
          if transaction_log:
            transaction_log.user_id = db_subscription.user_id
            transaction_log.organization_id = db_subscription.organization_id
            await session.commit()

          self.logger.info(f"✅ Successfully activated subscription {subscription_id} for organization {db_subscription.organization_id}")

          return {
            "status": "success",
            "message": "Subscription activated successfully",
            "data": {"subscription_id": subscription_id, "organization_id": str(db_subscription.organization_id), "is_active": True},
          }
        else:
          self.logger.info(f"Subscription {subscription_id} is already active (is_active={db_subscription.is_active})")
          return {"status": "success", "message": "Subscription already active"}

      except Exception as e:
        self.logger.error(f"Error activating subscription: {str(e)}")
        return {"status": "error", "message": f"Error activating subscription: {str(e)}"}

    # Handle subscription status changes (paused, cancelled, resumed)
    if webhook_event in [
      RazorpayWebhookEvent.SUBSCRIPTION_PAUSED,
      RazorpayWebhookEvent.SUBSCRIPTION_CANCELLED,
      RazorpayWebhookEvent.SUBSCRIPTION_RESUMED,
    ]:
      try:
        # Find subscription record in our database
        subscription_query = select(SubscriptionModel).where(SubscriptionModel.subscription_id == subscription_id)
        subscription_result = await session.execute(subscription_query)
        db_subscription = subscription_result.scalar_one_or_none()

        if db_subscription:
          # Update subscription status based on webhook event
          if webhook_event in [RazorpayWebhookEvent.SUBSCRIPTION_PAUSED, RazorpayWebhookEvent.SUBSCRIPTION_CANCELLED]:
            db_subscription.is_active = False
          elif webhook_event == RazorpayWebhookEvent.SUBSCRIPTION_RESUMED:
            db_subscription.is_active = True

          # Update subscription settings with latest status
          if db_subscription.settings:
            db_subscription.settings.update({
              "status": status,
              f"{webhook_event.value.split('.')[-1]}_at": datetime.utcnow().isoformat(),
              "webhook_event": webhook_event.value,
            })

          await session.commit()
          await session.refresh(db_subscription)

          # Update transaction log with subscription info
          if transaction_log:
            transaction_log.user_id = db_subscription.user_id
            transaction_log.organization_id = db_subscription.organization_id
            await session.commit()

          self.logger.info(f"Updated subscription {subscription_id} status to {webhook_event.value}: is_active={db_subscription.is_active}")

          return {
            "status": "success",
            "message": f"Subscription {webhook_event.value.split('.')[-1]} successfully",
            "data": {
              "subscription_id": subscription_id,
              "organization_id": str(db_subscription.organization_id),
              "is_active": db_subscription.is_active,
              "event": webhook_event.value,
            },
          }
        else:
          self.logger.warning(f"Subscription record not found for {webhook_event.value}: {subscription_id}")

      except Exception as e:
        self.logger.error(f"Error updating subscription status for {webhook_event.value}: {str(e)}")
        return {"status": "error", "message": f"Error updating subscription: {str(e)}"}

    # Handle subscription.charged event - this is where we add credits
    if webhook_event == RazorpayWebhookEvent.SUBSCRIPTION_CHARGED and payment_id and payment_amount:
      try:
        # Get Razorpay payment provider
        razorpay_provider = await PaymentProviderModel.get_by_name("razorpay", session)
        if not razorpay_provider:
          self.logger.error("Razorpay payment provider not found in database")
          return {"status": "error", "message": "Payment provider not found"}

        # Find customer in our database
        customer_query = select(CustomerModel).where(
          CustomerModel.customer_id == customer_id, CustomerModel.provider_id == razorpay_provider.id, CustomerModel.is_active
        )
        customer_result = await session.execute(customer_query)
        customer = customer_result.scalar_one_or_none()

        if not customer:
          self.logger.warning(f"Customer not found for customer_id: {customer_id}, attempting to find by subscription_id: {subscription_id}")

          # Try to find customer by subscription_id in our subscription records
          subscription_query = select(SubscriptionModel).where(SubscriptionModel.subscription_id == subscription_id, SubscriptionModel.is_active)
          subscription_result = await session.execute(subscription_query)
          subscription_record = subscription_result.scalar_one_or_none()

          if subscription_record:
            # Found subscription, now get the associated customer
            customer_by_user_query = select(CustomerModel).where(
              CustomerModel.user_id == subscription_record.user_id, CustomerModel.provider_id == razorpay_provider.id, CustomerModel.is_active
            )
            customer_by_user_result = await session.execute(customer_by_user_query)
            customer = customer_by_user_result.scalar_one_or_none()

            if customer:
              # Update customer record with new customer_id from Razorpay
              old_customer_id = customer.customer_id
              customer.customer_id = customer_id

              # Update provider metadata if it exists
              if customer.provider_metadata:
                customer.provider_metadata["previous_customer_id"] = old_customer_id
                customer.provider_metadata["customer_id_updated_at"] = datetime.utcnow().isoformat()
              else:
                customer.provider_metadata = {"previous_customer_id": old_customer_id, "customer_id_updated_at": datetime.utcnow().isoformat()}

              await session.commit()
              await session.refresh(customer)

              self.logger.info(f"Updated customer {customer.id} customer_id from {old_customer_id} to {customer_id}")
            else:
              self.logger.error(f"No customer record found for subscription {subscription_id} and user {subscription_record.user_id}")
              return {"status": "error", "message": "Customer record not found for subscription"}
          else:
            self.logger.error(f"No subscription record found for subscription_id: {subscription_id}")
            return {"status": "error", "message": "Subscription not found in database"}

        if not customer:
          self.logger.error(f"Unable to resolve customer for customer_id: {customer_id}")
          return {"status": "error", "message": "Customer not found"}

        # Convert amount from paise to rupees
        amount_inr = payment_amount / 100

        # Calculate credits (12 credits per rupee for INR)
        credit_amount = int(amount_inr * 12)

        # Find user's organization (assuming first active organization)
        user_query = select(UserModel).where(UserModel.id == customer.user_id)
        user_result = await session.execute(user_query)
        user = user_result.scalar_one_or_none()

        if not user:
          self.logger.error(f"User not found for user_id: {customer.user_id}")
          return {"status": "error", "message": "User not found"}

        # Get user's organizations

        org_member_query = select(OrganizationMemberModel).where(OrganizationMemberModel.user_id == user.id)
        org_member_result = await session.execute(org_member_query)
        org_member = org_member_result.scalar_one_or_none()

        if not org_member:
          self.logger.error(f"No organization found for user: {user.id}")
          return {"status": "error", "message": "No organization found for user"}

        organization_id = org_member.organization_id

        # Add credits to organization wallet
        await self._add_credits(
          user_id=user.id,
          org_id=organization_id,
          amount=credit_amount,
          description=f"Subscription payment via Razorpay (Plan: {plan_id}, Payment: {payment_id})",
          session=session,
        )

        # Create a transaction record for this subscription payment
        transaction = TransactionModel(
          user_id=user.id,
          organization_id=organization_id,
          type=TransactionType.CREDIT,
          status=TransactionStatus.COMPLETED,
          amount=amount_inr,
          credits=credit_amount,
          description="Subscription payment via Razorpay",
          payment_provider="razorpay",
          payment_metadata={
            "subscription_id": subscription_id,
            "payment_id": payment_id,
            "plan_id": plan_id,
            "customer_id": customer_id,
            "webhook_event": webhook_event.value,
          },
          transaction_metadata={
            "subscription_charged": True,
            "amount_inr": amount_inr,
            "payment_amount_paise": payment_amount,
            "processed_at": datetime.utcnow().isoformat(),
          },
        )

        session.add(transaction)
        await session.commit()
        await session.refresh(transaction)

        self.logger.info(
          f"Successfully processed subscription payment: {payment_id}, added {credit_amount} credits to organization {organization_id}"
        )

        # Update transaction log with user and organization info
        if transaction_log:
          transaction_log.user_id = user.id
          transaction_log.organization_id = organization_id
          await session.commit()

        return {
          "status": "success",
          "message": "Subscription payment processed successfully",
          "data": {"credits_added": credit_amount, "amount_inr": amount_inr, "transaction_id": str(transaction.id)},
        }

      except Exception as e:
        self.logger.error(f"Error processing subscription payment: {str(e)}")
        return {"status": "error", "message": f"Error processing subscription payment: {str(e)}"}

    # For other subscription events, just log them
    subscription_metadata = {
      "subscription_id": subscription_id,
      "plan_id": plan_id,
      "customer_id": customer_id,
      "status": status,
      "webhook_event": webhook_event.value,
      "processed_at": datetime.utcnow().isoformat(),
    }

    if payment_id and payment_amount:
      subscription_metadata.update({
        "payment_id": payment_id,
        "payment_amount": payment_amount,
      })

    self.logger.info(f"Subscription {webhook_event.value} event logged: {subscription_metadata}")
    return {"status": "success", "message": f"Subscription {webhook_event.value} event processed"}

  async def _get_or_create_razorpay_customer_with_storage(self, user_id: UUID, email: str, name: str, session: Optional[AsyncSession] = None) -> str:
    """Get or create a Razorpay customer and store in customer table."""
    # Use provided session or fall back to self.session
    if not session:
      raise ValueError("No database session available")

    # Get Razorpay payment provider
    razorpay_provider = await PaymentProviderModel.get_by_name("razorpay", session)
    if not razorpay_provider:
      self.logger.error("Razorpay payment provider not found in database")
      raise ValueError("Razorpay payment provider not found")

    # Check if customer exists in our customer table for this user
    customer_query = select(CustomerModel).where(
      CustomerModel.user_id == user_id, CustomerModel.provider_id == razorpay_provider.id, CustomerModel.is_active
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
        CustomerModel.customer_id == razorpay_customer_id, CustomerModel.provider_id == razorpay_provider.id, CustomerModel.is_active
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
      CustomerModel.customer_id == razorpay_customer_id, CustomerModel.provider_id == razorpay_provider.id, CustomerModel.is_active
    )
    check_result = await session.execute(check_existing_query)
    existing_record = check_result.scalar_one_or_none()

    if not existing_record:
      # Store customer in our database
      new_customer = CustomerModel(
        id=uuid4(),
        user_id=user_id,
        provider_id=razorpay_provider.id,
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
