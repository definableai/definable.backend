from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import stripe
from fastapi import Depends, HTTPException, Request
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC
from models import (
  BillingPlanModel,
  TransactionModel,
  TransactionStatus,
  TransactionType,
  WalletModel,
)
from services.__base.acquire import Acquire

from ..auth.model import UserModel
from .schema import (
  BillingPlanResponseSchema,
  CheckoutSessionCreateSchema,
  CreditCalculationResponseSchema,
  TransactionResponseSchema,
  WalletResponseSchema,
)


class BillingService:
  """Billing service for managing credits, transactions, and billable functions."""

  http_exposed = [
    "get=wallet",
    "get=plans",
    "get=calculate_credits",
    "get=invoice",
    "get=transactions",
    "get=usage_history",
    "post=checkout",
    "post=checkout_cancel",
    "post=stripe_webhook",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.credits_per_usd = 1000  # 1 USD = 1000 credits
    stripe.api_key = settings.stripe_secret_key
    self.session: Optional[AsyncSession] = None
    self.request_id = str(uuid4())  # Add a unique ID per service instance
    self.logger = acquire.logger
    self.logger.info("BillingService initialized", request_id=self.request_id)

  async def get_wallet(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> WalletResponseSchema:
    """Get user's credit balance with spent credits tracking."""
    user_id = UUID(user["id"])
    self.logger.info("Starting wallet retrieval", method="get_wallet", user_id=str(user_id), org_id=str(org_id), request_id=self.request_id)
    self.session = session

    try:
      wallet = await self._get_or_create_wallet(user_id)

      # Calculate low balance indicator
      low_balance = False

      if wallet.balance <= 0:
        self.logger.info("Balance is 0 for user {user_id}, resetting spent credits", user_id=str(user_id))
        low_balance = True
        wallet.credits_spent = 0
        wallet.last_reset_date = datetime.utcnow()
        await session.commit()
        await session.refresh(wallet)
      else:
        total_cycle_credits = wallet.balance + (wallet.credits_spent or 0)
        low_balance = wallet.balance / total_cycle_credits < 0.1 if total_cycle_credits > 0 else False

      self.logger.info("Wallet retrieved successfully", balance=wallet.balance, spent=wallet.credits_spent or 0, user_id=str(user_id))

      return WalletResponseSchema(
        id=UUID(str(wallet.id)),
        user_id=UUID(str(wallet.user_id)),
        balance=wallet.balance,
        hold=wallet.hold,
        credits_spent=wallet.credits_spent or 0,
        low_balance=low_balance,
      )

    except Exception as e:
      self.logger.error("Failed to retrieve wallet", error=str(e), user_id=str(user_id), request_id=self.request_id, exc_info=True)
      raise HTTPException(status_code=500, detail=f"Failed to retrieve wallet information: {str(e)}")

  async def get_plans(
    self,
    org_id: UUID,
    include_inactive: bool = False,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> List[BillingPlanResponseSchema]:
    """Get all available billing plans."""
    try:
      self.logger.info("Starting get_billing_plans")
      query = select(BillingPlanModel)

      if not include_inactive:
        query = query.where(BillingPlanModel.is_active)

      self.logger.info(f"Executing query: {query}")
      result = await session.execute(query)
      plans = result.scalars().all()
      self.logger.info(f"Found {len(plans)} billing plans")

      return [BillingPlanResponseSchema.from_orm(plan) for plan in plans]

    except Exception as e:
      self.logger.error(f"Error in get_plans: {str(e)}", exc_info=True)
      raise HTTPException(status_code=500, detail=f"Failed to retrieve billing plans: {str(e)}")

  async def get_calculate_credits(
    self,
    org_id: UUID,
    amount_usd: float,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> CreditCalculationResponseSchema:
    """Calculate credits for a given USD amount with applicable discounts."""
    # Determine discount percentage based on amount

    discount_percentage = 0.0

    # Base credits (using existing ratio)
    base_credits = int(float(amount_usd) * self.credits_per_usd)

    # Apply discount as bonus credits
    bonus_credits = int(base_credits * discount_percentage / 100)
    total_credits = base_credits + bonus_credits

    return CreditCalculationResponseSchema(
      amount_usd=amount_usd,
      credits=total_credits,
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
    self.logger.info(f"Getting invoice URL for transaction: {transaction_id}, user: {user_id}")

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

    if not transaction.stripe_invoice_id:
      self.logger.warning(f"No invoice ID found for transaction {transaction_id}")
      raise HTTPException(status_code=404, detail="No invoice available for this transaction")

    try:
      # Get the invoice from Stripe
      invoice = stripe.Invoice.retrieve(transaction.stripe_invoice_id)
      self.logger.info(f"Retrieved Stripe invoice: {invoice.id} for transaction {transaction_id}")

      # Return the hosted invoice URL
      invoice_url = invoice.hosted_invoice_url or ""
      return {
        "invoice_url": invoice_url,
        "status": "success",
        "message": "Invoice retrieved successfully",
      }

    except stripe.StripeError as e:
      self.logger.error(f"Stripe invoice error for transaction {transaction_id}: {str(e)}")
      raise HTTPException(status_code=400, detail=f"Failed to retrieve invoice: {str(e)}")

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
    """Get user's transaction history with filtering options."""
    user_id = UUID(user["id"])
    self.logger.info(f"Fetching transactions for user {user_id} with limit={limit}, offset={offset}")

    # Start with base query
    query = select(TransactionModel).where(TransactionModel.user_id == user_id)
    self.logger.debug(f"Base query: {query}")

    # Apply filters
    if transaction_type:
      self.logger.info(f"Filtering by transaction type: {transaction_type}")
      query = query.where(TransactionModel.type == transaction_type)

    if date_from:
      self.logger.info(f"Filtering by date from: {date_from}")
      query = query.where(TransactionModel.created_at >= date_from)

    if date_to:
      self.logger.info(f"Filtering by date to: {date_to}")
      query = query.where(TransactionModel.created_at <= date_to)

    # Get total count for pagination
    self.logger.debug("Executing count query")
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.execute(count_query)
    total_count = total.scalar_one()
    self.logger.info(f"Total matching transactions: {total_count}")

    # Apply pagination
    query = query.order_by(TransactionModel.created_at.desc()).limit(limit).offset(offset)
    self.logger.debug(f"Final query with pagination: {query}")

    # Execute query
    self.logger.debug("Executing main query")
    result = await session.execute(query)
    transactions = result.scalars().all()
    self.logger.info(f"Retrieved {len(transactions)} transactions")

    # Include invoice URL in response if available
    transactions_response = []
    for tx in transactions:
      self.logger.debug(f"Processing transaction {tx.id}, type={tx.type}, status={tx.status}")
      try:
        tx_dict = TransactionResponseSchema.from_orm(tx).dict()
        if tx.stripe_invoice_id:
          self.logger.debug(f"Transaction {tx.id} has invoice: {tx.stripe_invoice_id}")
          tx_dict["has_invoice"] = True
        else:
          tx_dict["has_invoice"] = False
        transactions_response.append(tx_dict)
      except Exception as e:
        self.logger.error(f"Error processing transaction {tx.id}: {str(e)}")
        # Continue processing other transactions

    self.logger.info(f"Returning {len(transactions_response)} transactions")
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

    try:
      # Base query - get all DEBIT transactions
      query = select(TransactionModel).where(
        TransactionModel.user_id == user_id,
        TransactionModel.type == TransactionType.DEBIT,
        # Exclude Stripe payment transactions
        TransactionModel.stripe_payment_intent_id.is_(None),
      )

      # Apply date filters
      if date_from:
        query = query.where(TransactionModel.created_at >= date_from)

      if date_to:
        query = query.where(TransactionModel.created_at <= date_to)

      # Get total count for pagination
      count_query = select(func.count()).select_from(query.subquery())
      total = await session.execute(count_query)
      total_count = total.scalar_one()

      # Apply sorting and pagination
      query = query.order_by(TransactionModel.created_at.desc())
      query = query.limit(limit).offset(offset)

      # Execute query
      result = await session.execute(query)
      transactions = result.scalars().all()

      # Process results
      usage_items = []
      total_credits_used = 0

      if group_by:
        # Manual grouping by the specified field
        grouped_data: Dict[str, Any] = {}

        for tx in transactions:
          metadata = tx.transaction_metadata or {}

          if group_by == "day":
            key = tx.created_at.strftime("%Y-%m-%d")
          elif group_by == "month":
            key = tx.created_at.strftime("%Y-%m")
          elif group_by == "service":
            key = metadata.get("service", "unknown")
          elif group_by == "transaction_type":  # Add grouping by transaction type
            key = tx.type
          else:
            key = "unknown"

          if key not in grouped_data:
            grouped_data[key] = {
              "credits": 0,
              "transactions": 0,
              "types": dict(),  # Initialize as a proper dictionary
            }

          # Check if credits exists and is not None before adding
          if tx.credits is not None:
            grouped_data[key]["credits"] += tx.credits
            total_credits_used += tx.credits

          grouped_data[key]["transactions"] += 1

          # Count by transaction type
          tx_type = tx.type
          if tx_type is not None:
            types_dict = grouped_data[key]["types"]
            if not isinstance(types_dict, dict):
              grouped_data[key]["types"] = dict()
              types_dict = grouped_data[key]["types"]

            tx_type_str = str(tx_type) if tx_type is not None else "unknown"
            if tx_type_str not in types_dict:
              types_dict[tx_type_str] = 0
            types_dict[tx_type_str] += 1

        # Format the grouped data
        for key, data in grouped_data.items():
          credit_amount = data["credits"] or 0
          item = {
            "group_key": key,
            "credits_used": credit_amount,
            "transaction_count": data["transactions"],
            "cost_usd": round(credit_amount / self.credits_per_usd, 2),
          }

          if group_by == "day" or group_by == "month":
            item["period"] = key
          elif group_by == "service":
            item["service"] = key

          usage_items.append(item)
      else:
        # No grouping - return individual transactions
        for tx in transactions:
          metadata = tx.transaction_metadata or {}
          charge_name = metadata.get("charge_name", "Unknown Operation")
          service = metadata.get("service", "Unknown Service")

          item = {
            "id": str(tx.id),
            "timestamp": tx.created_at.isoformat(),
            "description": tx.description or charge_name,
            "service": service,
            "credits_used": tx.credits or 0,
            "cost_usd": round((tx.credits or 0) / self.credits_per_usd, 2),
            "transaction_type": tx.type,
            "status": tx.status,
          }

          # Add relevant metadata fields
          if "action" in metadata:
            item["action"] = metadata["action"]

          if "qty" in metadata:
            item["quantity"] = metadata["qty"]

          usage_items.append(item)
          if tx.credits is not None:
            total_credits_used += tx.credits
          else:
            # If we reach here and tx.credits is None,
            # we need to handle it appropriately
            pass

      # Return the formatted response
      return {
        "usage_history": usage_items,
        "total_credits_used": total_credits_used,
        "total_cost_usd": round(total_credits_used / self.credits_per_usd, 2),
        "pagination": {"total": total_count, "limit": limit, "offset": offset},
      }
    except Exception as e:
      # Log the error and return a helpful message
      self.logger.error(f"Error in get_usage_history: {str(e)}", exc_info=True)
      raise HTTPException(status_code=500, detail=f"Failed to retrieve usage history: {str(e)}")

  async def post_checkout(
    self,
    org_id: UUID,
    checkout_data: Union[CheckoutSessionCreateSchema, Dict[str, Any]],
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, str]:
    """Create a Stripe checkout session for purchasing credits from plan or custom amount."""
    self.logger.info(f"Raw checkout data received: {checkout_data}")
    user_id = UUID(user["id"])
    self.logger.info(f"Creating checkout session for user {user_id}")

    # Handle nested checkout_data structure
    if isinstance(checkout_data, dict) and "checkout_data" in checkout_data:
      checkout_data = checkout_data["checkout_data"]

    # Handle both existing schema and dict input
    amount_usd = None
    plan_id = None
    customer_email = None
    success_url = settings.stripe_success_url
    cancel_url = settings.stripe_cancel_url

    if isinstance(checkout_data, CheckoutSessionCreateSchema):
      plan_id = getattr(checkout_data, "plan_id", None)
      amount_usd = checkout_data.amount_usd
      customer_email = checkout_data.customer_email
    else:
      amount_usd = checkout_data.get("amount_usd")
      plan_id = checkout_data.get("plan_id")
      customer_email = checkout_data.get("customer_email")

    if amount_usd is not None:
      amount_usd = float(str(amount_usd))

    # Handle plan-based purchase
    credit_amount = None
    if plan_id:
      self.logger.info(f"Plan-based purchase requested with plan_id: {plan_id}")
      plan_query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
      plan_result = await session.execute(plan_query)
      plan = plan_result.scalar_one_or_none()

      if not plan:
        self.logger.warning(f"Plan not found: {plan_id}")
        raise HTTPException(status_code=404, detail="Billing plan not found or inactive")

      amount_usd = float(plan.amount_usd)
      credit_amount = int(plan.credits)
      self.logger.info(f"Using plan: {plan.name}, amount: ${amount_usd}, credits: {credit_amount}")
    elif amount_usd:
      self.logger.info(f"Custom amount purchase requested: ${amount_usd}")
      # Calculate credits for custom amount
      calculation = await self.get_calculate_credits(org_id=org_id, amount_usd=amount_usd, session=session, user=user)

      credit_amount = int(calculation.credits)
      self.logger.info(f"Calculated credits for ${amount_usd}: {credit_amount} credits")
    else:
      self.logger.error("Neither amount_usd nor plan_id provided in checkout request")
      raise HTTPException(
        status_code=400,
        detail="Either amount_usd or plan_id must be provided for checkout",
      )

    # Ensure amount_usd is a proper float
    amount_usd = float(amount_usd)

    # Calculate the unit amount in cents with safeguards
    unit_amount = int(round(amount_usd * 100))

    # Rest of implementation follows existing pattern
    try:
      # Check if customer_email is provided in the input
      if not customer_email:
        # If email is not in the token payload, we need to fetch the user data
        user_query = select(UserModel).where(UserModel.id == user_id)
        user_result = await session.execute(user_query)
        user_db = user_result.scalar_one_or_none()
        if user_db:
          customer_email = user_db.email

      # Now use the email (either from input, token, or fetched from database)
      if not customer_email:
        raise HTTPException(
          status_code=400,
          detail="Email is required to create a Stripe customer. Please provide customer_email.",
        )

      # Get or create Stripe customer with the obtained email
      customer = await self._get_or_create_stripe_customer(user_id, customer_email, session)
      # Get or create Stripe customer

      self.logger.debug("About to create Stripe session", customer=customer.id, credits=credit_amount, amount_usd=amount_usd)

      # Convert URLs to strings for Stripe
      success_url_str = str(success_url) if success_url else ""
      cancel_url_str = str(cancel_url) if cancel_url else ""

      # Create Stripe checkout session with invoice
      stripe_session = stripe.checkout.Session.create(
        customer=customer.id,
        payment_method_types=["card"],
        line_items=[
          {
            "price_data": {
              "currency": "usd",
              "product_data": {
                "name": f"Purchase {credit_amount} Credits",
              },
              "unit_amount": unit_amount,  # Safe integer in cents
            },
            "quantity": 1,
          }
        ],
        mode="payment",
        success_url=success_url_str,
        cancel_url=cancel_url_str,
        invoice_creation={"enabled": True},  # Enable invoice creation
        metadata={
          "user_id": str(user_id),
          "org_id": str(org_id),
          "credits": str(credit_amount),
        },
      )

      # Store the session ID instead of payment intent
      transaction = TransactionModel(
        user_id=user_id,
        type=TransactionType.CREDIT,
        status=TransactionStatus.PENDING,
        amount_usd=amount_usd,
        credits=credit_amount,
        stripe_payment_intent_id=None,  # We'll update this later in the webhook
        stripe_customer_id=customer.id,
        stripe_invoice_id=None,  # This will also be updated in the webhook
        description=f"Purchase of {credit_amount} credits",
        transaction_metadata={
          "org_id": str(org_id),
          "checkout_session_id": stripe_session.id,  # Store the session ID
        },
      )

      session.add(transaction)
      await session.commit()

      # Ensure URL and session ID are not None
      checkout_url = stripe_session.url or ""
      session_id = stripe_session.id or ""

      return {"checkout_url": checkout_url, "session_id": session_id}

    except Exception as e:
      self.logger.error(f"Error creating Stripe session: {str(e)}")
      await session.rollback()
      # Handle Stripe errors with 400 status code, other errors with 500
      if "stripe" in str(e).lower():
        raise HTTPException(status_code=400, detail=str(e))

      raise HTTPException(status_code=500, detail="Internal server error")

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

    Args:
        org_id: Organization ID
        session_id: The Stripe checkout session ID
        cancel_reason: Reason for cancellation (optional)
        session: Database session
        user: Authenticated user info

    Returns:
        Status information
    """
    self.session = session
    user_id = UUID(user["id"])
    self.logger.info(f"Processing user-cancelled checkout session: {session_id} for user {user_id}")

    try:
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

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error handling checkout cancellation: {str(e)}", exc_info=True)
      await session.rollback()
      raise HTTPException(status_code=500, detail=f"Failed to process cancellation: {str(e)}")

  async def post_stripe_webhook(self, request: Request, session: AsyncSession = Depends(get_db)) -> None:
    """Handle Stripe webhook events."""
    self.session = session
    log_context = {"method": "post_stripe_webhook", "request_id": self.request_id}

    # Get the webhook payload and signature header
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
      # Verify webhook signature
      event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
      log_context["event_id"] = event.id
      log_context["event_type"] = event.type

      self.logger.info("Processing Stripe webhook event", extra=log_context)

      # Handle checkout.session.completed event
      if event.type == "checkout.session.completed":
        session_obj = event.data.object
        self.logger.info(f"Processing completed checkout session: {session_obj.id}")

        # Get metadata from the session
        metadata = session_obj.metadata
        user_id = metadata.get("user_id")
        credit_amount = metadata.get("credits")

        if not user_id or not credit_amount:
          self.logger.error("Missing user_id or credits in session metadata")

        # Convert to proper types
        user_id = UUID(user_id)
        credit_amount = int(credit_amount)

        # Find the pending transaction with matching session ID
        query = (
          select(TransactionModel)
          .where(and_(text("transaction_metadata->>'checkout_session_id' = :session_id"), TransactionModel.status == TransactionStatus.PENDING))
          .params(session_id=session_obj.id)
        )

        result = await session.execute(query)
        transaction = result.scalar_one_or_none()

        self.logger.info(f"Found transaction: {session_obj}")

        if transaction:
          # Update transaction status
          transaction.status = TransactionStatus.COMPLETED
          transaction.stripe_payment_intent_id = session_obj.payment_intent
          transaction.stripe_invoice_id = session_obj.invoice

          # Add credits to user's wallet
          await self._add_credits(user_id, credit_amount, f"Purchase of {credit_amount} credits", session)

          await session.commit()
          self.logger.info(f"Added {credit_amount} credits to user {user_id}")
        else:
          self.logger.error(f"No matching transaction found for session: {session_obj.id}")

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

      # Handle other webhook events as needed
      # ...

    except stripe.SignatureVerificationError:
      self.logger.error("Invalid webhook signature", exc_info=True, extra=log_context)
      raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
      self.logger.error(f"Webhook handler error: {str(e)}", exc_info=True, extra=log_context)
      raise HTTPException(status_code=500, detail=str(e))

  async def _get_or_create_stripe_customer(
    self,
    user_id: UUID,
    email: Optional[str],
    session: AsyncSession,
  ) -> stripe.Customer:
    """Get or create a Stripe customer for the user."""
    # Check if customer exists
    transaction_result = await session.execute(
      select(TransactionModel.stripe_customer_id)
      .where(TransactionModel.user_id == user_id)
      .where(TransactionModel.stripe_customer_id.isnot(None))
      .order_by(TransactionModel.created_at.desc())
    )

    # Change this line:
    # customer_id = transaction_result.scalar_one_or_none()

    # To this:
    result = transaction_result.first()
    customer_id = result[0] if result else None

    if customer_id:
      return stripe.Customer.retrieve(customer_id)

    # Create new customer
    if not email:
      raise HTTPException(status_code=400, detail="Email is required to create a customer")

    customer = stripe.Customer.create(email=email, metadata={"user_id": str(user_id)})
    return customer

  async def _get_or_create_wallet(self, user_id: UUID, session: Optional[AsyncSession] = None) -> WalletModel:
    """Get or create user's wallet."""
    try:
      self.logger.info(f"Triggered _get_or_create_wallet for user {user_id}")
      # Use provided session or fall back to self.session
      db_session = session or self.session
      if not db_session:
        raise ValueError("No database session available")

      # First verify the user exists
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await db_session.execute(user_query)
      user = user_result.scalar_one_or_none()

      if not user:
        self.logger.error(f"User {user_id} not found in database")
        raise HTTPException(status_code=404, detail="User not found")

      # Then try to get existing wallet
      query = select(WalletModel).where(WalletModel.user_id == user_id)
      self.logger.info("BillingService: Executing wallet query")
      result = await db_session.execute(query)
      self.logger.info("BillingService: Query executed")
      wallet = result.scalar_one_or_none()
      self.logger.info(f"BillingService: Wallet query result: {wallet}")

      if wallet:
        self.logger.info(f"Found existing wallet for user {user_id}: {wallet.balance}")
        return wallet

      self.logger.info(f"No wallet found for user {user_id}, creating new entry")
      # Create new wallet entry
      new_wallet = WalletModel(user_id=user_id, balance=0, hold=0, credits_spent=0)

      self.logger.info(f"Creating new wallet {new_wallet}")

      db_session.add(new_wallet)
      self.logger.info(f"Adding new wallet {new_wallet}")
      await db_session.commit()
      self.logger.info(f"Committing new wallet {new_wallet}")
      await db_session.refresh(new_wallet)
      self.logger.info(f"Successfully created new wallet for user {user_id}")
      return new_wallet

    except Exception as e:
      self.logger.error(f"Error in _get_or_create_wallet: {str(e)}")
      if db_session:
        await db_session.rollback()
      raise HTTPException(status_code=500, detail="Error managing wallet")

  async def _add_credits(self, user_id: UUID, amount: int, description: Optional[str] = None, session: Optional[AsyncSession] = None) -> bool:
    """Add credits to user's wallet."""
    if amount <= 0:
      return True

    db_session = session or self.session
    if not db_session:
      raise ValueError("No database session available")

    wallet = await self._get_or_create_wallet(user_id, db_session)

    # Create CREDIT transaction
    transaction = TransactionModel(
      user_id=user_id,
      type=TransactionType.CREDIT,
      status=TransactionStatus.COMPLETED,
      credits=amount,
      description=description or "Credit purchase",
    )

    # Add to session
    db_session.add(transaction)

    # Update balance
    wallet.balance += amount

    await db_session.commit()

    return True
