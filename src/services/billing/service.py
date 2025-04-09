# import asyncio
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

# from dependencies.usage import Usage  # Import the Usage dependency at the top of the file
from models import BillingPlanModel, TransactionModel, TransactionStatus, TransactionType, UserModel, WalletModel
from services.__base.acquire import Acquire

# from utils.charge import Charge
from .schema import (
  BillingPlanResponseSchema,
  CheckoutSessionCreateSchema,
  CreditCalculationResponseSchema,
  TransactionWithInvoiceSchema,
  WalletResponseSchema,
)


class BillingService:
  """Billing service for managing credits, transactions, and billable functions."""

  http_exposed = [
    "get=wallet",
    "get=wallet_test",
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

  # async def get_wallet_test(
  #   self,
  #   org_id: UUID,
  #   background_tasks: BackgroundTasks,
  #   qty: int = 1,
  #   session: AsyncSession = Depends(get_db),
  #   # Set background=True to let us handle the charge
  #   usage: dict = Depends(Usage("pdf_extraction", background=True, metadata={"operation": "wallet_test"})),
  # ) -> dict:
  #   """
  #   Test charge creation and incremental update functionality.
  #   """
  #   user_id = UUID(usage["id"])
  #   self.logger.info(f"Starting wallet test: user_id={user_id}, org_id={org_id}, initial_qty={qty}")
  #   self.session = session

  #   try:
  #     # Get the charge from usage dependency
  #     charge = usage["charge"]

  #     # Add our custom background task to handle the charge
  #     background_tasks.add_task(self._test_increment_charge, charge=charge, qty_per_step=1)

  #     return {
  #       "status": "success",
  #       "message": f"Created charge with qty={qty} and scheduled increment steps",
  #       "transaction_id": str(charge.transaction_id),
  #     }

  #   except Exception as e:
  #     self.logger.error(f"Error in wallet test: {str(e)}")
  #     # If there's an error, release the hold manually
  #     try:
  #       charge = usage["charge"]
  #       await charge.delete(reason=f"Error in wallet test: {str(e)}")
  #     except Exception as release_error:
  #       self.logger.error(f"Failed to release charge: {str(release_error)}")
  #     return {"status": "error", "message": str(e)}

  # async def _test_increment_charge(self, charge: Charge, steps: int = 3, qty_per_step: int = 1):
  #   """
  #   Test the quantity increment feature by incrementing in a loop.
  #   """
  #   self.logger.info(f"Starting incremental test for transaction {charge.transaction_id} with {steps} steps")

  #   try:
  #     # Loop through incremental steps
  #     for step in range(1, 5):  # Use steps parameter correctly
  #       await asyncio.sleep(2)  # Wait between increments

  #       try:
  #         self.logger.info(f"Step {step}/{steps}: Incrementing quantity by {qty_per_step}")
  #         await charge.update(additional_metadata={"increment_step": step}, qty_increment=qty_per_step)
  #         self.logger.info(f"Step {step}/{steps} completed successfully")
  #       except HTTPException as e:
  #         # If we get insufficient credits, stop the loop but continue to finalize
  #         self.logger.warning(f"Step {step}/{steps} failed: {str(e.detail)}")
  #         break
  #       except Exception as e:
  #         self.logger.error(f"Step {step}/{steps} failed: {str(e)}")
  #         break

  #     # After all steps (or if we broke out early), finalize the transaction
  #     await asyncio.sleep(2)  # Wait before finalizing
  #     self.logger.info(f"Finalizing transaction {charge.transaction_id}")
  #     await charge.update(additional_metadata={"increment_test": "completed"})
  #     self.logger.info(f"Transaction {charge.transaction_id} finalized successfully")

  #   except Exception as e:
  #     self.logger.error(f"Error during increment test: {str(e)}")
  #     # Clean up by releasing the hold
  #     try:
  #       await charge.delete(reason=f"Increment test failure: {str(e)}")
  #       self.logger.info(f"Released hold for transaction {charge.transaction_id}")
  #     except Exception as cleanup_error:
  #       self.logger.error(f"Failed to clean up: {str(cleanup_error)}")

  async def get_plans(
    self,
    org_id: UUID,
    include_inactive: bool = False,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> List[BillingPlanResponseSchema]:
    """Get all available billing plans."""
    self.logger.debug("Starting get_billing_plans")
    query = select(BillingPlanModel)

    if not include_inactive:
      query = query.where(BillingPlanModel.is_active)

    self.logger.debug(f"Executing query: {query}")
    result = await session.execute(query)
    plans = result.scalars().all()
    self.logger.debug(f"Found {len(plans)} billing plans")

    return [BillingPlanResponseSchema.from_orm(plan) for plan in plans]

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

    if not transaction.stripe_invoice_id:
      self.logger.warning(f"No invoice ID found for transaction {transaction_id}")
      raise HTTPException(status_code=404, detail="No invoice available for this transaction")

    # Get the invoice from Stripe
    invoice = stripe.Invoice.retrieve(transaction.stripe_invoice_id)
    self.logger.debug(f"Retrieved Stripe invoice: {invoice.id} for transaction {transaction_id}")

    # Return the hosted invoice URL
    invoice_url = invoice.hosted_invoice_url or ""
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
    """Get user's transaction history with filtering options."""
    user_id = UUID(user["id"])
    self.logger.debug(f"Fetching transactions for user {user_id} with limit={limit}, offset={offset}")

    # Start with base query
    query = select(TransactionModel).where(TransactionModel.user_id == user_id)
    self.logger.debug(f"Base query: {query}")

    # Apply filters
    if transaction_type:
      self.logger.debug(f"Filtering by transaction type: {transaction_type}")
      query = query.where(TransactionModel.type == transaction_type)

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
    self.logger.debug(f"Total matching transactions: {total_count}")

    # Apply pagination
    query = query.order_by(TransactionModel.created_at.desc()).limit(limit).offset(offset)
    self.logger.debug(f"Final query with pagination: {query}")

    # Execute query
    self.logger.debug("Executing main query")
    result = await session.execute(query)
    transactions = result.scalars().all()
    self.logger.debug(f"Retrieved {len(transactions)} transactions")

    try:
      # Use a list comprehension for elegance and efficiency
      transactions_response = [TransactionWithInvoiceSchema.from_transaction(tx).dict() for tx in transactions]
    except Exception as e:
      self.logger.error(f"Error processing transactions: {e}")
      raise HTTPException(status_code=500, detail="Error processing transactions")

    self.logger.debug(f"Returning {len(transactions_response)} transactions")
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
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, str]:
    """Create a Stripe checkout session for purchasing credits from plan or custom amount."""
    self.logger.debug(f"Raw checkout data received: {checkout_data}")
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
      self.logger.debug(f"Plan-based purchase requested with plan_id: {plan_id}")
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
      self.logger.debug(f"Custom amount purchase requested: ${amount_usd}")
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

    self.logger.info(f"Calculated credits: {credit_amount}")
    # Calculate the unit amount in cents with safeguards
    unit_amount = int(round(amount_usd * 100))

    self.logger.info(f"Calculated unit amount: {unit_amount}")
    # Check if customer_email is provided in the input
    if not customer_email:
      # If email is not in the token payload, we need to fetch the user data
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      user_db = user_result.scalar_one_or_none()
      if user_db:
        customer_email = user_db.email

    self.logger.debug(f"Customer email: {customer_email}")

    # Now use the email (either from input, token, or fetched from database)
    if not customer_email:
      raise HTTPException(
        status_code=400,
        detail="Email is required to create a Stripe customer. Please provide customer_email.",
      )

    # Get or create Stripe customer with the obtained email
    customer = await self._get_or_create_stripe_customer(user_id, customer_email, session)

    self.logger.info("Creating Stripe session", customer=customer.id, credits=credit_amount, amount_usd=amount_usd)

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

    # Store the session ID instead of payment intent
    transaction = TransactionModel(
      id=uuid4(),
      user_id=user_id,
      organization_id=org_id,
      type=TransactionType.CREDIT,
      status=TransactionStatus.PENDING,
      amount_usd=amount_usd,
      credits=credit_amount,
      stripe_payment_intent_id=None,
      stripe_customer_id=customer.id,
      stripe_invoice_id=None,
      description=f"Purchase of {credit_amount} credits",
      transaction_metadata={
        "org_id": str(org_id),
        "checkout_session_id": stripe_session.id,
      },
    )

    session.add(transaction)
    await session.commit()

    self.logger.info(f"Created PENDING transaction {transaction.id} with checkout_session_id: {stripe_session.id}")

    return {"checkout_url": stripe_session.url or "", "session_id": stripe_session.id or ""}

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
        transaction.stripe_payment_intent_id = session_obj.payment_intent
        transaction.stripe_invoice_id = session_obj.invoice

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

    self.logger.debug(f"Looking for transaction with checkout_session_id: {session_id}", query=str(query))

  async def _get_or_create_stripe_customer(
    self,
    user_id: UUID,
    email: Optional[str],
    session: AsyncSession,
  ) -> stripe.Customer:
    """Get or create a Stripe customer for the user."""
    self.logger.debug("Getting or creating Stripe customer", user_id=str(user_id), email=email)

    # Check if customer exists
    transaction_result = await session.execute(
      select(TransactionModel.stripe_customer_id)
      .where(TransactionModel.user_id == user_id)
      .where(TransactionModel.stripe_customer_id.isnot(None))
      .order_by(TransactionModel.created_at.desc())
    )

    result = transaction_result.first()
    customer_id = result[0] if result else None
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
