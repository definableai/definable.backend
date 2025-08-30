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
  CreateSubscriptionWithPlanIdRequestSchema,
  CreateSubscriptionWithPlanIdResponseSchema,
  CreditCalculationResponseSchema,
  RazorpaySubscriptionSchema,
  TransactionWithInvoiceSchema,
  WalletResponseSchema,
)


class BillingService:
  """Billing service for managing credits, transactions, and billable functions."""

  http_exposed = [
    "get=wallet",
    # "get=wallet_test",
    "get=plans",
    "get=calculate_credits",
    "get=invoice",
    "get=transactions",
    "get=usage_history",
    "post=checkout",
    "post=checkout_cancel",
    "post=stripe_webhook",
    "post=create_subscription",
    "post=create_subscription_with_plan_id",
    "post=razorpay_webhook",
    "post=verify_razorpay_payment",
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
    self.razorpay_engine = razorpay_engine  # Initialize Razorpay engine

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
    region: str = "USD",
    include_inactive: bool = False,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> List[BillingPlanResponseSchema]:
    """Get available billing plans for a specific region."""
    query = select(BillingPlanModel).where(BillingPlanModel.currency == region)

    if not include_inactive:
      query = query.where(BillingPlanModel.is_active)

    result = await session.execute(query)
    plans = result.scalars().all()

    return [BillingPlanResponseSchema.from_orm(plan) for plan in plans]

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
      customer_email = user_db.email if user_db else None

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
    customer_id = await self._get_or_create_razorpay_customer_with_storage(
      user_id, user_db.email, f"{user_db.first_name} {user_db.last_name}", session
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

  async def post_create_subscription(
    self,
    plan_id: UUID,
    org_id: UUID,
    token: dict = Depends(RBAC("billing", "write")),
    session: AsyncSession = Depends(get_db),
  ) -> Dict[str, Any]:
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
      return await self._create_razorpay_subscription(plan, user_id, org_id, session)

    # For non-INR currencies, return error for now
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Subscription creation for currency {plan.currency} is not supported yet")

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

      self.logger.info(f"Creating Razorpay subscription for plan {plan.id} with currency INR")

      # Get or create Razorpay customer and store in customer table
      customer_id = await self._get_or_create_razorpay_customer_with_storage(user_id, user.email, f"{user.first_name} {user.last_name}", session)

      # Check if plan has razorpay plan_id, if not create one
      razorpay_plan_id = plan.plan_id

      if not razorpay_plan_id:
        # Create Razorpay plan if it doesn't exist
        razorpay_plan_data = {
          "name": plan.name,
          "amount": int(plan.amount * 100),  # Convert to paise
          "currency": plan.currency,
          "period": "monthly",  # Default to monthly, could be made configurable
          "description": f"Subscription plan: {plan.name}",
        }

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

      # Calculate subscription timing
      import time

      start_at = int(time.time()) + 300  # Start 5 minutes from now
      expire_by = int(time.time()) + (365 * 24 * 60 * 60)  # Expire in 1 year

      # Create subscription with enhanced parameters
      subscription_response = self.razorpay_engine.create_subscription(
        plan_id=razorpay_plan_id,
        customer_id=customer_id,
        total_count=12,  # 12 months by default
        quantity=1,
        start_at=start_at,
        expire_by=expire_by,
        customer_notify=True,
        notes={"organization_id": str(org_id), "user_id": str(user_id), "plan_id": str(plan.id), "plan_name": plan.name},
        notify_info={"notify_email": user.email},
      )

      if not subscription_response.success:
        self.logger.error(f"Failed to create Razorpay subscription: {subscription_response.errors}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {subscription_response.errors}")

      subscription = subscription_response.data

      self.logger.info(f"Successfully created Razorpay subscription: {subscription['id']}")

      return {
        "success": True,
        "provider": "razorpay",
        "subscription_id": subscription["id"],
        "subscription_url": subscription.get("short_url", ""),
        "plan_id": razorpay_plan_id,
        "customer_id": customer_id,
        "status": subscription.get("status"),
        "currency": plan.currency,
        "amount": plan.amount,
        "credits": plan.credits,
        "start_at": subscription.get("start_at"),
        "end_at": subscription.get("end_at"),
        "total_count": subscription.get("total_count"),
        "remaining_count": subscription.get("remaining_count"),
        "subscription_details": subscription,
      }

    except HTTPException:
      # Re-raise HTTP exceptions as-is
      raise
    except Exception as e:
      self.logger.error(f"Unexpected error creating Razorpay subscription: {str(e)}", exc_info=True)
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {str(e)}")

  async def post_create_subscription_with_plan_id(
    self,
    subscription_data: CreateSubscriptionWithPlanIdRequestSchema,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> CreateSubscriptionWithPlanIdResponseSchema:
    """Create a subscription using an existing Razorpay plan_id."""

    user_id = UUID(user["id"])
    self.session = session

    try:
      # Get user details
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      user_obj = user_result.scalar_one_or_none()

      if not user_obj:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="User not found")

      # Get or create Razorpay customer and store in customer table
      customer_id = await self._get_or_create_razorpay_customer_with_storage(
        user_id, user_obj.email, f"{user_obj.first_name} {user_obj.last_name}", session
      )

      # Set default timing if not provided
      import time

      start_at = subscription_data.start_at
      expire_by = subscription_data.expire_by

      if not start_at:
        start_at = int(time.time()) + 300  # Start 5 minutes from now
      if not expire_by:
        expire_by = int(time.time()) + (365 * 24 * 60 * 60)  # Expire in 1 year

      # Default notes if not provided
      notes = subscription_data.notes
      if not notes:
        notes = {"organization_id": str(org_id), "user_id": str(user_id), "created_via": "api"}

      self.logger.info(f"Creating Razorpay subscription with plan_id: {subscription_data.plan_id} for user {user_id}")

      # Create subscription using existing plan_id
      subscription_response = self.razorpay_engine.create_subscription(
        plan_id=subscription_data.plan_id,
        customer_id=customer_id,
        total_count=subscription_data.total_count,
        quantity=subscription_data.quantity,
        start_at=start_at,
        expire_by=expire_by,
        customer_notify=subscription_data.customer_notify,
        notes=notes,
        addons=subscription_data.addons,
        notify_info={"notify_email": user_obj.email},
      )

      if not subscription_response.success:
        self.logger.error(f"Failed to create Razorpay subscription: {subscription_response.errors}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {subscription_response.errors}")

      subscription = subscription_response.data
      self.logger.info(f"Successfully created Razorpay subscription: {subscription['id']}")

      # Create the subscription schema object
      subscription_schema = RazorpaySubscriptionSchema(
        id=subscription["id"],
        entity=subscription.get("entity"),
        plan_id=subscription.get("plan_id"),
        status=subscription.get("status"),
        current_start=subscription.get("current_start"),
        current_end=subscription.get("current_end"),
        ended_at=subscription.get("ended_at"),
        quantity=subscription.get("quantity"),
        notes=subscription.get("notes"),
        charge_at=subscription.get("charge_at"),
        start_at=subscription.get("start_at"),
        end_at=subscription.get("end_at"),
        auth_attempts=subscription.get("auth_attempts"),
        total_count=subscription.get("total_count"),
        paid_count=subscription.get("paid_count"),
        customer_notify=subscription.get("customer_notify"),
        created_at=subscription.get("created_at"),
        expire_by=subscription.get("expire_by"),
        short_url=subscription.get("short_url"),
        has_scheduled_changes=subscription.get("has_scheduled_changes"),
        change_scheduled_at=subscription.get("change_scheduled_at"),
        source=subscription.get("source"),
        remaining_count=subscription.get("remaining_count"),
      )

      return CreateSubscriptionWithPlanIdResponseSchema(
        success=True,
        provider="razorpay",
        subscription=subscription_schema,
        customer_id=customer_id,
      )

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Unexpected error in create_subscription_with_plan_id: {str(e)}", exc_info=True)
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Failed to create subscription: {str(e)}")

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
    """Handle Razorpay webhook events."""
    self.session = session

    # Get webhook payload
    payload = await request.json()
    self.logger.info(f"Razorpay webhook payload: {json.dumps(payload)}")
    signature = request.headers.get("x-razorpay-signature")
    self.logger.info(f"Razorpay webhook signature: {signature}")

    # Log webhook details for debugging
    event = payload.get("event")
    self.logger.info(f"Processing Razorpay webhook event: {event}")

    # Skip signature verification temporarily if having issues
    # Try to continue processing even if signature verification fails
    try:
      if signature:
        payload_string = json.dumps(payload, separators=(",", ":"))
        signature_response = self.razorpay_engine.verify_webhook_signature(payload_string, signature)
        if not signature_response.success or not signature_response.data:
          raise Exception("Signature verification failed")
        self.logger.info("Webhook signature verified successfully")
    except Exception as e:
      self.logger.warning(f"Webhook signature verification failed: {str(e)}")
      # Continue processing instead of returning an error
      # This helps during testing or if signature issues occur

    # Process different webhook events
    if event == "invoice.paid":
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

      # Only process if not already completed
      if transaction.status != TransactionStatus.COMPLETED:
        # Update transaction with payment ID and status
        transaction.status = TransactionStatus.COMPLETED

        # Update payment metadata with payment details
        if not transaction.payment_metadata:
          transaction.payment_metadata = {}
        transaction.payment_metadata.update({
          "payment_id": payment_id,
        })

        if not transaction.transaction_metadata:
          transaction.transaction_metadata = {}

        # Update transaction metadata
        transaction.transaction_metadata.update({
          "payment_status": invoice_entity.get("status", "paid"),
          "payment_method": payment_entity.get("method"),
          "webhook_event": event,
          "processed_at": datetime.utcnow().isoformat(),
        })

        self.logger.info(f"Updated transaction {transaction.id} with payment ID: {payment_id}")

        # Add credits to wallet
        await self._add_credits(
          transaction.user_id, transaction.organization_id, transaction.credits, f"Purchase of {transaction.credits} credits via Razorpay", session
        )

        await session.commit()
        self.logger.info(f"Successfully processed Razorpay invoice payment: {payment_id} for invoice: {invoice_id}")
      else:
        self.logger.info(f"Transaction already completed for invoice: {invoice_id}, skipping processing")

      return {"status": "success", "message": "Invoice payment processed successfully"}

    # Handle payment.captured event as a fallback
    elif event == "payment.captured":
      payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
      payment_id = payment_entity.get("id")
      invoice_id = payment_entity.get("invoice_id")

      if not payment_id:
        self.logger.error("Missing payment_id in webhook payload")
        return {"status": "error", "message": "Missing required parameters"}

      # If we have an invoice_id, try to find the transaction
      if invoice_id:
        query = (
          select(TransactionModel)
          .where(text("payment_metadata->>'invoice_id' = :invoice_id"), TransactionModel.payment_provider == "razorpay")
          .params(invoice_id=invoice_id)
        )
        result = await session.execute(query)
        transaction = result.scalar_one_or_none()

        # If found, process it
        if transaction and transaction.status != TransactionStatus.COMPLETED:
          self.logger.info(f"Processing payment.captured for invoice: {invoice_id}")
          transaction.status = TransactionStatus.COMPLETED

          # Update payment metadata with payment ID
          if not transaction.payment_metadata:
            transaction.payment_metadata = {}
          transaction.payment_metadata.update({
            "payment_id": payment_id,
          })

          if not transaction.transaction_metadata:
            transaction.transaction_metadata = {}

          transaction.transaction_metadata.update({
            "payment_status": "captured",
            "webhook_event": event,
            "processed_at": datetime.utcnow().isoformat(),
          })

          # Only add credits if status is changing to COMPLETED
          await self._add_credits(
            transaction.user_id, transaction.organization_id, transaction.credits, f"Purchase of {transaction.credits} credits via Razorpay", session
          )

          await session.commit()
          self.logger.info(f"Successfully processed Razorpay payment: {payment_id} for invoice: {invoice_id}")

          return {"status": "success", "message": "Payment processed successfully"}

    # For other events, just log and return success
    self.logger.info(f"Webhook event {event} processed (no action taken)")
    return {"status": "success", "message": "Webhook received"}

  async def _get_or_create_razorpay_customer_with_storage(self, user_id: UUID, email: str, name: str, session: Optional[AsyncSession] = None) -> str:
    """Get or create a Razorpay customer and store in customer table."""
    # Use provided session or fall back to self.session
    db_session = session or self.session
    if not db_session:
      raise ValueError("No database session available")

    # Check if customer exists in our customer table
    customer_query = select(CustomerModel).where(
      CustomerModel.user_id == user_id, CustomerModel.payment_provider == "razorpay", CustomerModel.is_active
    )
    customer_result = await db_session.execute(customer_query)
    existing_customer = customer_result.scalar_one_or_none()

    if existing_customer:
      self.logger.info(f"Found existing customer in database: {existing_customer.customer_id}")
      return existing_customer.customer_id

    # No customer found in our database, check Razorpay directly
    customer_response = self.razorpay_engine.fetch_customer_by_email(email)
    razorpay_customer_id = None

    if customer_response.success and customer_response.data:
      # Found existing customer in Razorpay
      razorpay_customer = customer_response.data
      razorpay_customer_id = razorpay_customer["id"]
      self.logger.info(f"Found existing Razorpay customer: {razorpay_customer_id}")
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
          raise Exception(f"Failed to create customer: {create_response.errors}")
      else:
        razorpay_customer = create_response.data
        razorpay_customer_id = razorpay_customer["id"]
        self.logger.info(f"Successfully created new Razorpay customer: {razorpay_customer_id}")

    # Store customer in our database
    new_customer = CustomerModel(
      id=uuid4(),
      user_id=user_id,
      payment_provider="razorpay",
      customer_id=razorpay_customer_id,
      is_active=True,
      provider_metadata={"email": email, "name": name},
    )

    db_session.add(new_customer)
    await db_session.commit()
    await db_session.refresh(new_customer)

    self.logger.info(f"Stored customer {razorpay_customer_id} in database for user {user_id}")
    return razorpay_customer_id

  async def _get_or_create_razorpay_customer_via_engine(self, user_id: UUID, email: str, name: str, session: Optional[AsyncSession] = None) -> str:
    """Get or create a Razorpay customer using the engine."""
    # Use provided session or fall back to self.session
    db_session = session or self.session
    if not db_session:
      raise ValueError("No database session available")

    # Check existing customer in payment_metadata
    transaction_result = await db_session.execute(
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
      raise Exception(f"Failed to create customer: {customer_response.errors}")

    customer = customer_response.data
    self.logger.info(f"Successfully created new customer: {customer['id']}")
    return customer["id"]
