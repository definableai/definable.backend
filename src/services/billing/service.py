import stripe
from fastapi import HTTPException, Depends, Body, Request
from sqlalchemy import select, update, and_, func, String
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime

from src.config.settings import settings
from src.database import get_db
from src.services.__base.acquire import Acquire
from src.dependencies.security import RBAC
from loguru import logger
from ..auth.model import UserModel
from .decorators import track_request

from .model import (
    CreditBalanceModel,
    TransactionModel,
    TransactionType,
    TransactionStatus,
    BillingPlanModel,
    ServiceModel,
    UserUsageModel,
)
from .schema import (
    CheckoutSessionCreateSchema,
    CreditBalanceResponseExtendedSchema,
    TransactionResponseSchema,
    BillingPlanResponseSchema,
    BillingPlanSchema,
    CreditCalculationResponseSchema,
    BillingPlanCreateSchema,
    ServiceCreateSchema,
    ServiceResponseSchema,
)


class BillingService:
    """Billing service for managing credits, transactions, and billable functions."""

    http_exposed = [
        "get=balance",
        "get=plans",
        "get=calculate_credits",
        "get=invoice",
        "get=transactions",
        "post=create_checkout_session",
        "post=register_service",
        "post=stripe_webhook",
        "post=create_billing_plan",
        "put=update_billing_plan",
        "delete=billing_plan",
    ]

    def __init__(self, acquire: Acquire):
        """Initialize service."""
        self.acquire = acquire
        self.utils = acquire.utils
        self.credits_per_usd = 1000  # 1 USD = 1000 credits
        stripe.api_key = settings.stripe_secret_key

    # GROUP 1: CREDIT MANAGEMENT
    @track_request("get-credit")
    async def get_balance(
        self,
        org_id: UUID,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "read")),
    ) -> CreditBalanceResponseExtendedSchema:
        """Get user's credit balance with spent credits tracking."""
        try:
            user_id = UUID(user["id"])
            logger.info(f"Fetching credit balance for user {user_id}")
            balance = await self._get_or_create_balance(user_id, session)

            # Calculate low balance indicator (less than 10% of total purchased in this cycle)
            low_balance = False
            if balance.balance > 0:
                total_cycle_credits = balance.balance + (balance.credits_spent or 0)
                low_balance = (
                    balance.balance / total_cycle_credits < 0.1
                    if total_cycle_credits > 0
                    else False
                )
                logger.debug(
                    f"Low balance calculation: {balance.balance}/{total_cycle_credits} = {balance.balance / total_cycle_credits if total_cycle_credits > 0 else 0}"
                )

            logger.info(
                f"Successfully retrieved balance for user {user_id}: {balance.balance} credits"
            )
            return CreditBalanceResponseExtendedSchema(
                balance=balance.balance,
                balance_usd=balance.balance / self.credits_per_usd,
                credits_spent=balance.credits_spent or 0,
                last_reset_date=balance.last_reset_date,
                low_balance=low_balance,
            )
        except Exception as e:
            logger.error(
                f"Error fetching credit balance for user {user['id']}: {str(e)}",
                exc_info=True,
            )
            # Return fallback data
            return CreditBalanceResponseExtendedSchema(
                balance=0,
                balance_usd=0,
                credits_spent=0,
                last_reset_date=datetime.utcnow(),
                low_balance=False,
            )

    async def _deduct_credits(
        self, user_id: UUID, credits: int, session: AsyncSession
    ) -> bool:
        """Deduct credits from user's balance with spent tracking."""
        if credits <= 0:
            logger.debug(
                f"No credits to deduct for user {user_id} (requested: {credits})"
            )
            return True

        try:
            # Get current balance
            logger.info(f"Attempting to deduct {credits} credits for user {user_id}")
            balance = await self._get_or_create_balance(user_id, session)

            # Ensure balance is sufficient
            if balance.balance < credits:
                logger.warning(
                    f"Insufficient balance for user {user_id}: {balance.balance} < {credits}"
                )
                return False

            # Track spent credits
            if balance.credits_spent is None:
                balance.credits_spent = 0

            balance.credits_spent += credits
            balance.balance -= credits

            logger.info(
                f"Updated balance for user {user_id}: {balance.balance} credits remaining, {balance.credits_spent} credits spent"
            )

            # If balance reaches zero, reset spent counter
            if balance.balance <= 0:
                logger.info(
                    f"Balance reached zero for user {user_id}, resetting spent counter"
                )
                balance.credits_spent = 0
                balance.last_reset_date = datetime.utcnow()

            await session.commit()
            return True

        except Exception as e:
            logger.error(
                f"Error deducting credits for user {user_id}: {str(e)}", exc_info=True
            )
            await session.rollback()
            raise HTTPException(
                status_code=500,
                detail="Failed to deduct credits: Internal server error",
            )

    async def _get_or_create_balance(
        self, user_id: UUID, session: AsyncSession
    ) -> CreditBalanceModel:
        """Get or create user's credit balance."""
        try:
            logger.info(f"Triggered _get_or_create_balance for user {user_id}")
            # First verify the user exists
            user_query = select(UserModel).where(UserModel.id == user_id)
            user_result = await session.execute(user_query)
            user = user_result.scalar_one_or_none()

            if not user:
                logger.error(f"User {user_id} not found in database")
                raise HTTPException(status_code=404, detail="User not found")

            # Then try to get existing balance
            query = select(CreditBalanceModel).where(
                CreditBalanceModel.user_id == user_id
            )
            logger.info("BillingService line 1173")
            result = await session.execute(query)
            logger.info("BillingService line 1175")
            balance = result.scalar_one_or_none()
            logger.info("BillingService line 1177", balance)
            print("BillingService line 1178 ======", balance)

            if balance:
                logger.info(
                    f"Found existing balance for user {user_id}: {balance.balance}"
                )
                return balance

            logger.info(f"No balance found for user {user_id}, creating new entry")
            # Create new balance entry
            new_balance = CreditBalanceModel(
                user_id=user_id, balance=0, credits_spent=0
            )

            logger.info(f"Creating new balance {new_balance}")

            session.add(new_balance)
            logger.info(f"Adding new balance {new_balance}")
            await session.commit()
            logger.info(f"Comminting new balance {new_balance}")
            await session.refresh(new_balance)
            logger.info(f"Successfully created new credit balance for user {user_id}")
            return new_balance

        except Exception as e:
            logger.error(f"Error in _get_or_create_balance: {str(e)}")
            await session.rollback()
            raise HTTPException(status_code=500, detail="Error managing credit balance")

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

        # Example discount tiers

        # if float(amount_usd) >= 500:
        #     discount_percentage = 15.0
        # elif float(amount_usd) >= 200:
        #     discount_percentage = 10.0
        # elif float(amount_usd) >= 100:
        #     discount_percentage = 5.0

        # Base credits (using existing ratio)
        base_credits = int(float(amount_usd) * self.credits_per_usd)

        # Apply discount as bonus credits
        bonus_credits = int(base_credits * discount_percentage / 100)
        total_credits = base_credits + bonus_credits

        return CreditCalculationResponseSchema(
            amount_usd=amount_usd,
            credits=total_credits,
        )

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

        # Start with base query
        query = select(TransactionModel).where(TransactionModel.user_id == user_id)

        # Apply filters
        if transaction_type:
            query = query.where(TransactionModel.type == transaction_type)

        if date_from:
            query = query.where(TransactionModel.created_at >= date_from)

        if date_to:
            query = query.where(TransactionModel.created_at <= date_to)

        # Get total count for pagination
        count_query = select(func.count()).select_from(query.subquery())
        total = await session.execute(count_query)
        total_count = total.scalar_one()

        # Apply pagination
        query = (
            query.order_by(TransactionModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        # Execute query
        result = await session.execute(query)
        transactions = result.scalars().all()

        # Include invoice URL in response if available
        transactions_response = []
        for tx in transactions:
            tx_dict = TransactionResponseSchema.from_orm(tx).dict()
            if tx.stripe_invoice_id:
                tx_dict["has_invoice"] = True
            else:
                tx_dict["has_invoice"] = False
            transactions_response.append(tx_dict)

        return {
            "transactions": transactions_response,
            "pagination": {"total": total_count, "limit": limit, "offset": offset},
        }

    # GROUP 3: STRIPE INTEGRATION
    async def post_create_checkout_session(
        self,
        org_id: UUID,
        checkout_data: Union[CheckoutSessionCreateSchema, Dict[str, Any]],
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "write")),
    ) -> Dict[str, str]:
        """Create a Stripe checkout session for purchasing credits from plan or custom amount."""
        user_id = UUID(user["id"])
        logger.info(f"Creating checkout session for user {user_id}")

        # Handle nested checkout_data structure
        if isinstance(checkout_data, dict) and "checkout_data" in checkout_data:
            checkout_data = checkout_data["checkout_data"]

        # Handle both existing schema and dict input
        amount_usd = None
        plan_id = None
        success_url = None
        cancel_url = None
        customer_email = None

        if isinstance(checkout_data, CheckoutSessionCreateSchema):
            plan_id = getattr(checkout_data, "plan_id", None)
            amount_usd = checkout_data.amount_usd
            success_url = checkout_data.success_url
            cancel_url = checkout_data.cancel_url
            customer_email = checkout_data.customer_email
        else:
            amount_usd = checkout_data.get("amount_usd")
            plan_id = checkout_data.get("plan_id")
            success_url = checkout_data.get("success_url")
            cancel_url = checkout_data.get("cancel_url")
            customer_email = checkout_data.get("customer_email")

        # Handle plan-based purchase
        credits = None
        if plan_id:
            logger.info(f"Plan-based purchase requested with plan_id: {plan_id}")
            plan_query = select(BillingPlanModel).where(BillingPlanModel.id == plan_id)
            plan_result = await session.execute(plan_query)
            plan = plan_result.scalar_one_or_none()

            if not plan:
                logger.warning(f"Plan not found: {plan_id}")
                raise HTTPException(
                    status_code=404, detail="Billing plan not found or inactive"
                )

            amount_usd = plan.amount_usd
            credits = plan.credits
            logger.info(
                f"Using plan: {plan.name}, amount: ${amount_usd}, credits: {credits}"
            )
        elif amount_usd:
            logger.info(f"Custom amount purchase requested: ${amount_usd}")
            # Calculate credits for custom amount
            calculation = await self.get_calculate_credits(
                org_id=org_id, amount_usd=amount_usd, session=session, user=user
            )

            credits = calculation.credits
            logger.info(f"Calculated credits for ${amount_usd}: {credits} credits")
        else:
            logger.error("Neither amount_usd nor plan_id provided in checkout request")
            raise HTTPException(
                status_code=400,
                detail="Either amount_usd or plan_id must be provided for checkout",
            )

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
            customer = await self._get_or_create_stripe_customer(
                user_id, customer_email, session
            )
            # Get or create Stripe customer

            logger.debug(
                "About to create Stripe session with params: %s",
                {"customer": customer.id, "credits": credits, "amount_usd": amount_usd},
            )

            # Add validation for amount_usd
            if isinstance(amount_usd, str):
                amount_usd = float(
                    amount_usd.split(".")[0] + "." + amount_usd.split(".")[1]
                )
            amount_usd = float(amount_usd)  # Ensure it's a float

            # Create Stripe checkout session with invoice
            stripe_session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=["card"],
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": f"Purchase {credits} Credits",
                            },
                            "unit_amount": int(amount_usd * 100),  # Stripe uses cents
                        },
                        "quantity": 1,
                    }
                ],
                mode="payment",
                success_url=success_url,
                cancel_url=cancel_url,
                invoice_creation={"enabled": True},  # Enable invoice creation
                metadata={
                    "user_id": str(user_id),
                    "org_id": str(org_id),
                    "credits": credits,
                },
            )

            logger.debug("Stripe session created successfully: %s", stripe_session.id)

            # Get invoice ID if available
            invoice_id = None
            if hasattr(stripe_session, "invoice") and stripe_session.invoice:
                invoice_id = stripe_session.invoice

            # Create pending transaction
            # Store the session ID instead of payment intent
            transaction = TransactionModel(
                user_id=user_id,
                type=TransactionType.CREDIT_PURCHASE,
                status=TransactionStatus.PENDING,
                amount_usd=amount_usd,
                credits=credits,
                stripe_payment_intent_id=None,  # We'll update this later in the webhook
                stripe_customer_id=customer.id,
                stripe_invoice_id=None,  # This will also be updated in the webhook
                description=f"Purchase of {credits} credits",
                transaction_metadata={
                    "org_id": str(org_id),
                    "checkout_session_id": stripe_session.id,  # Store the session ID
                },
            )

            session.add(transaction)
            await session.commit()

            return {"checkout_url": stripe_session.url, "session_id": stripe_session.id}

        except stripe.error.StripeError as e:
            logger.error("Failed to create Stripe session: %s", str(e))
            logger.error(f"Create checkout session error : {str(e)}")
            await session.rollback()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Fix the logging format - remove the %s
            logger.error("Unexpected error during checkout creation: " + str(e))
            # logger.exception("Full traceback:")  # This will log the full stack trace
            await session.rollback()
            raise HTTPException(status_code=500, detail="Internal server error")

    async def post_stripe_webhook(
        self, request: Request, session: AsyncSession = Depends(get_db)
    ) -> None:
        """Handle Stripe webhook events."""
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")

        try:
            # Log raw payload with proper formatting
            logger.info("Webhook raw payload received")

            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.stripe_webhook_secret
            )
            # Log constructed event
            logger.info(f"Stripe webhook event received: {event.type} (ID: {event.id})")

            # Handle different event types
            if event.type == "checkout.session.completed":
                session_obj = event.data.object
                logger.info(
                    f"Processing completed checkout session ID: {session_obj.id}"
                )

                # Find transaction by checkout session ID
                transaction = await session.execute(
                    select(TransactionModel).where(
                        and_(
                            # Add explicit JSON extraction and comparison
                            func.json_extract_path_text(
                                TransactionModel.transaction_metadata,
                                "checkout_session_id",
                            )
                            == session_obj.id,
                            TransactionModel.status == TransactionStatus.PENDING,
                        )
                    )
                )
                transaction = transaction.scalar_one_or_none()

                if transaction:
                    logger.info(f"Found matching transaction: {transaction.id}")
                    # Update transaction with payment intent and status
                    transaction.stripe_payment_intent_id = session_obj.payment_intent
                    transaction.stripe_invoice_id = session_obj.invoice
                    transaction.status = TransactionStatus.COMPLETED

                    # Commit these changes first
                    await session.commit()
                    await session.refresh(transaction)

                    logger.info(
                        f"Updated transaction {transaction.id} with payment intent: {session_obj.payment_intent}"
                    )

                    # Update credit balance
                    await self._handle_successful_payment(
                        user_id=UUID(session_obj.metadata["user_id"]),
                        payment_intent_id=session_obj.payment_intent,
                        credits=int(session_obj.metadata["credits"]),
                        session=session,
                    )

                    logger.info(
                        f"Successfully processed payment for transaction: {transaction.id}, added {session_obj.metadata['credits']} credits"
                    )
                else:
                    logger.error(
                        f"No matching transaction found for checkout session: {session_obj.id}"
                    )
                    return {
                        "status": "error",
                        "message": "No matching transaction found for this checkout session",
                    }

            elif event.type == "checkout.session.expired":
                session_obj = event.data.object
                logger.info(f"Processing expired checkout session: {session_obj.id}")

                # Find and update the transaction status
                transaction = await session.execute(
                    select(TransactionModel).where(
                        and_(
                            TransactionModel.transaction_metadata[
                                "checkout_session_id"
                            ].astext
                            == session_obj.id,
                            TransactionModel.status == TransactionStatus.PENDING,
                        )
                    )
                )
                transaction = transaction.scalar_one_or_none()

                if transaction:
                    transaction.status = TransactionStatus.FAILED
                    transaction.transaction_metadata["failure_reason"] = (
                        "Checkout session expired"
                    )
                    await session.commit()
                    logger.info(
                        "Transaction marked as failed due to session expiration: %s",
                        transaction.id,
                    )
                else:
                    logger.error(
                        "No matching transaction found for expired session: %s",
                        session_obj.id,
                    )

            elif event.type == "checkout.session.async_payment_succeeded":
                session_obj = event.data.object
                logger.info(f"Processing async payment success: {session_obj.id}")

                # Similar handling to checkout.session.completed
                transaction = await session.execute(
                    select(TransactionModel).where(
                        and_(
                            TransactionModel.transaction_metadata[
                                "checkout_session_id"
                            ].astext
                            == session_obj.id,
                            TransactionModel.status == TransactionStatus.PENDING,
                        )
                    )
                )
                transaction = transaction.scalar_one_or_none()

                if transaction:
                    transaction.stripe_payment_intent_id = session_obj.payment_intent
                    transaction.status = TransactionStatus.COMPLETED
                    transaction.stripe_invoice_id = session_obj.invoice

                    await self._handle_successful_payment(
                        user_id=UUID(session_obj.metadata["user_id"]),
                        payment_intent_id=session_obj.payment_intent,
                        credits=int(session_obj.metadata["credits"]),
                        session=session,
                    )
                else:
                    logger.error(
                        "No matching transaction found for async payment success: %s",
                        session_obj.id,
                    )

            elif event.type == "checkout.session.async_payment_failed":
                session_obj = event.data.object
                logger.info(f"Processing async payment failure: {session_obj.id}")

                # Update transaction status to failed
                transaction = await session.execute(
                    select(TransactionModel).where(
                        and_(
                            TransactionModel.transaction_metadata[
                                "checkout_session_id"
                            ].astext
                            == session_obj.id,
                            TransactionModel.status == TransactionStatus.PENDING,
                        )
                    )
                )
                transaction = transaction.scalar_one_or_none()

                if transaction:
                    transaction.status = TransactionStatus.FAILED
                    transaction.transaction_metadata["failure_reason"] = (
                        "Async payment failed"
                    )
                    await session.commit()
                    logger.info(
                        "Transaction marked as failed due to async payment failure: %s",
                        transaction.id,
                    )
                else:
                    logger.error(
                        "No matching transaction found for async payment failure: %s",
                        session_obj.id,
                    )

            else:
                logger.info(f"Unhandled event type received: {event.type}")

        except (ValueError, stripe.error.SignatureVerificationError) as e:
            logger.error(f"Invalid webhook signature: {str(e)}")
            logger.error(f"Signature header: {sig_header}")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}")
            logger.error(f"Full error details: ", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_or_create_stripe_customer(
        self,
        user_id: UUID,
        email: Optional[str],
        session: AsyncSession,
    ) -> stripe.Customer:
        """Get or create a Stripe customer for the user."""
        # Check if customer exists

        transaction = await session.execute(
            select(TransactionModel)
            .where(TransactionModel.user_id == user_id)
            .where(TransactionModel.stripe_customer_id.isnot(None))
            .order_by(TransactionModel.created_at.desc())
        )
        # Get the first transaction with a stripe_customer_id for this user, or None if none exists
        transaction = transaction.scalars().first()

        if transaction and transaction.stripe_customer_id:
            return stripe.Customer.retrieve(transaction.stripe_customer_id)

        # Create new customer
        if not email:
            raise HTTPException(
                status_code=400, detail="Email is required to create a customer"
            )

        customer = stripe.Customer.create(
            email=email, metadata={"user_id": str(user_id)}
        )
        return customer

    async def _handle_successful_payment(
        self, user_id: UUID, payment_intent_id: str, credits: int, session: AsyncSession
    ):
        """Handle successful payment from Stripe webhook."""
        try:
            logger.info(
                f"Handling successful payment for user {user_id}, payment intent: {payment_intent_id}"
            )
            # Update transaction status if not already completed
            update_result = await session.execute(
                update(TransactionModel)
                .where(
                    and_(
                        TransactionModel.stripe_payment_intent_id == payment_intent_id,
                        TransactionModel.status != TransactionStatus.COMPLETED,
                    )
                )
                .values(status=TransactionStatus.COMPLETED)
            )

            rows_affected = update_result.rowcount
            logger.info(f"Updated {rows_affected} transactions to COMPLETED status")

            # Update credit balance with spent tracking reset logic
            balance = await self._get_or_create_balance(user_id, session)
            logger.info(
                f"Current balance for user {user_id}: {balance.balance} credits"
            )

            # If balance is zero or negative, this is a fresh start
            if balance.balance <= 0:
                logger.info(
                    f"Balance was zero or negative, resetting spent tracking for user {user_id}"
                )
                balance.credits_spent = 0
                balance.last_reset_date = datetime.utcnow()

            previous_balance = balance.balance
            balance.balance += credits

            logger.info(
                f"Updated balance for user {user_id}: {previous_balance} → {balance.balance} credits (+{credits})"
            )

            await session.commit()
            logger.info(
                f"Successfully handled payment for user {user_id}, added {credits} credits"
            )
            return {
                "status": "success",
                "message": f"Successfully added {credits} credits to your account",
            }
        except Exception as e:
            logger.error(
                f"Error handling successful payment for user {user_id}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to process payment: {str(e)}"
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
        logger.info(
            f"Getting invoice URL for transaction: {transaction_id}, user: {user_id}"
        )

        # Get the transaction
        query = select(TransactionModel).where(
            TransactionModel.id == transaction_id,
            TransactionModel.user_id == user_id,
        )
        result = await session.execute(query)
        transaction = result.scalar_one_or_none()

        if not transaction:
            logger.warning(f"Transaction {transaction_id} not found for user {user_id}")
            raise HTTPException(
                status_code=404,
                detail="Transaction not found or you don't have permission to access it",
            )

        if not transaction.stripe_invoice_id:
            logger.warning(f"No invoice ID found for transaction {transaction_id}")
            raise HTTPException(
                status_code=404, detail="No invoice available for this transaction"
            )

        try:
            # Get the invoice from Stripe
            invoice = stripe.Invoice.retrieve(transaction.stripe_invoice_id)
            logger.info(
                f"Retrieved Stripe invoice: {invoice.id} for transaction {transaction_id}"
            )

            # Return the hosted invoice URL
            return {
                "invoice_url": invoice.hosted_invoice_url,
                "status": "success",
                "message": "Invoice retrieved successfully",
            }

        except stripe.error.StripeError as e:
            logger.error(
                f"Stripe invoice error for transaction {transaction_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=400, detail=f"Failed to retrieve invoice: {str(e)}"
            )

    # GROUP 5: USAGE AND PRICING
    async def process_usage(
        self, transaction_id: UUID, result: Any, session: AsyncSession
    ) -> int:
        """Process usage and calculate credits based on function configuration."""
        # Get transaction and function details
        transaction = await session.execute(
            select(TransactionModel).where(TransactionModel.id == transaction_id)
        )
        transaction = transaction.scalar_one_or_none()

        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")

        function = await self.get_billable_function(transaction.function_id, session)

        if not function:
            raise HTTPException(status_code=404, detail="Function not found")

        # Extract usage data from result
        usage_data = self._extract_usage_data(result, function.pricing_type)

        # Calculate credits
        credits_used = await self._calculate_credits(function, usage_data, session)

        # Complete transaction
        transaction.status = "COMPLETED"  # Using string instead of enum
        # Use credits_used attribute if it exists, otherwise update credits
        if hasattr(transaction, "credits_used"):
            transaction.credits_used = credits_used
        else:
            transaction.credits = credits_used

        if transaction.transaction_metadata is None:
            transaction.transaction_metadata = {}
        transaction.transaction_metadata.update(
            {"usage_data": usage_data, "completed_at": datetime.utcnow().isoformat()}
        )

        # Deduct credits
        await self._deduct_credits(transaction.user_id, credits_used, session)

        await session.commit()

        return credits_used

    def _extract_usage_data(self, result: Any, pricing_type: str) -> Dict[str, Any]:
        """Extract usage data from function result."""
        usage_data = {}

        if pricing_type == "per_token":
            # Handle token-based pricing
            if isinstance(result, dict):
                usage_data["tokens"] = result.get("tokens_used", 0)
                usage_data["input_tokens"] = result.get("input_tokens", 0)
                usage_data["output_tokens"] = result.get("output_tokens", 0)
            else:
                # Default token counting if not provided
                usage_data["tokens"] = len(str(result).split()) if result else 0

        elif pricing_type == "fixed":
            # Fixed pricing doesn't need usage data
            usage_data["fixed"] = True

        elif pricing_type == "custom":
            # Custom pricing should provide its own usage data
            if isinstance(result, dict):
                usage_data = result.get("usage_data", {})

        return usage_data

    # GROUP 6: BILLING PLAN MANAGEMENT
    async def get_plans(
        self,
        org_id: UUID,
        include_inactive: bool = False,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "read")),
    ) -> List[BillingPlanResponseSchema]:
        """Get all available billing plans."""
        try:
            logger.info("Starting get_billing_plans")
            query = select(BillingPlanModel)

            if not include_inactive:
                query = query.where(BillingPlanModel.is_active == True)

            logger.info(f"Executing query: {query}")
            result = await session.execute(query)
            plans = result.scalars().all()
            logger.info(f"Found {len(plans)} billing plans")

            return [BillingPlanResponseSchema.from_orm(plan) for plan in plans]

        except Exception as e:
            logger.error(f"Error in get_plans: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve billing plans: {str(e)}"
            )

    async def post_create_billing_plan(
        self,
        org_id: UUID,
        plan_data: BillingPlanCreateSchema,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "admin")),
    ) -> BillingPlanResponseSchema:
        """Create a new billing plan."""
        plan = BillingPlanSchema(
            name=plan_data.name,
            description=plan_data.description,
            amount_usd=plan_data.amount_usd,
            credits=plan_data.credits,
            discount_percentage=plan_data.discount_percentage,
            is_active=plan_data.is_active,
        )

        session.add(plan)
        await session.commit()
        await session.refresh(plan)

        return BillingPlanResponseSchema.from_orm(plan)

    async def put_update_billing_plan(
        self,
        org_id: UUID,
        plan_id: UUID,
        plan_data: BillingPlanCreateSchema,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "admin")),
    ) -> BillingPlanResponseSchema:
        """Update an existing billing plan."""
        # Get the plan
        query = select(BillingPlanSchema).where(BillingPlanSchema.id == plan_id)
        result = await session.execute(query)
        plan = result.scalar_one_or_none()

        if not plan:
            raise HTTPException(status_code=404, detail="Billing plan not found")

        # Update fields
        plan.name = plan_data.name
        plan.description = plan_data.description
        plan.amount_usd = plan_data.amount_usd
        plan.credits = plan_data.credits
        plan.discount_percentage = plan_data.discount_percentage
        plan.is_active = plan_data.is_active
        plan.updated_at = datetime.utcnow()

        await session.commit()
        await session.refresh(plan)

        return BillingPlanResponseSchema.from_orm(plan)

    async def delete_billing_plan(
        self,
        org_id: UUID,
        plan_id: UUID,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "admin")),
    ) -> Dict[str, Any]:
        """Soft-delete a billing plan by setting is_active to False."""
        # Get the plan
        query = select(BillingPlanSchema).where(BillingPlanSchema.id == plan_id)
        result = await session.execute(query)
        plan = result.scalar_one_or_none()

        if not plan:
            raise HTTPException(status_code=404, detail="Billing plan not found")

        # Soft delete
        plan.is_active = False
        plan.updated_at = datetime.utcnow()

        await session.commit()

        return {
            "status": "success",
            "message": f"Billing plan '{plan.name}' has been deactivated",
        }

    async def fail_transaction(
        self, transaction_id: UUID, error: str, session: AsyncSession
    ):
        """Mark transaction as failed."""
        transaction = await session.execute(
            select(TransactionModel).where(TransactionModel.id == transaction_id)
        )
        transaction = transaction.scalar_one_or_none()

        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")

        transaction.status = "FAILED"  # Using string instead of enum
        transaction.credits_used = 0
        if transaction.transaction_metadata is None:
            transaction.transaction_metadata = {}
        transaction.transaction_metadata.update(
            {"error": error, "failed_at": datetime.utcnow().isoformat()}
        )

        await session.commit()

    async def _get_service_by_name(
        self, service_name: str, session: AsyncSession
    ) -> Optional[ServiceModel]:
        """Private helper method to get service by name (globally)."""
        query = select(ServiceModel).where(ServiceModel.name == service_name)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def get_service_by_name(
        self, org_id: UUID, service_name: str, session: AsyncSession
    ) -> Optional[ServiceModel]:
        """Get service details by name."""
        # Keep this compatibility method that takes org_id but doesn't use it
        return await self._get_service_by_name(service_name, session)

    async def post_register_service(
        self,
        org_id: UUID,  # Keep org_id for permission check only
        service_data: ServiceCreateSchema,
        session: AsyncSession = Depends(get_db),
        user: dict = Depends(RBAC("billing", "admin")),
    ) -> ServiceResponseSchema:
        """Register a new service that can be billed."""
        # Check if service with same name already exists (globally)
        existing = await self._get_service_by_name(service_data.name, session)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Service with name '{service_data.name}' already exists",
            )

        # Create new service (no org_id)
        service = ServiceModel(
            name=service_data.name,
            credit_cost=service_data.credit_cost,
            description=service_data.description,
            # No org_id here
        )
        session.add(service)
        await session.commit()
        await session.refresh(service)

        # Return the Pydantic schema instead of the SQLAlchemy model
        return ServiceResponseSchema.from_orm(service)

    async def track_service_usage(
        self,
        org_id: UUID,
        user_id: UUID,
        service_id: UUID,
        credits_used: int,
        session: AsyncSession,
    ) -> None:
        """Track service usage for a user within an organization."""
        current_month = datetime.utcnow().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        query = (
            select(UserUsageModel)
            .where(
                and_(
                    UserUsageModel.user_id == user_id,
                    UserUsageModel.org_id == org_id,
                    UserUsageModel.service_id == service_id,
                    UserUsageModel.month == current_month,
                )
            )
            .with_for_update()
        )

        result = await session.execute(query)
        usage = result.scalar_one_or_none()

        if not usage:
            usage = UserUsageModel(
                user_id=user_id,
                org_id=org_id,
                service_id=service_id,
                month=current_month,
                total_requests=1,
                total_credits=credits_used,
            )
            session.add(usage)
        else:
            usage.total_requests += 1
            usage.total_credits += credits_used

        await session.commit()
