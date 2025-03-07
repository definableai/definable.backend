import stripe
from fastapi import HTTPException, Depends
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from src.config.settings import settings
from src.database import get_db
from src.services.billing.models import (
    CreditBalance, PaymentCard, Transaction,
    TransactionType, TransactionStatus,
    BillableService, BillableFunction, BillingTransaction,
    FunctionVersion
)
from typing import List, Optional, Dict, Any
import uuid

stripe.api_key = settings.stripe_secret_key

class BillingManager:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.credits_per_usd = 1000  # 1 USD = 1000 credits

    async def register_service(
        self,
        name: str,
        description: Optional[str] = None
    ) -> BillableService:
        service = BillableService(name=name, description=description)
        self.db.add(service)
        await self.db.commit()
        return service

    async def register_function(
        self,
        service_id: uuid.UUID,
        name: str,
        path: str,
        pricing_type: str,
        base_price_credits: int,
        pricing_config: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ) -> BillableFunction:
        function = BillableFunction(
            service_id=service_id,
            name=name,
            path=path,
            version=version,
            pricing_type=pricing_type,
            base_price_credits=base_price_credits,
            pricing_config=pricing_config
        )
        self.db.add(function)
        await self.db.commit()
        return function

    async def start_transaction(
        self,
        user_id: uuid.UUID,
        function_id: uuid.UUID,
        parent_transaction_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingTransaction:
        # Get function details
        function = await self.db.execute(
            select(BillableFunction).where(BillableFunction.id == function_id)
        )
        function = function.scalar_one_or_none()
        
        if not function:
            raise HTTPException(status_code=404, detail="Billable function not found")
            
        # Create transaction
        transaction = BillingTransaction(
            user_id=user_id,
            function_id=function_id,
            function_version=function.version,
            parent_transaction_id=parent_transaction_id,
            type=TransactionType.CREDIT_USAGE,
            status=TransactionStatus.PENDING,
            credits_used=0,  # Will be updated on completion
            metadata=metadata
        )
        self.db.add(transaction)
        await self.db.commit()
        return transaction

    async def complete_transaction(
        self,
        transaction_id: uuid.UUID,
        credits_used: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BillingTransaction:
        transaction = await self.db.execute(
            select(BillingTransaction).where(BillingTransaction.id == transaction_id)
        )
        transaction = transaction.scalar_one_or_none()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
            
        # Update transaction
        transaction.credits_used = credits_used
        transaction.status = TransactionStatus.COMPLETED
        if metadata:
            transaction.metadata = {**(transaction.metadata or {}), **metadata}
            
        await self.db.commit()
        return transaction

    async def calculate_credits(
        self,
        function_id: uuid.UUID,
        usage_data: Dict[str, Any]
    ) -> int:
        function = await self.db.execute(
            select(BillableFunction).where(BillableFunction.id == function_id)
        )
        function = function.scalar_one_or_none()
        
        if not function:
            raise HTTPException(status_code=404, detail="Billable function not found")
            
        if function.pricing_type == "per_token":
            return function.base_price_credits * usage_data.get("tokens", 0)
        elif function.pricing_type == "fixed":
            return function.base_price_credits
        else:
            # Custom pricing logic based on pricing_config
            # Implement your custom pricing logic here
            return function.base_price_credits

    async def get_credit_balance(self, user_id: uuid.UUID) -> CreditBalance:
        query = select(CreditBalance).where(CreditBalance.user_id == user_id)
        result = await self.db.execute(query)
        balance = result.scalar_one_or_none()
        
        if not balance:
            balance = CreditBalance(user_id=user_id, balance=0)
            self.db.add(balance)
            await self.db.commit()
            
        return balance

    async def create_checkout_session(
        self,
        user_id: uuid.UUID,
        amount_usd: float,
        success_url: str,
        cancel_url: str,
        customer_email: Optional[str] = None
    ) -> Dict[str, str]:
        try:
            # Get or create Stripe customer
            customer = await self._get_or_create_stripe_customer(user_id, customer_email)

            # Create Stripe checkout session
            session = stripe.checkout.Session.create(
                customer=customer.id,
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'Purchase {int(amount_usd * self.credits_per_usd)} Credits',
                        },
                        'unit_amount': int(amount_usd * 100),  # Stripe uses cents
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': str(user_id),
                    'credits': int(amount_usd * self.credits_per_usd)
                }
            )

            # Create pending transaction
            transaction = Transaction(
                user_id=user_id,
                type=TransactionType.CREDIT_PURCHASE,
                status=TransactionStatus.PENDING,
                amount_usd=amount_usd,
                credits=int(amount_usd * self.credits_per_usd),
                stripe_payment_intent_id=session.payment_intent,
                stripe_customer_id=customer.id,
                description=f"Purchase of {int(amount_usd * self.credits_per_usd)} credits"
            )
            self.db.add(transaction)
            await self.db.commit()

            return {
                'checkout_url': session.url,
                'session_id': session.id
            }

        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def create_customer_portal_session(
        self,
        user_id: uuid.UUID,
        return_url: str
    ) -> Dict[str, str]:
        try:
            # Get Stripe customer ID
            transaction = await self.db.execute(
                select(Transaction)
                .where(Transaction.user_id == user_id)
                .where(Transaction.stripe_customer_id.isnot(None))
                .order_by(Transaction.created_at.desc())
            )
            transaction = transaction.scalar_one_or_none()

            if not transaction or not transaction.stripe_customer_id:
                raise HTTPException(
                    status_code=404,
                    detail="No Stripe customer found for this user"
                )

            # Create portal session
            session = stripe.billing_portal.Session.create(
                customer=transaction.stripe_customer_id,
                return_url=return_url
            )

            return {'url': session.url}

        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> None:
        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                settings.stripe_webhook_secret
            )

            if event.type == 'checkout.session.completed':
                session = event.data.object
                await self._handle_successful_payment(
                    user_id=uuid.UUID(session.metadata['user_id']),
                    payment_intent_id=session.payment_intent,
                    credits=int(session.metadata['credits'])
                )

        except (ValueError, stripe.error.SignatureVerificationError):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

    async def _get_or_create_stripe_customer(
        self,
        user_id: uuid.UUID,
        email: Optional[str] = None
    ) -> stripe.Customer:
        # Check if customer exists
        transaction = await self.db.execute(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .where(Transaction.stripe_customer_id.isnot(None))
            .order_by(Transaction.created_at.desc())
        )
        transaction = transaction.scalar_one_or_none()

        if transaction and transaction.stripe_customer_id:
            return stripe.Customer.retrieve(transaction.stripe_customer_id)

        # Create new customer
        customer = stripe.Customer.create(
            email=email,
            metadata={'user_id': str(user_id)}
        )
        return customer

    async def _handle_successful_payment(
        self,
        user_id: uuid.UUID,
        payment_intent_id: str,
        credits: int
    ):
        # Update transaction status
        await self.db.execute(
            update(Transaction)
            .where(Transaction.stripe_payment_intent_id == payment_intent_id)
            .values(status=TransactionStatus.COMPLETED)
        )

        # Update credit balance
        balance = await self.get_credit_balance(user_id)
        balance.balance += credits
        
        await self.db.commit()

    async def get_transactions(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[Transaction]:
        query = (
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        return result.scalars().all()

async def get_billing_manager(db: AsyncSession = Depends(get_db)) -> BillingManager:
    return BillingManager(db)
