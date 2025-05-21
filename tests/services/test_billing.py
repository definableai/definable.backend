import pytest
import pytest_asyncio
import sys
import os
from uuid import uuid4
from datetime import datetime, timezone
from decimal import Decimal

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
import stripe
from sqlalchemy import text

from src.services.billing.service import BillingService
from src.services.billing.schema import (
    TransactionType,
    TransactionStatus,
    BillingPlanResponseSchema,
    CheckoutSessionCreateSchema,
    WalletResponseSchema,
    CreditCalculationResponseSchema,
)
from src.services.__base.acquire import Acquire

# Define function to check for integration tests
def is_integration_test():
    """Check if we're running in integration test mode."""
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# TestAcquire - mock of the Acquire class for service initialization
class TestAcquire(Acquire):
    def __init__(self):
        self.settings = type('Settings', (), {})()
        self.settings.stripe_secret_key = "test_stripe_key"
        self.settings.stripe_webhook_secret = "test_webhook_secret"
        self.logger = MagicMock()
        self.utils = MagicMock()

# Only import actual models in integration test mode
if is_integration_test():
    from sqlalchemy import select, text
    from models import (
        TransactionModel,
        TransactionStatus as DBTransactionStatus,
        TransactionType as DBTransactionType
    )
else:
    # Mock modules to prevent SQLAlchemy issues
    sys.modules["database"] = MagicMock()
    sys.modules["database.postgres"] = MagicMock()
    sys.modules["src.database"] = MagicMock()
    sys.modules["src.database.postgres"] = MagicMock()
    sys.modules["dependencies.security"] = MagicMock()

# Mock model classes
class MockWalletModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.organization_id = kwargs.get('organization_id', uuid4())
        self.balance = kwargs.get('balance', 0)
        self.hold = kwargs.get('hold', 0)
        self.credits_spent = kwargs.get('credits_spent', 0)
        self.last_reset_date = kwargs.get('last_reset_date')
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))

        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

class MockTransactionModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.user_id = kwargs.get('user_id', uuid4())
        self.organization_id = kwargs.get('organization_id', uuid4())
        self.type = kwargs.get('type', TransactionType.CREDIT_PURCHASE)
        self.status = kwargs.get('status', TransactionStatus.COMPLETED)
        self.amount_usd = kwargs.get('amount_usd', Decimal('10.00'))
        self.credits = kwargs.get('credits', 10000)
        self.description = kwargs.get('description', 'Test transaction')
        self.stripe_session_id = kwargs.get('stripe_session_id', None)
        self.stripe_invoice_id = kwargs.get('stripe_invoice_id', None)
        self.stripe_customer_id = kwargs.get('stripe_customer_id', None)
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.transaction_metadata = kwargs.get('transaction_metadata', {})

        # Add any other fields needed for mocking
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

class MockBillingPlanModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', "Test Plan")
        self.description = kwargs.get('description', "Test billing plan")
        self.amount_usd = kwargs.get('amount_usd', 10.0)
        self.credits = kwargs.get('credits', 10000)
        self.discount_percentage = kwargs.get('discount_percentage', 0.0)
        self.is_active = kwargs.get('is_active', True)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))

        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

class MockUserModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.email = kwargs.get('email', "test@example.com")
        self.first_name = kwargs.get('first_name', "Test")
        self.last_name = kwargs.get('last_name', "User")

        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

# Mock Stripe response objects
class MockStripeCustomer:
    def __init__(self, customer_id="cus_test123", email="test@example.com", name="Test User"):
        self.id = customer_id
        self.email = email
        self.name = name

class MockStripeCheckoutSession:
    def __init__(self, session_id="cs_test123", url="https://checkout.stripe.com/test", **kwargs):
        self.id = session_id
        self.url = url
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockStripeEvent:
    def __init__(self, event_type="checkout.session.completed", data=None):
        self.type = event_type
        self.id = "evt_test123"  # Add ID attribute for proper mocking

        # Format data correctly for webhook handling
        if data is None:
            data = {"object": {"id": "cs_test_webhook"}}

        # Create proper nested objects instead of using type() constructor
        class EventObject:
            def __init__(self, props):
                for key, value in props.items():
                    setattr(self, key, value)

        class EventData:
            def __init__(self, obj_data):
                self.object = EventObject(obj_data["object"])

        self.data = EventData(data)

@pytest.fixture
def billing_service():
    """Create a BillingService instance."""
    return BillingService(acquire=TestAcquire())

@pytest.fixture
def mock_db_session():
    """Create a mock database session with properly structured results."""
    session = MagicMock()

    # Set up add and delete methods
    session.add = MagicMock()
    session.delete = AsyncMock()

    # Set up transaction methods
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()

    # Set up scalar method
    session.scalar = AsyncMock()

    # Create a properly structured mock for database queries
    # For scalars().all() pattern
    scalars_mock = MagicMock()
    scalars_mock.all = MagicMock(return_value=[])
    scalars_mock.first = MagicMock(return_value=None)

    # For unique().scalar_one_or_none() pattern
    unique_mock = MagicMock()
    unique_mock.scalar_one_or_none = MagicMock(return_value=None)
    unique_mock.scalar_one = MagicMock(return_value=0)
    unique_mock.scalars = MagicMock(return_value=scalars_mock)

    # For direct scalar_one_or_none
    execute_mock = AsyncMock()
    execute_mock.scalar = MagicMock(return_value=0)
    execute_mock.scalar_one_or_none = MagicMock(return_value=None)
    execute_mock.scalar_one = MagicMock(return_value=0)
    execute_mock.scalars = MagicMock(return_value=scalars_mock)
    execute_mock.unique = MagicMock(return_value=unique_mock)
    execute_mock.all = MagicMock(return_value=[])

    # Set up session execute to return the mock
    session.execute = AsyncMock(return_value=execute_mock)

    # Set up get method
    session.get = AsyncMock(return_value=None)

    return session

@pytest.fixture
def test_user():
    """Create a test user dictionary."""
    user_id = uuid4()
    return {
        "id": str(user_id),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "org_id": str(uuid4()),
        "is_admin": True
    }

@pytest.fixture
def test_org_id():
    """Create a test organization ID."""
    return uuid4()

@pytest.fixture
def mock_wallet():
    """Create a mock wallet."""
    return MockWalletModel(balance=10000, credits_spent=5000)

@pytest.fixture
def mock_billing_plan():
    """Create a mock billing plan."""
    return MockBillingPlanModel()

@pytest.fixture
def mock_transaction():
    """Create a mock transaction."""
    return MockTransactionModel()

@pytest.mark.asyncio
class TestBillingService:
    """Tests for the BillingService."""

    async def test_get_wallet_success(self, billing_service, mock_db_session, test_org_id, mock_wallet, test_user):
        """Test getting wallet successfully."""
        # Setup
        org_id = test_org_id

        # Mock the _get_or_create_wallet method to return a real wallet
        with patch.object(billing_service, '_get_or_create_wallet', return_value=mock_wallet):
            # Execute
            result = await billing_service.get_wallet(
                org_id=org_id,
                session=mock_db_session,
                user=test_user
            )

            # Verify
            assert isinstance(result, WalletResponseSchema)
            assert result.balance == mock_wallet.balance - mock_wallet.hold
            assert result.credits_spent == mock_wallet.credits_spent
            assert result.id == mock_wallet.id

    async def test_get_plans_success(self, billing_service, mock_db_session, test_user, mock_billing_plan):
        """Test getting all available billing plans."""
        # Setup
        org_id = uuid4()
        mock_plans = [mock_billing_plan]

        # Mock database to return plans
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_plans

        # Execute
        result = await billing_service.get_plans(
            org_id=org_id,
            session=mock_db_session,
            user=test_user
        )

        # Verify
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], BillingPlanResponseSchema)
        assert result[0].id == mock_billing_plan.id
        assert result[0].name == mock_billing_plan.name

    async def test_get_plans_include_inactive(self, billing_service, mock_db_session, test_user):
        """Test getting all billing plans including inactive ones."""
        # Setup
        org_id = uuid4()
        mock_active_plan = MockBillingPlanModel(name="Active Plan", is_active=True)
        mock_inactive_plan = MockBillingPlanModel(name="Inactive Plan", is_active=False)
        mock_plans = [mock_active_plan, mock_inactive_plan]

        # Mock database to return plans
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_plans

        # Execute with include_inactive=True
        result = await billing_service.get_plans(
            org_id=org_id,
            include_inactive=True,
            session=mock_db_session,
            user=test_user
        )

        # Verify
        assert isinstance(result, list)
        assert len(result) == 2  # Both active and inactive

    async def test_calculate_credits(self, billing_service, mock_db_session, test_user):
        """Test credit calculation for a given USD amount."""
        # Setup
        org_id = uuid4()
        amount_usd = 10.0

        # Execute
        result = await billing_service.get_calculate_credits(
            org_id=org_id,
            amount_usd=amount_usd,
            session=mock_db_session,
            user=test_user
        )

        # Verify
        assert isinstance(result, CreditCalculationResponseSchema)
        assert result.amount_usd == amount_usd
        assert result.credits == amount_usd * billing_service.credits_per_usd

    # Additional test placeholders

    @patch('stripe.checkout.Session.create')
    async def test_post_checkout_with_plan(self, mock_session_create, billing_service, mock_db_session, test_user, mock_billing_plan):
        """Test creating a checkout session with a billing plan."""
        # Setup
        org_id = uuid4()
        plan_id = uuid4()
        mock_billing_plan.id = plan_id
        mock_billing_plan.is_active = True

        checkout_data = CheckoutSessionCreateSchema(
            plan_id=plan_id,
            customer_email="test@example.com"
        )

        # Use execute().scalar_one_or_none() pattern to return the mock plan
        scalar_one_or_none_mock = MagicMock(return_value=mock_billing_plan)
        mock_db_session.execute.return_value.scalar_one_or_none = scalar_one_or_none_mock

        # Mock Stripe session creation
        mock_session_create.return_value = MockStripeCheckoutSession(id="cs_test_plan")

        # Mock _get_or_create_stripe_customer
        with patch.object(billing_service, '_get_or_create_stripe_customer', return_value=MockStripeCustomer()):
            # Execute
            result = await billing_service.post_checkout(
                org_id=org_id,
                checkout_data=checkout_data,
                session=mock_db_session,
                user=test_user
            )

            # Verify
            assert isinstance(result, dict)
            assert "checkout_url" in result
            assert result["checkout_url"] == "https://checkout.stripe.com/test"
            mock_session_create.assert_called_once()
            mock_db_session.add.assert_called_once()  # Should add transaction record

    @patch('stripe.Webhook.construct_event')
    async def test_post_stripe_webhook_invalid_signature(self, mock_construct_event, billing_service, mock_db_session):
        """Test handling a webhook with invalid signature."""
        # Setup
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Stripe-Signature": "invalid_signature"}
        mock_request.body = AsyncMock()
        mock_request.body.return_value = b'{}'

        # Mock construct_event to raise an exception
        mock_construct_event.side_effect = stripe.error.SignatureVerificationError("Invalid signature", "test_sig")

        # Execute and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await billing_service.post_stripe_webhook(
                request=mock_request,
                session=mock_db_session
            )

        # Verify
        assert excinfo.value.status_code == 400
        assert "Invalid signature" in excinfo.value.detail

    async def test_get_or_create_wallet_existing(self, billing_service, mock_db_session, mock_wallet):
        """Test retrieving an existing wallet."""
        # Setup
        org_id = uuid4()

        # Set up the mock wallet
        mock_wallet.organization_id = org_id  # Make sure org_id matches

        # Configure mock to return our mock wallet
        scalar_mock = MagicMock()
        scalar_mock.return_value = mock_wallet
        mock_db_session.execute.return_value.scalar_one_or_none = scalar_mock

        # Execute
        result = await billing_service._get_or_create_wallet(org_id, mock_db_session)

        # Verify the result - use more flexible assertions
        assert result.organization_id == org_id
        assert hasattr(result, 'balance')  # Should have balance attribute
        # Just verify it's the same instance, not comparing IDs which might change
        assert result is mock_wallet

        mock_db_session.add.assert_not_called()  # Should not create a new wallet

    async def test_add_credits(self, billing_service, mock_db_session, mock_wallet, mock_transaction):
        """Test adding credits to a wallet."""
        # Setup
        user_id = uuid4()
        org_id = uuid4()
        amount = 5000
        description = "Test credit addition"

        # Reset mock calls
        mock_db_session.add.reset_mock()
        mock_db_session.commit.reset_mock()

        # Make sure wallet is properly mocked
        initial_balance = 10000
        mock_wallet.balance = initial_balance

        # Mock _get_or_create_wallet to return the wallet
        with patch.object(billing_service, '_get_or_create_wallet', return_value=mock_wallet):
            # Execute
            result = await billing_service._add_credits(
                user_id=user_id,
                org_id=org_id,
                amount=amount,
                description=description,
                session=mock_db_session
            )

            # Verify result and balance update
            assert result is True
            assert mock_wallet.balance == initial_balance + amount

            # Verify commit was called (service doesn't always add a transaction)
            assert mock_db_session.commit.called

@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for billing integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    test_org_id = None
    test_user_id = None
    session = None

    # Get session from the generator without exhausting it
    try:
        session_gen = db_session.__aiter__()
        session = await session_gen.__anext__()

        # Create necessary database tables if they don't exist
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS organizations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                slug VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS wallets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                organization_id UUID NOT NULL,
                balance INTEGER NOT NULL DEFAULT 0,
                hold INTEGER NOT NULL DEFAULT 0,
                credits_spent INTEGER DEFAULT 0,
                last_reset_date TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS billing_plans (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                amount_usd DECIMAL(10, 2) NOT NULL,
                credits INTEGER NOT NULL,
                discount_percentage DECIMAL(5, 2) DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS transactions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                organization_id UUID NOT NULL,
                type VARCHAR(50) NOT NULL,
                status VARCHAR(50) NOT NULL,
                amount_usd DECIMAL(10, 2) NOT NULL,
                credits INTEGER NOT NULL,
                description TEXT,
                stripe_session_id VARCHAR(255),
                stripe_invoice_id VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                first_name VARCHAR(255),
                last_name VARCHAR(255),
                stytch_id VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                organization_id UUID NOT NULL,
                subscription_id VARCHAR(255) NOT NULL,
                subscription_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Commit schema changes first
        await session.commit()

        # Clean up any existing test data
        await session.execute(
            text("""
                DELETE FROM transactions
                WHERE organization_id IN (
                    SELECT id FROM organizations
                    WHERE name LIKE 'Test Billing%'
                )
            """)
        )
        await session.execute(text("DELETE FROM wallets WHERE organization_id IN (SELECT id FROM organizations WHERE name LIKE 'Test Billing%')"))
        # Delete previous test organizations
        await session.execute(text("DELETE FROM organizations WHERE name LIKE 'Test Billing%'"))
        await session.commit()

        # Create a test organization with unique slug
        test_org_id = uuid4()
        random_suffix = uuid4().hex[:8]  # Add random suffix to prevent unique constraint violations
        await session.execute(
            text("""
                INSERT INTO organizations (id, name, slug)
                VALUES (:id, :name, :slug)
            """),
            {
                "id": str(test_org_id),
                "name": f"Test Billing Organization {random_suffix}",
                "slug": f"test-billing-org-{random_suffix}"
            }
        )

        # Create a test user
        test_user_id = uuid4()
        test_stytch_id = f"test-stytch-id-{test_user_id}"
        await session.execute(
            text("""
                INSERT INTO users (id, email, first_name, last_name, stytch_id)
                VALUES (:id, :email, :first_name, :last_name, :stytch_id)
            """),
            {
                "id": str(test_user_id),
                "email": f"test-billing-{random_suffix}@example.com",
                "first_name": "Billing",
                "last_name": "Test",
                "stytch_id": test_stytch_id
            }
        )

        # Delete any existing test plans to avoid conflicts
        await session.execute(text("DELETE FROM billing_plans WHERE name IN ('Standard', 'Premium')"))

        # Create billing plans - excluding the problematic description field
        standard_plan_id = uuid4()
        premium_plan_id = uuid4()

        # Debug the SQL execution
        try:
            await session.execute(
                text("""
                    INSERT INTO billing_plans (id, name, amount_usd, credits, discount_percentage, is_active)
                    VALUES (:id, :name, :amount_usd, :credits, :discount, :is_active)
                """),
                {
                    "id": str(standard_plan_id),
                    "name": "Standard",
                    "amount_usd": 10.00,
                    "credits": 10000,
                    "discount": 0,
                    "is_active": True
                }
            )

            await session.execute(
                text("""
                    INSERT INTO billing_plans (id, name, amount_usd, credits, discount_percentage, is_active)
                    VALUES (:id, :name, :amount_usd, :credits, :discount, :is_active)
                """),
                {
                    "id": str(premium_plan_id),
                    "name": "Premium",
                    "amount_usd": 50.00,
                    "credits": 60000,
                    "discount": 20,
                    "is_active": True
                }
            )
        except Exception as e:
            print(f"Error inserting billing plans: {e}")
            raise

        await session.commit()

    except Exception as e:
        print(f"Error in setup: {e}")
        if session:
            await session.rollback()
        raise

    # Return test data to tests
    yield {
        "org_id": test_org_id,
        "user_id": test_user_id,
        "stytch_id": test_stytch_id,
        "standard_plan_id": standard_plan_id,
        "premium_plan_id": premium_plan_id,
        "db_session": db_session
    }

    # Clean up after tests
    try:
        # Get a new session for cleanup
        async for cleanup_session in db_session:
            try:
                # Delete test data
                await cleanup_session.execute(
                    text("""
                        DELETE FROM user_subscriptions
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await cleanup_session.execute(
                    text("""
                        DELETE FROM transactions
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await cleanup_session.execute(
                    text("""
                        DELETE FROM wallets
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await cleanup_session.execute(
                    text("""
                        DELETE FROM organizations
                        WHERE name LIKE 'Test Billing%'
                    """)
                )
                await cleanup_session.commit()
                break  # Only process the first yielded session
            except Exception as e:
                print(f"Error in cleanup: {e}")
                await cleanup_session.rollback()
    except Exception as e:
        print(f"Could not acquire session for cleanup: {e}")

@pytest.mark.asyncio
class TestBillingServiceIntegration:
    """Integration tests for Billing service with real database."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    # Integration tests would be implemented here
    async def test_get_wallet_integration(self, billing_service, setup_test_db_integration):
        """Test getting wallet with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Set up a test user for RBAC dependency
                current_user = {"id": str(test_data["user_id"])}

                # Execute
                response = await billing_service.get_wallet(
                    org_id=org_id,
                    session=session,
                    user=current_user
                )

                # Assert
                assert response is not None
                assert isinstance(response, WalletResponseSchema)
                assert response.balance >= 0  # Should have at least 0 balance

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_wallet_integration: {e}")
                raise

    @patch('stripe.checkout.Session.create')
    async def test_post_checkout_integration(self, mock_session_create, billing_service, setup_test_db_integration):
        """Test creating a checkout session with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]
        plan_id = test_data["standard_plan_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Mock Stripe session creation
                mock_session_create.return_value = MockStripeCheckoutSession(id="cs_test_integration")

                # Create checkout data
                checkout_data = CheckoutSessionCreateSchema(plan_id=plan_id)

                # Set up a test user for RBAC dependency
                current_user = {"id": str(test_data["user_id"])}

                # Execute
                response = await billing_service.post_checkout(
                    org_id=org_id,
                    checkout_data=checkout_data,
                    session=session,
                    user=current_user
                )

                # Assert
                assert response is not None
                assert "checkout_url" in response
                assert response["checkout_url"] == "https://checkout.stripe.com/test"

                # Verify transaction was created
                query = select(TransactionModel).where(
                    TransactionModel.organization_id == org_id,
                    TransactionModel.stripe_session_id == "cs_test_integration"
                )
                result = await session.execute(query)
                transaction = result.scalar_one_or_none()
                assert transaction is not None
                assert transaction.status == DBTransactionStatus.PENDING
                assert transaction.type == DBTransactionType.CREDIT_PURCHASE

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_post_checkout_integration: {e}")
                raise

    async def test_get_plans_integration(self, billing_service, setup_test_db_integration):
        """Test retrieving billing plans with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Set up a test user for RBAC dependency
                current_user = {"id": str(test_data["user_id"])}

                # Execute
                response = await billing_service.get_plans(
                    org_id=org_id,
                    session=session,
                    user=current_user
                )

                # Assert
                assert response is not None
                assert len(response) >= 2  # Should have at least the two plans we created
                assert all(isinstance(plan, BillingPlanResponseSchema) for plan in response)

                # Verify plan details
                standard_plan = next((p for p in response if p.name == "Standard"), None)
                premium_plan = next((p for p in response if p.name == "Premium"), None)

                assert standard_plan is not None
                assert premium_plan is not None
                assert standard_plan.amount_usd == 10.0
                assert premium_plan.amount_usd == 50.0
                assert standard_plan.credits == 10000
                assert premium_plan.credits == 60000
                assert premium_plan.discount_percentage == 20.0

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_plans_integration: {e}")
                raise

    # Delete test data
    async def test_delete_test_data(self, billing_service, setup_test_db_integration):
        """Test deleting test data from the database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]

        # Get a new session
        async for session in db_session:
            try:
                # Delete test data
                await session.execute(
                    text("""
                        DELETE FROM user_subscriptions
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await session.execute(
                    text("""
                        DELETE FROM transactions
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await session.execute(
                    text("""
                        DELETE FROM wallets
                        WHERE organization_id IN (
                            SELECT id FROM organizations
                            WHERE name LIKE 'Test Billing%'
                        )
                    """)
                )
                await session.execute(
                    text("""
                        DELETE FROM organizations
                        WHERE name LIKE 'Test Billing%'
                    """)
                )
                await session.commit()

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_delete_test_data: {e}")
                raise