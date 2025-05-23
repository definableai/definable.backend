import datetime
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete

from models import BillingPlanModel
from src.services.__base.acquire import Acquire
from src.services.plans.schema import BillingPlanCreateSchema, BillingPlanResponseSchema, BillingPlanUpdateSchema
from src.services.plans.service import BillingPlansService


# Define a function to check if we're running integration tests
def is_integration_test():
  """Check if we're running in integration test mode.

  This is controlled by the INTEGRATION_TEST environment variable.
  Set it to 1 or true to run integration tests.
  """
  integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
  return integration_env in ("1", "true", "yes")


# TestAcquire - mock of the Acquire class for service initialization
class TestAcquire(Acquire):
  def __init__(self):
    self.settings = type("Settings", (), {})()
    self.logger = MagicMock()
    self.utils = MagicMock()


# Mock these modules to prevent SQLAlchemy issues when running unit tests
sys.modules["database"] = MagicMock()
sys.modules["database.postgres"] = MagicMock()
sys.modules["src.database"] = MagicMock()
sys.modules["src.database.postgres"] = MagicMock()
sys.modules["dependencies.security"] = MagicMock()

###########################################
# UNIT TESTS
###########################################


@pytest.fixture
def mock_db_session():
  """Create a mock database session."""
  session = AsyncMock()
  session.execute = AsyncMock()
  session.commit = AsyncMock()
  session.scalar = AsyncMock()
  session.get = AsyncMock()
  session.delete = AsyncMock()
  session.refresh = AsyncMock()
  return session


@pytest.fixture
def plans_service():
  """Create a BillingPlansService instance."""
  return BillingPlansService(acquire=TestAcquire())


@pytest.fixture
def sample_user():
  """Create a sample user for testing."""
  return {"id": UUID("11111111-1111-1111-1111-111111111111"), "email": "test@example.com", "first_name": "Test", "last_name": "User"}


@pytest.fixture
def sample_plan_model():
  """Create a sample BillingPlanModel for testing."""
  plan = MagicMock()
  plan.id = UUID("22222222-2222-2222-2222-222222222222")
  plan.name = "Test Plan"
  plan.amount = 29.99
  plan.credits = 1000
  plan.discount_percentage = 0.0
  plan.currency = "USD"
  plan.is_active = True
  plan.created_at = datetime.datetime.now()
  plan.updated_at = datetime.datetime.now()
  return plan


@pytest.fixture
def sample_plan_create():
  """Create a sample BillingPlanCreateSchema for testing."""
  return BillingPlanCreateSchema(name="Test Plan", amount=29.99, credits=1000, discount_percentage=0.0, currency="USD", is_active=True)


@pytest.fixture
def sample_plan_update():
  """Create a sample BillingPlanUpdateSchema for testing."""
  return BillingPlanUpdateSchema(name="Updated Plan", amount=39.99, credits=2000, discount_percentage=10.0, is_active=False, currency="EUR")


class TestBillingPlansService:
  """Unit tests for BillingPlansService."""

  @pytest.mark.asyncio
  async def test_get_plan(self, plans_service, mock_db_session, sample_user, sample_plan_model):
    """Test get_plan method."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock
    mock_db_session.get.return_value = sample_plan_model

    # Call the method
    result = await plans_service.get_plan(plan_id=plan_id, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    assert result.id == sample_plan_model.id
    assert result.name == sample_plan_model.name
    assert result.amount == sample_plan_model.amount
    assert result.credits == sample_plan_model.credits
    assert result.discount_percentage == sample_plan_model.discount_percentage
    assert result.currency == sample_plan_model.currency
    assert result.is_active == sample_plan_model.is_active

  @pytest.mark.asyncio
  async def test_get_plan_not_found(self, plans_service, mock_db_session, sample_user):
    """Test get_plan method when plan is not found."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock to return None (plan not found)
    mock_db_session.get.return_value = None

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.get_plan(plan_id=plan_id, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Billing plan not found" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_get_plans(self, plans_service, mock_db_session, sample_user, sample_plan_model):
    """Test get_plans method."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [sample_plan_model, sample_plan_model]
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method
    result = await plans_service.get_plans(session=mock_db_session, user=sample_user)

    # Assertions
    assert mock_db_session.execute.call_count == 2  # One for query, one for count
    assert len(result["items"]) == 2
    assert result["pagination"]["total"] == 2
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["limit"] == 100

  @pytest.mark.asyncio
  async def test_get_plans_with_filters(self, plans_service, mock_db_session, sample_user, sample_plan_model):
    """Test get_plans method with filters."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [sample_plan_model]
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method with filters
    result = await plans_service.get_plans(is_active=True, currency="USD", limit=10, offset=5, session=mock_db_session, user=sample_user)

    # Assertions
    assert mock_db_session.execute.call_count == 2  # One for query, one for count
    assert len(result["items"]) == 1
    assert result["pagination"]["total"] == 1
    assert result["pagination"]["offset"] == 5
    assert result["pagination"]["limit"] == 10

  @pytest.mark.asyncio
  async def test_post_create(self, plans_service, mock_db_session, sample_user, sample_plan_create):
    """Test post_create method."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None  # No existing plan with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Create a mock for the new plan with proper datetime fields
    mock_new_plan = MagicMock()
    mock_new_plan.id = UUID("22222222-2222-2222-2222-222222222222")
    mock_new_plan.name = sample_plan_create.name
    mock_new_plan.amount = sample_plan_create.amount
    mock_new_plan.credits = sample_plan_create.credits
    mock_new_plan.discount_percentage = sample_plan_create.discount_percentage
    mock_new_plan.currency = sample_plan_create.currency
    mock_new_plan.is_active = sample_plan_create.is_active
    mock_new_plan.created_at = datetime.datetime.now()
    mock_new_plan.updated_at = datetime.datetime.now()

    # Mock the refresh to set our mock_new_plan as the refreshed object
    def mock_refresh_side_effect(obj):
      for key, value in mock_new_plan.__dict__.items():
        if not key.startswith("_"):
          setattr(obj, key, value)
      return None

    mock_db_session.refresh.side_effect = mock_refresh_side_effect

    # Call the method with mocked uuid4
    with patch("src.services.plans.service.uuid4", return_value=UUID("22222222-2222-2222-2222-222222222222")):
      result = await plans_service.post_create(plan_data=sample_plan_create, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()
    assert isinstance(result, BillingPlanResponseSchema)
    assert result.id == UUID("22222222-2222-2222-2222-222222222222")
    assert result.name == sample_plan_create.name

  @pytest.mark.asyncio
  async def test_post_create_duplicate_name(self, plans_service, mock_db_session, sample_user, sample_plan_create, sample_plan_model):
    """Test post_create method with duplicate name."""
    # Setup mocks to simulate existing plan with same name
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = sample_plan_model  # Existing plan with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.post_create(plan_data=sample_plan_create, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_patch_plan(self, plans_service, mock_db_session, sample_user, sample_plan_model, sample_plan_update):
    """Test patch_plan method."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_plan_model
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None  # No existing plan with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method
    result = await plans_service.patch_plan(plan_id=plan_id, plan_data=sample_plan_update, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()
    assert isinstance(result, BillingPlanResponseSchema)

  @pytest.mark.asyncio
  async def test_patch_plan_not_found(self, plans_service, mock_db_session, sample_user, sample_plan_update):
    """Test patch_plan method when plan is not found."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock to return None (plan not found)
    mock_db_session.get.return_value = None

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.patch_plan(plan_id=plan_id, plan_data=sample_plan_update, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Billing plan not found" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_patch_plan_duplicate_name(self, plans_service, mock_db_session, sample_user, sample_plan_model, sample_plan_update):
    """Test patch_plan method with duplicate name."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_plan_model

    # Mock to simulate existing plan with same name
    existing_plan = MagicMock()
    existing_plan.id = UUID("33333333-3333-3333-3333-333333333333")  # Different ID
    existing_plan.name = "Updated Plan"  # Same name as in update

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = existing_plan  # Existing plan with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.patch_plan(plan_id=plan_id, plan_data=sample_plan_update, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_delete_plan(self, plans_service, mock_db_session, sample_user, sample_plan_model):
    """Test delete_plan method."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_plan_model

    # For the debug query in delete_plan
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [sample_plan_model]
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method
    result = await plans_service.delete_plan(plan_id=plan_id, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    mock_db_session.delete.assert_called_once_with(sample_plan_model)
    mock_db_session.commit.assert_called_once()
    assert "deleted successfully" in result["message"]

  @pytest.mark.asyncio
  async def test_delete_plan_not_found(self, plans_service, mock_db_session, sample_user):
    """Test delete_plan method when plan is not found."""
    plan_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []  # No plans found
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result
    mock_db_session.get.return_value = None  # Plan not found

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.delete_plan(plan_id=plan_id, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Billing plan not found" in excinfo.value.detail


###########################################
# INTEGRATION TESTS
###########################################


@pytest.fixture
def test_integration_plan():
  """Create a plan data for integration tests."""
  return {"name": "Integration Test Plan", "amount": 49.99, "credits": 5000, "discount_percentage": 5.0, "currency": "USD", "is_active": True}


@pytest.mark.skipif(not is_integration_test(), reason="Integration tests disabled")
class TestBillingPlansServiceIntegration:
  """Integration tests for BillingPlansService."""

  @pytest_asyncio.fixture
  async def session(self, db_session):
    """Get the actual session from the db_session fixture."""
    async for session in db_session:
      yield session

  async def _clean_test_data(self, session):
    """Helper to clean test data."""
    stmt = delete(BillingPlanModel).where(BillingPlanModel.name.like("Integration Test%"))
    await session.execute(stmt)
    await session.commit()

  @pytest.mark.asyncio
  async def test_plan_crud_integration(self, plans_service, session):
    """Test full CRUD operations for plans in integration."""
    # Clean up any existing test data first
    await self._clean_test_data(session)

    user = {"id": uuid4()}

    # 1. Create a new plan
    create_data = BillingPlanCreateSchema(
      name="Integration Test Plan", amount=49.99, credits=5000, discount_percentage=5.0, currency="USD", is_active=True
    )

    created_plan = await plans_service.post_create(plan_data=create_data, session=session, user=user)

    assert created_plan.name == "Integration Test Plan"
    assert created_plan.amount == 49.99
    assert created_plan.credits == 5000
    plan_id = created_plan.id

    # 2. Get the plan by ID
    retrieved_plan = await plans_service.get_plan(plan_id=plan_id, session=session, user=user)

    assert retrieved_plan.id == plan_id
    assert retrieved_plan.name == "Integration Test Plan"

    # 3. Get all plans and verify our plan is in the list
    all_plans = await plans_service.get_plans(session=session, user=user)

    plan_ids = [plan.id for plan in all_plans["items"]]
    assert plan_id in plan_ids

    # 4. Update the plan
    update_data = BillingPlanUpdateSchema(
      name="Integration Test Plan Updated", amount=59.99, credits=6000, discount_percentage=5.0, is_active=True, currency="USD"
    )

    updated_plan = await plans_service.patch_plan(plan_id=plan_id, plan_data=update_data, session=session, user=user)

    assert updated_plan.name == "Integration Test Plan Updated"
    assert updated_plan.amount == 59.99
    assert updated_plan.credits == 6000

    # 5. Delete the plan
    delete_result = await plans_service.delete_plan(plan_id=plan_id, session=session, user=user)

    assert "deleted successfully" in delete_result["message"]

    # 6. Verify the plan is deleted
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.get_plan(plan_id=plan_id, session=session, user=user)

    assert excinfo.value.status_code == 404

  @pytest.mark.asyncio
  async def test_plans_filtering_integration(self, plans_service, session):
    """Test filtering plans in integration."""
    # Clean up any existing test data first
    await self._clean_test_data(session)

    user = {"id": uuid4()}

    # Create multiple plans with different attributes
    plans_data = [
      BillingPlanCreateSchema(name="Integration Test Plan USD", amount=49.99, credits=5000, currency="USD", is_active=True),
      BillingPlanCreateSchema(name="Integration Test Plan EUR", amount=45.99, credits=5000, currency="EUR", is_active=True),
      BillingPlanCreateSchema(name="Integration Test Plan Inactive", amount=29.99, credits=3000, currency="USD", is_active=False),
    ]

    created_plans = []
    for plan_data in plans_data:
      plan = await plans_service.post_create(plan_data=plan_data, session=session, user=user)
      created_plans.append(plan)

    # Test filtering by currency
    usd_plans = await plans_service.get_plans(currency="USD", session=session, user=user)

    usd_plan_names = [plan.name for plan in usd_plans["items"]]
    assert "Integration Test Plan USD" in usd_plan_names
    assert "Integration Test Plan EUR" not in usd_plan_names

    # Test filtering by active status
    active_plans = await plans_service.get_plans(is_active=True, session=session, user=user)

    active_plan_names = [plan.name for plan in active_plans["items"]]
    assert "Integration Test Plan USD" in active_plan_names
    assert "Integration Test Plan EUR" in active_plan_names
    assert "Integration Test Plan Inactive" not in active_plan_names

    # Test combined filters
    active_usd_plans = await plans_service.get_plans(is_active=True, currency="USD", session=session, user=user)

    active_usd_plan_names = [plan.name for plan in active_usd_plans["items"]]
    assert "Integration Test Plan USD" in active_usd_plan_names
    assert "Integration Test Plan EUR" not in active_usd_plan_names
    assert "Integration Test Plan Inactive" not in active_usd_plan_names

    # Clean up
    for plan in created_plans:
      await plans_service.delete_plan(plan_id=plan.id, session=session, user=user)

  @pytest.mark.asyncio
  async def test_plan_duplicate_name_integration(self, plans_service, session, test_integration_plan):
    """Test duplicate name handling in integration."""
    # Clean up any existing test data first
    await self._clean_test_data(session)

    user = {"id": uuid4()}

    # Create a plan
    create_data = BillingPlanCreateSchema(**test_integration_plan)
    created_plan = await plans_service.post_create(plan_data=create_data, session=session, user=user)

    # Try to create another plan with the same name
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.post_create(plan_data=create_data, session=session, user=user)

    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

    # Try to update another plan to have the same name
    # First create another plan
    another_plan_data = BillingPlanCreateSchema(name="Integration Test Plan 2", amount=19.99, credits=2000, currency="USD")
    another_plan = await plans_service.post_create(plan_data=another_plan_data, session=session, user=user)

    # Try to update it to have the same name as the first plan
    update_data = BillingPlanUpdateSchema(
      name=test_integration_plan["name"],
      amount=test_integration_plan["amount"],
      credits=test_integration_plan["credits"],
      discount_percentage=test_integration_plan["discount_percentage"],
      is_active=test_integration_plan["is_active"],
      currency=test_integration_plan["currency"],
    )
    with pytest.raises(HTTPException) as excinfo:
      await plans_service.patch_plan(plan_id=another_plan.id, plan_data=update_data, session=session, user=user)

    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

    # Clean up
    await plans_service.delete_plan(plan_id=created_plan.id, session=session, user=user)
    await plans_service.delete_plan(plan_id=another_plan.id, session=session, user=user)
