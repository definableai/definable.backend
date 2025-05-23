import datetime
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException

from src.services.__base.acquire import Acquire
from src.services.charges.schema import ChargeCreateSchema, ChargeResponseSchema, ChargeUpdateSchema
from src.services.charges.service import ChargesService


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
def charges_service():
  """Create a ChargesService instance."""
  return ChargesService(acquire=TestAcquire())


@pytest.fixture
def sample_user():
  """Create a sample user for testing."""
  return {"id": UUID("11111111-1111-1111-1111-111111111111"), "email": "test@example.com", "first_name": "Test", "last_name": "User"}


@pytest.fixture
def sample_charge_model():
  """Create a sample ChargeModel for testing."""
  charge = MagicMock()
  charge.id = UUID("22222222-2222-2222-2222-222222222222")
  charge.name = "text_generation"
  charge.amount = 50
  charge.unit = "credit"
  charge.measure = "tokens"
  charge.service = "llm"
  charge.action = "completion"
  charge.description = "Credits per 1000 tokens for text generation"
  charge.is_active = True
  charge.created_at = datetime.datetime.now()
  charge.updated_at = datetime.datetime.now()
  return charge


@pytest.fixture
def sample_charge_create():
  """Create a sample ChargeCreateSchema for testing."""
  return ChargeCreateSchema(
    name="text_generation",
    amount=50,
    unit="credit",
    measure="tokens",
    service="llm",
    action="completion",
    description="Credits per 1000 tokens for text generation",
    is_active=True,
  )


@pytest.fixture
def sample_charge_update():
  """Create a sample ChargeUpdateSchema for testing."""
  return ChargeUpdateSchema(
    name="text_generation_updated",
    amount=75,
    unit="credit",
    measure="tokens",
    service="llm",
    action="completion",
    description="Updated description",
    is_active=False,
  )


class TestChargesService:
  """Unit tests for ChargesService."""

  @pytest.mark.asyncio
  async def test_get_charge(self, charges_service, mock_db_session, sample_user, sample_charge_model):
    """Test get_charge method."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock
    mock_db_session.get.return_value = sample_charge_model

    # Call the method
    result = await charges_service.get_charge(charge_id=charge_id, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    assert result.id == sample_charge_model.id
    assert result.name == sample_charge_model.name
    assert result.amount == sample_charge_model.amount
    assert result.unit == sample_charge_model.unit
    assert result.measure == sample_charge_model.measure
    assert result.service == sample_charge_model.service
    assert result.action == sample_charge_model.action
    assert result.description == sample_charge_model.description
    assert result.is_active == sample_charge_model.is_active

  @pytest.mark.asyncio
  async def test_get_charge_not_found(self, charges_service, mock_db_session, sample_user):
    """Test get_charge method when charge is not found."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock to return None (charge not found)
    mock_db_session.get.return_value = None

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.get_charge(charge_id=charge_id, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Charge not found" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_get_charges(self, charges_service, mock_db_session, sample_user, sample_charge_model):
    """Test get_charges method."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [sample_charge_model, sample_charge_model]
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method
    result = await charges_service.get_charges(session=mock_db_session, user=sample_user)

    # Assertions
    assert mock_db_session.execute.call_count == 2  # One for query, one for count
    assert len(result["items"]) == 2
    assert result["pagination"]["total"] == 2
    assert result["pagination"]["offset"] == 0
    assert result["pagination"]["limit"] == 100

  @pytest.mark.asyncio
  async def test_get_charges_with_filters(self, charges_service, mock_db_session, sample_user, sample_charge_model):
    """Test get_charges method with filters."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [sample_charge_model]
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method with filters
    result = await charges_service.get_charges(is_active=True, service="llm", limit=10, offset=5, session=mock_db_session, user=sample_user)

    # Assertions
    assert mock_db_session.execute.call_count == 2  # One for query, one for count
    assert len(result["items"]) == 1
    assert result["pagination"]["total"] == 1
    assert result["pagination"]["offset"] == 5
    assert result["pagination"]["limit"] == 10

  @pytest.mark.asyncio
  async def test_post_charge(self, charges_service, mock_db_session, sample_user, sample_charge_create):
    """Test post_charge method."""
    # Setup mocks
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None  # No existing charge with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Create a mock for the new charge with proper datetime fields
    mock_new_charge = MagicMock()
    mock_new_charge.id = UUID("22222222-2222-2222-2222-222222222222")
    mock_new_charge.name = sample_charge_create.name
    mock_new_charge.amount = sample_charge_create.amount
    mock_new_charge.unit = sample_charge_create.unit
    mock_new_charge.measure = sample_charge_create.measure
    mock_new_charge.service = sample_charge_create.service
    mock_new_charge.action = sample_charge_create.action
    mock_new_charge.description = sample_charge_create.description
    mock_new_charge.is_active = sample_charge_create.is_active
    mock_new_charge.created_at = datetime.datetime.now()
    mock_new_charge.updated_at = datetime.datetime.now()

    # Mock the refresh to set our mock_new_charge as the refreshed object
    def mock_refresh_side_effect(obj):
      for key, value in mock_new_charge.__dict__.items():
        if not key.startswith("_"):
          setattr(obj, key, value)
      return None

    mock_db_session.refresh.side_effect = mock_refresh_side_effect

    # Call the method with mocked uuid4
    with patch("uuid.uuid4", return_value=UUID("22222222-2222-2222-2222-222222222222")):
      result = await charges_service.post_charge(charge_data=sample_charge_create, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()
    assert isinstance(result, ChargeResponseSchema)
    assert result.id == UUID("22222222-2222-2222-2222-222222222222")
    assert result.name == sample_charge_create.name

  @pytest.mark.asyncio
  async def test_post_charge_duplicate_name(self, charges_service, mock_db_session, sample_user, sample_charge_create, sample_charge_model):
    """Test post_charge method with duplicate name."""
    # Setup mocks to simulate existing charge with same name
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = sample_charge_model  # Existing charge with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.post_charge(charge_data=sample_charge_create, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_patch_charge(self, charges_service, mock_db_session, sample_user, sample_charge_model, sample_charge_update):
    """Test patch_charge method."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_charge_model
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None  # No existing charge with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method
    result = await charges_service.patch_charge(charge_id=charge_id, charge_data=sample_charge_update, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()
    assert isinstance(result, ChargeResponseSchema)

  @pytest.mark.asyncio
  async def test_patch_charge_not_found(self, charges_service, mock_db_session, sample_user, sample_charge_update):
    """Test patch_charge method when charge is not found."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock to return None (charge not found)
    mock_db_session.get.return_value = None

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.patch_charge(charge_id=charge_id, charge_data=sample_charge_update, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Charge not found" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_patch_charge_duplicate_name(self, charges_service, mock_db_session, sample_user, sample_charge_model, sample_charge_update):
    """Test patch_charge method with duplicate name."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_charge_model

    # Mock to simulate existing charge with same name
    existing_charge = MagicMock()
    existing_charge.id = UUID("33333333-3333-3333-3333-333333333333")  # Different ID
    existing_charge.name = "text_generation_updated"  # Same name as in update

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = existing_charge  # Existing charge with same name
    mock_result.scalars.return_value = mock_scalars
    mock_db_session.execute.return_value = mock_result

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.patch_charge(charge_id=charge_id, charge_data=sample_charge_update, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

  @pytest.mark.asyncio
  async def test_delete_charge(self, charges_service, mock_db_session, sample_user, sample_charge_model):
    """Test delete_charge method."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mocks
    mock_db_session.get.return_value = sample_charge_model

    # Call the method
    result = await charges_service.delete_charge(charge_id=charge_id, session=mock_db_session, user=sample_user)

    # Assertions
    mock_db_session.get.assert_called_once()
    mock_db_session.delete.assert_called_once_with(sample_charge_model)
    mock_db_session.commit.assert_called_once()
    assert "deleted successfully" in result["message"]

  @pytest.mark.asyncio
  async def test_delete_charge_not_found(self, charges_service, mock_db_session, sample_user):
    """Test delete_charge method when charge is not found."""
    charge_id = UUID("22222222-2222-2222-2222-222222222222")

    # Setup mock to return None (charge not found)
    mock_db_session.get.return_value = None

    # Call the method and expect exception
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.delete_charge(charge_id=charge_id, session=mock_db_session, user=sample_user)

    # Assertions
    assert excinfo.value.status_code == 404
    assert "Charge not found" in excinfo.value.detail


###########################################
# INTEGRATION TESTS
###########################################


@pytest.fixture
def test_integration_charge():
  """Create a charge data for integration tests."""
  return {
    "name": "integration_test_charge",
    "amount": 100,
    "unit": "credit",
    "measure": "tokens",
    "service": "test_service",
    "action": "test_action",
    "description": "Integration test charge",
    "is_active": True,
  }


@pytest_asyncio.fixture
async def setup_test_db_integration():
  """Setup and teardown for integration tests."""
  from sqlalchemy import delete, text

  from models import ChargeModel

  # Import directly from conftest to get the session maker
  from tests.conftest import TestingSessionLocal

  # Create a new session directly
  async with TestingSessionLocal() as session:
    # First check if charges table exists
    try:
      # Try to create the charges table if it doesn't exist
      await session.execute(
        text("""
        CREATE TABLE IF NOT EXISTS charges (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          name VARCHAR NOT NULL UNIQUE,
          amount INTEGER NOT NULL,
          unit VARCHAR NOT NULL DEFAULT 'credit',
          measure VARCHAR NOT NULL,
          service VARCHAR NOT NULL,
          action VARCHAR NOT NULL,
          description VARCHAR NULL,
          is_active BOOLEAN NOT NULL DEFAULT true,
          created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
      """)
      )
      await session.commit()
    except Exception as e:
      await session.rollback()
      print(f"Error creating charges table: {e}")

    # Clean up any existing test data before tests
    try:
      stmt = delete(ChargeModel).where(ChargeModel.name.like("integration_test%"))
      await session.execute(stmt)
      await session.commit()
    except Exception as e:
      await session.rollback()
      print(f"Error cleaning test data: {e}")

    # Return the session for use in tests
    yield session

    # Rollback at the end for cleanup
    await session.rollback()


@pytest.mark.skipif(not is_integration_test(), reason="Integration tests disabled")
class TestChargesServiceIntegration:
  """Integration tests for ChargesService."""

  @pytest.mark.asyncio
  async def test_charge_crud_integration(self, charges_service, setup_test_db_integration):
    """Test full CRUD operations for charges in integration."""
    db_session = setup_test_db_integration
    user = {"id": uuid4()}

    # 1. Create a new charge
    create_data = ChargeCreateSchema(
      name="integration_test_charge",
      amount=100,
      unit="credit",
      measure="tokens",
      service="test_service",
      action="test_action",
      description="Integration test charge",
      is_active=True,
    )

    created_charge = await charges_service.post_charge(charge_data=create_data, session=db_session, user=user)

    assert created_charge.name == "integration_test_charge"
    assert created_charge.amount == 100
    assert created_charge.service == "test_service"
    charge_id = created_charge.id

    # 2. Get the charge by ID
    retrieved_charge = await charges_service.get_charge(charge_id=charge_id, session=db_session, user=user)

    assert retrieved_charge.id == charge_id
    assert retrieved_charge.name == "integration_test_charge"

    # 3. Get all charges and verify our charge is in the list
    all_charges = await charges_service.get_charges(session=db_session, user=user)

    charge_ids = [charge.id for charge in all_charges["items"]]
    assert charge_id in charge_ids

    # 4. Update the charge
    update_data = ChargeUpdateSchema(
      name="integration_test_charge_updated",
      amount=150,
      unit="credit",
      measure="tokens",
      service="test_service",
      action="test_action",
      description="Updated integration test charge",
      is_active=True,
    )

    updated_charge = await charges_service.patch_charge(charge_id=charge_id, charge_data=update_data, session=db_session, user=user)

    assert updated_charge.name == "integration_test_charge_updated"
    assert updated_charge.amount == 150
    assert updated_charge.description == "Updated integration test charge"

    # 5. Delete the charge
    delete_result = await charges_service.delete_charge(charge_id=charge_id, session=db_session, user=user)

    assert "deleted successfully" in delete_result["message"]

    # 6. Verify the charge is deleted
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.get_charge(charge_id=charge_id, session=db_session, user=user)

    assert excinfo.value.status_code == 404

  @pytest.mark.asyncio
  async def test_charges_filtering_integration(self, charges_service, setup_test_db_integration):
    """Test filtering charges in integration."""
    db_session = setup_test_db_integration
    user = {"id": uuid4()}

    # Create multiple charges with different attributes
    charges_data = [
      ChargeCreateSchema(
        name="integration_test_charge_llm",
        amount=100,
        unit="credit",
        measure="tokens",
        service="llm",
        action="completion",
        description="LLM charge",
        is_active=True,
      ),
      ChargeCreateSchema(
        name="integration_test_charge_vision",
        amount=200,
        unit="credit",
        measure="images",
        service="vision",
        action="generation",
        description="Vision charge",
        is_active=True,
      ),
      ChargeCreateSchema(
        name="integration_test_charge_inactive",
        amount=50,
        unit="credit",
        measure="tokens",
        service="llm",
        action="embedding",
        description="Inactive charge",
        is_active=False,
      ),
    ]

    created_charges = []
    for charge_data in charges_data:
      charge = await charges_service.post_charge(charge_data=charge_data, session=db_session, user=user)
      created_charges.append(charge)

    # Test filtering by service
    llm_charges = await charges_service.get_charges(service="llm", session=db_session, user=user)

    llm_charge_names = [charge.name for charge in llm_charges["items"]]
    assert "integration_test_charge_llm" in llm_charge_names
    assert "integration_test_charge_vision" not in llm_charge_names

    # Test filtering by active status
    active_charges = await charges_service.get_charges(is_active=True, session=db_session, user=user)

    active_charge_names = [charge.name for charge in active_charges["items"]]
    assert "integration_test_charge_llm" in active_charge_names
    assert "integration_test_charge_vision" in active_charge_names
    assert "integration_test_charge_inactive" not in active_charge_names

    # Test combined filters
    active_llm_charges = await charges_service.get_charges(is_active=True, service="llm", session=db_session, user=user)

    active_llm_charge_names = [charge.name for charge in active_llm_charges["items"]]
    assert "integration_test_charge_llm" in active_llm_charge_names
    assert "integration_test_charge_vision" not in active_llm_charge_names
    assert "integration_test_charge_inactive" not in active_llm_charge_names

    # Clean up
    for charge in created_charges:
      await charges_service.delete_charge(charge_id=charge.id, session=db_session, user=user)

  @pytest.mark.asyncio
  async def test_charge_duplicate_name_integration(self, charges_service, setup_test_db_integration, test_integration_charge):
    """Test duplicate name handling in integration."""
    db_session = setup_test_db_integration
    user = {"id": uuid4()}

    # Create a charge
    create_data = ChargeCreateSchema(**test_integration_charge)
    created_charge = await charges_service.post_charge(charge_data=create_data, session=db_session, user=user)

    # Try to create another charge with the same name
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.post_charge(charge_data=create_data, session=db_session, user=user)

    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

    # Try to update another charge to have the same name
    # First create another charge
    another_charge_data = ChargeCreateSchema(
      name="integration_test_charge_2", amount=75, unit="credit", measure="tokens", service="llm", action="embedding", description="Another charge"
    )
    another_charge = await charges_service.post_charge(charge_data=another_charge_data, session=db_session, user=user)

    # Try to update it to have the same name as the first charge
    update_data = ChargeUpdateSchema(
      name=test_integration_charge["name"],
      amount=75,
      unit="credit",
      measure="tokens",
      service="llm",
      action="embedding",
      description="Another charge",
      is_active=True,
    )
    with pytest.raises(HTTPException) as excinfo:
      await charges_service.patch_charge(charge_id=another_charge.id, charge_data=update_data, session=db_session, user=user)

    assert excinfo.value.status_code == 400
    assert "already exists" in excinfo.value.detail

    # Clean up
    await charges_service.delete_charge(charge_id=created_charge.id, session=db_session, user=user)
    await charges_service.delete_charge(charge_id=another_charge.id, session=db_session, user=user)
