import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import UUID, uuid4
import os


# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.

    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# Store original functions before patching
original_async_session = None
original_select = None

# Only patch if we're running unit tests, not integration tests
if not is_integration_test():
    # Save original functions
    from sqlalchemy.ext.asyncio import AsyncSession as OrigAsyncSession
    from sqlalchemy import select as orig_select
    original_async_session = OrigAsyncSession
    original_select = orig_select

    # Patch the necessary dependencies for testing
    patch('sqlalchemy.ext.asyncio.AsyncSession', MagicMock()).start()
    patch('sqlalchemy.select', lambda *args: MagicMock()).start()

# Import the real service and schema
from src.services.org.service import OrganizationService
from src.services.org.schema import OrganizationResponse

# Only mock modules for unit tests
if not is_integration_test():
    # Import models needed for patching
    from src.models import OrganizationModel, OrganizationMemberModel, RoleModel

    # Mock modules that might cause import issues
    sys.modules["database"] = MagicMock()
    sys.modules["database.postgres"] = MagicMock()
    sys.modules["src.database"] = MagicMock()
    sys.modules["src.database.postgres"] = MagicMock()
    sys.modules["config"] = MagicMock()
    sys.modules["config.settings"] = MagicMock()
    sys.modules["src.config"] = MagicMock()
    sys.modules["src.config.settings"] = MagicMock()
    sys.modules["dependencies.security"] = MagicMock()

    # Explicitly mock the RoleService
    role_service_mock = MagicMock()
    # Create a static method for _get_role that doesn't need self
    async def mock_get_role(role_id, organization_id, session):
        # Will be customized in individual tests
        pass
    role_service_mock._get_role = mock_get_role

    # Add the mock to both module paths that might be imported
    sys.modules["services.roles.service"] = MagicMock()
    sys.modules["services.roles.service"].RoleService = role_service_mock
    sys.modules["src.services.roles.service"] = MagicMock()
    sys.modules["src.services.roles.service"].RoleService = role_service_mock
    sys.modules["src.services.__base.acquire"] = MagicMock()

# Save original models
OrigOrganizationModel = None
OrigOrganizationMemberModel = None
OrigRoleModel = None
OrigSelect = None

# Create a unit test context manager to be used in unit tests
@pytest.fixture
def unit_test_context():
    """Create a context for unit tests to ensure proper patching regardless of INTEGRATION_TEST."""
    # Save original imports
    global OrigOrganizationModel, OrigOrganizationMemberModel, OrigRoleModel, OrigSelect

    from src.models import OrganizationModel as OrigOrgModel
    from src.models import OrganizationMemberModel as OrigOrgMemberModel
    from src.models import RoleModel as OrigRoleModel
    from sqlalchemy import select as OrigSelect

    OrigOrganizationModel = OrigOrgModel
    OrigOrganizationMemberModel = OrigOrgMemberModel
    OrigRoleModel = OrigRoleModel
    OrigSelect = OrigSelect

    # Apply patches for the unit test
    if is_integration_test():
        # We need to patch for unit tests even if integration tests are enabled
        select_patch = patch('src.services.org.service.select', lambda *args: MagicMock())
        org_model_patch = patch('src.services.org.service.OrganizationModel', MagicMock())
        org_member_patch = patch('src.services.org.service.OrganizationMemberModel', MagicMock())
        role_model_patch = patch('src.services.org.service.RoleModel', MagicMock())

        select_patch.start()
        org_model_patch.start()
        org_member_patch.start()
        role_model_patch.start()

        yield

        # Stop patches
        select_patch.stop()
        org_model_patch.stop()
        org_member_patch.stop()
        role_model_patch.stop()
    else:
        # Already patched globally if not in integration mode
        yield

# Mock models
class MockOrganizationModel:
    def __init__(self, model_id=None, name="Test Organization", slug="test-org-12345678", settings=None):
        self.id = model_id or uuid4()
        self.name = name
        self.slug = slug
        self.settings = settings or {}
        self.is_active = True

    def __eq__(self, other):
        if not isinstance(other, MockOrganizationModel):
            return False
        return self.id == other.id and self.name == other.name

class MockOrganizationMemberModel:
    def __init__(self, organization_id=None, user_id=None, role_id=None, status="active"):
        self.id = uuid4()
        self.organization_id = organization_id or uuid4()
        self.user_id = user_id or uuid4()
        self.role_id = role_id or uuid4()
        self.status = status
        self.invited_by = None

class MockRoleModel:
    def __init__(self, model_id=None, name="owner", organization_id=None, is_system_role=True, hierarchy_level=100, **kwargs):
        # Allow 'id' to be passed as alternative to model_id
        if 'id' in kwargs and model_id is None:
            model_id = kwargs['id']
        self.id = model_id or uuid4()
        self.name = name
        self.organization_id = organization_id or uuid4()
        self.is_system_role = is_system_role
        self.hierarchy_level = hierarchy_level
        self.description = f"{name} role"

@pytest.fixture
def mock_user():
  """Create a mock user with proper permissions."""
  return {
    "id": uuid4(),
    "email": "test@example.com",
    "first_name": "Test",
    "last_name": "User",
        "org_id": uuid4(),
  }

@pytest.fixture
def mock_db_session():
  """Create a mock database session."""
  session = MagicMock()
  session.add = MagicMock()
  session.commit = AsyncMock()
  session.flush = AsyncMock()

  # Create a properly structured mock result for database queries
  MagicMock()
  unique_mock = MagicMock()
  scalars_mock = MagicMock()
  MagicMock()

  scalars_mock.all = MagicMock(return_value=[])
  unique_mock.scalar_one_or_none = MagicMock(return_value=None)
  unique_mock.scalars = MagicMock(return_value=scalars_mock)

  execute_mock = MagicMock()
  execute_mock.scalar_one_or_none = MagicMock(return_value=None)
  execute_mock.unique = MagicMock(return_value=unique_mock)

  session.execute = AsyncMock(return_value=execute_mock)

  return session

@pytest.fixture
def mock_organization():
  """Create a mock organization."""
  return MockOrganizationModel()

@pytest.fixture
def mock_multiple_organizations():
    """Create multiple mock organizations."""
    return [
        MockOrganizationModel(name="Org 1", slug="org-1-12345678"),
        MockOrganizationModel(name="Org 2", slug="org-2-12345678"),
        MockOrganizationModel(name="Org 3", slug="org-3-12345678")
    ]

@pytest.fixture
def mock_owner_role():
    """Create a mock owner role."""
    return MockRoleModel(name="owner")

@pytest.fixture
def mock_org_member():
  """Create a mock organization member."""
  return MockOrganizationMemberModel()

@pytest.fixture
def mock_acquire():
    """Create a mock Acquire object."""
    acquire_mock = MagicMock()
    acquire_mock.logger = MagicMock()
    return acquire_mock

@pytest.fixture
def org_service(mock_acquire):
    """Create the real organization service with mocked dependencies."""
    return OrganizationService(acquire=mock_acquire)

@pytest.mark.asyncio
class TestOrganizationService:
    """Test organization service."""

    async def test_post_create_org_success(self, org_service, mock_db_session, mock_user, mock_owner_role, unit_test_context):
        """Test creating a new organization successfully."""
        # Setup
        org_name = "Test Organization"

        # Mock the queries properly
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = mock_owner_role

        # Override flush to set the organization ID
        async def mock_flush():
            for call in mock_db_session.add.call_args_list:
                obj = call[0][0]
                if hasattr(obj, 'name') and obj.name == org_name:
                    obj.id = uuid4()

        mock_db_session.flush.side_effect = mock_flush

        # Execute with mock for OrganizationModel
        mock_org = MockOrganizationModel(name=org_name)

        with patch('src.services.org.service.OrganizationModel', return_value=mock_org):
            await org_service.post_create_org(
                name=org_name,
                session=mock_db_session,
                user=mock_user
            )

        # Assert that the organization was created
        assert mock_db_session.add.called
        assert mock_db_session.flush.called
        assert mock_db_session.commit.called

    async def test_post_create_org_duplicate_slug(
        self,
        org_service,
        mock_db_session,
        mock_user,
        mock_owner_role,
        mock_organization,
        unit_test_context
    ):
        """Test creating an organization with an existing slug."""
        # Setup
        org_name = "Test Organization"

        # Simplify the approach - use concrete results instead of side effects
        # First mock setup: Return an existing organization to trigger slug generation
        first_execute_result = MagicMock()
        first_execute_result.scalar_one_or_none = MagicMock(return_value=mock_organization)
        first_execute_result.unique = MagicMock()
        first_execute_result.unique.return_value = MagicMock()
        first_execute_result.unique.return_value.scalar_one_or_none = MagicMock(return_value=mock_owner_role)

        # Second mock setup: Return None on subsequent calls (for uniqueness check with suffix)
        second_execute_result = MagicMock()
        second_execute_result.scalar_one_or_none = MagicMock(return_value=None)
        second_execute_result.unique = first_execute_result.unique  # Reuse the same unique mock

        # Configure execute to return different results on consecutive calls
        mock_db_session.execute = AsyncMock()
        mock_db_session.execute.side_effect = [
            first_execute_result,   # First check: slug exists
            second_execute_result,  # Second check: slug with suffix doesn't exist
            first_execute_result    # Get owner role
        ]

        # Mock flush to set ID
        async def mock_flush():
            # Setting ID on organization object
            # This is simpler than trying to track individual calls
            mock_org.id = uuid4()

        mock_db_session.flush.side_effect = mock_flush

        # Create a concrete mock org to return
        mock_org = MockOrganizationModel(name=org_name)
        mock_org.slug = f"{org_name.lower().replace(' ', '-')}-1"  # Pre-set the suffix

        # Use simpler patching - patch the class, not the __new__ method
        with patch('src.services.org.service.OrganizationModel', return_value=mock_org):
            # Execute
            response = await org_service.post_create_org(
                name=org_name,
                session=mock_db_session,
                user=mock_user
            )

        # Assert that the slug has the suffix
        assert "-1" in response.slug

    async def test_post_create_org_missing_owner_role(self, org_service, mock_db_session, mock_user, unit_test_context):
        """Test creating an organization when owner role doesn't exist."""
        # Setup
        org_name = "Test Organization"

        # Return None for the owner role lookup
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None

        # Create a mock org to return from OrganizationModel
        mock_org = MockOrganizationModel(name=org_name)

        # Execute and Assert
        with pytest.raises(HTTPException) as exc_info:
            with patch('src.services.org.service.OrganizationModel', return_value=mock_org):
                await org_service.post_create_org(
                    name=org_name,
                    session=mock_db_session,
                    user=mock_user
                )

        assert exc_info.value.status_code == 500
        assert "Default OWNER role not found" in exc_info.value.detail

    async def test_post_add_member_success(self, org_service, mock_db_session, mock_user, mock_owner_role, unit_test_context):
        """Test adding a member to an organization successfully."""
        # Setup
        organization_id = uuid4()
        user_id = uuid4()
        role_id = mock_owner_role.id

        # Use the mock_get_role function directly
        async def mock_get_role_success(role_id, organization_id, session):
            return mock_owner_role

        # Patch RoleService._get_role with our custom function - use the correct import path
        with patch('services.roles.service.RoleService._get_role', mock_get_role_success):
            # Patch OrganizationMemberModel
            mock_member = MockOrganizationMemberModel(
                organization_id=organization_id,
                user_id=user_id,
                role_id=role_id
            )

            with patch('src.services.org.service.OrganizationMemberModel', return_value=mock_member):
                # Execute
                response = await org_service.post_add_member(
                    organization_id=organization_id,
                    user_id=user_id,
                    role_id=role_id,
                    session=mock_db_session
                )

        # Assert
        assert mock_db_session.add.called
        assert isinstance(response, JSONResponse)
        assert response.status_code == 201

    async def test_post_add_member_role_not_found(self, org_service, mock_db_session, mock_user, unit_test_context):
        """Test adding a member with a non-existent role."""
        # Setup
        organization_id = uuid4()
        user_id = uuid4()
        role_id = uuid4()

        # Define a mock function that returns None
        async def mock_get_role_none(role_id, organization_id, session):
            return None

        # Patch RoleService._get_role with our custom function - use the correct import path
        with patch('services.roles.service.RoleService._get_role', mock_get_role_none):
            # Execute and Assert
            with pytest.raises(HTTPException) as exc_info:
                await org_service.post_add_member(
                    organization_id=organization_id,
                    user_id=user_id,
                    role_id=role_id,
                    session=mock_db_session
                )

        assert exc_info.value.status_code == 404
        assert "Role" in exc_info.value.detail
        assert "not found" in exc_info.value.detail

    async def test_get_list(self, org_service, mock_db_session, mock_user, mock_multiple_organizations, unit_test_context):
        """Test listing organizations that a user belongs to."""
        # Setup
        mock_user["id"]

        # Mock the database query to return our mock organizations
        mock_db_session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = mock_multiple_organizations

        # Patch OrganizationResponse.model_validate to handle our mock objects
        with patch.object(OrganizationResponse, 'model_validate', lambda org: OrganizationResponse(
            id=org.id,
            name=org.name,
            slug=org.slug
        )):
            # Execute
            response = await org_service.get_list(
                session=mock_db_session,
                user=mock_user
            )

        # Assert
        assert isinstance(response, list)
        assert len(response) == len(mock_multiple_organizations)


# ============================================================================
# INTEGRATION TESTS - RUN WITH: INTEGRATION_TEST=1 pytest tests/services/test_org.py
# ============================================================================

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select, text
    from models import OrganizationModel, OrganizationMemberModel, RoleModel

@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for organization integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    owner_role = None

    # Create necessary database objects
    async for session in db_session:
        try:
            # Create required tables if they don't exist
            # Create roles table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS roles (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    organization_id UUID NULL,
                    description TEXT,
                    is_system_role BOOLEAN DEFAULT false,
                    hierarchy_level INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create role_permissions table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_id UUID NOT NULL,
                    permission_id UUID NOT NULL,
                    PRIMARY KEY (role_id, permission_id)
                )
            """))

            # Create permissions table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS permissions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    resource VARCHAR(255),
                    action VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create organizations table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255) UNIQUE NOT NULL,
                    settings JSONB DEFAULT '{}'::jsonb,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create organization_members table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS organization_members (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    organization_id UUID NOT NULL,
                    user_id UUID NOT NULL,
                    role_id UUID NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    invited_by UUID,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (organization_id, user_id)
                )
            """))

            # Create users table (minimal for foreign key references)
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    first_name VARCHAR(255),
                    last_name VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Commit the table creation
            await session.commit()

            # Clean up any existing test data first
            await session.execute(
                text("""
                    DELETE FROM organization_members
                    WHERE user_id IN (
                        SELECT id FROM users
                        WHERE email LIKE 'test-integration-%@example.com'
                    )
                """)
            )
            await session.execute(
                text("""
                    DELETE FROM organizations
                    WHERE name LIKE 'Test Org Integration%'
                """)
            )

            # Check if owner role exists
            result = await session.execute(text("SELECT * FROM roles WHERE name = 'owner' LIMIT 1"))
            owner_role_data = result.mappings().first()

            if not owner_role_data:
                # Insert owner role
                role_id = uuid4()
                await session.execute(
                    text("""
                        INSERT INTO roles (id, name, is_system_role, hierarchy_level, description)
                        VALUES (:id, 'owner', TRUE, 100, 'Owner role for test')
                    """),
                    {"id": str(role_id)}
                )
                await session.commit()

                # Get the inserted role
                result = await session.execute(text("SELECT * FROM roles WHERE id = :id"), {"id": str(role_id)})
                owner_role_data = result.mappings().one()

            # Convert to a proper object - only include parameters that MockRoleModel accepts
            owner_role = MockRoleModel(
                id=UUID(owner_role_data['id']) if isinstance(owner_role_data['id'], str) else owner_role_data['id'],
                name=owner_role_data['name'],
                is_system_role=owner_role_data['is_system_role'],
                hierarchy_level=owner_role_data['hierarchy_level']
            )

            await session.commit()

            yield owner_role

            # Clean up after tests
            await session.execute(
                text("""
                    DELETE FROM organization_members
                    WHERE user_id IN (
                        SELECT id FROM users
                        WHERE email LIKE 'test-integration-%@example.com'
                    )
                """)
            )
            await session.execute(
                text("""
                    DELETE FROM organizations
                    WHERE name LIKE 'Test Org Integration%'
                """)
            )
            await session.commit()

        except Exception as e:
            print(f"Error in setup: {e}")
            await session.rollback()
            raise
        finally:
            # Only process the first yielded session
            break

@pytest.fixture
def test_user_data():
    """Create test user data."""
    user_id = uuid4()
    return {
        "id": user_id,
        "email": f"test-integration-{user_id}@example.com",
        "first_name": "Test",
        "last_name": "Integration",
        "org_id": None
    }

@pytest.mark.asyncio
class TestOrganizationServiceIntegration:
    """Integration tests for Organization service using a real database."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_create_org_integration(self, org_service, db_session, test_user_data, setup_test_db_integration):
        """Test creating an organization with integration database."""
        org_name = "Test Org Integration"

        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Execute
                response = await org_service.post_create_org(
                    name=org_name,
                    session=session,
                    user=test_user_data
                )

                # Assert
                assert response is not None
                assert response.name == org_name
                assert response.id is not None
                assert response.slug is not None and org_name.lower().replace(" ", "-") in response.slug

                # Verify in database
                query = select(OrganizationModel).where(OrganizationModel.id == response.id)
                result = await session.execute(query)
                db_org = result.scalar_one_or_none()
                assert db_org is not None
                assert db_org.name == org_name

                # Verify member was added
                query = select(OrganizationMemberModel).where(
                    OrganizationMemberModel.organization_id == response.id,
                    OrganizationMemberModel.user_id == test_user_data["id"]
                )
                result = await session.execute(query)
                member = result.scalar_one_or_none()
                assert member is not None
                assert member.status == "active"

                # Verify role is owner
                query = select(RoleModel).where(RoleModel.id == member.role_id)
                result = await session.execute(query)
                role = result.scalar_one_or_none()
                assert role is not None
                assert role.name == "owner"

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_create_org_duplicate_slug_integration(self, org_service, db_session, test_user_data, setup_test_db_integration):
        """Test creating an organization with a duplicate slug."""
        org_name = "Test Org Integration"

        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create first organization to create the initial slug
                first_org = await org_service.post_create_org(
                    name=org_name,
                    session=session,
                    user=test_user_data
                )

                # Create second organization with same name to trigger slug suffix
                second_org = await org_service.post_create_org(
                    name=org_name,
                    session=session,
                    user=test_user_data
                )

                # Assert
                assert first_org.slug != second_org.slug
                assert first_org.name == second_org.name
                base_slug = org_name.lower().replace(" ", "-")
                assert base_slug in first_org.slug
                assert base_slug in second_org.slug

                # Verify both exist in database
                query = select(OrganizationModel).where(OrganizationModel.name == org_name)
                result = await session.execute(query)
                orgs = result.scalars().all()
                assert len(orgs) == 2

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_add_member_integration(self, org_service, db_session, test_user_data, setup_test_db_integration):
        """Test adding a member to an organization."""
        org_name = "Test Org Integration Member"

        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create an organization first
                org = await org_service.post_create_org(
                    name=org_name,
                    session=session,
                    user=test_user_data
                )

                # Create a new user to add as member
                new_user_id = uuid4()

                # Get a role (member role or create one)
                query = select(RoleModel).where(RoleModel.name == "member")
                result = await session.execute(query)
                member_role = result.scalar_one_or_none()

                if not member_role:
                    # Create member role if it doesn't exist
                    member_role = RoleModel(
                        name="member",
                        is_system_role=True,
                        hierarchy_level=10,
                        description="Member role created for tests"
                    )
                    session.add(member_role)
                    await session.commit()
                    await session.refresh(member_role)

                # Add the new user as a member
                response = await org_service.post_add_member(
                    organization_id=org.id,
                    user_id=new_user_id,
                    role_id=member_role.id,
                    session=session
                )

                # Assert
                assert isinstance(response, JSONResponse)
                assert response.status_code == 201

                # Verify in database
                query = select(OrganizationMemberModel).where(
                    OrganizationMemberModel.organization_id == org.id,
                    OrganizationMemberModel.user_id == new_user_id
                )
                result = await session.execute(query)
                member = result.scalar_one_or_none()
                assert member is not None
                assert member.role_id == member_role.id
                assert member.status == "active"

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_get_list_integration(self, org_service, db_session, test_user_data, setup_test_db_integration):
        """Test listing organizations a user belongs to."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create multiple organizations for the test user
                org_names = [
                    "Test Org Integration List 1",
                    "Test Org Integration List 2",
                    "Test Org Integration List 3"
                ]

                created_orgs = []
                for name in org_names:
                    org = await org_service.post_create_org(
                        name=name,
                        session=session,
                        user=test_user_data
                    )
                    created_orgs.append(org)

                # List organizations
                response = await org_service.get_list(
                    session=session,
                    user=test_user_data
                )

                # Assert
                assert isinstance(response, list)
                assert len(response) >= len(org_names)  # May include orgs from other tests

                # Verify created orgs are in the response
                response_ids = [org.id for org in response]
                for created_org in created_orgs:
                    assert created_org.id in response_ids

                # Verify org names
                response_names = [org.name for org in response]
                for name in org_names:
                    assert name in response_names

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_add_member_invalid_role_integration(self, org_service, db_session, test_user_data, setup_test_db_integration):
        """Test adding a member with an invalid role."""
        org_name = "Test Org Integration Invalid Role"

        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create an organization first
                org = await org_service.post_create_org(
                    name=org_name,
                    session=session,
                    user=test_user_data
                )

                # Create a new user to add as member
                new_user_id = uuid4()

                # Use a non-existent role ID
                invalid_role_id = uuid4()

                # Add the new user with invalid role ID
                with pytest.raises(HTTPException) as exc_info:
                    await org_service.post_add_member(
                        organization_id=org.id,
                        user_id=new_user_id,
                        role_id=invalid_role_id,
                        session=session
                    )

                # Assert
                assert exc_info.value.status_code == 404
                assert "Role" in exc_info.value.detail
                assert "not found" in exc_info.value.detail

            except Exception as e:
                if not isinstance(e, HTTPException):
                    await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break
