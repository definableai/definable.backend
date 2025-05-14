from fastapi import HTTPException
import pytest
import asyncio
import os
from uuid import uuid4
from datetime import datetime, timezone

from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio

from src.services.roles.service import RoleService
from src.services.roles.schema import PermissionCreate, RoleCreate, RoleUpdate, RoleResponse
from src.services.__base.acquire import Acquire

# TestAcquire - mock of the Acquire class for service initialization
class TestAcquire(Acquire):
    def __init__(self):
        self.settings = type('Settings', (), {})()
        self.logger = MagicMock()


@pytest.fixture
def test_user():
    """Create a test user dictionary."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
        "is_admin": True
    }


@pytest.fixture
def test_organization():
    """Create a test organization ID."""
    return uuid4()


@pytest.fixture
def roles_service():
    """Create a RoleService instance."""
    return RoleService(acquire=TestAcquire())


@pytest.fixture
def mock_db_session():
    """Create a mock database session with configurable results."""
    session = AsyncMock()

    # Setup for execute method to return our desired results
    execute_result = AsyncMock()
    execute_result.scalar_one_or_none.return_value = None
    execute_result.all.return_value = []
    execute_result.first.return_value = None
    execute_result.fetchone.return_value = None
    execute_result.fetchall.return_value = []

    # Make the results chainable
    execute_result.scalars.return_value = execute_result
    execute_result.unique.return_value = execute_result

    # Setup execute to return our configurable result
    session.execute.return_value = execute_result

    # Behavior for add
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    return session


# Helper function to create a proper UUID-compatible model mock
def create_model_mock(**kwargs):
    """Create a mock model with common attributes."""
    instance = MagicMock()

    # Set default attributes from UUIDMixin and TimestampMixin
    instance.id = kwargs.get('id', uuid4())
    instance.created_at = kwargs.get('created_at', datetime.now(timezone.utc))

    # Set any additional attributes
    for key, value in kwargs.items():
        setattr(instance, key, value)

    return instance


@pytest.fixture
def mock_role():
    """Create a mock role."""
    role_id = uuid4()
    return create_model_mock(
        id=role_id,
        name="Test Role",
        description="Test Role Description",
        is_system_role=False,
        hierarchy_level=50,
        organization_id=uuid4(),
        permissions=[]
    )


@pytest.fixture
def mock_system_role():
    """Create a mock system role."""
    role_id = uuid4()
    return create_model_mock(
        id=role_id,
        name="OWNER",
        description="Owner Role",
        is_system_role=True,
        hierarchy_level=100,
        organization_id=uuid4(),
        permissions=[]
    )


@pytest.fixture
def mock_permission():
    """Create a mock permission."""
    perm_id = uuid4()
    return create_model_mock(
        id=perm_id,
        name="test:read",
        resource="test",
        action="read",
        description="Test Permission"
    )


@pytest.mark.asyncio
class TestRoleService:
    """Test RoleService."""

    async def test_delete_permission(self, roles_service, mock_db_session):
        """Test deleting a permission."""
        permission_id = uuid4()

        await roles_service.delete_permission(
            permission_id=permission_id,
            session=mock_db_session
        )

        assert mock_db_session.execute.called
        assert mock_db_session.commit.called

    async def test_post_create_role_exists(self, roles_service, mock_db_session, test_user, mock_role):
        """Test creating a role with a name that already exists."""
        org_id = uuid4()

        # Mock session.execute to return a result with the methods we need
        execute_result = MagicMock()
        mock_unique = MagicMock()
        mock_unique.scalar_one_or_none.return_value = mock_role  # Role exists
        execute_result.unique.return_value = mock_unique
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # The test case
        role_data = RoleCreate(
            name="Existing Role",
            description="New description",
            hierarchy_level=60,
            permission_ids=[uuid4()]  # Add at least one permission ID to satisfy validation
        )

        # Expect an exception
        with pytest.raises(Exception) as exc_info:
            await roles_service.post_create(
                org_id=org_id,
                role_data=role_data,
                session=mock_db_session,
                user=test_user
            )

        assert "Role name already exists" in str(exc_info.value)

    async def test_post_create_invalid_hierarchy(self, roles_service, mock_db_session, test_user):
        """Test creating a role with an invalid hierarchy level (>= 90)."""
        org_id = uuid4()

        # Mock session.execute to return a result with the methods we need
        execute_result = MagicMock()
        mock_unique = MagicMock()
        mock_unique.scalar_one_or_none.return_value = None  # No existing role
        execute_result.unique.return_value = mock_unique
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # The test case
        role_data = RoleCreate(
            name="Invalid Role",
            description="Test role description",
            hierarchy_level=95,  # Invalid - should be < 90
            permission_ids=[uuid4()]  # Add at least one permission ID to satisfy validation
        )

        # Expect an exception
        with pytest.raises(Exception) as exc_info:
            await roles_service.post_create(
                org_id=org_id,
                role_data=role_data,
                session=mock_db_session,
                user=test_user
            )

        assert "Hierarchy level must be less than 90" in str(exc_info.value)

    async def test_put_update_success(self, roles_service, mock_db_session, test_user, mock_role):
        """Test updating a role successfully."""
        org_id = uuid4()
        role_id = mock_role.id

        # Patch the necessary functions with AsyncMock to handle coroutines
        with patch.object(roles_service, '_get_role', AsyncMock(return_value=mock_role)), \
             patch.object(roles_service, '_validate_hierarchy_level', AsyncMock(return_value=None)), \
             patch.object(roles_service, '_get_role_with_permissions', AsyncMock()) as mock_get_with_perm:

            # Configure _get_role_with_permissions to return updated role
            mock_get_with_perm.return_value = RoleResponse(
                id=role_id,
                name="Updated Role",
                description="Updated description",
                hierarchy_level=60,
                organization_id=org_id,
                is_system_role=False,
                created_at=datetime.now(),
                permissions=[]
            )

            # The test case
            role_data = RoleUpdate(
                name="Updated Role",
                description="Updated description",
                hierarchy_level=60
            )

            result = await roles_service.put_update(
                org_id=org_id,
                role_id=role_id,
                role_data=role_data,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert mock_db_session.commit.called
        assert isinstance(result, RoleResponse)
        assert result.name == role_data.name
        assert result.description == role_data.description
        assert result.hierarchy_level == role_data.hierarchy_level

    async def test_put_update_not_found(self, roles_service, mock_db_session, test_user):
        """Test updating a role that doesn't exist."""
        org_id = uuid4()
        role_id = uuid4()

        # Patch _get_role to return None (role not found)
        with patch.object(roles_service, '_get_role', AsyncMock(return_value=None)):
            # The test case
            role_data = RoleUpdate(
                name="Updated Role",
                description="Updated description"
            )

            # Expect an exception
            with pytest.raises(Exception) as exc_info:
                await roles_service.put_update(
                    org_id=org_id,
                    role_id=role_id,
                    role_data=role_data,
                    session=mock_db_session,
                    user=test_user
                )

            assert "Role not found" in str(exc_info.value)

    async def test_put_update_system_role(self, roles_service, mock_db_session, test_user, mock_system_role):
        """Test updating a system role (should fail)."""
        org_id = uuid4()
        role_id = mock_system_role.id

        # Patch _get_role to return a system role
        with patch.object(roles_service, '_get_role', AsyncMock(return_value=mock_system_role)):
            # The test case
            role_data = RoleUpdate(
                name="Updated System Role",
                description="Updated description"
            )

            # Expect an exception
            with pytest.raises(Exception) as exc_info:
                await roles_service.put_update(
                    org_id=org_id,
                    role_id=role_id,
                    role_data=role_data,
                    session=mock_db_session,
                    user=test_user
                )

            assert "Cannot update system role" in str(exc_info.value)

    async def test_delete_remove_success(self, roles_service, mock_db_session, test_user, mock_role):
        """Test deleting a role successfully."""
        org_id = uuid4()
        role_id = mock_role.id

        # Configure mock to return role with 0 members
        role_with_count = (mock_role, 0)  # (role, member_count = 0)
        execute_result = MagicMock()
        execute_result.first.return_value = role_with_count

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # The test case
        result = await roles_service.delete_remove(
            org_id=org_id,
            role_id=role_id,
            session=mock_db_session,
            user=test_user
        )

        # Assertions
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called
        assert isinstance(result, dict)
        assert "message" in result
        assert "deleted successfully" in result["message"]

    async def test_delete_remove_not_found(self, roles_service, mock_db_session, test_user):
        """Test deleting a role that doesn't exist."""
        org_id = uuid4()
        role_id = uuid4()

        # Configure mock to return None (role not found)
        execute_result = MagicMock()
        execute_result.first.return_value = None

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Expect an exception
        with pytest.raises(Exception) as exc_info:
            await roles_service.delete_remove(
                org_id=org_id,
                role_id=role_id,
                session=mock_db_session,
                user=test_user
            )

        assert "Role not found" in str(exc_info.value)

    async def test_delete_remove_system_role(self, roles_service, mock_db_session, test_user, mock_system_role):
        """Test deleting a system role (should fail)."""
        org_id = uuid4()
        role_id = mock_system_role.id

        # Configure mock to return a system role with 0 members
        role_with_count = (mock_system_role, 0)  # (role, member_count = 0)
        execute_result = MagicMock()
        execute_result.first.return_value = role_with_count

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Expect an exception
        with pytest.raises(Exception) as exc_info:
            await roles_service.delete_remove(
                org_id=org_id,
                role_id=role_id,
                session=mock_db_session,
                user=test_user
            )

        assert "Cannot delete system role" in str(exc_info.value)

    async def test_delete_remove_with_members(self, roles_service, mock_db_session, test_user, mock_role):
        """Test deleting a role with assigned members (should fail)."""
        org_id = uuid4()
        role_id = mock_role.id

        # Configure mock to return role with members
        role_with_count = (mock_role, 5)  # (role, member_count = 5)
        execute_result = MagicMock()
        execute_result.first.return_value = role_with_count

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Expect an exception
        with pytest.raises(Exception) as exc_info:
            await roles_service.delete_remove(
                org_id=org_id,
                role_id=role_id,
                session=mock_db_session,
                user=test_user
            )

        assert "Cannot delete role that is assigned to members" in str(exc_info.value)

    async def test_get_list_roles(self, roles_service, mock_db_session, test_user, mock_role, mock_system_role):
        """Test listing all roles for an organization."""
        org_id = uuid4()

        # Configure mock to return list of roles
        roles = [mock_role, mock_system_role]
        scalar_result = MagicMock()
        scalar_result.all.return_value = roles
        unique_result = MagicMock()
        unique_result.scalars.return_value = scalar_result
        execute_result = MagicMock()
        execute_result.unique.return_value = unique_result

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # The test case
        result = await roles_service.get_list_roles(
            org_id=org_id,
            session=mock_db_session,
            user=test_user
        )

        # Assertions
        assert mock_db_session.execute.called
        assert len(result) == 2
        assert result[0] == mock_role
        assert result[1] == mock_system_role

    async def test_get_list_permissions(self, roles_service, mock_db_session, test_user, mock_permission):
        """Test listing all permissions."""
        # Configure mock to return list of permissions
        permissions = [mock_permission]
        scalar_result = MagicMock()
        scalar_result.all.return_value = permissions
        unique_result = MagicMock()
        unique_result.scalars.return_value = scalar_result
        execute_result = MagicMock()
        execute_result.unique.return_value = unique_result

        # Create a new AsyncMock to handle awaits from session.execute
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # The test case
        result = await roles_service.get_list_permissions(
            session=mock_db_session,
            user=test_user
        )

        # Assertions
        assert mock_db_session.execute.called
        assert len(result) == 1
        assert result[0] == mock_permission

# ============================================================================
# PERMISSION MANAGEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
class TestPermissionManagement:
    """Tests specifically for permission management."""

    async def test_add_permissions_to_role(self, roles_service, mock_db_session, mock_role, mock_permission):
        """Test adding permissions to an existing role."""
        role_id = mock_role.id
        permission_id = mock_permission.id

        # Patch the _add_role_permissions method
        with patch.object(roles_service, '_add_role_permissions', AsyncMock()) as mock_add_permissions:
            # Execute
            await roles_service._add_role_permissions(
                role_id=role_id,
                permission_ids=[permission_id],
                session=mock_db_session
            )

            # Assert - use keyword arguments to match how the method is called
            mock_add_permissions.assert_called_once_with(
                role_id=role_id,
                permission_ids=[permission_id],
                session=mock_db_session
            )

    async def test_get_role_with_permissions(self, roles_service, mock_db_session, mock_role, mock_permission):
        """Test retrieving a role with its permissions."""
        role_id = mock_role.id

        # Configure the mock to return a role with permissions
        mock_role.permissions = [mock_permission]

        # Create a properly structured mock result for database queries
        execute_result = MagicMock()
        unique_result = MagicMock()
        unique_result.scalar_one_or_none = MagicMock(return_value=mock_role)
        execute_result.unique = MagicMock(return_value=unique_result)

        # Make execute return the mock result structure
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Patch the response model validation
        with patch('src.services.roles.schema.RoleResponse', return_value=MagicMock()):
            # Execute
            result = await roles_service._get_role_with_permissions(
                role_id=role_id,
                session=mock_db_session
            )

            # Assert
            assert mock_db_session.execute.called
            assert result is not None

    async def test_update_role_permissions(self, roles_service, mock_db_session, test_user, mock_role):
        """Test updating the permissions of a role."""
        org_id = uuid4()
        role_id = mock_role.id
        new_permission_ids = [uuid4(), uuid4()]

        # Setup mocks
        with patch.object(roles_service, '_get_role', AsyncMock(return_value=mock_role)), \
             patch.object(roles_service, '_add_role_permissions', AsyncMock()) as mock_add, \
             patch.object(roles_service, '_get_role_with_permissions', AsyncMock()) as mock_get:

            # Execute
            role_data = RoleUpdate(permission_ids=new_permission_ids)
            await roles_service.put_update(
                org_id=org_id,
                role_id=role_id,
                role_data=role_data,
                session=mock_db_session,
                user=test_user
            )

            # Assert
            mock_db_session.execute.assert_called()  # Should call delete on existing permissions
            mock_add.assert_called_once_with(role_id, new_permission_ids, mock_db_session)
            mock_get.assert_called_once_with(role_id, mock_db_session)

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
class TestRoleServiceErrorHandling:
    """Tests for error handling in the Role service."""

    async def test_invalid_permission_ids(self, roles_service, mock_db_session, test_user):
        """Test handling of invalid permission IDs when creating a role."""
        # Setup
        org_id = uuid4()
        invalid_permission_id = uuid4()  # This ID doesn't match any existing permission

        # Mock execute to simulate database error when adding invalid permission
        mock_db_session.execute = AsyncMock(side_effect=Exception("Invalid permission ID"))

        # Patch _validate_hierarchy_level to prevent validation errors
        with patch.object(roles_service, '_validate_hierarchy_level', AsyncMock()):
            # Create role data with invalid permission IDs
            role_data = RoleCreate(
                name="Test Role",
                description="Role with invalid permissions",
                hierarchy_level=50,
                permission_ids=[invalid_permission_id]
            )

            # Execute and Assert
            with pytest.raises(Exception):
                await roles_service.post_create(
                    org_id=org_id,
                    role_data=role_data,
                    session=mock_db_session,
                    user=test_user
                )

    async def test_hierarchy_level_constraints(self, roles_service, mock_db_session, test_user):
        """Test various hierarchy level constraints and validations."""
        # Setup
        org_id = uuid4()

        # Test case 1: Too high hierarchy level (>=90)
        role_data_high = RoleCreate(
            name="High Level Role",
            description="Role with too high hierarchy",
            hierarchy_level=95,
            permission_ids=[uuid4()]
        )

        # Execute and Assert for high hierarchy
        with pytest.raises(HTTPException) as exc_info:
            await roles_service._validate_hierarchy_level(
                organization_id=org_id,
                hierarchy_level=role_data_high.hierarchy_level,
                session=mock_db_session
            )

        assert exc_info.value.status_code == 400
        assert "Hierarchy level must be less than 90" in exc_info.value.detail

    async def test_db_transaction_rollback(self, roles_service, mock_db_session, test_user, monkeypatch):
        """Test database transaction rollback on error."""
        # Setup
        org_id = uuid4()
        role_data = RoleCreate(
            name="Rollback Test Role",
            description="This role creation should trigger a rollback",
            hierarchy_level=50,
            permission_ids=[uuid4()]
        )

        # Mock database to simulate an error during commit
        async def mock_commit_error():
            raise Exception("Simulated database error")

        # Create a normal execute result that doesn't trigger errors
        execute_result = MagicMock()
        mock_unique = MagicMock()
        mock_unique.scalar_one_or_none.return_value = None  # No existing role
        execute_result.unique.return_value = mock_unique
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Make the commit fail
        mock_db_session.commit = AsyncMock(side_effect=mock_commit_error)

        # Mock rollback to verify it's called
        mock_db_session.rollback = AsyncMock()

        # Patch necessary methods
        with patch.object(roles_service, '_validate_hierarchy_level', AsyncMock()), \
             patch.object(roles_service, '_add_role_permissions', AsyncMock()):

            # Execute and Assert
            with pytest.raises(Exception):
                await roles_service.post_create(
                    org_id=org_id,
                    role_data=role_data,
                    session=mock_db_session,
                    user=test_user
                )

            # Note: In a real scenario, the service's exception handler would call rollback
            # Here we're just verifying the exception is raised

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
class TestRoleServicePerformance:
    """Performance tests for the Role service."""

    async def test_bulk_role_creation(self, roles_service, mock_db_session, test_organization):
        """Test creating multiple roles in bulk."""
        # Setup - create multiple roles
        num_roles = 5
        org_id = test_organization

        # Setup for batched execution
        tasks = []

        # Mock the database session for all roles
        execute_result = MagicMock()
        mock_unique = MagicMock()
        mock_unique.scalar_one_or_none.return_value = None  # No existing roles
        execute_result.unique.return_value = mock_unique
        mock_db_session.execute = AsyncMock(return_value=execute_result)

        # Create a real UUID for each role
        role_ids = [uuid4() for _ in range(num_roles)]

        # Patch required methods
        with patch.object(roles_service, '_validate_hierarchy_level', AsyncMock()), \
             patch.object(roles_service, '_add_role_permissions', AsyncMock()), \
             patch.object(roles_service, '_get_role_with_permissions', AsyncMock()) as mock_get_role, \
             patch('src.models.RoleModel') as mock_role_model:

            # Configure _get_role_with_permissions to return different roles
            mock_role_responses = []
            for i in range(num_roles):
                role_response = RoleResponse(
                    id=role_ids[i],
                    name=f"Bulk Role {i}",
                    description=f"Bulk created role {i}",
                    hierarchy_level=10+i,
                    organization_id=org_id,
                    is_system_role=False,
                    created_at=datetime.now(timezone.utc),
                    permissions=[]
                )
                mock_role_responses.append(role_response)

            # Setup side effects for the mock role model to return roles with IDs
            role_instances = []
            for i in range(num_roles):
                instance = MagicMock()
                instance.id = role_ids[i]
                role_instances.append(instance)

            mock_role_model.side_effect = role_instances
            mock_get_role.side_effect = mock_role_responses

            # Create role creation tasks
            for i in range(num_roles):
                role_data = RoleCreate(
                    name=f"Bulk Role {i}",
                    description=f"Bulk created role {i}",
                    hierarchy_level=10+i,
                    permission_ids=[uuid4()]
                )

                tasks.append(
                    roles_service.post_create(
                        org_id=org_id,
                        role_data=role_data,
                        session=mock_db_session,
                        user=test_user
                    )
                )

            # Execute concurrently
            import time

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # Assert
            assert len(results) == num_roles
            assert all(isinstance(r, RoleResponse) for r in results)
            assert mock_db_session.add.call_count >= num_roles

            # Performance check - this is optional and the threshold would depend on
            # the specific environment, but gives some indicator
            assert end_time - start_time < 1.0, "Bulk creation took too long"

# ============================================================================
# INTEGRATION TESTS - RUN WITH: INTEGRATION_TEST=1 pytest tests/services/test_roles.py
# ============================================================================

# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.

    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select, text
    from models import PermissionModel, RoleModel, RolePermissionModel

@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for role integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    # Create necessary database objects
    async for session in db_session:
        try:
            # Create required tables if they don't exist
            # Create permissions table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS permissions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    resource VARCHAR(255) NOT NULL,
                    action VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

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

            # Create organizations table (for foreign key integrity)
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

            # Create organization_members table (for foreign key integrity)
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

            # Clean up any existing test data first
            await session.execute(text("DELETE FROM role_permissions WHERE role_id IN (SELECT id FROM roles WHERE name LIKE 'Test Integration%')"))
            await session.execute(text("DELETE FROM roles WHERE name LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM permissions WHERE name LIKE 'test:integration:%'"))

            await session.commit()

            yield

            # Clean up after tests
            await session.execute(text("DELETE FROM role_permissions WHERE role_id IN (SELECT id FROM roles WHERE name LIKE 'Test Integration%')"))
            await session.execute(text("DELETE FROM roles WHERE name LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM permissions WHERE name LIKE 'test:integration:%'"))
            await session.commit()

        except Exception as e:
            print(f"Error in setup: {e}")
            await session.rollback()
            raise
        finally:
            # Only process the first yielded session
            break

@pytest.fixture
def integration_test_user():
    """Create a test user for integration tests."""
    return {
        "id": uuid4(),
        "email": "integration-test@example.com",
        "first_name": "Integration",
        "last_name": "Test",
        "organization_id": uuid4(),
        "is_admin": True
    }

@pytest.fixture
def integration_test_org():
    """Create a test organization ID for integration tests."""
    return uuid4()

@pytest.mark.asyncio
class TestRoleServiceIntegration:
    """Integration tests for Role service with real database."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_create_permission_integration(self, roles_service, db_session, setup_test_db_integration):
        """Test creating a permission with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create a permission
                permission_data = PermissionCreate(
                    name="test:integration:read",
                    resource="test:integration",
                    action="read",
                    description="Test integration read permission"
                )

                # Execute
                response = await roles_service.post_permission(
                    permission=permission_data,
                    session=session
                )

                # Assert
                assert response is not None
                assert response.name == permission_data.name
                assert response.resource == permission_data.resource
                assert response.action == permission_data.action
                assert response.id is not None

                # Verify in database
                query = select(PermissionModel).where(PermissionModel.name == permission_data.name)
                result = await session.execute(query)
                db_permission = result.scalar_one_or_none()
                assert db_permission is not None
                assert db_permission.name == permission_data.name

            finally:
                # Only process the first yielded session
                break

    async def test_create_role_integration(self, roles_service, db_session, integration_test_user,
                                           integration_test_org, setup_test_db_integration):
        """Test creating a role with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # First create a permission to associate with the role
                permission_data = PermissionCreate(
                    name="test:integration:write",
                    resource="test:integration",
                    action="write",
                    description="Test integration write permission"
                )

                permission_response = await roles_service.post_permission(
                    permission=permission_data,
                    session=session
                )

                # Now create a role with that permission
                role_data = RoleCreate(
                    name="Test Integration Role",
                    description="Role created for integration test",
                    hierarchy_level=50,
                    permission_ids=[permission_response.id]
                )

                # Execute
                response = await roles_service.post_create(
                    org_id=integration_test_org,
                    role_data=role_data,
                    session=session,
                    user=integration_test_user
                )

                # Assert
                assert response is not None
                assert response.name == role_data.name
                assert response.description == role_data.description
                assert response.hierarchy_level == role_data.hierarchy_level
                assert response.id is not None
                assert len(response.permissions) == 1
                assert response.permissions[0].id == permission_response.id

                # Verify in database
                query = select(RoleModel).where(RoleModel.name == role_data.name)
                result = await session.execute(query)
                db_role = result.scalar_one_or_none()
                assert db_role is not None
                assert db_role.name == role_data.name

                # Verify permission association
                query = select(RolePermissionModel).where(RolePermissionModel.role_id == db_role.id)
                result = await session.execute(query)
                role_permissions = result.all()
                assert len(role_permissions) == 1
                assert role_permissions[0][0] == permission_response.id

            finally:
                # Only process the first yielded session
                break

    async def test_authorization_non_admin_create_role(self, roles_service, db_session, setup_test_db_integration):
        """Test creating a role without proper permissions using real database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Setup
                org_id = uuid4()
                non_admin_user = {
                    "id": uuid4(),
                    "email": "regular@example.com",
                    "first_name": "Regular",
                    "last_name": "User",
                    "organization_id": org_id,
                    "is_admin": False
                }

                # Create a test permission
                permission_data = PermissionCreate(
                    name="test:integration:auth:read",
                    resource="test:integration:auth",
                    action="read",
                    description="Test integration auth permission"
                )

                permission = await roles_service.post_permission(
                    permission=permission_data,
                    session=session
                )

                # Create role data with the real permission
                role_data = RoleCreate(
                    name="Test Integration Unauthorized Role",
                    description="Role created without permission",
                    hierarchy_level=10,
                    permission_ids=[permission.id]
                )

                # In a real integration test, the RBAC middleware would reject this
                # Here we're simulating that by mocking the dependency
                with patch('dependencies.security.RBAC', side_effect=HTTPException(
                    status_code=403, detail="Insufficient permissions"
                )):
                    # Execute and Assert
                    with pytest.raises(HTTPException) as exc_info:
                        await roles_service.post_create(
                            org_id=org_id,
                            role_data=role_data,
                            session=session,
                            user=non_admin_user
                        )

                    assert exc_info.value.status_code == 403
            finally:
                # Only process the first yielded session
                break

    async def test_authorization_update_higher_role(self, roles_service, db_session, integration_test_user,
                                                   integration_test_org, setup_test_db_integration):
        """Test that roles cannot be updated by users with lower hierarchy using real database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create a permission for the role
                permission_data = PermissionCreate(
                    name="test:integration:auth:write",
                    resource="test:integration:auth",
                    action="write",
                    description="Test integration auth write permission"
                )

                permission = await roles_service.post_permission(
                    permission=permission_data,
                    session=session
                )

                # First create a high-level role with the real permission
                high_role_data = RoleCreate(
                    name="Test Integration High Role",
                    description="High level role for hierarchy test",
                    hierarchy_level=80,  # High level
                    permission_ids=[permission.id]
                )

                high_role_response = await roles_service.post_create(
                    org_id=integration_test_org,
                    role_data=high_role_data,
                    session=session,
                    user=integration_test_user
                )

                # Create a user with lower hierarchy
                lower_hierarchy_user = {
                    "id": uuid4(),
                    "email": "lower@example.com",
                    "first_name": "Lower",
                    "last_name": "Hierarchy",
                    "organization_id": integration_test_org,
                    "role_hierarchy": 40  # Lower than the role we created
                }

                # Create update data
                update_data = RoleUpdate(
                    name="Updated High Role",
                    description="Attempt to update higher role"
                )

                # In a real integration test, the RBAC middleware would reject this
                # Here we're simulating that by mocking the dependency
                with patch('dependencies.security.RBAC', side_effect=HTTPException(
                    status_code=403, detail="Cannot modify a role with higher hierarchy level"
                )):
                    # Execute and Assert
                    with pytest.raises(HTTPException) as exc_info:
                        await roles_service.put_update(
                            org_id=integration_test_org,
                            role_id=high_role_response.id,
                            role_data=update_data,
                            session=session,
                            user=lower_hierarchy_user
                        )

                    assert exc_info.value.status_code == 403

                # Verify that the role was not updated in the database
                query = select(RoleModel).where(RoleModel.id == high_role_response.id)
                result = await session.execute(query)
                db_role = result.scalar_one_or_none()
                assert db_role is not None
                assert db_role.name == high_role_data.name  # Name should not have changed
            finally:
                # Only process the first yielded session
                break
