import pytest
import pytest_asyncio
import sys
import os
from uuid import uuid4, UUID

from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from src.services.users.service import UserService
from src.services.users.schema import (
    InviteSignup,
    StytchUser,
    UserDetailResponse,
    OrganizationInfo
)
from src.services.__base.acquire import Acquire

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
        self.settings = type('Settings', (), {})()
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
    return session

@pytest.fixture
def mock_stytch():
    """Mock the stytch_base module."""
    with patch('src.services.users.service.stytch_base') as mock:
        mock.invite_user = AsyncMock()
        yield mock

@pytest.fixture
def users_service():
    """Create a UserService instance."""
    return UserService(acquire=TestAcquire())

@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return {
        "id": UUID("11111111-1111-1111-1111-111111111111"),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User"
    }

@pytest.fixture
def sample_user_model():
    """Create a sample UserModel for testing."""
    user = MagicMock()
    user.id = UUID("11111111-1111-1111-1111-111111111111")
    user.email = "test@example.com"
    user.first_name = "Test"
    user.last_name = "User"
    user.full_name = "Test User"
    return user

@pytest.fixture
def sample_org_member():
    """Create a sample OrganizationMemberModel."""
    member = MagicMock()
    member.organization_id = UUID("22222222-2222-2222-2222-222222222222")
    member.role_id = UUID("33333333-3333-3333-3333-333333333333")
    member.user_id = UUID("11111111-1111-1111-1111-111111111111")
    member.status = "active"
    return member

@pytest.fixture
def sample_org():
    """Create a sample OrganizationModel."""
    org = MagicMock()
    org.id = UUID("22222222-2222-2222-2222-222222222222")
    org.name = "Test Organization"
    org.slug = "test-org"
    return org

@pytest.fixture
def sample_role():
    """Create a sample RoleModel."""
    role = MagicMock()
    role.id = UUID("33333333-3333-3333-3333-333333333333")
    role.name = "MEMBER"
    return role

class TestUserService:
    """Unit tests for UserService."""

    @pytest.mark.asyncio
    async def test_get_me(self, users_service, mock_db_session, sample_user, sample_user_model, sample_org_member, sample_org, sample_role):
        """Test get_me method."""
        # Set full organization info in user response
        org_info = OrganizationInfo(
            id=sample_org.id,
            name=sample_org.name,
            slug=sample_org.slug,
            role_name=sample_role.name,
            role_id=sample_role.id
        )

        # Create a user response with organizations
        user_response = UserDetailResponse(
            id=sample_user_model.id,
            email=sample_user_model.email,
            first_name=sample_user_model.first_name,
            last_name=sample_user_model.last_name,
            full_name=sample_user_model.full_name,
            organizations=[org_info]
        )

        # Patch the _get_user_details directly to return our prepared response
        with patch.object(users_service, '_get_user_details', return_value=user_response) as mock_get_details:
            # Call the method
            result = await users_service.get_me(
                current_user=sample_user,
                session=mock_db_session
            )

            # Verify _get_user_details was called with the right parameters
            mock_get_details.assert_called_once_with(sample_user["id"], mock_db_session)

            # Assertions
            assert result.id == sample_user["id"]
            assert result.email == sample_user["email"]
            assert result.first_name == sample_user["first_name"]
            assert result.last_name == sample_user["last_name"]
            assert len(result.organizations) == 1
            assert result.organizations[0].id == sample_org.id
            assert result.organizations[0].name == sample_org.name
            assert result.organizations[0].role_name == sample_role.name

    @pytest.mark.asyncio
    async def test_get_user_details_not_found(self, users_service, mock_db_session):
        """Test _get_user_details when user is not found."""
        # Mock user query to return None
        mock_result_user = MagicMock()
        mock_result_user.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result_user

        # Call the method and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await users_service._get_user_details(
                user_id=UUID("11111111-1111-1111-1111-111111111111"),
                session=mock_db_session
            )

        # Assertions
        assert excinfo.value.status_code == 404
        assert "User not found" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_get_user_details_no_organizations(self, users_service, mock_db_session, sample_user_model):
        """Test _get_user_details when user has no organizations."""
        # Mock user query
        mock_result_user = MagicMock()
        mock_result_user.scalar_one_or_none.return_value = sample_user_model

        # Mock member query to return empty list
        mock_result_members = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result_members.scalars.return_value = mock_scalars

        # Configure session.execute for both queries
        mock_db_session.execute.side_effect = [
            mock_result_user,  # For user query
            mock_result_members,  # For members query
        ]

        # Call the method
        result = await users_service._get_user_details(
            user_id=sample_user_model.id,
            session=mock_db_session
        )

        # Assertions
        assert result.id == sample_user_model.id
        assert result.email == sample_user_model.email
        assert len(result.organizations) == 0

    @pytest.mark.asyncio
    async def test_get_list(self, users_service, mock_db_session, sample_user_model, sample_org_member, sample_org, sample_role):
        """Test get_list method."""
        org_id = UUID("22222222-2222-2222-2222-222222222222")

        # Mock count query
        mock_db_session.scalar.return_value = 1

        # Mock user IDs query
        user_ids = [sample_user_model.id]
        mock_result_users = MagicMock()
        mock_scalars_users = MagicMock()
        mock_scalars_users.all.return_value = user_ids
        mock_result_users.scalars.return_value = mock_scalars_users

        # Mock user details query (reuse the test_get_me results)
        mock_result_user = MagicMock()
        mock_result_user.scalar_one_or_none.return_value = sample_user_model

        mock_result_members = MagicMock()
        mock_scalars_members = MagicMock()
        mock_scalars_members.all.return_value = [sample_org_member]
        mock_result_members.scalars.return_value = mock_scalars_members

        mock_result_org = MagicMock()
        mock_result_org.scalar_one_or_none.return_value = sample_org

        mock_result_role = MagicMock()
        mock_result_role.scalar_one_or_none.return_value = sample_role

        # Configure the session.execute
        mock_db_session.execute.side_effect = [
            mock_result_users,  # For user IDs query
            mock_result_user,  # For user query
            mock_result_members,  # For members query
            mock_result_org,  # For organization query
            mock_result_role,  # For role query
        ]

        # Call the method
        result = await users_service.get_list(
            org_id=org_id,
            offset=0,
            limit=10,
            session=mock_db_session,
            user={"id": uuid4(), "org_id": org_id}
        )

        # Assertions
        assert result.total == 1
        assert len(result.users) == 1
        assert result.users[0].id == sample_user_model.id
        assert result.users[0].email == sample_user_model.email

    @pytest.mark.asyncio
    async def test_post_invite(self, users_service, mock_db_session, mock_stytch):
        """Test post_invite method."""
        # Test data
        org_id = UUID("22222222-2222-2222-2222-222222222222")
        invite_data = InviteSignup(
            email="new@example.com",
            first_name="New",
            last_name="User",
            role="MEMBER"
        )

        # Mock stytch invite response
        mock_stytch.invite_user.return_value = {
            "status_code": 200,
            "user_id": "stytch-user-123"
        }

        # Call the method
        result = await users_service.post_invite(
            user_data=invite_data,
            org_id=org_id,
            token_payload={"id": uuid4(), "org_id": org_id},
            session=mock_db_session
        )

        # Assertions
        assert result.status_code == 200
        assert "User invited successfully" in result.body.decode()
        mock_stytch.invite_user.assert_awaited_once_with(
            email=invite_data.email,
            first_name=invite_data.first_name,
            last_name=invite_data.last_name
        )

###########################################
# INTEGRATION TESTS
###########################################

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select, text
    from models import OrganizationMemberModel
else:
    # Mock modules to prevent SQLAlchemy issues when running without integration flag
    sys.modules["database"] = MagicMock()
    sys.modules["database.postgres"] = MagicMock()
    sys.modules["src.database"] = MagicMock()
    sys.modules["src.database.postgres"] = MagicMock()
    sys.modules["dependencies.security"] = MagicMock()

@pytest.fixture
def test_integration_user():
    """Create a test user for integration tests."""
    user_id = uuid4()
    org_id = uuid4()
    return {
        "id": user_id,
        "email": f"test-integration-{user_id}@example.com",
        "first_name": "Integration",
        "last_name": "Test",
        "organization_id": org_id,
        "is_admin": True
    }

@pytest.fixture
def test_integration_org():
    """Create a test organization ID for integration tests."""
    return uuid4()

@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for users integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    test_org_id = None
    test_role_id = None
    test_user_id = None
    test_stytch_id = None
    session = None

    # Get session from the generator without exhausting it
    try:
        session_gen = db_session.__aiter__()
        session = await session_gen.__anext__()

        # Create necessary database tables if they don't exist
        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                first_name VARCHAR(255),
                last_name VARCHAR(255),
                stytch_id VARCHAR(255) UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

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

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS roles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                organization_id UUID NOT NULL,
                description TEXT,
                is_system_role BOOLEAN DEFAULT false,
                hierarchy_level INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))

        await session.execute(text("""
            CREATE TABLE IF NOT EXISTS organization_members (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                organization_id UUID NOT NULL REFERENCES organizations(id),
                user_id UUID NOT NULL REFERENCES users(id),
                role_id UUID REFERENCES roles(id),
                status VARCHAR(50) DEFAULT 'active',
                invited_by UUID,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (organization_id, user_id)
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
        await session.execute(text("DELETE FROM users WHERE email LIKE 'test-integration-%@example.com'"))
        await session.execute(text("DELETE FROM roles WHERE name = 'MEMBER'"))
        await session.execute(text("DELETE FROM organizations WHERE name LIKE 'Test Integration%'"))
        await session.commit()

        # Create a test organization
        test_org_id = uuid4()
        await session.execute(
            text("""
                INSERT INTO organizations (id, name, slug)
                VALUES (:id, 'Test Integration Organization', 'test-integration-org')
            """),
            {"id": str(test_org_id)}
        )

        # Create role with the organization_id set correctly
        test_role_id = uuid4()
        await session.execute(
            text("""
                INSERT INTO roles (id, name, is_system_role, hierarchy_level, description, organization_id)
                VALUES (:id, 'MEMBER', TRUE, 10, 'Member role', :org_id)
            """),
            {"id": str(test_role_id), "org_id": str(test_org_id)}
        )

        # Create a test user directly in database
        test_user_id = uuid4()
        test_stytch_id = f"test-stytch-id-{test_user_id}"
        await session.execute(
            text("""
                INSERT INTO users (id, email, first_name, last_name, stytch_id)
                VALUES (:id, 'test-integration-user@example.com', 'Integration', 'Test', :stytch_id)
            """),
            {"id": str(test_user_id), "stytch_id": test_stytch_id}
        )

        # Connect user to organization
        member_id = uuid4()
        await session.execute(
            text("""
                INSERT INTO organization_members (id, organization_id, user_id, role_id, status)
                VALUES (:id, :org_id, :user_id, :role_id, 'active')
            """),
            {
                "id": str(member_id),
                "org_id": str(test_org_id),
                "user_id": str(test_user_id),
                "role_id": str(test_role_id)
            }
        )

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
        "role_id": test_role_id,
        "stytch_id": test_stytch_id,
        "db_session": db_session  # Return the session generator for test use
    }

    # Clean up after tests
    try:
        # Get a new session for cleanup
        async for cleanup_session in db_session:
            try:
                await cleanup_session.execute(
                    text("""
                        DELETE FROM invitations
                        WHERE email LIKE 'test-integration-%@example.com'
                    """)
                )
                await cleanup_session.execute(
                    text("""
                        DELETE FROM organization_members
                        WHERE user_id IN (
                            SELECT id FROM users
                            WHERE email LIKE 'test-integration-%@example.com'
                        )
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
class TestUserServiceIntegration:
    """Integration tests for User service with real database."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_get_me_integration(self, users_service, setup_test_db_integration):
        """Test getting current user details with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]

        # Get a new session
        async for session in db_session:
            try:
                # Setup current user
                current_user = {"id": test_data["user_id"]}

                # Execute
                response = await users_service.get_me(
                    current_user=current_user,
                    session=session
                )

                # Assert
                assert response is not None
                assert response.id == test_data["user_id"]
                assert response.email == "test-integration-user@example.com"
                assert response.first_name == "Integration"
                assert response.last_name == "Test"
                assert len(response.organizations) == 1
                assert response.organizations[0].id == test_data["org_id"]
                assert response.organizations[0].role_id == test_data["role_id"]

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_me_integration: {e}")
                raise

    async def test_get_list_integration(self, users_service, setup_test_db_integration):
        """Test listing users with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Create a few more users for pagination testing
                for i in range(3):
                    user_id = uuid4()
                    stytch_id = f"test-stytch-id-{user_id}"
                    await session.execute(
                        text("""
                            INSERT INTO users (id, email, first_name, last_name, stytch_id)
                            VALUES (:id, :email, :first_name, :last_name, :stytch_id)
                        """),
                        {
                            "id": str(user_id),
                            "email": f"test-integration-{i}@example.com",
                            "first_name": f"Test{i}",
                            "last_name": "User",
                            "stytch_id": stytch_id
                        }
                    )

                    # Connect user to organization
                    member_id = uuid4()
                    await session.execute(
                        text("""
                            INSERT INTO organization_members (id, organization_id, user_id, role_id, status)
                            VALUES (:id, :org_id, :user_id, :role_id, 'active')
                        """),
                        {
                            "id": str(member_id),
                            "org_id": str(org_id),
                            "user_id": str(user_id),
                            "role_id": str(test_data["role_id"])
                        }
                    )

                await session.commit()

                # Setup current user
                current_user = {"id": test_data["user_id"]}

                # Execute
                response = await users_service.get_list(
                    org_id=org_id,
                    offset=0,
                    limit=10,
                    session=session,
                    user=current_user
                )

                # Assert
                assert response is not None
                assert response.total == 4  # Original user + 3 new ones
                assert len(response.users) == 4

                # Test pagination
                paginated_response = await users_service.get_list(
                    org_id=org_id,
                    offset=0,
                    limit=2,
                    session=session,
                    user=current_user
                )

                assert paginated_response.total == 4
                assert len(paginated_response.users) == 2

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_list_integration: {e}")
                raise

    async def test_post_invite_integration(self, users_service, setup_test_db_integration):
        """Test inviting a user with integration database and real Stytch credentials."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Create user data for invitation with a unique email to avoid conflicts
                test_email = f"test-invite-{uuid4()}@example.com"
                user_data = InviteSignup(
                    email=test_email,
                    first_name="Invited",
                    last_name="User",
                    role="MEMBER"
                )

                # Execute with the current user from test data
                current_user = {"id": test_data["user_id"]}

                # Call the service method with real Stytch credentials
                response = await users_service.post_invite(
                    user_data=user_data,
                    org_id=org_id,
                    token_payload=current_user,
                    session=session
                )

                # Assert successful response
                assert isinstance(response, JSONResponse)
                assert response.status_code == 200
                assert "User invited successfully" in response.body.decode('utf-8')

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_post_invite_integration: {e}")
                raise

    async def test_get_user_details_integration(self, users_service, setup_test_db_integration):
        """Test getting user details directly."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        user_id = test_data["user_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Execute the internal method directly
                result = await users_service._get_user_details(
                    user_id=user_id,
                    session=session
                )

                # Assert
                assert result is not None
                assert result.id == user_id
                assert result.email == "test-integration-user@example.com"
                assert result.first_name == "Integration"
                assert result.last_name == "Test"
                assert len(result.organizations) == 1

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_user_details_integration: {e}")
                raise

    async def test_get_user_details_no_organizations_integration(self, users_service, setup_test_db_integration):
        """Test getting user details when user has no organizations."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]

        # Get a new session
        async for session in db_session:
            try:
                # Create a user without organization memberships
                user_id = uuid4()
                stytch_id = f"test-stytch-id-no-org-{user_id}"

                await session.execute(
                    text("""
                        INSERT INTO users (id, email, first_name, last_name, stytch_id)
                        VALUES (:id, :email, :first_name, :last_name, :stytch_id)
                    """),
                    {
                        "id": str(user_id),
                        "email": "test-integration-no-orgs@example.com",
                        "first_name": "NoOrg",
                        "last_name": "User",
                        "stytch_id": stytch_id
                    }
                )
                await session.commit()

                # Execute the internal method directly
                result = await users_service._get_user_details(
                    user_id=user_id,
                    session=session
                )

                # Assert
                assert result is not None
                assert result.id == user_id
                assert result.email == "test-integration-no-orgs@example.com"
                assert result.first_name == "NoOrg"
                assert result.last_name == "User"
                assert len(result.organizations) == 0

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_user_details_no_organizations_integration: {e}")
                raise

    async def test_get_user_details_user_not_found_integration(self, users_service, setup_test_db_integration):
        """Test getting user details when user doesn't exist."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]

        # Get a new session
        async for session in db_session:
            try:
                # Generate a UUID that doesn't exist
                non_existent_id = uuid4()

                # Execute and verify exception
                with pytest.raises(HTTPException) as excinfo:
                    await users_service._get_user_details(
                        user_id=non_existent_id,
                        session=session
                    )

                # Verify the exception
                assert excinfo.value.status_code == 404
                assert "User not found" in excinfo.value.detail

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_get_user_details_user_not_found_integration: {e}")
                raise

    async def test_setup_user_integration(self, users_service, setup_test_db_integration):
        """Test setting up a user with integration database."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Create a new user via _setup_user
                stytch_user = StytchUser(
                    email="new-integration-user@example.com",
                    stytch_id="new-stytch-id-integration",
                    first_name="New",
                    last_name="Integration"
                )

                # Execute
                result = await users_service._setup_user(
                    user_data=stytch_user,
                    org_id=org_id,
                    role="MEMBER",
                    session=session
                )

                # Assert
                assert result is not None
                assert result.email == stytch_user.email
                assert result.first_name == stytch_user.first_name
                assert result.last_name == stytch_user.last_name
                assert result.stytch_id == stytch_user.stytch_id

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_setup_user_integration: {e}")
                raise

    async def test_setup_existing_user_integration(self, users_service, setup_test_db_integration):
        """Test setting up an existing user with a new organization."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]

        # Get a new session
        async for session in db_session:
            try:
                # Create a new organization
                new_org_id = uuid4()
                await session.execute(
                    text("""
                        INSERT INTO organizations (id, name, slug)
                        VALUES (:id, 'Test Integration Org 2', 'test-integration-org-2')
                    """),
                    {"id": str(new_org_id)}
                )
                await session.commit()

                # Use existing user
                existing_user_id = test_data["user_id"]
                existing_stytch_id = test_data["stytch_id"]

                # Create StytchUser from existing user
                stytch_user = StytchUser(
                    email="test-integration-user@example.com",
                    stytch_id=existing_stytch_id,
                    first_name="Integration",
                    last_name="Test"
                )

                # Execute _setup_user
                result = await users_service._setup_user(
                    user_data=stytch_user,
                    org_id=new_org_id,
                    role="MEMBER",
                    session=session
                )

                # Assert we got the existing user back
                assert result.id == existing_user_id

                # Verify new organization membership was created
                query = select(OrganizationMemberModel).where(
                    OrganizationMemberModel.user_id == existing_user_id,
                    OrganizationMemberModel.organization_id == new_org_id
                )
                result = await session.execute(query)
                membership = result.scalar_one_or_none()
                assert membership is not None
                assert membership.status == "active"

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_setup_existing_user_integration: {e}")
                raise

    async def test_setup_organization_integration(self, users_service, setup_test_db_integration):
        """Test setting up an organization directly."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        user_id = test_data["user_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Create a new organization
                new_org_id = uuid4()
                await session.execute(
                    text("""
                        INSERT INTO organizations (id, name, slug)
                        VALUES (:id, 'Test Setup Organization', 'test-setup-org')
                    """),
                    {"id": str(new_org_id)}
                )
                await session.commit()

                # Execute the internal method directly
                await users_service._setup_organization(
                    user_id=user_id,
                    org_id=new_org_id,
                    role="MEMBER",
                    session=session
                )

                # Verify organization membership was created
                query = select(OrganizationMemberModel).where(
                    OrganizationMemberModel.user_id == user_id,
                    OrganizationMemberModel.organization_id == new_org_id
                )
                result = await session.execute(query)
                membership = result.scalar_one_or_none()

                # Assert
                assert membership is not None
                assert membership.status == "active"

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_setup_organization_integration: {e}")
                raise

    async def test_organization_not_found_integration(self, users_service, setup_test_db_integration):
        """Test setup organization with non-existent org."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        user_id = test_data["user_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Generate a UUID that doesn't exist
                non_existent_org_id = uuid4()

                # Execute and verify exception
                with pytest.raises(HTTPException) as excinfo:
                    await users_service._setup_organization(
                        user_id=user_id,
                        org_id=non_existent_org_id,
                        role="MEMBER",
                        session=session
                    )

                # Verify the exception
                assert excinfo.value.status_code == 404
                assert "Organization not found" in excinfo.value.detail

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_organization_not_found_integration: {e}")
                raise

    async def test_role_not_found_integration(self, users_service, setup_test_db_integration):
        """Test setup organization with non-existent role."""
        # Get setup data
        test_data = setup_test_db_integration
        db_session = test_data["db_session"]
        user_id = test_data["user_id"]
        org_id = test_data["org_id"]

        # Get a new session
        async for session in db_session:
            try:
                # Use a role name that doesn't exist
                non_existent_role = "NONEXISTENT_ROLE"

                # Execute and verify exception
                with pytest.raises(HTTPException) as excinfo:
                    await users_service._setup_organization(
                        user_id=user_id,
                        org_id=org_id,
                        role=non_existent_role,
                        session=session
                    )

                # Verify the exception
                assert excinfo.value.status_code == 500
                assert "Role not found" in excinfo.value.detail

                # Only process the first session
                break
            except Exception as e:
                print(f"Error in test_role_not_found_integration: {e}")
                raise