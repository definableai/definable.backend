import pytest
import sys
import json
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from src.services.auth.service import AuthService
from src.services.auth.schema import (
    StytchUser,
    TestLogin,
    TestResponse,
    TestSignup,
    InviteSignup
)
from src.services.__base.acquire import Acquire
from src.libs.response import LibResponse

# Import the actual Stytch client to use for tests
from src.libs.stytch.v1 import stytch_base
from src.config.settings import settings

# Mock modules to prevent SQLAlchemy issues
sys.modules["database"] = MagicMock()
sys.modules["database.postgres"] = MagicMock()
sys.modules["src.database"] = MagicMock()
sys.modules["src.database.postgres"] = MagicMock()


# TestAcquire - mock of the Acquire class for service initialization
class TestAcquire(Acquire):
    def __init__(self):
        self.settings = settings
        self.logger = MagicMock()
        self.utils = MagicMock()


# Mock model classes
class MockUserModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.stytch_id = kwargs.get('stytch_id', f"stytch_id_{uuid4().hex[:8]}")
        self.email = kwargs.get('email', "test@example.com")
        self.first_name = kwargs.get('first_name', "Test")
        self.last_name = kwargs.get('last_name', "User")
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        self._metadata = kwargs.get('_metadata', {})
        
        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class MockOrganizationModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', "Test Organization")
        self.slug = kwargs.get('slug', "test-organization")
        self.settings = kwargs.get('settings', {})
        self.is_active = kwargs.get('is_active', True)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        
        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class MockRoleModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', "owner")
        self.description = kwargs.get('description', "Owner Role")
        self.is_system_role = kwargs.get('is_system_role', True)
        self.hierarchy_level = kwargs.get('hierarchy_level', 100)
        self.organization_id = kwargs.get('organization_id', None)
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        
        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class MockOrganizationMemberModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.organization_id = kwargs.get('organization_id', uuid4())
        self.user_id = kwargs.get('user_id', uuid4())
        self.role_id = kwargs.get('role_id', uuid4())
        self.status = kwargs.get('status', "active")
        self.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))
        
        # Add any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


# Create a class with model_dump_json for update_user method
class MockUpdateUserResponseData:
    def __init__(self, user_id=None):
        self.user_id = user_id or str(uuid4())
    
    def model_dump_json(self):
        return json.dumps({"user_id": self.user_id})


@pytest.fixture
def auth_service():
    """Create an AuthService instance."""
    return AuthService(acquire=TestAcquire())


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
def mock_request():
    """Create a mock FastAPI request with webhook data."""
    request = MagicMock()
    request.headers = {
        "svix-id": "test-id",
        "svix-timestamp": "test-timestamp",
        "svix-signature": "test-signature"
    }
    # Set up body method to return JSON webhook data
    request.body = AsyncMock(return_value=json.dumps({
        "action": "CREATE",
        "user": {
            "user_id": f"user-test-{uuid4().hex[:8]}",
            "emails": [{"email": "test@example.com"}],
            "name": {
                "first_name": "Test",
                "last_name": "User"
            },
            "untrusted_metadata": {}
        }
    }).encode())
    
    return request


# Generate a unique email for tests to avoid conflicts
@pytest.fixture
def test_email():
    """Generate a unique email for testing."""
    unique_id = uuid4().hex[:8]
    return f"test.user.{unique_id}@example.com"


@pytest.mark.asyncio
class TestAuthService:
    """Test the AuthService class."""
    
    # List to keep track of Stytch user IDs created during tests
    created_stytch_users = []
    
    @classmethod
    async def async_teardown_class(cls):
        """Clean up test users created in Stytch after tests are done."""
        print(f"Cleaning up {len(cls.created_stytch_users)} Stytch test users...")
        for user_id in cls.created_stytch_users:
            try:
                # Try to delete the user from Stytch
                response = await stytch_base.client.users.delete_async(user_id)
                print(f"Deleted user {user_id}: {response.status_code}")
            except Exception as e:
                print(f"Error deleting user {user_id}: {str(e)}")
    
    @classmethod
    def teardown_class(cls):
        """Non-async wrapper for teardown to clean up users."""
        import asyncio
        asyncio.run(cls.async_teardown_class())
    
    async def test_post_webhook_create_user(self, auth_service, mock_db_session, mock_request):
        """Test processing a webhook POST request for user creation."""
        
        # Mock the verify_svix_signature function to return True
        with patch("src.services.auth.service.verify_svix_signature", return_value=True):
            # Mock the _create_new_user method to return a user
            mock_user = MockUserModel()
            with patch.object(auth_service, '_create_new_user', AsyncMock(return_value=mock_user)):
                # Mock stytch_base.update_user
                update_response_data = MockUpdateUserResponseData()
                with patch("src.services.auth.service.stytch_base.update_user", 
                          AsyncMock(return_value=LibResponse(success=True, data=update_response_data))):
                    
                    response = await auth_service.post(mock_request, mock_db_session)
                    
                    assert isinstance(response, JSONResponse)
                    assert response.status_code == 200
                    content = json.loads(response.body)
                    assert content["message"] == "User created successfully"
    
    async def test_post_webhook_invalid_signature(self, auth_service, mock_db_session, mock_request):
        """Test webhook with invalid signature."""
        
        # Mock the verify_svix_signature function to return False (invalid signature)
        with patch("src.services.auth.service.verify_svix_signature", return_value=False):
            with pytest.raises(HTTPException) as excinfo:
                await auth_service.post(mock_request, mock_db_session)
            
            assert excinfo.value.status_code == 400
            assert "Invalid signature" in excinfo.value.detail

    async def test_post_webhook_temp_user(self, auth_service, mock_db_session, mock_request):
        """Test webhook with temp user flag."""
        
        # Modify the mock request to have temp flag in untrusted_metadata
        mock_request.body = AsyncMock(return_value=json.dumps({
            "action": "CREATE",
            "user": {
                "user_id": f"user-test-{uuid4().hex[:8]}",
                "emails": [{"email": "test@example.com"}],
                "name": {
                    "first_name": "Test",
                    "last_name": "User"
                },
                "untrusted_metadata": {"temp": True}
            }
        }).encode())
        
        # Mock the verify_svix_signature function to return True
        with patch("src.services.auth.service.verify_svix_signature", return_value=True):
            response = await auth_service.post(mock_request, mock_db_session)
            
            assert isinstance(response, JSONResponse)
            assert response.status_code == 200
            content = json.loads(response.body)
            assert content["message"] == "User created from temp"

    async def test_post_webhook_no_email(self, auth_service, mock_db_session, mock_request):
        """Test webhook with no email in user data."""
        
        # Modify the mock request to have empty emails array
        mock_request.body = AsyncMock(return_value=json.dumps({
            "action": "CREATE",
            "user": {
                "user_id": f"user-test-{uuid4().hex[:8]}",
                "emails": [],
                "name": {
                    "first_name": "Test",
                    "last_name": "User"
                },
                "untrusted_metadata": {}
            }
        }).encode())
        
        # Mock the verify_svix_signature function to return True
        with patch("src.services.auth.service.verify_svix_signature", return_value=True):
            response = await auth_service.post(mock_request, mock_db_session)
            
            assert isinstance(response, JSONResponse)
            assert response.status_code == 200
            content = json.loads(response.body)
            assert content["message"] == "No email found"
    
    async def test_post_webhook_invalid_action(self, auth_service, mock_db_session, mock_request):
        """Test webhook with invalid action."""
        
        # Modify the mock request to have an unsupported action
        mock_request.body = AsyncMock(return_value=json.dumps({
            "action": "UPDATE",
            "user": {
                "user_id": f"user-test-{uuid4().hex[:8]}",
                "emails": [{"email": "test@example.com"}],
                "name": {
                    "first_name": "Test",
                    "last_name": "User"
                },
                "untrusted_metadata": {}
            }
        }).encode())
        
        # Mock the verify_svix_signature function to return True
        with patch("src.services.auth.service.verify_svix_signature", return_value=True):
            response = await auth_service.post(mock_request, mock_db_session)
            
            assert isinstance(response, JSONResponse)
            assert response.status_code == 200
            content = json.loads(response.body)
            assert content["message"] == "Invalid action"

    async def test_post_signup_invite(self, auth_service, mock_db_session):
        """Test signup invite endpoint."""
        
        # Create test invite data
        invite_data = InviteSignup(
            first_name="John",
            last_name="Doe",
            email="john.doe@example.com"
        )
        
        # Mock token payload
        token_payload = {"sub": "test-user", "org_id": str(uuid4())}
        
        # Call the method
        response = await auth_service.post_signup_invite(invite_data, token_payload, mock_db_session)
        
        # Check response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        content = json.loads(response.body)
        assert content["status"] == "success"
    
    async def test_post_test_signup_success(self, auth_service, mock_db_session, test_email):
        """Test test signup endpoint success case with real Stytch integration."""
        
        # Create test signup data with unique email
        test_signup = TestSignup(
            first_name="Test",
            last_name="User",
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Mock database user that will be created from the test signup
        mock_user = MockUserModel(
            id=uuid4(),
            email=test_email
        )
        
        # Mock _create_new_user to return the mock user
        with patch.object(auth_service, '_create_new_user', AsyncMock(return_value=mock_user)):
            # Call the method to use the real Stytch client
            response = await auth_service.post_test_signup(test_signup, mock_db_session)
            
            # Check response
            assert isinstance(response, TestResponse)
            assert response.user_id == mock_user.id
            assert response.email == test_email
            assert response.stytch_token is not None
            assert response.stytch_user_id is not None
            
            # Store the created stytch_id for cleanup
            mock_user.stytch_id = response.stytch_user_id
            
            # Add to the list of created users for cleanup
            self.created_stytch_users.append(response.stytch_user_id)
            
            # Return the stytch_id and session token for cleanup
            return response.stytch_user_id, response.stytch_token
    
    async def test_post_test_signup_db_error(self, auth_service, mock_db_session, test_email):
        """Test test signup endpoint with database error."""
        
        # Create test signup data with unique email
        test_signup = TestSignup(
            first_name="Test",
            last_name="User",
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Mock _create_new_user to return None (user not created in db)
        with patch.object(auth_service, '_create_new_user', AsyncMock(return_value=None)):
            # Call the method and expect HTTPException
            with pytest.raises(HTTPException) as excinfo:
                await auth_service.post_test_signup(test_signup, mock_db_session)
            
            # Check exception
            assert excinfo.value.status_code == 500
            assert "Failed to create user in database" in excinfo.value.detail
    
    async def test_post_test_login_success(self, auth_service, mock_db_session, test_email):
        """Test test login endpoint success case with real Stytch integration."""
        
        # First create a user with Stytch to login with
        signup_data = TestSignup(
            first_name="Test",
            last_name="User",
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Create mock database user
        mock_user = MockUserModel(
            id=uuid4(),
            email=test_email
        )
        
        # Mock the database user creation for signup
        with patch.object(auth_service, '_create_new_user', AsyncMock(return_value=mock_user)):
            # Create the test user with real Stytch
            signup_response = await auth_service.post_test_signup(signup_data, mock_db_session)
            
            # Store the stytch_id for the created user
            mock_user.stytch_id = signup_response.stytch_user_id
            
            # Add to the list of created users for cleanup
            self.created_stytch_users.append(signup_response.stytch_user_id)
        
        # Create test login data
        test_login = TestLogin(
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Mock db query to return the user
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=mock_user)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Call the login method with real Stytch
        response = await auth_service.post_test_login(test_login, mock_db_session)
        
        # Check response
        assert isinstance(response, TestResponse)
        assert response.user_id == mock_user.id
        assert response.email == mock_user.email
        assert response.stytch_token is not None
        assert response.stytch_user_id == mock_user.stytch_id
    
    async def test_post_test_login_user_not_found(self, auth_service, mock_db_session, test_email):
        """Test test login endpoint with user not found in database."""
        
        # First create a user to login with
        signup_data = TestSignup(
            first_name="Test",
            last_name="User",
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Create mock database user
        mock_user = MockUserModel(
            id=uuid4(),
            email=test_email
        )
        
        # Mock the database user creation for signup
        with patch.object(auth_service, '_create_new_user', AsyncMock(return_value=mock_user)):
            # Create the test user with real Stytch
            signup_response = await auth_service.post_test_signup(signup_data, mock_db_session)
            
            # Add to the list of created users for cleanup
            self.created_stytch_users.append(signup_response.stytch_user_id)
        
        # Create test login data
        test_login = TestLogin(
            email=test_email,
            password="Password123*(^$!*(^&@*(&!"
        )
        
        # Mock db query to return None (user not found)
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=None)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Call the method and expect HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.post_test_login(test_login, mock_db_session)
        
        # Check exception
        assert excinfo.value.status_code == 500
        assert "User not found in database" in excinfo.value.detail
    
    async def test_create_new_user_success(self, auth_service, mock_db_session):
        """Test _create_new_user method success case."""
        
        # Create test stytch user data
        stytch_user = StytchUser(
            email="john.doe@example.com",
            stytch_id="user-test-123",
            first_name="John",
            last_name="Doe",
            metadata={}
        )
        
        # Mock db query to return None (user doesn't exist yet)
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=None)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Mock the _setup_default_organization method
        with patch.object(auth_service, '_setup_default_organization', AsyncMock()):
            
            # Call the method
            result = await auth_service._create_new_user(stytch_user, mock_db_session)
            
            # Check result
            assert result is not None
            assert result.email == "john.doe@example.com"
            assert result.stytch_id == "user-test-123"
            assert result.first_name == "John"
            assert result.last_name == "Doe"
            
            # Verify session methods were called
            mock_db_session.add.assert_called_once()
            mock_db_session.flush.assert_called_once()
            mock_db_session.commit.assert_called_once()
            mock_db_session.refresh.assert_called_once()
    
    async def test_create_new_user_existing_user(self, auth_service, mock_db_session):
        """Test _create_new_user method with existing user."""
        
        # Create test stytch user data
        stytch_user = StytchUser(
            email="john.doe@example.com",
            stytch_id="user-test-123",
            first_name="John",
            last_name="Doe",
            metadata={}
        )
        
        # Create existing user
        existing_user = MockUserModel(
            email="john.doe@example.com",
            stytch_id="user-test-123"
        )
        
        # Mock db query to return existing user
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=existing_user)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Call the method
        result = await auth_service._create_new_user(stytch_user, mock_db_session)
        
        # Check that None is returned (user already exists)
        assert result is None
        
        # Verify session methods were not called
        mock_db_session.add.assert_not_called()
        mock_db_session.flush.assert_not_called()
        mock_db_session.commit.assert_not_called()
        mock_db_session.refresh.assert_not_called()
    
    async def test_setup_default_organization(self, auth_service, mock_db_session):
        """Test _setup_default_organization method."""
        
        # Create test user
        user = MockUserModel(id=uuid4())
        
        # Create test role
        owner_role = MockRoleModel(
            id=uuid4(),
            name="owner",
            hierarchy_level=100
        )
        
        # Mock db query to return owner role
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=owner_role)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Call the method
        result = await auth_service._setup_default_organization(user, mock_db_session)
        
        # Check result
        assert result is not None
        assert result.name == "Default Org"
        assert "default_" in result.slug
        assert result.is_active is True
        
        # Verify session methods were called
        assert mock_db_session.add.call_count == 2  # Once for org, once for member
        assert mock_db_session.flush.call_count == 2  # Once after org creation, once after member
    
    async def test_setup_default_organization_no_owner_role(self, auth_service, mock_db_session):
        """Test _setup_default_organization method with no owner role found."""
        
        # Create test user
        user = MockUserModel(id=uuid4())
        
        # Mock db query to return None (no owner role)
        unique_mock = MagicMock()
        unique_mock.scalar_one_or_none = MagicMock(return_value=None)
        
        execute_mock = AsyncMock()
        execute_mock.unique = MagicMock(return_value=unique_mock)
        
        mock_db_session.execute = AsyncMock(return_value=execute_mock)
        
        # Call the method and expect HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service._setup_default_organization(user, mock_db_session)
        
        # Check exception
        assert excinfo.value.status_code == 500
        assert "Default owner role not found" in excinfo.value.detail
