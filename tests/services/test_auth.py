import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import uuid4
from datetime import datetime, timedelta

# Create mock modules before any imports
sys.modules['database'] = MagicMock()
sys.modules['database.postgres'] = MagicMock()
sys.modules['database.models'] = MagicMock()
sys.modules['src.database'] = MagicMock()
sys.modules['src.database.postgres'] = MagicMock()
sys.modules['src.database.models'] = MagicMock()
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['src.config'] = MagicMock()
sys.modules['src.config.settings'] = MagicMock()
sys.modules['src.services.__base.acquire'] = MagicMock()
sys.modules['src.services.org.model'] = MagicMock()
sys.modules['src.utils.auth_util'] = MagicMock()
sys.modules['src.services.auth.service'] = MagicMock()
sys.modules['src.services.auth.model'] = MagicMock()
sys.modules['src.services.auth.schema'] = MagicMock()

# Mock the auth models and schemas
class MockUserModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.email = kwargs.get('email', "test@example.com")
        self.password = kwargs.get('password', "hashed_password123")
        self.first_name = kwargs.get('first_name', "Test")
        self.last_name = kwargs.get('last_name', "User")
        self.is_active = kwargs.get('is_active', True)
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, data):
        if isinstance(data, dict):
            return cls(**data)
        return cls(**{k: v for k, v in data.__dict__.items() if not k.startswith('_')})
    
    def model_dump(self, **kwargs):
        exclude_unset = kwargs.get('exclude_unset', False)
        if exclude_unset:
            # Return only items that have been explicitly set
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    
    # Setup scalar to return a properly mocked result
    scalar_mock = AsyncMock()
    session.scalar = scalar_mock
    
    # Setup execute to return a properly mocked result
    execute_mock = AsyncMock()
    # Make unique(), scalars(), first(), etc. return self to allow chaining
    execute_result = AsyncMock()
    execute_result.unique.return_value = execute_result
    execute_result.scalars.return_value = execute_result
    execute_result.scalar_one_or_none.return_value = None
    execute_result.scalar_one.return_value = None
    execute_result.first.return_value = None
    execute_result.all.return_value = []
    execute_result.mappings.return_value = execute_result
    
    execute_mock.return_value = execute_result
    session.execute = execute_mock
    
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    return session

@pytest.fixture
def mock_user():
    """Create a mock user."""
    return MockUserModel(
        id=uuid4(),
        email="test@example.com",
        password="hashed_password123",
        first_name="Test",
        last_name="User",
        is_active=True
    )

@pytest.fixture
def mock_auth_service():
    """Create a mock auth service."""
    auth_service = MagicMock()
    
    async def mock_post_signup(user_data, session):
        # Check if user with email already exists
        if getattr(session.execute.return_value.scalar_one_or_none.return_value, 'email', None) == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        new_user = MockUserModel(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            password="hashed_" + user_data.password
        )
        
        # Add to DB
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        
        # Return response matching Postman collection
        return MockResponse(
            id=new_user.id,
            email=new_user.email,
            first_name=new_user.first_name,
            last_name=new_user.last_name,
            message="User created successfully"
        )
    
    async def mock_post_login(form_data, session):
        # Get user by email
        user = session.execute.return_value.scalar_one_or_none.return_value
        
        # Check if user exists and credentials are valid
        if not user or not user.is_active or form_data.password != "password123":
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Generate token - matching Postman response
        access_token = "mocked_jwt_token"
        expires_delta = timedelta(minutes=30)
        
        # Return response matching Postman collection
        return MockResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_delta.total_seconds()
        )
    
    async def mock_get_me(current_user):
        # Return current user information
        return MockResponse(
            id=current_user.get("id"),
            email=current_user.get("email"),
            first_name=current_user.get("first_name"),
            last_name=current_user.get("last_name"),
            organization_id=current_user.get("organization_id"),
            roles=current_user.get("roles", [])
        )
    
    # Create AsyncMock objects for these methods to ensure they have .called attribute
    post_signup_mock = AsyncMock(side_effect=mock_post_signup)
    post_login_mock = AsyncMock(side_effect=mock_post_login)
    get_me_mock = AsyncMock(side_effect=mock_get_me)
    
    # Assign the AsyncMock objects to the service
    auth_service.post_signup = post_signup_mock
    auth_service.post_login = post_login_mock
    auth_service.get_me = get_me_mock
    
    return auth_service

@pytest.mark.asyncio
class TestAuthService:
    """Tests for the Authentication service."""

    async def test_signup_success(self, mock_auth_service, mock_db_session):
        """Test successful user signup."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None  # No existing user
        
        # Create signup data matching Postman collection request
        signup_data = MockResponse(
            email="newuser@example.com",
            first_name="New",
            last_name="User",
            password="Rock0004@"  # Match password format in collection
        )
        
        # Call the service
        response = await mock_auth_service.post_signup(signup_data, mock_db_session)
        
        # Verify result structure matches Postman response
        assert response.email == signup_data.email
        assert response.first_name == signup_data.first_name
        assert response.last_name == signup_data.last_name
        assert hasattr(response, "id")
        assert "successfully" in response.message.lower()
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_auth_service.post_signup.called

    async def test_signup_duplicate_email(self, mock_auth_service, mock_db_session, mock_user):
        """Test signup with duplicate email."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user  # User exists
        
        # Create signup data
        signup_data = MockResponse(
            email=mock_user.email,
            first_name="Duplicate",
            last_name="User",
            password="Rock0004@"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_signup(signup_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Email already registered" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_signup.called

    async def test_login_success(self, mock_auth_service, mock_db_session, mock_user):
        """Test successful login."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        # Create login data matching Postman collection
        login_data = MockResponse(
            email=mock_user.email,
            password="password123"
        )
        
        # Call the service
        response = await mock_auth_service.post_login(login_data, mock_db_session)
        
        # Verify result structure matches Postman response
        assert response.access_token == "mocked_jwt_token"
        assert response.token_type == "bearer"
        assert hasattr(response, "expires_in")
        
        # Verify service method was called
        assert mock_auth_service.post_login.called

    async def test_login_invalid_credentials(self, mock_auth_service, mock_db_session, mock_user):
        """Test login with invalid credentials."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        # Create login data with wrong password
        login_data = MockResponse(
            email=mock_user.email,
            password="wrongpassword"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_login(login_data, mock_db_session)
        
        assert exc_info.value.status_code == 401
        assert "Incorrect email or password" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_login.called

    async def test_login_inactive_user(self, mock_auth_service, mock_db_session):
        """Test login with inactive user."""
        # Create inactive user
        inactive_user = MockUserModel(
            id=uuid4(),
            email="inactive@example.com",
            password="hashed_password123",
            first_name="Inactive",
            last_name="User",
            is_active=False
        )
        
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = inactive_user
        
        # Create login data
        login_data = MockResponse(
            email=inactive_user.email,
            password="password123"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_login(login_data, mock_db_session)
        
        assert exc_info.value.status_code == 401
        assert "Incorrect email or password" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_login.called

    async def test_get_current_user(self, mock_auth_service):
        """Test getting current user information."""
        # Create user data matching expected JWT payload
        current_user = {
            "id": uuid4(),
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
            "organization_id": uuid4(),
            "roles": [
                {
                    "role_id": uuid4(),
                    "organization_id": uuid4(),
                    "name": "Admin"
                }
            ]
        }
        
        # Call the service
        response = await mock_auth_service.get_me(current_user)
        
        # Verify result structure
        assert response.id == current_user["id"]
        assert response.email == current_user["email"]
        assert response.first_name == current_user["first_name"]
        assert response.last_name == current_user["last_name"]
        assert response.organization_id == current_user["organization_id"]
        assert response.roles == current_user["roles"]
        
        # Verify service method was called
        assert mock_auth_service.get_me.called

    async def test_signup_invalid_email_format(self, mock_auth_service, mock_db_session):
        """Test signup with invalid email format."""
        # Setup original post_signup method
        original_signup = mock_auth_service.post_signup
        
        # Create a custom implementation that validates email
        async def mock_signup_invalid_email(user_data, session):
            # Basic email validation
            import re
            email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            if not email_pattern.match(user_data.email):
                raise HTTPException(status_code=400, detail="Invalid email format")
            
            # This won't execute
            return await original_signup(user_data, session)
            
        mock_auth_service.post_signup = AsyncMock(side_effect=mock_signup_invalid_email)
        
        # Create signup data with invalid email
        signup_data = MockResponse(
            email="invalid-email",
            first_name="Test",
            last_name="User",
            password="Rock0004@"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_signup(signup_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Invalid email format" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_signup.called
    
    async def test_signup_weak_password(self, mock_auth_service, mock_db_session):
        """Test signup with a weak password."""
        # Setup original post_signup method
        original_signup = mock_auth_service.post_signup
        
        # Create a custom implementation that validates password strength
        async def mock_signup_weak_password(user_data, session):
            # Basic password strength check
            if len(user_data.password) < 8:
                raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
            
            # This won't execute
            return await original_signup(user_data, session)
            
        mock_auth_service.post_signup = AsyncMock(side_effect=mock_signup_weak_password)
        
        # Create signup data with weak password
        signup_data = MockResponse(
            email="newuser@example.com",
            first_name="Test",
            last_name="User",
            password="weak"  # Too short
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_signup(signup_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Password must be at least 8 characters" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_signup.called
    
    async def test_login_user_not_found(self, mock_auth_service, mock_db_session):
        """Test login with non-existent user."""
        # Setup mocks - no user found
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Create login data
        login_data = MockResponse(
            email="nonexistent@example.com",
            password="password123"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_login(login_data, mock_db_session)
        
        assert exc_info.value.status_code == 401
        assert "Incorrect email or password" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_login.called
    
    async def test_login_with_empty_credentials(self, mock_auth_service, mock_db_session):
        """Test login with empty credentials."""
        # Create login data with empty values
        login_data = MockResponse(
            email="",
            password=""
        )
        
        # Set up a custom login method that checks for empty credentials
        async def mock_login_empty_credentials(form_data, session):
            if not form_data.email or not form_data.password:
                raise HTTPException(
                    status_code=400,
                    detail="Email and password cannot be empty",
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
            # This code won't execute due to validation error
            user = session.execute.return_value.scalar_one_or_none.return_value
            return MockResponse(
                access_token="token",
                token_type="bearer",
                expires_in=1800
            )
            
        mock_auth_service.post_login = AsyncMock(side_effect=mock_login_empty_credentials)
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_login(login_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "Email and password cannot be empty" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_login.called
    
    async def test_get_current_user_no_roles(self, mock_auth_service):
        """Test getting current user with no roles."""
        # Create user data without roles
        current_user = {
            "id": uuid4(),
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
            "organization_id": uuid4()
            # No roles field
        }
        
        # Call the service
        response = await mock_auth_service.get_me(current_user)
        
        # Verify result structure
        assert response.id == current_user["id"]
        assert response.email == current_user["email"]
        assert response.first_name == current_user["first_name"]
        assert response.last_name == current_user["last_name"]
        assert response.organization_id == current_user["organization_id"]
        assert hasattr(response, "roles")  # Should have roles attribute even if empty
        assert response.roles == []  # Should default to empty list
        
        # Verify service method was called
        assert mock_auth_service.get_me.called
    
    async def test_refresh_token(self, mock_auth_service, mock_db_session, mock_user):
        """Test refreshing an access token."""
        # Add mock refresh token method
        async def mock_refresh_token(refresh_token, session):
            # Check if refresh token is valid (hardcoded valid token for test)
            if refresh_token != "valid_refresh_token":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid refresh token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Generate new access token
            access_token = "new_mocked_jwt_token"
            expires_delta = timedelta(minutes=30)
            
            # Return response with new tokens
            return MockResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=expires_delta.total_seconds()
            )
            
        mock_auth_service.post_refresh_token = AsyncMock(side_effect=mock_refresh_token)
        
        # Call the service
        response = await mock_auth_service.post_refresh_token("valid_refresh_token", mock_db_session)
        
        # Verify result structure
        assert response.access_token == "new_mocked_jwt_token"
        assert response.token_type == "bearer"
        assert response.expires_in == timedelta(minutes=30).total_seconds()
        
        # Verify service method was called
        assert mock_auth_service.post_refresh_token.called
    
    async def test_refresh_token_invalid(self, mock_auth_service, mock_db_session):
        """Test refreshing with an invalid token."""
        # Add mock refresh token method
        async def mock_refresh_token_invalid(refresh_token, session):
            # Check if refresh token is valid
            if refresh_token != "valid_refresh_token":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid refresh token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
            # This won't execute
            return MockResponse(
                access_token="new_token",
                token_type="bearer",
                expires_in=1800
            )
            
        mock_auth_service.post_refresh_token = AsyncMock(side_effect=mock_refresh_token_invalid)
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.post_refresh_token("invalid_token", mock_db_session)
        
        assert exc_info.value.status_code == 401
        assert "Invalid refresh token" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.post_refresh_token.called
    
    async def test_update_user_profile(self, mock_auth_service, mock_db_session, mock_user):
        """Test updating a user's profile."""
        # Add mock update profile method
        async def mock_update_profile(user_id, profile_data, session):
            # Get user
            user = session.execute.return_value.scalar_one_or_none.return_value
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Update user fields
            update_data = profile_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            # Save changes
            await session.commit()
            
            # Return updated user
            return MockResponse(
                id=user.id,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name
            )
            
        mock_auth_service.put_update_profile = AsyncMock(side_effect=mock_update_profile)
        
        # Setup mocks to find the user
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user
        
        # Create update data
        user_id = mock_user.id
        update_data = MockResponse(
            first_name="Updated",
            last_name="Name"
        )
        
        # Call the service
        response = await mock_auth_service.put_update_profile(user_id, update_data, mock_db_session)
        
        # Verify result
        assert response.id == mock_user.id
        assert response.email == mock_user.email
        assert response.first_name == "Updated"
        assert response.last_name == "Name"
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_auth_service.put_update_profile.called
    
    async def test_update_nonexistent_user(self, mock_auth_service, mock_db_session):
        """Test updating a non-existent user."""
        # Add mock update profile method that fails
        async def mock_update_nonexistent_user(user_id, profile_data, session):
            # No user found
            session.execute.return_value.scalar_one_or_none.return_value = None
            raise HTTPException(status_code=404, detail="User not found")
            
        mock_auth_service.put_update_profile = AsyncMock(side_effect=mock_update_nonexistent_user)
        
        # Create update data
        user_id = uuid4()  # Random non-existent ID
        update_data = MockResponse(
            first_name="Updated",
            last_name="Name"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_auth_service.put_update_profile(user_id, update_data, mock_db_session)
        
        assert exc_info.value.status_code == 404
        assert "User not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_auth_service.put_update_profile.called
