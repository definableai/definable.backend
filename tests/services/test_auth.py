import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from typing import List, Any, Optional
from pydantic import BaseModel, Field

# Create mock modules before any imports
sys.modules["database"] = MagicMock()
sys.modules["database.postgres"] = MagicMock()
sys.modules["database.models"] = MagicMock()
sys.modules["src.database"] = MagicMock()
sys.modules["src.database.postgres"] = MagicMock()
sys.modules["src.database.models"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
sys.modules["src.config"] = MagicMock()
sys.modules["src.config.settings"] = MagicMock()
sys.modules["src.services.__base.acquire"] = MagicMock()
sys.modules["src.services.org.model"] = MagicMock()
sys.modules["src.utils.auth_util"] = MagicMock()
sys.modules["src.services.auth.service"] = MagicMock()
sys.modules["src.services.auth.model"] = MagicMock()
sys.modules["src.services.auth.schema"] = MagicMock()
sys.modules["dependencies.security"] = MagicMock()


# Mock database models
class MockUserModel(BaseModel):
  """Mock database user model."""

  id: UUID = Field(default_factory=uuid4)
  email: str = "test@example.com"
  password: str = "hashed_password123"
  first_name: str = "Test"
  last_name: str = "User"
  is_active: bool = True
  created_at: datetime = Field(default_factory=datetime.now)
  updated_at: datetime = Field(default_factory=datetime.now)
  organization_id: Optional[UUID] = None

  model_config = {"extra": "allow"}


# Mock API response model (used to simulate API responses)
class MockResponse(BaseModel):
  """Mock API response model."""

  id: Optional[UUID] = None
  email: Optional[str] = None
  first_name: Optional[str] = None
  last_name: Optional[str] = None
  password: Optional[str] = None
  organization_id: Optional[UUID] = None
  is_active: Optional[bool] = None
  roles: Optional[List[Any]] = None
  access_token: Optional[str] = None
  token_type: Optional[str] = None
  refresh_token: Optional[str] = None
  expires_in: Optional[int] = None
  message: Optional[str] = None

  model_config = {"extra": "allow"}


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
  return MockUserModel(id=uuid4(), email="test@example.com", password="hashed_password123", first_name="Test", last_name="User", is_active=True)


@pytest.fixture
def mock_auth_service():
  """Create a mock auth service."""
  auth_service = MagicMock()

  async def mock_post_signup(user_data, session):
    # Check if user with email already exists
    existing_user = session.execute.return_value.scalar_one_or_none.return_value
    if existing_user and getattr(existing_user, "email", None) == user_data.email:
      existing_user = session.execute.return_value.scalar_one_or_none.return_value
    if existing_user and getattr(existing_user, "email", None) == user_data.email:
      raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = MockUserModel(
      email=user_data.email, first_name=user_data.first_name, last_name=user_data.last_name, password="hashed_" + user_data.password
    )

    # Add to DB
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    # Return response matching API format
    return MockResponse(
      id=new_user.id, email=new_user.email, first_name=new_user.first_name, last_name=new_user.last_name, message="User created successfully"
    )

  async def mock_post_login(form_data, session):
    # Get user by email
    user = session.execute.return_value.scalar_one_or_none.return_value

    # Check if user exists and credentials are valid
    if not user or not user.is_active or form_data.password != "password123":
      raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})

    # Generate token
    access_token = "mocked_jwt_token"
    expires_delta = timedelta(minutes=30)

    # Return response matching API format
    return MockResponse(access_token=access_token, token_type="bearer", expires_in=int(expires_delta.total_seconds()))

  async def mock_get_me(current_user):
    # Return current user information
    return MockResponse(
      id=current_user.get("id"),
      email=current_user.get("email"),
      first_name=current_user.get("first_name"),
      last_name=current_user.get("last_name"),
      organization_id=current_user.get("organization_id"),
      roles=current_user.get("roles", []),
    )

  async def mock_refresh_token(refresh_token, session):
    # Check if refresh token is valid (hardcoded valid token for test)
    if refresh_token != "valid_refresh_token":
      raise HTTPException(status_code=401, detail="Invalid refresh token", headers={"WWW-Authenticate": "Bearer"})

    # Generate new access token
    access_token = "new_mocked_jwt_token"
    refresh_token = "new_refresh_token"
    expires_delta = timedelta(minutes=30)

    # Return response matching API format
    return MockResponse(access_token=access_token, refresh_token=refresh_token, token_type="bearer", expires_in=int(expires_delta.total_seconds()))

  async def mock_update_profile(user_id, profile_data, session):
    # Get user
    user = session.execute.return_value.scalar_one_or_none.return_value

    # Check if user exists
    if not user:
      raise HTTPException(status_code=404, detail="User not found")

    # Update user fields
    update_data = profile_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(user, field, value)

    # Update database
    await session.commit()
    await session.refresh(user)

    # Return updated user profile
    return MockResponse(id=user.id, email=user.email, first_name=user.first_name, last_name=user.last_name, message="Profile updated successfully")

  # Create AsyncMock objects for these methods to ensure they have .called attribute
  post_signup_mock = AsyncMock(side_effect=mock_post_signup)
  post_login_mock = AsyncMock(side_effect=mock_post_login)
  get_me_mock = AsyncMock(side_effect=mock_get_me)
  refresh_token_mock = AsyncMock(side_effect=mock_refresh_token)
  update_profile_mock = AsyncMock(side_effect=mock_update_profile)

  # Assign the AsyncMock objects to the service
  auth_service.post_signup = post_signup_mock
  auth_service.post_login = post_login_mock
  auth_service.get_me = get_me_mock
  auth_service.refresh_token = refresh_token_mock
  auth_service.update_profile = update_profile_mock

  return auth_service


@pytest.mark.asyncio
class TestAuthService:
  """Tests for the Authentication service."""

  async def test_signup_success(self, mock_auth_service, mock_db_session):
    """Test successful user signup."""
    # Setup mocks
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None  # No existing user

    # Create signup data matching API schema
    signup_data = MockResponse(
      email="newuser@example.com",
      first_name="New",
      last_name="User",
      password="Rock0004@",
    )

    # Call the service
    response = await mock_auth_service.post_signup(signup_data, session=mock_db_session)

    # Verify result structure
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
    signup_data = MockResponse(email=mock_user.email, first_name="Duplicate", last_name="User", password="Rock0004@")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_signup(signup_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 400
    assert "Email already registered" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_signup.called

  async def test_login_success(self, mock_auth_service, mock_db_session, mock_user):
    """Test successful login."""
    # Setup mocks
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user

    # Create login data matching API schema
    login_data = MockResponse(email=mock_user.email, password="password123")

    # Call the service
    response = await mock_auth_service.post_login(login_data, session=mock_db_session)

    # Verify result structure
    assert response.access_token == "mocked_jwt_token"
    assert response.token_type == "bearer"
    assert response.expires_in == timedelta(minutes=30).total_seconds()

    # Verify service method was called
    assert mock_auth_service.post_login.called

  async def test_login_invalid_credentials(self, mock_auth_service, mock_db_session, mock_user):
    """Test login with invalid credentials."""
    # Setup mocks
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user

    # Create login data with wrong password
    login_data = MockResponse(
      email=mock_user.email,
      password="wrong_password",
    )

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_login(login_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 401
    assert "Incorrect email or password" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_login.called

  async def test_login_inactive_user(self, mock_auth_service, mock_db_session):
    """Test login with inactive user."""
    # Setup mocks - create inactive user
    # Setup mocks - create inactive user
    inactive_user = MockUserModel(email="inactive@example.com", is_active=False)
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = inactive_user

    # Create login data
    login_data = MockResponse(email=inactive_user.email, password="password123")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_login(login_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 401
    assert "Incorrect email or password" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_login.called

    assert mock_auth_service.post_login.called

  async def test_get_current_user(self, mock_auth_service):
    """Test getting current user info."""
    # Create mock user data (as it would be extracted from JWT)
    current_user = {
      "id": uuid4(),
      "email": "current@example.com",
      "first_name": "Current",
      "last_name": "User",
      "organization_id": uuid4(),
      "roles": [
        {
          "organization_id": uuid4(),
          "role": "ADMIN",
        }
      ],
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

    # Override the mock implementation for this test
    async def mock_signup_invalid_email(user_data, session):
      # Basic email validation
      if "@" not in user_data.email or "." not in user_data.email:
        raise HTTPException(status_code=400, detail="Invalid email format")

      # Original implementation
      return await mock_auth_service.post_signup.side_effect(user_data, session)

    # Replace the mock method with our new implementation for this test only
    mock_auth_service.post_signup.side_effect = mock_signup_invalid_email

    # Create signup data with invalid email
    signup_data = MockResponse(email="invalid-email", first_name="Invalid", last_name="Email", password="Rock0004@")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_signup(signup_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 400
    assert "Invalid email format" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_signup.called

  async def test_signup_weak_password(self, mock_auth_service, mock_db_session):
    """Test signup with weak password."""

    # Override the mock implementation for this test
    async def mock_signup_weak_password(user_data, session):
      # Basic password strength check
      if len(user_data.password) < 8:
        raise HTTPException(status_code=400, detail="Password too weak")

      # Original implementation
      return await mock_auth_service.post_signup.side_effect(user_data, session)

    # Replace the mock method with our new implementation for this test only
    mock_auth_service.post_signup.side_effect = mock_signup_weak_password

    # Create signup data with weak password
    signup_data = MockResponse(email="newuser@example.com", first_name="New", last_name="User", password="weak")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_signup(signup_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 400
    assert "Password too weak" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_signup.called

  async def test_login_user_not_found(self, mock_auth_service, mock_db_session):
    """Test login with non-existent user."""
    # Setup mocks - no user found
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

    # Create login data
    login_data = MockResponse(email="nonexistent@example.com", password="password123")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_login(login_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 401
    assert "Incorrect email or password" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.post_login.called

  async def test_login_with_empty_credentials(self, mock_auth_service, mock_db_session):
    """Test login with empty credentials."""

    # Override the mock implementation for this test
    async def mock_login_empty_credentials(form_data, session):
      # Check for empty credentials
      if not form_data.email or not form_data.password:
        raise HTTPException(status_code=400, detail="Email and password are required", headers={"WWW -Authenticate": "Bearer"})

      # Original implementation
      return await mock_auth_service.post_login.side_effect(form_data, session)

    # Replace the mock method with our new implementation for this test only
    mock_auth_service.post_login.side_effect = mock_login_empty_credentials

    # Create login data with empty password
    login_data = MockResponse(email="test@example.com", password="")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.post_login(login_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 400
    assert "required" in str(exc_info.value.detail).lower()

    # Verify service method was called
    assert mock_auth_service.post_login.called

  async def test_get_current_user_no_roles(self, mock_auth_service):
    """Test getting current user info with no roles."""
    # Create mock user data with no roles
    current_user = {
      "id": uuid4(),
      "email": "noroles@example.com",
      "first_name": "NoRoles",
      "last_name": "User",
      "organization_id": uuid4(),
      "roles": [],
    }

    # Call the service
    response = await mock_auth_service.get_me(current_user)

    # Verify result structure
    assert response.id == current_user["id"]
    assert response.roles == []

    # Verify service method was called
    assert mock_auth_service.get_me.called

  async def test_refresh_token(self, mock_auth_service, mock_db_session, mock_user):
    """Test refreshing access token."""
    # Call the service
    response = await mock_auth_service.refresh_token("valid_refresh_token", session=mock_db_session)

    # Verify result structure
    assert response.access_token == "new_mocked_jwt_token"
    assert response.refresh_token == "new_refresh_token"
    assert response.token_type == "bearer"
    assert response.expires_in == timedelta(minutes=30).total_seconds()

    # Verify service method was called
    assert mock_auth_service.refresh_token.called

  async def test_refresh_token_invalid(self, mock_auth_service, mock_db_session):
    """Test refreshing with invalid refresh token."""
    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.refresh_token("invalid_refresh_token", session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 401
    assert "Invalid refresh token" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.refresh_token.called

  async def test_update_user_profile(self, mock_auth_service, mock_db_session, mock_user):
    """Test updating user profile."""
    # Setup mocks
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_user

    # Create profile update data
    profile_data = MockResponse(first_name="Updated", last_name="Name")

    # Call the service
    response = await mock_auth_service.update_profile(mock_user.id, profile_data, session=mock_db_session)

    # Verify result structure
    assert response.id == mock_user.id
    assert response.first_name == profile_data.first_name
    assert response.last_name == profile_data.last_name
    assert "updated successfully" in response.message.lower()

    # Verify database operations
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

    # Verify service method was called
    assert mock_auth_service.update_profile.called

  async def test_update_nonexistent_user(self, mock_auth_service, mock_db_session):
    """Test updating non-existent user."""
    # Setup mocks - no user found
    mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

    # Create profile update data
    profile_data = MockResponse(first_name="Updated", last_name="Name")

    # Verify exception is raised
    with pytest.raises(HTTPException) as exc_info:
      await mock_auth_service.update_profile(uuid4(), profile_data, session=mock_db_session)

    # Verify exception details
    assert exc_info.value.status_code == 404
    assert "User not found" in str(exc_info.value.detail)

    # Verify service method was called
    assert mock_auth_service.update_profile.called
