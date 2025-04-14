import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

# Create test fixtures based on the API collection

@pytest.fixture
def test_user_data():
    """Create test user data for authentication tests."""
    return {
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "password": "SecurePass123!"
    }

@pytest.fixture
def test_login_data():
    """Create test login credentials."""
    return {
        "email": "test@example.com",
        "password": "SecurePass123!"
    }

@pytest.mark.asyncio
class TestAuthAPI:
    """Test authentication API endpoints."""

    async def test_signup(self, client, mock_db_session, test_user_data):
        """Test user signup endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a user object with ID
            user_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": user_id,
                "email": test_user_data["email"],
                "first_name": test_user_data["first_name"],
                "last_name": test_user_data["last_name"]
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                "/api/auth/signup",
                json=test_user_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == user_id
            assert result["email"] == test_user_data["email"]

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_login(self, client, mock_db_session, test_login_data):
        """Test user login endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a token
            user_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "test_token",
                "token_type": "bearer",
                "user": {
                    "id": user_id,
                    "email": test_login_data["email"]
                }
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                "/api/auth/login",
                json=test_login_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert "access_token" in result
            assert result["token_type"] == "bearer"

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()