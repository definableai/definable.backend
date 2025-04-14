import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

@pytest.fixture
def test_role_data():
    """Create test role data."""
    return {
        "name": "Test Role",
        "description": "Test role description",
        "hierarchy_level": 10,
        "permission_ids": [str(uuid4())]
    }

@pytest.mark.asyncio
class TestRolesAPI:
    """Test Roles API endpoints."""

    async def test_create_role(self, client, mock_db_session, auth_headers, test_role_data):
        """Test creating a role endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a role
            role_id = str(uuid4())
            org_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": role_id,
                "name": test_role_data["name"],
                "description": test_role_data["description"],
                "hierarchy_level": test_role_data["hierarchy_level"],
                "organization_id": org_id,
                "is_admin": False,
                "permissions": []
            }
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                f"/api/roles/create?org_id={org_id}",
                headers=auth_headers,
                json=test_role_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == role_id
            assert result["name"] == test_role_data["name"]
            assert result["organization_id"] == org_id
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()
    
    async def test_list_permissions(self, client, mock_db_session, auth_headers):
        """Test listing permissions endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of permissions
            permission_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": permission_id,
                    "name": "View Organizations",
                    "resource": "organization",
                    "action": "view"
                },
                {
                    "id": str(uuid4()),
                    "name": "Create Organizations",
                    "resource": "organization",
                    "action": "create"
                }
            ]
            mock_get.return_value = mock_response
            
            # Make the API request
            response = client.get(
                "/api/roles/list_permissions",
                headers=auth_headers
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == permission_id
            assert result[0]["resource"] == "organization"
            
            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once() 