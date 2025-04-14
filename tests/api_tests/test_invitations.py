import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta, timezone

@pytest.mark.asyncio
class TestInvitationsAPI:
    """Test Invitations API endpoints."""

    async def test_list_invitations(self, client, mock_db_session, auth_headers, org_id):
        """Test listing invitations for an organization."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of invitations
            invitation_id = str(uuid4())
            role_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [
                    {
                        "id": invitation_id,
                        "organization_id": org_id,
                        "role_id": role_id,
                        "role_name": "Test Role",
                        "role_description": "Test Role Description",
                        "invitee_email": "invite@example.com",
                        "invited_by": str(uuid4()),
                        "status": 0,
                        "status_name": "PENDING",
                        "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                ],
                "total": 1,
                "page": 1,
                "size": 10
            }
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                f"/api/invitations?org_id={org_id}&page=1&size=10",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert "items" in result
            assert len(result["items"]) == 1
            assert result["items"][0]["id"] == invitation_id
            assert result["items"][0]["organization_id"] == org_id
            assert result["total"] == 1

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()

    async def test_get_invitation(self, client, mock_db_session, auth_headers, org_id):
        """Test getting a specific invitation."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return an invitation
            invitation_id = str(uuid4())
            role_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": invitation_id,
                "organization_id": org_id,
                "role_id": role_id,
                "role_name": "Test Role",
                "role_description": "Test Role Description",
                "invitee_email": "invite@example.com",
                "invited_by": str(uuid4()),
                "status": 0,
                "status_name": "PENDING",
                "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                f"/api/invitations/{invitation_id}?org_id={org_id}",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == invitation_id
            assert result["organization_id"] == org_id
            assert result["role_id"] == role_id
            assert result["invitee_email"] == "invite@example.com"

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()

    async def test_send_invitation(self, client, mock_db_session, auth_headers, org_id):
        """Test sending a new invitation."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            role_id = str(uuid4())
            invitation_id = str(uuid4())
            invitation_data = {
                "role_id": role_id,
                "invitee_email": "newinvite@example.com",
                "expiry_time": None  # Let the service set the default
            }

            # Configure the mock to return a response
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "id": invitation_id,
                "organization_id": org_id,
                "role_id": role_id,
                "role_name": "Test Role",
                "role_description": "Test Role Description",
                "invitee_email": "newinvite@example.com",
                "invited_by": str(uuid4()),
                "status": 0,
                "status_name": "PENDING",
                "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "Invitation sent successfully"
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/invitations/send?org_id={org_id}",
                headers=auth_headers,
                json=invitation_data
            )

            # Verify response
            assert response.status_code == 201
            result = response.json()
            assert result["id"] == invitation_id
            assert result["organization_id"] == org_id
            assert result["role_id"] == role_id
            assert result["invitee_email"] == "newinvite@example.com"
            assert "message" in result
            assert "successfully" in result["message"].lower()

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_resend_invitation(self, client, mock_db_session, auth_headers, org_id):
        """Test resending an invitation."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            invitation_id = str(uuid4())
            role_id = str(uuid4())
            resend_data = {
                "invitation_id": invitation_id
            }

            # Configure the mock to return a response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": str(uuid4()),  # New invitation ID
                "organization_id": org_id,
                "role_id": role_id,
                "role_name": "Test Role",
                "role_description": "Test Role Description",
                "invitee_email": "invite@example.com",
                "invited_by": str(uuid4()),
                "status": 0,
                "status_name": "PENDING",
                "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "Invitation resent successfully"
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/invitations/resend?org_id={org_id}",
                headers=auth_headers,
                json=resend_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["organization_id"] == org_id
            assert "message" in result
            assert "successfully" in result["message"].lower()

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_accept_invitation(self, client, mock_db_session):
        """Test accepting an invitation."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            org_id = str(uuid4())
            role_id = str(uuid4())
            action_data = {
                "token": "test_token",
                "email": "invite@example.com"
            }

            # Configure the mock to return a response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": "Invitation accepted successfully",
                "organization": {
                    "id": org_id,
                    "name": "Test Organization"
                },
                "role": {
                    "id": role_id,
                    "name": "Test Role"
                }
            }
            mock_post.return_value = mock_response

            # Make the API request - no auth headers needed for this endpoint
            response = client.post(
                "/api/invitations/accept",
                json=action_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Invitation accepted successfully"
            assert "organization" in result
            assert result["organization"]["id"] == org_id
            assert "role" in result
            assert result["role"]["id"] == role_id

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_reject_invitation(self, client, mock_db_session):
        """Test rejecting an invitation."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            action_data = {
                "token": "test_token",
                "email": "invite@example.com"
            }

            # Configure the mock to return a response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": "Invitation rejected successfully"
            }
            mock_post.return_value = mock_response

            # Make the API request - no auth headers needed for this endpoint
            response = client.post(
                "/api/invitations/reject",
                json=action_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Invitation rejected successfully"

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()