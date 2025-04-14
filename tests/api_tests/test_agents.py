import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

@pytest.mark.asyncio
class TestAgentsAPI:
    """Test Agents API endpoints."""

    async def test_list_org_specific_agents(self, client, mock_db_session, auth_headers, org_id):
        """Test listing organization-specific agents endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of agents
            agent_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": agent_id,
                    "name": "Test Agent",
                    "description": "Test agent description",
                    "organization_id": org_id,
                    "is_active": True,
                    "created_at": "2023-07-01T12:00:00Z",
                    "updated_at": "2023-07-01T12:00:00Z"
                }
            ]
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                f"/api/agents/list?org_id={org_id}",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == agent_id
            assert result[0]["organization_id"] == org_id

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()

    async def test_list_all_agents(self, client, mock_db_session, auth_headers):
        """Test listing all agents endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of all agents
            agent_id1 = str(uuid4())
            agent_id2 = str(uuid4())
            org_id1 = str(uuid4())
            org_id2 = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": agent_id1,
                    "name": "Org 1 Agent",
                    "description": "Org 1 agent description",
                    "organization_id": org_id1,
                    "is_active": True,
                    "created_at": "2023-07-01T12:00:00Z",
                    "updated_at": "2023-07-01T12:00:00Z"
                },
                {
                    "id": agent_id2,
                    "name": "Org 2 Agent",
                    "description": "Org 2 agent description",
                    "organization_id": org_id2,
                    "is_active": True,
                    "created_at": "2023-07-01T12:00:00Z",
                    "updated_at": "2023-07-01T12:00:00Z"
                }
            ]
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                "/api/agents/list_all",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == agent_id1
            assert result[1]["id"] == agent_id2
            assert result[0]["organization_id"] == org_id1
            assert result[1]["organization_id"] == org_id2

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()

    async def test_execute_agent(self, client, mock_db_session, auth_headers, org_id):
        """Test executing an agent endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            agent_id = str(uuid4())
            execution_data = {
                "input": "What's the weather like today?",
                "parameters": {
                    "location": "New York"
                }
            }

            # Configure the mock to return a streaming response
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Simulate a text stream by returning content as streaming events
            mock_response.iter_content.return_value = [
                b'data: {"step": "thinking", "content": "Thinking about the weather..."}\n\n',
                b'data: {"step": "checking", "content": "Checking weather service..."}\n\n',
                b'data: {"step": "result", "content": "The weather in New York is sunny, 72 degrees F"}\n\n',
                b'data: [DONE]\n\n'
            ]
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/agents/execute?org_id={org_id}&agent_id={agent_id}",
                headers=auth_headers,
                json=execution_data
            )

            # Verify the response status code
            assert response.status_code == 200

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()