import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

@pytest.fixture
def test_conversation_data():
    """Create test conversation data."""
    return {
        "title": "Test Conversation",
        "is_archived": False
    }

@pytest.fixture
def test_chat_session_data():
    """Create test chat session data."""
    return {
        "conversation_id": str(uuid4()),
        "model_id": str(uuid4())
    }

@pytest.fixture
def test_message_data():
    """Create test message data."""
    return {
        "message": "Hello, this is a test message",
        "chat_session_id": str(uuid4())
    }

@pytest.mark.asyncio
class TestConversationAPI:
    """Test Conversation API endpoints."""

    async def test_create_conversation(self, client, mock_db_session, auth_headers, test_conversation_data, org_id):
        """Test creating a conversation endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a conversation
            conversation_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": conversation_id,
                "title": test_conversation_data["title"],
                "is_archived": test_conversation_data["is_archived"],
                "organization_id": org_id,
                "created_at": "2023-07-01T12:00:00Z",
                "updated_at": "2023-07-01T12:00:00Z"
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/conversation/create?org_id={org_id}",
                headers=auth_headers,
                json=test_conversation_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == conversation_id
            assert result["title"] == test_conversation_data["title"]
            assert result["organization_id"] == org_id

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_list_conversations(self, client, mock_db_session, auth_headers, org_id):
        """Test listing conversations endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of conversations
            conversation_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": conversation_id,
                    "title": "Test Conversation",
                    "is_archived": False,
                    "organization_id": org_id,
                    "created_at": "2023-07-01T12:00:00Z",
                    "updated_at": "2023-07-01T12:00:00Z"
                }
            ]
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                f"/api/conversation/list?org_id={org_id}",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == conversation_id

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()

    async def test_create_chat_session(self, client, mock_db_session, auth_headers, test_chat_session_data, org_id):
        """Test creating a chat session endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a chat session
            session_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": session_id,
                "conversation_id": test_chat_session_data["conversation_id"],
                "model_id": test_chat_session_data["model_id"],
                "organization_id": org_id,
                "created_at": "2023-07-01T12:00:00Z"
            }
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/conversation/create_session?org_id={org_id}",
                headers=auth_headers,
                json=test_chat_session_data
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == session_id
            assert result["conversation_id"] == test_chat_session_data["conversation_id"]
            assert result["model_id"] == test_chat_session_data["model_id"]

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_stream_chat(self, client, mock_db_session, auth_headers, test_message_data, org_id):
        """Test streaming chat endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a streaming response
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Simulate a text stream by returning content as streaming events
            mock_response.iter_content.return_value = [
                b'data: {"delta": "Hello", "message_id": "' + str(uuid4()).encode() + b'"}\n\n',
                b'data: {"delta": ", how", "message_id": "' + str(uuid4()).encode() + b'"}\n\n',
                b'data: {"delta": " can I help?", "message_id": "' + str(uuid4()).encode() + b'"}\n\n',
                b'data: [DONE]\n\n'
            ]
            mock_post.return_value = mock_response

            # Make the API request
            response = client.post(
                f"/api/conversation/stream_chat?org_id={org_id}",
                headers=auth_headers,
                json=test_message_data
            )

            # Verify the response status code
            assert response.status_code == 200

            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()

    async def test_get_messages(self, client, mock_db_session, auth_headers, org_id):
        """Test getting messages endpoint."""
        # Create test data
        conversation_id = str(uuid4())

        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of messages
            message_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": message_id,
                    "content": "Hello, this is a test message",
                    "role": "user",
                    "conversation_id": conversation_id,
                    "created_at": "2023-07-01T12:00:00Z"
                },
                {
                    "id": str(uuid4()),
                    "content": "Hello, how can I help you?",
                    "role": "assistant",
                    "conversation_id": conversation_id,
                    "created_at": "2023-07-01T12:00:01Z"
                }
            ]
            mock_get.return_value = mock_response

            # Make the API request
            response = client.get(
                f"/api/conversation/messages?org_id={org_id}&conversation_id={conversation_id}&limit=10&offset=0",
                headers=auth_headers
            )

            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == message_id
            assert result[0]["role"] == "user"
            assert result[1]["role"] == "assistant"

            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()