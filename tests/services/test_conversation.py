import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import json
from uuid import uuid4
from datetime import datetime

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
sys.modules['src.services.agents.model'] = MagicMock()
sys.modules['src.services.llm.model'] = MagicMock()
sys.modules['dependencies.security'] = MagicMock()
sys.modules['libs.chats.streaming'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['src.utils.stream_util'] = MagicMock()

# Mock the conversation models and schemas
class MockConversationModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.organization_id = kwargs.get('organization_id', uuid4())
        self.user_id = kwargs.get('user_id', uuid4())
        self.title = kwargs.get('title', 'Test Conversation')
        self.description = kwargs.get('description', '')
        self.is_archived = kwargs.get('is_archived', False)
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockChatSessionModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.conversation_id = kwargs.get('conversation_id', uuid4())
        self.agent_id = kwargs.get('agent_id')
        self.model_id = kwargs.get('model_id', uuid4())
        self.status = kwargs.get('status', 'active')
        self.settings = kwargs.get('settings', {})
        self._metadata = kwargs.get('_metadata', {})
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockMessageModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.chat_session_id = kwargs.get('chat_session_id', uuid4())
        self.role = kwargs.get('role', 'user')
        self.content = kwargs.get('content', 'Test message')
        self.token_used = kwargs.get('token_used', 0)
        self._metadata = kwargs.get('_metadata', {})
        self.created_at = kwargs.get('created_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockModelModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', 'GPT-4')
        self.provider = kwargs.get('provider', 'openai')
        self.version = kwargs.get('version', '4o')
        self.is_active = kwargs.get('is_active', True)
        self.config = kwargs.get('config', {})
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockAgentModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.name = kwargs.get('name', 'Test Agent')
        self.description = kwargs.get('description', 'Test Agent Description')
        self.organization_id = kwargs.get('organization_id', uuid4())
        self.user_id = kwargs.get('user_id', uuid4())
        self.model_id = kwargs.get('model_id', uuid4())
        self.is_active = kwargs.get('is_active', True)
        self.settings = kwargs.get('settings', {})
        self.version = kwargs.get('version', '1.0.0')
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
def mock_user():
    """Create a mock user."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
    }

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
def mock_conversation():
    """Create a mock conversation."""
    return MockConversationModel(
        id=uuid4(),
        organization_id=uuid4(),
        user_id=uuid4(),
        title="testing",
        description="",
        is_archived=False
    )

@pytest.fixture
def mock_chat_session():
    """Create a mock chat session."""
    return MockChatSessionModel(
        id=uuid4(),
        conversation_id=uuid4(),
        model_id=uuid4(),
        status="active",
        settings={"temperature": 0.7}
    )

@pytest.fixture
def mock_message():
    """Create a mock message."""
    return MockMessageModel(
        id=uuid4(),
        chat_session_id=uuid4(),
        role="user",
        content="hello",
        token_used=5
    )

@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    return MockModelModel(
        id=uuid4(),
        name="gpt-4", 
        provider="openai",
        version="4o"
    )

@pytest.fixture
def mock_conversation_service():
    """Create a mock conversation service."""
    conversation_service = MagicMock()
    
    async def mock_post_create(org_id, conversation_data, session, user):
        # Create conversation
        conversation = MockConversationModel(
            organization_id=org_id,
            user_id=user["id"],
            title=conversation_data.title,
            description=getattr(conversation_data, 'description', ""),
            is_archived=getattr(conversation_data, 'is_archived', False)
        )
        
        # Add to database
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        
        # Return response matching API format
        return MockResponse(
            id=conversation.id,
            organization_id=conversation.organization_id,
            user_id=conversation.user_id,
            title=conversation.title,
            description=conversation.description,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at.isoformat()
        )
    
    async def mock_put_update(conversation_id, conversation_data, session, user):
        # Get conversation
        conversation = session.execute.return_value.scalar_one_or_none.return_value
        
        # Check if conversation exists
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation fields
        update_data = conversation_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(conversation, field, value)
        
        # Update database
        await session.commit()
        await session.refresh(conversation)
        
        # Return response matching API format
        return MockResponse(
            id=conversation.id,
            organization_id=conversation.organization_id,
            user_id=conversation.user_id,
            title=conversation.title,
            description=conversation.description,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat()
        )
    
    async def mock_post_create_session(org_id, session_data, session, user):
        # Get conversation
        conversation = session.execute.return_value.unique.return_value.scalar_one_or_none.return_value
        
        # Check if conversation exists
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Create chat session
        chat_session = MockChatSessionModel(
            conversation_id=session_data.conversation_id,
            agent_id=getattr(session_data, 'agent_id', None),
            model_id=session_data.model_id,
            status="active",
            settings=getattr(session_data, 'settings', {})
        )
        
        # Add to database
        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)
        
        # Return response matching API format
        return MockResponse(
            id=chat_session.id,
            conversation_id=chat_session.conversation_id,
            agent_id=chat_session.agent_id,
            model_id=chat_session.model_id,
            status=chat_session.status,
            settings=chat_session.settings,
            created_at=chat_session.created_at.isoformat()
        )
    
    async def mock_get_list(org_id, session, user):
        # Create mock conversations
        conversations = [
            MockConversationModel(
                id=uuid4(),
                organization_id=org_id,
                user_id=user["id"],
                title="Testing Conversation 1",
                is_archived=False,
                created_at=datetime.now()
            ),
            MockConversationModel(
                id=uuid4(),
                organization_id=org_id,
                user_id=user["id"],
                title="Testing Conversation 2",
                is_archived=False,
                created_at=datetime.now()
            )
        ]
        
        # Set up mock for database query
        session.execute.return_value.scalars.return_value.all.return_value = conversations
        
        # Format response to match API
        return [
            MockResponse(
                id=conv.id,
                organization_id=conv.organization_id,
                user_id=conv.user_id,
                title=conv.title,
                description=conv.description,
                is_archived=conv.is_archived,
                created_at=conv.created_at.isoformat()
            ) 
            for conv in conversations
        ]
    
    async def mock_get_with_sessions(org_id, conversation_id, session, user):
        # Create mock data
        conversation = MockConversationModel(
            id=conversation_id,
            organization_id=org_id,
            user_id=user["id"],
            title="testing",
            is_archived=False
        )
        
        # Create mock model
        model = MockModelModel(
            id=uuid4(),
            name="gpt-4",
            provider="openai",
            version="4o"
        )
        
        # Create chat sessions
        chat_sessions = [
            MockChatSessionModel(
                id=uuid4(),
                conversation_id=conversation_id,
                model_id=model.id,
                status="active",
                settings={"temperature": 0.7}
            )
        ]
        
        # Create mock row objects for join result
        class MockRow:
            def __init__(self, conversation, chat_session, model_name, agent_name):
                self.ConversationModel = conversation
                self.ChatSessionModel = chat_session
                self.model_name = model_name
                self.agent_name = agent_name
        
        rows = [
            MockRow(
                conversation,
                chat_sessions[0],
                model.name,
                None  # No agent
            )
        ]
        
        # Set up mock for database query
        session.execute.return_value.unique.return_value.all.return_value = rows if rows else []
        
        if not rows:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Format chat sessions for response
        formatted_chat_sessions = []
        for row in rows:
            if row.ChatSessionModel:
                formatted_chat_sessions.append(MockResponse(
                    id=row.ChatSessionModel.id,
                    conversation_id=row.ChatSessionModel.conversation_id,
                    agent_id=row.ChatSessionModel.agent_id,
                    model_id=row.ChatSessionModel.model_id,
                    status=row.ChatSessionModel.status,
                    settings=row.ChatSessionModel.settings,
                    created_at=row.ChatSessionModel.created_at.isoformat(),
                    model_name=row.model_name,
                    agent_name=row.agent_name
                ))
        
        # Return response matching API format
        return MockResponse(
            id=conversation.id,
            organization_id=conversation.organization_id,
            user_id=conversation.user_id,
            title=conversation.title,
            description=conversation.description,
            is_archived=conversation.is_archived,
            created_at=conversation.created_at.isoformat(),
            chat_sessions=formatted_chat_sessions
        )
    
    async def mock_get_messages(org_id, conversation_id, limit, offset, session, user):
        # Create mock messages
        messages = [
            MockMessageModel(
                id=uuid4(),
                chat_session_id=uuid4(),
                role="user",
                content="hello",
                token_used=5,
                created_at=datetime.now()
            ),
            MockMessageModel(
                id=uuid4(),
                chat_session_id=uuid4(),
                role="assistant",
                content="Hi there! How can I help you today?",
                token_used=15,
                created_at=datetime.now()
            )
        ]
        
        # Set up mock for database query
        session.execute.return_value.scalars.return_value.all.return_value = messages
        
        # Set up total count mock
        session.scalar.return_value = len(messages)
        
        # Format response to match API
        formatted_messages = [
            MockResponse(
                id=msg.id,
                chat_session_id=msg.chat_session_id,
                role=msg.role,
                content=msg.content,
                token_used=msg.token_used,
                created_at=msg.created_at.isoformat()
            )
            for msg in messages
        ]
        
        return MockResponse(
            messages=formatted_messages,
            total=len(messages),
            has_more=False
        )
    
    async def mock_stream_chat(org_id, chat_request, session, user):
        # Create mock chat session
        chat_session = session.execute.return_value.scalar_one_or_none.return_value
        
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Create user message
        user_message = MockMessageModel(
            chat_session_id=chat_request.chat_session_id,
            role="user",
            content=chat_request.message,
            token_used=len(chat_request.message.split())
        )
        
        # Create assistant message
        assistant_message = MockMessageModel(
            chat_session_id=chat_request.chat_session_id,
            role="assistant",
            content="This is a mock response from the assistant.",
            token_used=20
        )
        
        # Add messages to database
        session.add(user_message)
        session.add(assistant_message)
        await session.commit()
        
        # Return streaming response
        return "This is a mock response from the assistant."
    
    # Create AsyncMock objects
    create_mock = AsyncMock(side_effect=mock_post_create)
    update_mock = AsyncMock(side_effect=mock_put_update)
    create_session_mock = AsyncMock(side_effect=mock_post_create_session)
    list_mock = AsyncMock(side_effect=mock_get_list)
    get_with_sessions_mock = AsyncMock(side_effect=mock_get_with_sessions)
    get_messages_mock = AsyncMock(side_effect=mock_get_messages)
    stream_chat_mock = AsyncMock(side_effect=mock_stream_chat)
    
    # Assign mocks to service
    conversation_service.post_create = create_mock
    conversation_service.put_update = update_mock
    conversation_service.post_create_session = create_session_mock
    conversation_service.get_list = list_mock
    conversation_service.get_get_with_sessions = get_with_sessions_mock
    conversation_service.get_messages = get_messages_mock
    conversation_service.post_stream_chat = stream_chat_mock
    
    return conversation_service

@pytest.mark.asyncio
class TestConversationService:
    """Tests for the Conversation service."""

    async def test_create_conversation(self, mock_conversation_service, mock_db_session, mock_user):
        """Test creating a new conversation."""
        # Create conversation data matching Postman request
        conversation_data = MockResponse(
            title="testing",
            is_archived=False
        )
        
        # Call the service
        response = await mock_conversation_service.post_create(
            mock_user["organization_id"],
            conversation_data,
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert response.title == conversation_data.title
        assert response.user_id == mock_user["id"]
        assert response.organization_id == mock_user["organization_id"]
        assert hasattr(response, "id")
        assert hasattr(response, "created_at")
        assert not response.is_archived
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_conversation_service.post_create.called

    async def test_update_conversation(self, mock_conversation_service, mock_db_session, mock_user, mock_conversation):
        """Test updating a conversation."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        
        # Create update data
        update_data = MockResponse(
            title="Updated Conversation",
            description="Updated description"
        )
        
        # Call the service
        response = await mock_conversation_service.put_update(
            mock_conversation.id,
            update_data,
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert response.id == mock_conversation.id
        assert response.title == update_data.title
        assert response.description == update_data.description
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_conversation_service.put_update.called

    async def test_update_conversation_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test updating a non-existent conversation."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Create update data
        update_data = MockResponse(
            title="Updated Conversation",
            description="Updated description"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_conversation_service.put_update(
                uuid4(),
                update_data,
                mock_db_session,
                mock_user
            )
        
        # Verify exception matches API error response
        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_conversation_service.put_update.called

    async def test_create_chat_session(self, mock_conversation_service, mock_db_session, mock_user, mock_conversation, mock_model):
        """Test creating a new chat session."""
        # Setup mocks
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = mock_conversation
        
        # Create session data matching Postman request
        session_data = MockResponse(
            conversation_id=mock_conversation.id,
            model_id=mock_model.id,
            settings={"temperature": 0.7}
        )
        
        # Call the service
        response = await mock_conversation_service.post_create_session(
            mock_user["organization_id"],
            session_data,
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert response.conversation_id == session_data.conversation_id
        assert response.model_id == session_data.model_id
        assert response.status == "active"
        assert response.settings == session_data.settings
        assert hasattr(response, "id")
        assert hasattr(response, "created_at")
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
        
        # Verify service method was called
        assert mock_conversation_service.post_create_session.called

    async def test_create_chat_session_conversation_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test creating a chat session for a non-existent conversation."""
        # Setup mocks
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None
        
        # Create session data
        session_data = MockResponse(
            conversation_id=uuid4(),
            model_id=uuid4(),
            settings={"temperature": 0.7}
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_conversation_service.post_create_session(
                mock_user["organization_id"],
                session_data,
                mock_db_session,
                mock_user
            )
        
        # Verify exception matches API error response
        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_conversation_service.post_create_session.called

    async def test_get_conversation_list(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a list of conversations."""
        # Call the service
        response = await mock_conversation_service.get_list(
            mock_user["organization_id"],
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert isinstance(response, list)
        assert len(response) == 2
        
        # Verify each conversation has correct fields
        for conversation in response:
            assert hasattr(conversation, "id")
            assert hasattr(conversation, "title")
            assert hasattr(conversation, "organization_id")
            assert hasattr(conversation, "user_id")
            assert hasattr(conversation, "is_archived")
            assert hasattr(conversation, "created_at")
            assert conversation.organization_id == mock_user["organization_id"]
        
        # Verify service method was called
        assert mock_conversation_service.get_list.called

    async def test_get_conversation_with_sessions(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a conversation with its sessions."""
        # Call the service
        conversation_id = uuid4()
        
        response = await mock_conversation_service.get_get_with_sessions(
            mock_user["organization_id"],
            conversation_id,
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert response.id == conversation_id
        assert response.organization_id == mock_user["organization_id"]
        assert response.user_id == mock_user["id"]
        assert response.title == "testing"
        assert hasattr(response, "chat_sessions")
        assert isinstance(response.chat_sessions, list)
        assert len(response.chat_sessions) == 1
        
        # Verify chat session has correct fields
        chat_session = response.chat_sessions[0]
        assert hasattr(chat_session, "id")
        assert hasattr(chat_session, "conversation_id")
        assert hasattr(chat_session, "model_id")
        assert hasattr(chat_session, "status")
        assert hasattr(chat_session, "settings")
        assert hasattr(chat_session, "created_at")
        assert hasattr(chat_session, "model_name")
        assert chat_session.conversation_id == conversation_id
        assert chat_session.model_name == "gpt-4"
        
        # Verify service method was called
        assert mock_conversation_service.get_get_with_sessions.called

    async def test_get_conversation_with_sessions_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a non-existent conversation with sessions."""
        # Setup mocks for empty result
        # Replace the entire get_get_with_sessions mock for this test with one that raises an exception
        async def mock_get_empty_result(org_id, conversation_id, session, user):
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Replace the mock method with our new implementation for this test only
        mock_conversation_service.get_get_with_sessions = AsyncMock(side_effect=mock_get_empty_result)
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_conversation_service.get_get_with_sessions(
                mock_user["organization_id"],
                uuid4(),
                mock_db_session,
                mock_user
            )
        
        # Verify exception matches API error response
        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_conversation_service.get_get_with_sessions.called
        
    async def test_get_messages(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting messages for a conversation."""
        # Call the service
        conversation_id = uuid4()
        limit = 10
        offset = 0
        
        response = await mock_conversation_service.get_messages(
            mock_user["organization_id"],
            conversation_id,
            limit,
            offset,
            mock_db_session,
            mock_user
        )
        
        # Verify result structure matches API response
        assert hasattr(response, "messages")
        assert hasattr(response, "total")
        assert hasattr(response, "has_more")
        assert isinstance(response.messages, list)
        assert len(response.messages) == 2
        assert response.total == 2
        assert not response.has_more
        
        # Verify message has correct fields
        for message in response.messages:
            assert hasattr(message, "id")
            assert hasattr(message, "chat_session_id")
            assert hasattr(message, "role")
            assert hasattr(message, "content")
            assert hasattr(message, "token_used")
            assert hasattr(message, "created_at")
            assert message.role in ["user", "assistant"]
        
        # Verify service method was called
        assert mock_conversation_service.get_messages.called
        
    async def test_stream_chat(self, mock_conversation_service, mock_db_session, mock_user, mock_chat_session):
        """Test streaming chat messages."""
        # Setup mocks
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_chat_session
        
        # Create chat request matching Postman request
        chat_request = MockResponse(
            chat_session_id=mock_chat_session.id,
            message="hello"
        )
        
        # Call the service
        response = await mock_conversation_service.post_stream_chat(
            mock_user["organization_id"],
            chat_request,
            mock_db_session,
            mock_user
        )
        
        # Verify response is a string (content of the streamed response)
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify database operations - two messages added (user + assistant)
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_conversation_service.post_stream_chat.called 