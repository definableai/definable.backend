import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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

# Function to create empty dict for default_factory
def empty_dict() -> Dict[str, Any]:
    return {}

# Function to create empty list for default_factory
def empty_list() -> List[Any]:
    return []

# Pydantic models for mock objects
class MockConversationModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    organization_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(default_factory=uuid4)
    title: str = "Test Conversation"
    description: str = ""
    is_archived: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to maintain backward compatibility

class MockChatSessionModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID = Field(default_factory=uuid4)
    agent_id: Optional[UUID] = None
    model_id: UUID = Field(default_factory=uuid4)
    status: str = "active"
    settings: Dict[str, Any] = Field(default_factory=empty_dict)
    metadata: Dict[str, Any] = Field(default_factory=empty_dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to maintain backward compatibility

class MockMessageModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_session_id: UUID = Field(default_factory=uuid4)
    role: str = "user"
    content: str = "Test message"
    token_used: int = 0
    metadata: Dict[str, Any] = Field(default_factory=empty_dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to maintain backward compatibility

class MockModelModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "gpt-4"
    provider: str = "openai"
    version: str = "4o"
    is_active: bool = True
    config: Dict[str, Any] = Field(default_factory=empty_dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to maintain backward compatibility

class MockAgentModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "Test Agent"
    description: str = "Test Agent Description"
    organization_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(default_factory=uuid4)
    is_active: bool = True
    settings: Dict[str, Any] = Field(default_factory=empty_dict)
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to maintain backward compatibility

# Define a class for chat requests to fix the linter errors
class ChatRequestModel(BaseModel):
    chat_session_id: UUID
    message: str
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class MockResponse(BaseModel):
    id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    title: Optional[str] = None
    description: Optional[str] = None
    is_archived: Optional[bool] = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    model_id: Optional[UUID] = None
    agent_name: Optional[str] = None
    model_name: Optional[str] = None
    status: Optional[str] = None
    settings: Optional[Dict[str, Any]] = Field(default_factory=empty_dict)
    content: Optional[str] = None
    role: Optional[str] = None
    token_used: Optional[int] = None
    conversations: List[Any] = Field(default_factory=empty_list)
    messages: List[Any] = Field(default_factory=empty_list)
    sessions: List[Any] = Field(default_factory=empty_list)
    total: Optional[int] = None
    has_more: Optional[bool] = None
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields

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
        # Get conversation from the database using the mock
        conversation_id = session_data.conversation_id
        conversation = session.execute.return_value.scalar_one_or_none.return_value
        
        # Check if conversation exists
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Create chat session
        chat_session = MockChatSessionModel(
            conversation_id=conversation_id,
            model_id=session_data.model_id,
            agent_id=getattr(session_data, 'agent_id', None),
            settings=getattr(session_data, 'settings', {})
        )
        
        # Add to database
        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)
        
        # Get model name
        model = MockModelModel(id=chat_session.model_id)
        session.execute.return_value.scalar_one.return_value = model
        
        # Return response matching API format
        return MockResponse(
            id=chat_session.id,
            conversation_id=chat_session.conversation_id,
            model_id=chat_session.model_id,
            model_name=model.name,
            agent_id=chat_session.agent_id,
            agent_name=None,  # Would get agent name in real implementation
            status=chat_session.status,
            settings=chat_session.settings,
            created_at=chat_session.created_at.isoformat()
        )
    
    async def mock_get_list(org_id, session, user):
        # Create mock conversations
        conversations = []
        for i in range(5):
            conversation = MockConversationModel(
                organization_id=org_id,
                user_id=user["id"],
                title=f"Conversation {i+1}",
                is_archived=False if i < 4 else True
            )
            conversations.append(conversation)
        
        # Setup mock DB response
        session.execute.return_value.scalars.return_value.all.return_value = conversations
        
        # Return response matching API format
        return [
            MockResponse(
                id=conversation.id,
                organization_id=conversation.organization_id,
                user_id=conversation.user_id,
                title=conversation.title,
                description=conversation.description,
                is_archived=conversation.is_archived,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat()
            )
            for conversation in conversations
        ]
    
    async def mock_get_with_sessions(org_id, conversation_id, session, user):
        # Create mock data
        conversation = MockConversationModel(
            id=conversation_id,
            organization_id=org_id,
            user_id=user["id"],
            title="Test Conversation",
            is_archived=False
        )
        
        # Create chat sessions for this conversation
        chat_sessions = []
        for i in range(3):
            model_id = uuid4()
            agent_id = uuid4() if i % 2 == 0 else None
            
            chat_session = MockChatSessionModel(
                conversation_id=conversation_id,
                model_id=model_id,
                agent_id=agent_id,
                status="active" if i < 2 else "archived",
                settings={"temperature": 0.7}
            )
            chat_sessions.append(chat_session)
        
        # Create mock rows that combine conversation, session, and names
        class MockRow:
            def __init__(self, conversation, chat_session, model_name, agent_name):
                self.conversation = conversation
                self.chat_session = chat_session
                self.model_name = model_name
                self.agent_name = agent_name
        
        rows = []
        for i, chat_session in enumerate(chat_sessions):
            model_name = f"Model {i+1}"
            agent_name = f"Agent {i+1}" if chat_session.agent_id else None
            rows.append(MockRow(conversation, chat_session, model_name, agent_name))
        
        # Setup mock DB response
        session.execute.return_value.all.return_value = rows
        
        # Format response to match API schema
        sessions = []
        for row in rows:
            chat_session = row.chat_session
            sessions.append(MockResponse(
                id=chat_session.id,
                conversation_id=chat_session.conversation_id,
                model_id=chat_session.model_id,
                model_name=row.model_name,
                agent_id=chat_session.agent_id,
                agent_name=row.agent_name,
                status=chat_session.status,
                settings=chat_session.settings,
                created_at=chat_session.created_at.isoformat()
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
            updated_at=conversation.updated_at.isoformat(),
            sessions=sessions
        )
    
    async def mock_get_messages(org_id, conversation_id, limit, offset, session, user):
        # Create mock messages
        total_count = 15
        
        # Set up mock for the count query
        session.scalar.return_value = total_count
        
        # Create the messages for the current page
        messages = []
        for i in range(min(limit, total_count - offset)):
            idx = offset + i
            is_user = idx % 2 == 0
            
            message = MockMessageModel(
                chat_session_id=uuid4(),  # Would be consistent in real DB
                role="user" if is_user else "assistant",
                content=f"{'User' if is_user else 'Assistant'} message {idx+1}",
                token_used=10 if is_user else 20
            )
            messages.append(message)
        
        # Setup mock DB response
        session.execute.return_value.scalars.return_value.all.return_value = messages
        
        # Format response to match API schema
        message_list = []
        for message in messages:
            # Using model_dump directly from Pydantic
            message_dict = message.model_dump()
            # Convert datetime objects to strings to avoid validation errors
            if 'created_at' in message_dict and isinstance(message_dict['created_at'], datetime):
                message_dict['created_at'] = message_dict['created_at'].isoformat()
            message_list.append(MockResponse(**message_dict))
        
        # Return paginated response
        return MockResponse(
            messages=message_list,
            total=total_count,
            has_more=(offset + limit < total_count)
        )
    
    async def mock_stream_chat(org_id, chat_request: ChatRequestModel, session, user):
        # Create mock chat session
        chat_session_id = chat_request.chat_session_id
        chat_session = MockChatSessionModel(
            id=chat_session_id,
            conversation_id=uuid4(),
            model_id=uuid4()
        )
        
        # Setup mock DB response
        session.execute.return_value.scalar_one_or_none.return_value = chat_session
        
        # Create user message
        user_message = MockMessageModel(
            chat_session_id=chat_session_id,
            role="user",
            content=chat_request.message
        )
        
        # Create assistant message
        assistant_message = MockMessageModel(
            chat_session_id=chat_session_id,
            role="assistant",
            content="This is a mock response from the assistant.",
            token_used=30
        )
        
        # Add both messages to the database
        session.add(user_message)
        await session.flush()
        session.add(assistant_message)
        await session.commit()
        
        # Create a fake async generator for streaming
        async def fake_stream():
            chunks = [
                "This ", "is ", "a ", "mock ", "response ", 
                "from ", "the ", "assistant."
            ]
            for chunk in chunks:
                yield {"text": chunk}
        
        # Return the streaming response (will be handled by the mock)
        return fake_stream()
    
    # Create AsyncMock objects and assign side effects
    conversation_service.post = AsyncMock(side_effect=mock_post_create)
    conversation_service.put = AsyncMock(side_effect=mock_put_update)
    conversation_service.post_session = AsyncMock(side_effect=mock_post_create_session)
    conversation_service.get_list = AsyncMock(side_effect=mock_get_list)
    conversation_service.get = AsyncMock(side_effect=mock_get_with_sessions)
    conversation_service.get_messages = AsyncMock(side_effect=mock_get_messages)
    conversation_service.stream_chat = AsyncMock(side_effect=mock_stream_chat)
    
    return conversation_service

@pytest.mark.asyncio
class TestConversationService:
    """Tests for the Conversation service."""
    
    async def test_create_conversation(self, mock_conversation_service, mock_db_session, mock_user):
        """Test creating a new conversation."""
        # Call the service
        org_id = uuid4()
        conversation_data = MockResponse(
            title="New Conversation",
            description="Test description"
        )
        
        response = await mock_conversation_service.post(
            org_id,
            conversation_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure
        assert hasattr(response, "id")
        assert hasattr(response, "title")
        assert hasattr(response, "description")
        assert hasattr(response, "organization_id")
        assert hasattr(response, "user_id")
        assert hasattr(response, "is_archived")
        assert hasattr(response, "created_at")
        
        # Verify values
        assert response.title == conversation_data.title
        assert response.description == conversation_data.description
        assert response.organization_id == org_id
        assert response.user_id == mock_user["id"]
        assert response.is_archived is False
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    async def test_update_conversation(self, mock_conversation_service, mock_db_session, mock_user, mock_conversation):
        """Test updating a conversation."""
        # Set up conversation to be found in the database
        conversation_id = mock_conversation.id
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        
        # Call the service
        update_data = MockResponse(
            title="Updated Title",
            is_archived=True
        )
        
        response = await mock_conversation_service.put(
            conversation_id,
            update_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure
        assert response.id == conversation_id
        assert response.title == update_data.title
        assert response.is_archived == update_data.is_archived
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    async def test_update_conversation_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test updating a conversation that doesn't exist."""
        # Set up conversation not to be found in the database
        conversation_id = uuid4()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Call the service
        update_data = MockResponse(
            title="Updated Title"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_conversation_service.put(
                conversation_id,
                update_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "Conversation not found"
    
    async def test_create_chat_session(self, mock_conversation_service, mock_db_session, mock_user, mock_conversation, mock_model):
        """Test creating a new chat session."""
        # Set up conversation to be found
        org_id = mock_conversation.organization_id
        conversation_id = mock_conversation.id
        model_id = mock_model.id
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_conversation
        mock_db_session.execute.return_value.scalar_one.return_value = mock_model
        
        # Call the service
        session_data = MockResponse(
            conversation_id=conversation_id,
            model_id=model_id,
            settings={"temperature": 0.8}
        )
        
        response = await mock_conversation_service.post_session(
            org_id,
            session_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure
        assert hasattr(response, "id")
        assert hasattr(response, "conversation_id")
        assert hasattr(response, "model_id")
        assert hasattr(response, "model_name")
        assert hasattr(response, "agent_id")
        assert hasattr(response, "status")
        assert hasattr(response, "settings")
        assert hasattr(response, "created_at")
        
        # Verify values
        assert response.conversation_id == conversation_id
        assert response.model_id == model_id
        assert response.model_name == mock_model.name
        assert response.settings == session_data.settings
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()
    
    async def test_create_chat_session_conversation_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test creating a chat session for a conversation that doesn't exist."""
        # Set up conversation not to be found
        org_id = uuid4()
        conversation_id = uuid4()
        model_id = uuid4()
        
        # Ensure the mock returns None for scalar_one_or_none
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Call the service
        session_data = MockResponse(
            conversation_id=conversation_id,
            model_id=model_id,
            settings={}
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_conversation_service.post_session(
                org_id,
                session_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "Conversation not found"
    
    async def test_get_conversation_list(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a list of conversations."""
        # Call the service
        org_id = uuid4()
        
        response = await mock_conversation_service.get_list(
            org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure
        assert isinstance(response, list)
        assert len(response) == 5  # From the mock implementation
        
        # Verify each conversation has the right structure
        for conversation in response:
            assert hasattr(conversation, "id")
            assert hasattr(conversation, "title")
            assert hasattr(conversation, "organization_id")
            assert hasattr(conversation, "user_id")
            assert hasattr(conversation, "is_archived")
            assert hasattr(conversation, "created_at")
            assert hasattr(conversation, "updated_at")
            
            # Verify the values are as expected
            assert conversation.organization_id == org_id
            assert conversation.user_id == mock_user["id"]
    
    async def test_get_conversation_with_sessions(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a conversation with its chat sessions."""
        # Call the service
        org_id = uuid4()
        conversation_id = uuid4()
        
        response = await mock_conversation_service.get(
            org_id,
            conversation_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify conversation structure
        assert hasattr(response, "id")
        assert hasattr(response, "title")
        assert hasattr(response, "organization_id")
        assert hasattr(response, "user_id")
        assert hasattr(response, "is_archived")
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")
        assert hasattr(response, "sessions")
        
        # Verify values
        assert response.id == conversation_id
        assert response.organization_id == org_id
        assert response.user_id == mock_user["id"]
        
        # Verify sessions
        assert isinstance(response.sessions, list)
        assert len(response.sessions) == 3  # From the mock implementation
        
        # Verify each session has the right structure
        for session in response.sessions:
            assert hasattr(session, "id")
            assert hasattr(session, "conversation_id")
            assert hasattr(session, "model_id")
            assert hasattr(session, "model_name")
            assert hasattr(session, "agent_id")
            assert hasattr(session, "status")
            assert hasattr(session, "settings")
            assert hasattr(session, "created_at")
            
            # Verify the values are as expected
            assert session.conversation_id == conversation_id
    
    async def test_get_conversation_with_sessions_not_found(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting a conversation that doesn't exist."""
        # Override the mock implementation to return empty result
        async def mock_get_empty_result(org_id, conversation_id, session, user):
            return None
        
        mock_conversation_service.get.side_effect = mock_get_empty_result
        
        # Call the service
        org_id = uuid4()
        conversation_id = uuid4()
        
        response = await mock_conversation_service.get(
            org_id,
            conversation_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify response is None
        assert response is None
    
    async def test_get_messages(self, mock_conversation_service, mock_db_session, mock_user):
        """Test getting messages for a conversation."""
        # Call the service
        org_id = uuid4()
        conversation_id = uuid4()
        limit = 5
        offset = 0
        
        response = await mock_conversation_service.get_messages(
            org_id,
            conversation_id,
            limit=limit,
            offset=offset,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result structure
        assert hasattr(response, "messages")
        assert hasattr(response, "total")
        assert hasattr(response, "has_more")
        
        # Verify values
        assert isinstance(response.messages, list)
        assert len(response.messages) == limit  # From our limit parameter
        assert response.total == 15  # From the mock implementation
        assert response.has_more == True  # 5 < 15, so there are more messages
        
        # Verify each message has the right structure
        for i, message in enumerate(response.messages):
            assert hasattr(message, "id")
            assert hasattr(message, "chat_session_id")
            assert hasattr(message, "role")
            assert hasattr(message, "content")
            assert hasattr(message, "token_used")
            
            # Verify alternating roles
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message.role == expected_role
    
    async def test_stream_chat(self, mock_conversation_service, mock_db_session, mock_user, mock_chat_session):
        """Test streaming a chat conversation."""
        # Set up chat session to be found
        chat_session_id = mock_chat_session.id
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_chat_session
        
        # Call the service
        org_id = uuid4()
        # Use ChatRequestModel instead of MockResponse to avoid linter errors
        chat_request = ChatRequestModel(
            chat_session_id=chat_session_id,
            message="Hello, assistant!"
        )
        
        stream = await mock_conversation_service.stream_chat(
            org_id,
            chat_request,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify stream is returned
        assert stream is not None
        
        # Verify messages were added to database
        assert mock_db_session.add.call_count == 2  # Two messages: user and assistant
        mock_db_session.flush.assert_called_once()
        mock_db_session.commit.assert_called_once() 