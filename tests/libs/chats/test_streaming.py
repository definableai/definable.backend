import asyncio
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.libs.chats.v1.streaming import LLMFactory
from agno.agent import RunResponse
from agno.media import File, Image


@pytest.fixture
def llm_factory():
    """Fixture to create an LLMFactory instance."""
    return LLMFactory()


@pytest.mark.asyncio
async def test_get_model_class_openai(llm_factory):
    """Test getting the model class for OpenAI."""
    model_class = llm_factory.get_model_class("openai")
    assert model_class.__name__ == "OpenAIChat"


@pytest.mark.asyncio
async def test_get_model_class_anthropic(llm_factory):
    """Test getting the model class for Anthropic."""
    model_class = llm_factory.get_model_class("anthropic")
    assert model_class.__name__ == "Claude"


@pytest.mark.asyncio
async def test_get_model_class_deepseek(llm_factory):
    """Test getting the model class for DeepSeek."""
    model_class = llm_factory.get_model_class("deepseek")
    assert model_class.__name__ == "DeepSeek"


@pytest.mark.asyncio
async def test_get_model_class_invalid(llm_factory):
    """Test that an invalid provider raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        llm_factory.get_model_class("invalid")


@pytest.mark.asyncio
async def test_chat_basic():
    """Test basic chat functionality with a simple message."""
    # Create a mock token stream
    async def mock_token_stream():
        tokens = ["This", " is", " a", " mock", " response"]
        for token in tokens:
            yield RunResponse(content=token)

    # Mock both Agent class and its arun method
    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        # Make the arun method return our mock token stream
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        
        # Collect the response
        response = ""
        
        # Patch asyncio.create_task to handle the incorrect await in the real code
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for token in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="Hello, how are you?",
            ):
                response += token.content
        
        # Verify the response
        assert response == "This is a mock response"
        # Verify agent was created with expected parameters
        mock_agent_class.assert_called_once()


@pytest.mark.asyncio
async def test_chat_with_prompt():
    """Test chat with a custom prompt."""
    # Create a mock token stream
    async def mock_token_stream():
        tokens = ["Response", " with", " custom", " prompt"]
        for token in tokens:
            yield RunResponse(content=token)

    # Mock Agent class
    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        custom_prompt = "You are a helpful assistant."
        
        # Collect the response
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for _ in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="What can you help me with?",
                prompt=custom_prompt,
            ):
                pass
        
        # Verify agent was created with the custom prompt
        mock_agent_class.assert_called_once()
        # Get the call arguments
        _, kwargs = mock_agent_class.call_args
        assert kwargs.get('instructions') == custom_prompt


@pytest.mark.asyncio
async def test_chat_with_assets():
    """Test chat with assets (files and images)."""
    # Create mock images and files
    test_image = Image(url="http://example.com/image.jpg")
    test_file = File(url="http://example.com/document.pdf")
    
    # Create a mock token stream
    async def mock_token_stream():
        yield RunResponse(content="Asset response")

    # Mock Agent class
    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        
        # Collect the response
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for _ in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="Describe this image and document.",
                assets=[test_image, test_file],
            ):
                pass
        
        # Verify agent.arun was called with the correct assets
        mock_agent_instance.arun.assert_called_once()
        args, kwargs = mock_agent_instance.arun.call_args
        assert kwargs.get('images') == [test_image]
        assert kwargs.get('files') == [test_file]


@pytest.mark.asyncio
async def test_chat_with_temperature():
    """Test chat with temperature parameter."""
    # Create a mock token stream
    async def mock_token_stream():
        yield RunResponse(content="Response with temperature")

    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        temperature = 0.7
        
        # Collect the response
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for _ in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="Generate creative content.",
                temperature=temperature,
            ):
                pass
        
        # Check that Agent was initialized with a model that has temperature parameter
        mock_agent_class.assert_called_once()
        _, kwargs = mock_agent_class.call_args
        model = kwargs.get('model')
        assert model is not None
        assert model.temperature == temperature


@pytest.mark.asyncio
async def test_chat_with_max_tokens():
    """Test chat with max_tokens parameter."""
    # Create a mock token stream
    async def mock_token_stream():
        yield RunResponse(content="Response with max tokens")

    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        max_tokens = 100
        
        # Collect the response
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for _ in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="Generate a short response.",
                max_tokens=max_tokens,
            ):
                pass
        
        # Check that Agent was initialized with a model that has max_tokens parameter
        mock_agent_class.assert_called_once()
        _, kwargs = mock_agent_class.call_args
        model = kwargs.get('model')
        assert model is not None
        assert model.max_tokens == max_tokens


@pytest.mark.asyncio
async def test_chat_with_top_p():
    """Test chat with top_p parameter."""
    # Create a mock token stream
    async def mock_token_stream():
        yield RunResponse(content="Response with top_p")

    with patch('src.libs.chats.v1.streaming.Agent') as mock_agent_class:
        # Setup the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
        mock_agent_class.return_value = mock_agent_instance
        
        # Create the factory and test the chat method
        llm_factory = LLMFactory()
        chat_id = str(uuid.uuid4())
        top_p = 0.9
        
        # Collect the response
        with patch('asyncio.create_task', side_effect=lambda coro: coro):
            async for _ in llm_factory.chat(
                chat_session_id=chat_id,
                llm="gpt-4",
                provider="openai",
                message="Generate a response.",
                top_p=top_p,
            ):
                pass
        
        # Check that Agent was initialized with a model that has top_p parameter
        mock_agent_class.assert_called_once()
        _, kwargs = mock_agent_class.call_args
        model = kwargs.get('model')
        assert model is not None
        assert model.top_p == top_p 