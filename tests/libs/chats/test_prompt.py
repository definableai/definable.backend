from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.libs.chats.v1.prompt import generate_prompts_stream, map_model_to_deepseek


class TestMapModelToDeepseek:
  """Tests for the map_model_to_deepseek function."""

  def test_map_chat_model(self):
    """Test mapping 'chat' model to deepseek-chat."""
    result = map_model_to_deepseek("chat")
    assert result == "deepseek-chat"

  def test_map_reason_model(self):
    """Test mapping 'reason' model to deepseek-reason."""
    result = map_model_to_deepseek("reason")
    assert result == "deepseek-reason"

  def test_map_unknown_model(self):
    """Test mapping an unknown model defaults to deepseek-chat."""
    result = map_model_to_deepseek("unknown-model")
    assert result == "deepseek-chat"


class TestGeneratePromptsStream:
  """Tests for the generate_prompts_stream function."""

  @pytest.mark.asyncio
  async def test_generate_prompts_task_type(self):
    """Test generating a task prompt."""

    # Create mock token generator
    async def mock_token_stream():
      tokens = [
        MagicMock(content="Create"),
        MagicMock(content=" a"),
        MagicMock(content=" function"),
        MagicMock(content=" that"),
        MagicMock(content=" generates"),
        MagicMock(content=" text"),
      ]
      for token in tokens:
        yield token

    # Create a mock for the DeepSeek model and Agent
    with patch("src.libs.chats.v1.prompt.DeepSeek") as mock_deepseek, patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent
      mock_agent_instance = MagicMock()
      # Return our mocked async generator
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Set up the prompt buffer size to control chunk size
      with patch("src.libs.chats.v1.prompt.settings.prompt_buffer_size", 2), patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call the generate_prompts_stream function
        text = "Write code to generate text"
        chunks = []
        async for chunk in generate_prompts_stream(text, prompt_type="task"):
          chunks.append(chunk)

        # We expect chunks of size 2 (buffer_size)
        assert len(chunks) == 3
        assert chunks[0] == "Create a"
        assert chunks[1] == " function that"
        assert chunks[2] == " generates text"

        # Verify DeepSeek was initialized with the correct model ID
        mock_deepseek.assert_called_once()
        assert mock_deepseek.call_args[1]["id"] == "deepseek-chat"

  @pytest.mark.asyncio
  async def test_generate_prompts_creative_type(self):
    """Test generating a creative prompt."""

    # Create mock token generator
    async def mock_token_stream():
      tokens = [
        MagicMock(content="Write"),
        MagicMock(content=" a"),
        MagicMock(content=" story"),
        MagicMock(content=" about"),
        MagicMock(content=" space"),
        MagicMock(content=" exploration"),
      ]
      for token in tokens:
        yield token

    with patch("src.libs.chats.v1.prompt.DeepSeek"), patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Set buffer size to 3 for this test
      with patch("src.libs.chats.v1.prompt.settings.prompt_buffer_size", 3), patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call generate_prompts_stream with creative type
        text = "space exploration"
        chunks = []
        async for chunk in generate_prompts_stream(text, prompt_type="creative"):
          chunks.append(chunk)

        # Verify the system prompt for creative type was used
        call_args = mock_agent_instance.arun.call_args
        user_prompt = call_args[0][0]  # First positional arg
        assert "You are a creative writing assistant" in user_prompt

        # We expect chunks of size 3 (buffer_size)
        assert len(chunks) == 2
        assert chunks[0] == "Write a story"
        assert chunks[1] == " about space exploration"

  @pytest.mark.asyncio
  async def test_generate_prompts_multiple(self):
    """Test generating multiple prompts."""

    # Create mock token generator
    async def mock_token_stream():
      tokens = [
        MagicMock(content="1. What are the ethical implications of AI?\n"),
        MagicMock(content="2. How does AI impact job markets?\n"),
        MagicMock(content="3. Discuss the future of AI regulation."),
      ]
      for token in tokens:
        yield token

    with patch("src.libs.chats.v1.prompt.DeepSeek"), patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Run with patch for asyncio.create_task
      with patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call generate_prompts_stream with num_prompts=3
        text = "artificial intelligence ethics"
        chunks = []
        async for chunk in generate_prompts_stream(text, prompt_type="question", num_prompts=3):
          chunks.append(chunk)

        # Verify num_prompts parameter was used
        call_args = mock_agent_instance.arun.call_args
        user_prompt = call_args[0][0]  # First positional arg
        assert "Generate 3 different" in user_prompt

        # Join all chunks to validate complete output
        complete_output = "".join(chunks)
        assert "1. What are the ethical implications of AI?" in complete_output
        assert "2. How does AI impact job markets?" in complete_output
        assert "3. Discuss the future of AI regulation." in complete_output

  @pytest.mark.asyncio
  async def test_generate_prompts_with_different_model(self):
    """Test generating prompts with a specific model."""

    # Create mock token generator
    async def mock_token_stream():
      tokens = [MagicMock(content="Sample"), MagicMock(content=" output")]
      for token in tokens:
        yield token

    with patch("src.libs.chats.v1.prompt.DeepSeek") as mock_deepseek, patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Run with patch for asyncio.create_task
      with patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call generate_prompts_stream with model="reason"
        text = "test text"
        async for _ in generate_prompts_stream(text, model="reason"):
          pass

        # Verify DeepSeek was initialized with the correct mapped model ID
        mock_deepseek.assert_called_once()
        assert mock_deepseek.call_args[1]["id"] == "deepseek-reason"

  @pytest.mark.asyncio
  async def test_empty_response(self):
    """Test handling an empty response from the model."""

    # Create empty token generator that is an actual async generator
    async def mock_token_stream():
      # Empty generator - yields nothing
      if False:  # This ensures it's an async generator but yields nothing
        yield MagicMock(content="")

    with patch("src.libs.chats.v1.prompt.DeepSeek"), patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent with empty response
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Run with patch for asyncio.create_task
      with patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call generate_prompts_stream
        text = "some input"
        chunks = []
        async for chunk in generate_prompts_stream(text):
          chunks.append(chunk)

        # No chunks should be yielded
        assert len(chunks) == 0

  @pytest.mark.asyncio
  async def test_extension_prompt_included(self):
    """Test that the extension prompt is included in the user prompt."""

    # Create mock token generator
    async def mock_token_stream():
      tokens = [MagicMock(content="Test"), MagicMock(content=" response")]
      for token in tokens:
        yield token

    with patch("src.libs.chats.v1.prompt.DeepSeek"), patch("src.libs.chats.v1.prompt.Agent") as mock_agent_class:
      # Setup mock agent to capture the prompt that was passed
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(return_value=mock_token_stream())
      mock_agent_class.return_value = mock_agent_instance

      # Run with patch for asyncio.create_task
      with patch("asyncio.create_task", side_effect=lambda coro: coro):
        # Call generate_prompts_stream
        text = "test input"
        async for _ in generate_prompts_stream(text):
          pass

        # Check that the extension prompt is included in the user prompt
        call_args = mock_agent_instance.arun.call_args
        user_prompt = call_args[0][0]  # First positional arg
        assert "easy for an AI system to understand" in user_prompt
