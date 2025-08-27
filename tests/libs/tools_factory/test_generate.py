import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.libs.tools_factory.v1.generate import ToolGenerator


class TestToolGenerator:
  """Tests for the ToolGenerator class."""

  @pytest.fixture
  def mock_agent(self):
    """Mock the Agent class."""
    with patch("src.libs.tools_factory.v1.generate.Agent") as mock_agent:
      # Create a mock agent instance
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock()
      mock_agent_instance.arun.return_value.content = """```python
from agno.tools.toolkit import Toolkit
import aiohttp

class WeatherToolkit(Toolkit):
  \"\"\"Tool to extract weather of a specific location\"\"\"
  def __init__(self, api_key: str):
    super().__init__(name="weather")
    self.api_key = api_key
    self.register(self.run)

  async def run(self, location: str, units: str = "metric") -> str:
    \"\"\"Get the current weather for a location
    Args:
        location (str): The city name or ZIP code
        units (str): The unit system (metric or imperial)
    Returns:
        str: A string with the weather information
    \"\"\"
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "units": units, "appid": self.api_key}

    async with aiohttp.ClientSession() as session:
      async with session.get(url, params=params) as response:
        if response.status != 200:
          return f"Error: Could not get weather data (Status {response.status})"

        data = await response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        city = data["name"]

        return f"Weather in {city}: {desc}, Temperature: {temp}°{"C" if units == "metric" else "F"}"
```"""
      mock_agent.return_value = mock_agent_instance
      yield mock_agent

  @pytest.fixture
  def mock_settings(self):
    """Mock the settings module."""
    with patch("src.libs.tools_factory.v1.generate.settings") as mock_settings:
      mock_settings.anthropic_api_key = "test-api-key"
      yield mock_settings

  @pytest.fixture
  def tool_json(self):
    """Sample tool JSON for testing."""
    return {
      "info": {"name": "Weather", "version": "1", "description": "Tool to extract weather of a specific location"},
      "function_info": {
        "name": "run",
        "is_async": True,
        "description": "Get the current weather for a location",
        "code": (
          "import aiohttp\n\n"
          'async def run(self, location: str, units: str = "metric") -> str:\n'
          '  """Get the current weather for a location\n\n'
          "  Args:\n"
          "      location (str): The city name or ZIP code\n"
          "      units (str): The unit system (metric or imperial)\n\n"
          "  Returns:\n"
          "      str: A string with the weather information\n"
          '  """\n'
          '  url = "https://api.openweathermap.org/data/2.5/weather"\n'
          '  params = {"q": location, "units": units, "appid": self.api_key}\n\n'
          "  async with aiohttp.ClientSession() as session:\n"
          "    async with session.get(url, params=params) as response:\n"
          "      if response.status != 200:\n"
          '        return f"Error: Could not get weather data (Status {response.status})"\n\n'
          "      data = await response.json()\n"
          '      temp = data["main"]["temp"]\n'
          '      desc = data["weather"][0]["description"]\n'
          '      city = data["name"]\n\n'
          '      return f"Weather in {city}: {desc}, Temperature: {temp}°{"C" if units == "metric" else "F"}"'
        ),
      },
      "configuration": [{"name": "api_key", "type": "str", "description": "The API key for the weather service", "required": True, "default": None}],
    }

  @pytest.mark.asyncio
  async def test_generate_toolkit_from_json(self, mock_agent, mock_settings, tool_json):
    """Test generating a toolkit from JSON."""
    # Create an instance of ToolGenerator
    generator = ToolGenerator()

    # Generate toolkit code
    toolkit_code = await generator.generate_toolkit_from_json(tool_json)

    # Verify that the Agent was instantiated correctly
    mock_agent.assert_called_once()

    # Verify that arun was called with the expected prompt
    mock_agent.return_value.arun.assert_called_once()
    prompt_arg = mock_agent.return_value.arun.call_args[0][0]

    # Check that the prompt contains key elements from the tool JSON
    assert "Weather" in prompt_arg
    assert "Tool to extract weather of a specific location" in prompt_arg
    assert "api_key" in prompt_arg

    # Verify that the generated code is as expected
    assert "class WeatherToolkit(Toolkit):" in toolkit_code
    assert "def __init__(self, api_key: str):" in toolkit_code
    assert 'async def run(self, location: str, units: str = "metric") -> str:' in toolkit_code
    assert "Weather in {city}: {desc}, Temperature: {temp}" in toolkit_code

  @pytest.mark.asyncio
  async def test_create_prompt(self, mock_agent, mock_settings, tool_json):
    """Test creating a prompt for the agent."""
    # Create an instance of ToolGenerator
    generator = ToolGenerator()

    # Create a prompt
    prompt = generator._create_prompt(tool_json)

    # Verify that the prompt contains key elements
    assert "Generate an Agno toolkit class" in prompt
    assert "JSON Input:" in prompt
    assert "Weather" in prompt
    assert "Tool to extract weather of a specific location" in prompt
    assert "api_key" in prompt
    assert "The toolkit class should extend the `Toolkit` class from Agno" in prompt

  @pytest.mark.asyncio
  async def test_generate_toolkit_handles_agent_errors(self, mock_settings):
    """Test that errors from the agent are properly handled."""
    # Create a mock agent that raises an exception
    with patch("src.libs.tools_factory.v1.generate.Agent") as mock_agent:
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock(side_effect=Exception("Agent error"))
      mock_agent.return_value = mock_agent_instance

      # Create an instance of ToolGenerator
      generator = ToolGenerator()

      # Try to generate toolkit code, which should raise an exception
      with pytest.raises(Exception) as excinfo:
        await generator.generate_toolkit_from_json({})

      assert "Agent error" in str(excinfo.value)

  @pytest.mark.asyncio
  async def test_generate_toolkit_with_different_model(self, mock_settings):
    """Test generating a toolkit with a non-default model."""
    # Create a mock for the Claude class
    with patch("src.libs.tools_factory.v1.generate.Claude") as mock_claude, patch("src.libs.tools_factory.v1.generate.Agent") as mock_agent:
      # Set up the mock Claude instance
      mock_claude_instance = MagicMock()
      mock_claude.return_value = mock_claude_instance

      # Set up the mock Agent instance
      mock_agent_instance = MagicMock()
      mock_agent_instance.arun = AsyncMock()
      mock_agent_instance.arun.return_value.content = "Generated toolkit code"
      mock_agent.return_value = mock_agent_instance

      # Create an instance of ToolGenerator
      generator = ToolGenerator()

      # Generate toolkit code
      await generator.generate_toolkit_from_json({})

      # Verify that Claude was instantiated with the correct parameters
      mock_claude.assert_called_once_with(id="claude-3-7-sonnet-latest", api_key=mock_settings.anthropic_api_key)
