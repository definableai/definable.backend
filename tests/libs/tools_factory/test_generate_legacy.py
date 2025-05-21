import pytest
from unittest.mock import MagicMock, patch

from src.libs.tools_factory.v1.generate_legacy import ToolGenerator


class TestGenerateLegacy:
    """Tests for the legacy toolkit generation functions."""

    @pytest.fixture
    def tool_generator(self):
        """Create a ToolGenerator instance."""
        return ToolGenerator()

    @pytest.fixture
    def tool_json(self):
        """Sample tool JSON for testing."""
        return {
            "info": {
                "name": "Weather",
                "version": "1",
                "description": "Tool to extract weather of a specific location"
            },
            "function_info": {
                "name": "run",
                "is_async": True,
                "description": "Get the current weather for a location",
                "code": (
                    'import aiohttp\n\n'
                    'async def run(self, location: str, units: str = "metric") -> str:\n'
                    '  """Get the current weather for a location\n\n'
                    '  Args:\n'
                    '      location (str): The city name or ZIP code\n'
                    '      units (str): The unit system (metric or imperial)\n\n'
                    '  Returns:\n'
                    '      str: A string with the weather information\n'
                    '  """\n'
                    '  url = "https://api.openweathermap.org/data/2.5/weather"\n'
                    '  params = {"q": location, "units": units, "appid": self.api_key}\n\n'
                    '  async with aiohttp.ClientSession() as session:\n'
                    '    async with session.get(url, params=params) as response:\n'
                    '      if response.status != 200:\n'
                    '        return f"Error: Could not get weather data (Status {response.status})"\n\n'
                    '      data = await response.json()\n'
                    '      temp = data["main"]["temp"]\n'
                    '      desc = data["weather"][0]["description"]\n'
                    '      city = data["name"]\n\n'
                    '      return f"Weather in {city}: {desc}, Temperature: {temp}°{"C" if units == "metric" else "F"}"'
                )
            },
            "configuration": [
                {
                    "name": "api_key",
                    "type": "str",
                    "description": "The API key for the weather service",
                    "required": True,
                    "default": None
                }
            ]
        }

    def test_generate_toolkit(self, tool_generator, tool_json):
        """Test generating a toolkit from JSON."""
        # Generate toolkit code
        response = tool_generator.generate_toolkit(tool_json)

        # Verify the generated code
        assert "class WeatherToolkit(Toolkit):" in response.toolkit_code
        assert "def __init__(self, api_key: str):" in response.toolkit_code
        assert "self.api_key = api_key" in response.toolkit_code
        assert "async def run" in response.toolkit_code
        assert "Weather in {city}: {desc}, Temperature: {temp}" in response.toolkit_code

        # Check the class name
        assert response.class_name == "WeatherToolkit"

    def test_generate_toolkit_with_custom_class_name(self, tool_generator):
        """Test generating a toolkit with a custom class name."""
        json_data = {
            "info": {
                "name": "CustomName",
                "version": "1",
                "description": "A custom tool"
            },
            "function_info": {
                "name": "run",
                "is_async": False,
                "description": "Run the tool",
                "code": "def run(self): return 'Hello'"
            }
        }

        response = tool_generator.generate_toolkit(json_data)
        assert response.class_name == "CustomnameToolkit"
        assert "class CustomnameToolkit(Toolkit):" in response.toolkit_code

    def test_generate_toolkit_with_configuration(self, tool_generator):
        """Test generating a toolkit with multiple configuration parameters."""
        json_data = {
            "info": {
                "name": "ConfigTool",
                "version": "1",
                "description": "A tool with configuration"
            },
            "function_info": {
                "name": "run",
                "is_async": False,
                "description": "Run the tool",
                "code": "def run(self): return 'Hello'"
            },
            "configuration": [
                {"name": "api_key", "type": "str", "description": "API key"},
                {"name": "timeout", "type": "int", "description": "Timeout", "default": 30},
                {"name": "debug", "type": "bool", "description": "Debug mode", "default": False}
            ]
        }

        response = tool_generator.generate_toolkit(json_data)
        toolkit_code = response.toolkit_code

        # Check init parameters
        assert "def __init__(self, api_key: str, timeout: int = 30, debug: bool = False):" in toolkit_code

        # Check parameter storage
        assert "self.api_key = api_key" in toolkit_code
        assert "self.timeout = timeout" in toolkit_code
        assert "self.debug = debug" in toolkit_code

    @patch('os.makedirs')
    def test_initialization(self, mock_makedirs):
        """Test ToolGenerator initialization."""
        generator = ToolGenerator("custom_tools_dir")
        mock_makedirs.assert_called_once_with("custom_tools_dir", exist_ok=True)
        assert generator.tools_dir == "custom_tools_dir"

        # Test with default directory
        generator = ToolGenerator()
        assert generator.tools_dir == "tools"

    @patch('json.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_create_toolkit_from_json(self, mock_open, mock_json_load, tool_generator, tool_json):
        """Test creating a toolkit from a JSON file."""
        # Setup mocks
        mock_json_load.return_value = tool_json
        tool_generator.generate_toolkit = MagicMock()
        tool_generator.load_toolkit = MagicMock()

        # Test the method
        tool_generator.create_toolkit_from_json("test.json", api_key="test_key")

        # Verify calls
        mock_open.assert_called_once_with("test.json", "r")
        mock_json_load.assert_called_once()
        tool_generator.generate_toolkit.assert_called_once_with(tool_json)
        tool_generator.load_toolkit.assert_called_once()