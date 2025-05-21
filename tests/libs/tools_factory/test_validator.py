import pytest
from pydantic import ValidationError

from src.libs.tools_factory.v1.validator import (
    ToolInfo,
    ConfigParameter,
    InputParameter,
    OutputSpec,
    FunctionInfo,
    DeploymentConfig,
    ToolSchema,
    validate_tool_json,
    is_valid_tool_json
)


class TestValidator:
    """Tests for the validator module."""

    def test_toolinfo_validation_success(self):
        """Test successful validation of ToolInfo."""
        info = ToolInfo(name="TestTool", description="A test tool", version="1.0.0")
        assert info.name == "TestTool"
        assert info.description == "A test tool"
        assert info.version == "1.0.0"

    def test_toolinfo_validation_missing_fields(self):
        """Test validation fails when required fields are missing."""
        with pytest.raises(ValidationError):
            ToolInfo(name="TestTool", description="A test tool")  # Missing version

        with pytest.raises(ValidationError):
            ToolInfo(name="TestTool", version="1.0.0")  # Missing description

        with pytest.raises(ValidationError):
            ToolInfo(description="A test tool", version="1.0.0")  # Missing name

    def test_config_parameter_validation_success(self):
        """Test successful validation of ConfigParameter."""
        param = ConfigParameter(
            name="api_key",
            type="str",
            description="API key for service",
            required=True,
            default=None
        )
        assert param.name == "api_key"
        assert param.type == "str"
        assert param.description == "API key for service"
        assert param.required is True
        assert param.default is None

    def test_config_parameter_invalid_name(self):
        """Test validation fails with invalid parameter name."""
        with pytest.raises(ValidationError) as excinfo:
            ConfigParameter(
                name="invalid-name",  # Invalid: contains hyphen
                type="str",
                description="Test parameter",
                required=True
            )
        assert "must be a valid Python identifier" in str(excinfo.value)

    def test_input_parameter_validation(self):
        """Test validation of InputParameter."""
        param = InputParameter(
            name="location",
            type="str",
            description="Location to check",
            required=True,
            enum=["New York", "Tokyo", "London"]
        )
        assert param.name == "location"
        assert param.enum == ["New York", "Tokyo", "London"]

    def test_output_spec_validation(self):
        """Test validation of OutputSpec."""
        output = OutputSpec(type="str", description="Weather information")
        assert output.type == "str"
        assert output.description == "Weather information"

    def test_function_info_validation_success(self):
        """Test successful validation of FunctionInfo."""
        func_info = FunctionInfo(
            name="get_weather",
            is_async=True,
            description="Get weather information",
            code="async def get_weather(location): return f'Weather in {location}'"
        )
        assert func_info.name == "get_weather"
        assert func_info.is_async is True
        assert func_info.description == "Get weather information"
        assert "async def get_weather" in func_info.code

    def test_function_info_invalid_name(self):
        """Test validation fails with invalid function name."""
        with pytest.raises(ValidationError) as excinfo:
            FunctionInfo(
                name="invalid-function",  # Invalid: contains hyphen
                description="Test function",
                code="def test(): pass"
            )
        assert "must be a valid Python identifier" in str(excinfo.value)

    def test_function_info_empty_code(self):
        """Test validation fails with empty code."""
        with pytest.raises(ValidationError) as excinfo:
            FunctionInfo(
                name="test_func",
                description="Test function",
                code=""  # Empty code
            )
        assert "Code must be a non-empty string" in str(excinfo.value)

    def test_deployment_config_validation(self):
        """Test validation of DeploymentConfig."""
        config = DeploymentConfig(framework="agno", toolkit_class=True, standalone_function=False)
        assert config.framework == "agno"
        assert config.toolkit_class is True
        assert config.standalone_function is False

        # Test invalid framework
        with pytest.raises(ValidationError):
            DeploymentConfig(framework="invalid_framework")

    def test_tool_schema_success(self):
        """Test successful validation of complete ToolSchema."""
        schema = ToolSchema(
            info=ToolInfo(name="WeatherTool", description="Weather tool", version="1.0"),
            configuration=[
                ConfigParameter(name="api_key", type="str", description="API key", required=True)
            ],
            inputs=[
                InputParameter(name="location", type="str", description="Location", required=True)
            ],
            output=OutputSpec(type="str", description="Weather information"),
            function_info=FunctionInfo(
                name="get_weather",
                is_async=True,
                description="Get weather info",
                code="async def get_weather(location): return f'Weather in {location}'"
            ),
            requirements=["aiohttp>=3.8.0"]
        )
        assert schema.info.name == "WeatherTool"
        assert len(schema.configuration) == 1
        assert len(schema.inputs) == 1
        assert schema.output.type == "str"
        assert schema.function_info.name == "get_weather"
        assert len(schema.requirements) == 1

    def test_tool_schema_validation_consistency(self):
        """Test schema consistency validation."""
        # Required param with default value should fail
        with pytest.raises(ValidationError) as excinfo:
            ToolSchema(
                info=ToolInfo(name="TestTool", description="Test tool", version="1.0"),
                configuration=[
                    ConfigParameter(
                        name="api_key",
                        type="str",
                        description="API key",
                        required=True,
                        default="default_key"  # Default value for required param
                    )
                ],
                inputs=[
                    InputParameter(name="input", type="str", description="Input", required=True)
                ],
                output=OutputSpec(type="str", description="Output"),
                function_info=FunctionInfo(
                    name="test_func",
                    description="Test function",
                    code="def test_func(): pass"
                )
            )
        assert "is marked as required but has a default value" in str(excinfo.value)

    def test_validate_tool_json_success(self):
        """Test successful validation of tool JSON."""
        valid_json = {
            "info": {"name": "TestTool", "description": "Test tool", "version": "1.0"},
            "configuration": [
                {"name": "api_key", "type": "str", "description": "API key", "required": True}
            ],
            "inputs": [
                {"name": "input", "type": "str", "description": "Input param", "required": True}
            ],
            "output": {"type": "str", "description": "Output description"},
            "function_info": {
                "name": "test_func",
                "description": "Test function",
                "code": "def test_func(): pass"
            }
        }

        validated = validate_tool_json(valid_json)
        assert validated.info.name == "TestTool"
        assert len(validated.configuration) == 1
        assert validated.configuration[0].name == "api_key"

    def test_validate_tool_json_failure(self):
        """Test validation failure of tool JSON."""
        invalid_json = {
            "info": {"name": "TestTool", "description": "Test tool", "version": "1.0"},
            # Missing required fields
            "inputs": [],
            "function_info": {
                "name": "test_func",
                "description": "Test function",
                "code": "def test_func(): pass"
            }
        }

        with pytest.raises(ValidationError):
            validate_tool_json(invalid_json)

    def test_is_valid_tool_json(self):
        """Test is_valid_tool_json function."""
        valid_json = {
            "info": {"name": "TestTool", "description": "Test tool", "version": "1.0"},
            "configuration": [],
            "inputs": [
                {"name": "input", "type": "str", "description": "Input param", "required": True}
            ],
            "output": {"type": "str", "description": "Output description"},
            "function_info": {
                "name": "test_func",
                "description": "Test function",
                "code": "def test_func(): pass"
            }
        }

        assert is_valid_tool_json(valid_json) is True

        invalid_json = {
            "info": {"name": "TestTool", "description": "Test tool"},  # Missing version
            "configuration": [],
            "inputs": [],
            "function_info": {}
        }

        assert is_valid_tool_json(invalid_json) is False