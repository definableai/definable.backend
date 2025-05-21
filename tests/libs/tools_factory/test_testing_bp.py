import pytest
from typing import Dict, Any, List

from src.libs.tools_factory.v1.testing_bp import parse_configuration, generate_boilerplate


class TestTestingBp:
    """Tests for the testing_bp module."""
    
    def test_parse_configuration_empty(self):
        """Test parsing empty configuration."""
        config_items = []
        result = parse_configuration(config_items)
        assert result == {}
    
    def test_parse_configuration_with_values(self):
        """Test parsing configuration with values."""
        config_items = [
            {"name": "api_key", "value": "test_key", "type": "str", "description": "API key"},
            {"name": "timeout", "value": 30, "type": "int", "description": "Timeout in seconds"}
        ]
        result = parse_configuration(config_items)
        assert result == {"api_key": "test_key", "timeout": 30}
    
    def test_parse_configuration_missing_values(self):
        """Test parsing configuration with missing values."""
        config_items = [
            {"name": "api_key", "type": "str", "description": "API key"},  # No value
            {"name": "timeout", "value": 30, "type": "int", "description": "Timeout in seconds"}
        ]
        result = parse_configuration(config_items)
        assert result == {"timeout": 30}
    
    def test_parse_configuration_missing_names(self):
        """Test parsing configuration with missing names."""
        config_items = [
            {"type": "str", "value": "test_key", "description": "API key"},  # No name
            {"name": "timeout", "value": 30, "type": "int", "description": "Timeout in seconds"}
        ]
        result = parse_configuration(config_items)
        assert result == {"timeout": 30}
    
    def test_generate_boilerplate_openai(self):
        """Test generating boilerplate code for OpenAI."""
        class_name = "WeatherTool"
        input_prompt = "What's the weather like in Paris?"
        model_name = "gpt-4o"
        provider = "openai"
        api_key = "test_api_key"
        module_name = "weather_tool"
        config_items = [
            {"name": "api_key", "value": "weather_api_key", "type": "str"}
        ]
        instructions = "You are a helpful weather assistant."
        
        boilerplate = generate_boilerplate(
            class_name=class_name,
            input_prompt=input_prompt,
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            module_name=module_name,
            config_items=config_items,
            instructions=instructions
        )
        
        # Check that the generated code contains the expected elements
        assert "from weather_tool import WeatherTool" in boilerplate
        assert "model=OpenAIChat" in boilerplate
        assert "id=\"gpt-4o\"" in boilerplate
        assert "api_key=\"test_api_key\"" in boilerplate
        assert "tools=[WeatherTool(**{'api_key': 'weather_api_key'})]" in boilerplate
        assert "instructions=\"\"\"You are a helpful weather assistant.\"\"\"" in boilerplate
        assert "await async_agent.arun(\"What's the weather like in Paris?\")" in boilerplate
    
    def test_generate_boilerplate_anthropic(self):
        """Test generating boilerplate code for Anthropic."""
        class_name = "TranslationTool"
        input_prompt = "Translate 'Hello world' to French."
        model_name = "claude-3-7-sonnet-latest"
        provider = "anthropic"
        api_key = "test_anthropic_key"
        module_name = "translation_tool"
        
        boilerplate = generate_boilerplate(
            class_name=class_name,
            input_prompt=input_prompt,
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            module_name=module_name
        )
        
        # Check that the generated code contains the expected elements
        assert "from translation_tool import TranslationTool" in boilerplate
        assert "model=Claude" in boilerplate
        assert "id=\"claude-3-7-sonnet-latest\"" in boilerplate
        assert "api_key=\"test_anthropic_key\"" in boilerplate
        assert "tools=[TranslationTool(**{})]" in boilerplate
        assert "await async_agent.arun(\"Translate 'Hello world' to French.\")" in boilerplate
    
    def test_generate_boilerplate_invalid_provider(self):
        """Test generating boilerplate with invalid provider."""
        with pytest.raises(ValueError) as excinfo:
            generate_boilerplate(
                class_name="TestTool",
                input_prompt="Test prompt",
                model_name="test-model",
                provider="invalid_provider",  # Invalid provider
                api_key="test_key"
            )
        assert "Invalid provider: invalid_provider" in str(excinfo.value)
    
    def test_generate_boilerplate_with_complex_config(self):
        """Test generating boilerplate with complex configuration."""
        config_items = [
            {"name": "api_key", "value": "test_key", "type": "str"},
            {"name": "timeout", "value": 30, "type": "int"},
            {"name": "features", "value": ["search", "translate"], "type": "list"}
        ]
        
        boilerplate = generate_boilerplate(
            class_name="ComplexTool",
            input_prompt="Test complex tool",
            model_name="gpt-4o",
            provider="openai",
            api_key="test_api_key",
            config_items=config_items
        )
        
        # Check that the configuration is properly formatted
        expected_config = "{'api_key': 'test_key', 'timeout': 30, 'features': ['search', 'translate']}"
        assert f"tools=[ComplexTool(**{expected_config})]" in boilerplate 