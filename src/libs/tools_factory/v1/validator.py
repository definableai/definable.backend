from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolInfo(BaseModel):
  """Basic information about the tool"""

  name: str
  description: str
  version: str


class ParameterBase(BaseModel):
  """Base class for parameters (configuration and inputs)"""

  name: str
  type: str
  description: str
  required: bool = True
  default: Optional[Any] = None
  enum: Optional[List[Any]] = None

  @field_validator("name")
  @classmethod
  def validate_name(cls, v):
    if not v.isidentifier():
      raise ValueError(f"Name '{v}' must be a valid Python identifier")
    return v


class ConfigParameter(ParameterBase):
  """Configuration parameter for tool initialization"""

  pass


class InputParameter(ParameterBase):
  """Input parameter for the tool function"""

  pass


class OutputSpec(BaseModel):
  """Output specification for the tool function"""

  type: str
  description: str


class FunctionInfo(BaseModel):
  """Information about the function implementation"""

  name: str = Field(..., description="Function name (must be a valid Python identifier)")
  is_async: bool = Field(default=False, description="Whether the function is async")
  description: str = Field(..., description="Description of what the function does")
  code: str = Field(..., description="Python code for the function implementation")

  @field_validator("name")
  @classmethod
  def validate_function_name(cls, v):
    if not v.isidentifier():
      raise ValueError(f"Function name '{v}' must be a valid Python identifier")
    return v

  @field_validator("code")
  @classmethod
  def validate_code(cls, v):
    if not v or not isinstance(v, str):
      raise ValueError("Code must be a non-empty string")
    return v


class DeploymentConfig(BaseModel):
  """Deployment configuration"""

  framework: Literal["agno", "langchain"] = "agno"
  toolkit_class: bool = True
  standalone_function: bool = False


class ToolSchema(BaseModel):
  """Full schema for a tool definition"""

  info: ToolInfo
  configuration: List[ConfigParameter] = Field(default_factory=list)
  inputs: List[InputParameter]
  output: OutputSpec
  function_info: FunctionInfo
  requirements: List[str] = Field(default_factory=list)
  deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

  @model_validator(mode="after")
  def validate_schema_consistency(self):
    # Check for required params with default values
    for param in self.inputs + self.configuration:
      if param.required and param.default is not None:
        raise ValueError(f"Parameter '{param.name}' is marked as required but has a default value")

    # More validations could be added here
    return self

  model_config = {
    "json_schema_extra": {
      "examples": [
        {
          "info": {"name": "Weather Tool", "description": "Tool for fetching weather information", "version": "1.0.0"},
          "configuration": [{"name": "api_key", "type": "str", "description": "API key for weather service", "required": True, "default": None}],
          "inputs": [{"name": "location", "type": "str", "description": "City name or zip code", "required": True}],
          "output": {"type": "str", "description": "Weather information"},
          "function_info": {
            "name": "get_weather",
            "is_async": True,
            "description": "Get weather for location",
            "code": "# Weather fetching code here",
          },
          "requirements": ["aiohttp>=3.8.0"],
        }
      ]
    }
  }


def validate_tool_json(json_data: Dict[str, Any]) -> ToolSchema:
  """
  Validate the tool JSON data and return a ToolSchema object.

  Args:
      json_data: The JSON data to validate

  Returns:
      ToolSchema: A validated Pydantic model
  """
  return ToolSchema.model_validate(json_data)


def is_valid_tool_json(json_data: Dict[str, Any]) -> bool:
  """
  Check if the tool JSON data is valid.

  Args:
      json_data: The JSON data to validate

  Returns:
      bool: True if valid, False otherwise
  """
  try:
    validate_tool_json(json_data)
    return True
  except Exception:
    return False


# Example usage:
if __name__ == "__main__":
  import json

  # Load a tool JSON file
  with open("/Users/hash/work/neuron-square/zyeta.backend/src/tests/tools.json", "r") as f:
    tool_data = json.load(f)

  try:
    # Validate the tool data
    validated_tool = validate_tool_json(tool_data)
    print("✅ Tool schema is valid!")

    # You can access the validated data
    print(f"Tool name: {validated_tool.info.name}")
    print(f"Function: {validated_tool.function_info.name}")

    # Get JSON schema
    print("\nJSON Schema:")
    schema = ToolSchema.model_json_schema()
    print(json.dumps(schema, indent=2))

  except Exception as e:
    print(f"❌ Invalid tool schema: {e}")
