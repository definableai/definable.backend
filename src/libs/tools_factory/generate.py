import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

# Template for generating toolkit code
TOOLKIT_TEMPLATE = '''
from agno.tools.toolkit import Toolkit
{imports}

class {class_name}(Toolkit):
  """{description}"""
  def __init__(self{init_params}):
    super().__init__(name="{toolkit_name}")
    {init_body}
    self.register(self.run)

  {function_body}
'''


class ToolGeneratorResponse(BaseModel):
  toolkit_code: str
  class_name: str


class ToolGenerator:
  def __init__(self, tools_dir="tools"):
    self.tools_dir = tools_dir
    os.makedirs(tools_dir, exist_ok=True)

  # TODO: change this incorporate ToolAPI response!
  def generate_toolkit(self, tool_json: Dict[str, Any]) -> ToolGeneratorResponse:
    """Generate a toolkit Python file from the JSON definition."""
    info = tool_json["info"]
    function_info = tool_json["function_info"]

    # Create class name from tool name (CapitalizedWords)
    tool_name = info["name"]
    class_name = "".join(word.capitalize() for word in tool_name.split())
    if not class_name.endswith(("Tool", "Tools", "Toolkit")):
      class_name += "Toolkit"

    # Extract imports and function code
    function_code = function_info["code"]
    imports = []
    function_lines = []

    for line in function_code.split("\n"):
      if line.strip().startswith(("import ", "from ")):
        imports.append(line)
      else:
        function_lines.append(line)

    # Format initialization parameters
    config_params = tool_json.get("configuration", [])
    init_params = ""
    if config_params:
      init_params = ", " + ", ".join(
        f"{param['name']}: {param['type']}" + (f" = {param['default']}" if param.get("default") is not None else "") for param in config_params
      )

    # Format parameter storage in __init__
    init_body = "\n  ".join(f"self.{param['name']} = {param['name']}" for param in config_params)

    # Format function body with proper indentation
    function_body = "\n".join(f"  {line}" for line in function_lines)

    # Generate the code
    toolkit_code = TOOLKIT_TEMPLATE.format(
      imports="\n".join(imports),
      class_name=class_name,
      description=info.get("description", ""),
      init_params=init_params,
      toolkit_name=tool_name.lower().replace(" ", "_"),
      init_body=init_body,
      function_name=function_info["name"],
      function_body=function_body,
    )

    return ToolGeneratorResponse(
      toolkit_code=toolkit_code,
      class_name=class_name,
    )

  def load_toolkit(self, toolkit_path: str, **kwargs):
    """Load a toolkit class from a Python file and instantiate it."""
    # Get the module name from file path
    module_name = Path(toolkit_path).stem

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, toolkit_path)
    if spec is None:
      raise ValueError(f"Failed to load module from {toolkit_path}")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
      raise ValueError(f"Failed to load module from {toolkit_path}")

    spec.loader.exec_module(module)

    # Find the toolkit class in the module
    toolkit_class = None
    for name in dir(module):
      obj = getattr(module, name)
      if isinstance(obj, type) and name.endswith(("Tool", "Tools", "Toolkit")):
        toolkit_class = obj
        break

    if toolkit_class is None:
      raise ValueError(f"No toolkit class found in {toolkit_path}")

    # Instantiate the toolkit
    return toolkit_class(**kwargs)

  def create_toolkit_from_json(self, json_path: str, **kwargs):
    """Generate and load a toolkit from a JSON file."""
    # Load the JSON
    with open(json_path, "r") as f:
      tool_json = json.load(f)

    # Generate the toolkit
    response = self.generate_toolkit(tool_json)

    # Load and return the toolkit
    return self.load_toolkit(response.toolkit_code, **kwargs)
