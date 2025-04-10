import logging
import re
from typing import Any, Dict, List


class FileProcessor:
  """Processes generated files to prepare them for deployment."""

  def __init__(self, logger=None):
    self.logger = logger or logging.getLogger(__name__)

  def process_files(self, files: Dict[str, str], tools: List[Dict[str, Any]], agent_name: str, version: str) -> Dict[str, str]:
    """Process generated files to replace environment variables with actual values and fix imports."""
    self.logger.info("Processing generated files")

    processed_files = {}

    if "agent.py" in files:
      agent_content = files["agent.py"]

      # Process agent.py (fix imports, parameters, etc.)
      agent_content = self._process_agent_file(agent_content, tools, agent_name, version)
      processed_files["agent.py"] = agent_content

    # Process compose.yml - replace environment variables with actual values
    if "compose.yml" in files:
      compose_content = files["compose.yml"]
      compose_content = self._process_compose_file(compose_content, tools)
      processed_files["compose.yml"] = compose_content

    # Process other files as-is
    for file_name, content in files.items():
      if file_name not in processed_files:
        processed_files[file_name] = content

    # Add __init__.py files
    processed_files["__init__.py"] = ""  # Empty file for root directory
    processed_files["tools/__init__.py"] = self._generate_tools_init(tools)

    return processed_files

  def _process_agent_file(self, agent_content: str, tools: List[Dict[str, Any]], agent_name: str, version: str) -> str:
    """Process the agent.py file to fix imports, parameters, etc."""
    # 1. Fix imports
    agent_content = re.sub(r"from \.tools\.([a-zA-Z0-9_]+) import ([a-zA-Z0-9_]+)", r"from .tools import \2", agent_content)

    # 2. Extract tool information
    tool_info = self._extract_tool_info(tools)

    # 3. Fix all imports and class references
    for file_name, info in tool_info.items():
      if "actual_class" not in info:
        continue

      actual_class = info["actual_class"]

      # Common incorrect class name patterns
      incorrect_patterns = [
        f"{actual_class}Toolkit",
        "".join(word.capitalize() for word in file_name.split("_")) + "Toolkit",
        "".join(word.capitalize() for word in file_name.split("_")),
        file_name.replace("_", "") + "Toolkit",
        "".join(word.capitalize() for word in file_name.split("_")),
      ]

      # Fix imports for all possible variations
      for incorrect_class in incorrect_patterns:
        agent_content = re.sub(rf"from \.tools\.{file_name} import {incorrect_class}", f"from .tools import {actual_class}", agent_content)
        # Fix all class instantiations
        agent_content = re.sub(rf"{incorrect_class}\s*\(", f"{actual_class}(", agent_content)

    # 4. Fix parameter handling in tool initializations
    agent_content = self._fix_tool_parameters(agent_content, tool_info)

    # 5. Fix Agent initialization parameters
    param_mappings = {
      "memory_config": "memory",
      "max_tokens": "max_memory_tokens",
      "num_responses": "num_history_runs",
    }

    for incorrect, correct in param_mappings.items():
      agent_content = re.sub(rf"({incorrect})\s*=", f"{correct}=", agent_content)

    # 6. Fix FastAPI integration issues
    if "agent = create_agent()" in agent_content:
      # Replace app.post that references agent.name before creation
      agent_content = re.sub(r'@app\.post\(f"/v1/\{agent_name\}"\)', f'@app.post(f"/{version}/{{agent_name}}")', agent_content)
      # Fix agent.process to agent.run
      agent_content = re.sub(r"agent\.process\(", "agent.run(", agent_content)

    # 7. Add FastAPI interface if missing
    if "from fastapi import" not in agent_content:
      # Import PromptBuilder to access the template
      from .prompt_builder import PromptBuilder

      # Get FastAPI template from PromptBuilder
      fastapi_template = PromptBuilder.FASTAPI_REQUIREMENTS

      # Remove header and formatting, extract just the code part
      code_pattern = r"```python(.*?)```"
      code_matches = re.findall(code_pattern, fastapi_template, re.DOTALL)

      if code_matches:
        fastapi_code = "\n".join(code_matches)
        # Replace template variables
        fastapi_code = fastapi_code.replace("{agent_name}", agent_name)
        fastapi_code = fastapi_code.replace("{version}", version)

        if 'if __name__ == "__main__":' not in agent_content:
          # Add server startup code if not present
          fastapi_code += """

# Run the server when this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
          agent_content += "\n" + fastapi_code

    return agent_content

  def _process_compose_file(self, compose_content: str, tools: List[Dict[str, Any]]) -> str:
    """Process compose.yml to replace environment variables with actual values."""
    for tool in tools:
      # Get the environment variable name
      tool_name = tool["file_name"].replace("_toolkit.py", "").upper().replace(" ", "_")
      env_var = f"${{{tool_name}_API_KEY}}"

      # If the API key is directly provided (not a variable reference), use it
      if (
        "api_key" in tool
        and tool["api_key"]
        and not isinstance(tool["api_key"], str)
        or (isinstance(tool["api_key"], str) and not tool["api_key"].startswith("${"))
      ):
        # Handle two cases in compose.yml:
        # 1. Replace ${TOOL_API_KEY} with the actual value if we have it
        compose_content = compose_content.replace(f"- {tool_name}_API_KEY={env_var}", f"- {tool_name}_API_KEY={tool['api_key']}")
        # 2. Or add the variable if it doesn't exist yet
        if f"- {tool_name}_API_KEY" not in compose_content:
          # Find the environment section to add the key
          env_pattern = r"environment:(\s*- .*)*"
          env_match = re.search(env_pattern, compose_content)
          if env_match:
            env_section = env_match.group(0)
            new_env_section = env_section + f"\n      - {tool_name}_API_KEY={tool['api_key']}"
            compose_content = compose_content.replace(env_section, new_env_section)

    return compose_content

  def _extract_tool_info(self, tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Extract information about tools from their code."""
    tool_info = {}
    for tool in tools:
      tool_code = tool["code"]
      file_name = tool["file_name"].replace(".py", "")

      # Extract actual class name
      class_match = re.search(r"class\s+([A-Za-z0-9_]+)(?:\(|\s*:)", tool_code)
      if class_match:
        actual_class_name = class_match.group(1)

        # Extract __init__ parameters
        init_match = re.search(r"def\s+__init__\s*\(\s*self\s*,\s*(.*?)\)\s*:", tool_code, re.DOTALL)
        if init_match:
          param_str = init_match.group(1).strip()
          # Extract parameter names, ignoring type hints and default values
          param_list = []
          for p in param_str.split(","):
            if not p.strip():
              continue
            param_name = p.split(":")[0].split("=")[0].strip()
            if param_name and param_name != "self":
              param_list.append(param_name)
          accepted_params = param_list
        else:
          accepted_params = ["api_key"]  # Default fallback

        tool_info[file_name] = {"actual_class": actual_class_name, "accepted_params": accepted_params}
        self.logger.info(f"Tool {file_name} uses class {actual_class_name} accepting {accepted_params}")

    return tool_info

  def _fix_tool_parameters(self, agent_content: str, tool_info: Dict[str, Dict[str, Any]]) -> str:
    """Fix parameter handling in tool initializations."""
    for file_name, info in tool_info.items():
      if "actual_class" not in info or "accepted_params" not in info:
        continue

      actual_class = info["actual_class"]
      accepted_params = list(info["accepted_params"])

      # Find all initializations of this class
      for match in re.finditer(rf"{actual_class}\s*\(\s*(.*?)\s*\)", agent_content, re.DOTALL):
        init_text = match.group(0)
        params_text = match.group(1)

        # Handle empty parameters case
        if not params_text.strip():
          continue

        # Parse parameters
        current_params = {}
        for param in params_text.split(","):
          parts = param.strip().split("=", 1)
          if len(parts) == 2:
            name, value = parts
            current_params[name.strip()] = value.strip()

        # Filter to only accepted params and handle None values
        valid_params = {}
        for k, v in current_params.items():
          if k in accepted_params:
            # Handle None values for string parameters
            if v.lower() == "none" and k == "api_key":
              valid_params[k] = '"default-key"'
            else:
              valid_params[k] = v

        # Build new initialization with only valid params
        new_params = ", ".join(f"{k}={v}" for k, v in valid_params.items())
        new_init = f"{actual_class}({new_params})"

        # Replace in content
        agent_content = agent_content.replace(init_text, new_init)

    return agent_content

  def _generate_tools_init(self, tools: List[Dict[str, Any]]) -> str:
    """Generate the contents of tools/__init__.py."""
    tools_init_content = ""
    class_names = []

    for tool in tools:
      tool_code = tool["code"]
      class_match = re.search(r"class\s+([A-Za-z0-9_]+)(?:\(|\s*:)", tool_code)

      if class_match:
        class_name = class_match.group(1)
        file_name = tool["file_name"].replace(".py", "")
        tools_init_content += f"from .{file_name} import {class_name}\n"
        class_names.append(f'"{class_name}"')

    # Add blank line and __all__ list
    if class_names:
      tools_init_content += f"\n__all__ = [{', '.join(class_names)}]\n"
    else:
      tools_init_content += "\n__all__ = []\n"

    return tools_init_content
