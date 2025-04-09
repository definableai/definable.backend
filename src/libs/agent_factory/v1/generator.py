import logging
import os
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.anthropic import Claude

from config.settings import settings

from ...tools_factory.v1 import ToolGenerator
from .deployment import DeploymentManager
from .file_processor import FileProcessor
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser


class AgentGenerator:
  """Generator for dynamic agent creation with tools using Claude LLM via Agno."""

  def __init__(self, logger=None):
    self.logger = logger or logging.getLogger(__name__)
    self.logger.info("Initializing AgentGenerator with Claude via Agno")

    # Initialize components
    self.agent = Agent(model=Claude(id="claude-3-7-sonnet-latest", api_key=settings.anthropic_api_key), show_tool_calls=True, markdown=True)
    self.tool_generator = ToolGenerator()
    self.deployment_manager = DeploymentManager()
    self.prompt_builder = PromptBuilder(logger=self.logger)
    self.response_parser = ResponseParser(logger=self.logger)
    self.file_processor = FileProcessor(logger=self.logger)
    self.logger.info("AgentGenerator initialized successfully")

  async def generate_agent(
    self,
    agent_name: str,
    provider: str,
    model_details: Dict[str, str],
    tools: List[Dict[str, Any]],
    description: str,
    system_prompt: str,
    instructions: str,
    expected_output: Dict[str, Any],
    memory_config: Optional[Dict[str, Any]] = None,
    knowledge_base: Optional[Dict[str, Any]] = None,
    version: str = "v1",
  ) -> Dict[str, Any]:
    """Generate a complete agent with tools, configuration and deployment files."""
    self.logger.info(f"Generating agent: {agent_name}, provider: {provider}, version: {version}")
    self.logger.info(f"Using model: {model_details['name']} ({model_details['id']})")

    # 1. Process and prepare tools
    generated_tools = self._prepare_tools(tools)

    # 2. Create prompt using prompt builder
    prompt = self.prompt_builder.create_prompt(
      agent_name=agent_name,
      version=version,
      provider=provider,
      model_details=model_details,
      description=description,
      system_prompt=system_prompt,
      instructions=instructions,
      expected_output=expected_output,
      memory_config=memory_config,
      knowledge_base=knowledge_base,
      tools=generated_tools,
    )

    # 3. Generate files using Agno agent
    try:
      self.logger.info("Calling Claude via Agno to generate agent code")
      response = await self.agent.arun(prompt, stream=False)
      self.logger.info("Claude response received successfully")

      # 4. Parse response to extract files
      agent_files = self.response_parser.parse_response(response.content)
      self.logger.info(f"Extracted {len(agent_files)} files from Claude response")

      # 5. Process generated files
      processed_files = self.file_processor.process_files(agent_files, generated_tools, agent_name, version)
      self.logger.info("Files processed successfully")

      # 6. Create deployment package
      deployment_info = self.deployment_manager.prepare_deployment(
        agent_name=agent_name,
        agent_code=processed_files.get("agent.py", ""),
        dockerfile=processed_files.get("Dockerfile", ""),
        compose_yml=processed_files.get("compose.yml", ""),
        tools=generated_tools,
        version=version,
        processed_files=processed_files,
      )

      # Add requirements.txt if provided
      if "requirements.txt" in processed_files:
        requirements_path = os.path.join(deployment_info["version_path"], "requirements.txt")
        with open(requirements_path, "w") as f:
          f.write(processed_files["requirements.txt"])
        self.logger.info(f"Added requirements.txt to {requirements_path}")

      self.logger.info(f"Deployment package created: {deployment_info['deployment_path']}")

    except Exception as e:
      self.logger.error(f"Error generating agent: {str(e)}")
      raise

    return {
      "agent_name": agent_name,
      "agent_code": processed_files.get("agent.py", ""),
      "dockerfile": processed_files.get("Dockerfile", ""),
      "compose_yml": processed_files.get("compose.yml", ""),
      "tools": generated_tools,
      "deployment": deployment_info,
    }

  def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process tools and prepare them for agent generation."""
    self.logger.info(f"Processing {len(tools)} tools")
    generated_tools = []

    for i, tool in enumerate(tools):
      self.logger.info(f"Processing tool {i + 1}/{len(tools)}: {tool['info']['name']}")
      try:
        tool_name = tool["info"]["name"]
        # Standardize the class name format - CamelCase convention
        class_name = "".join(word.capitalize() for word in tool_name.split()) + "Toolkit"

        # Extract or generate tool code
        if "generated_code" in tool and tool["generated_code"]:
          tool_code = tool["generated_code"]
          self.logger.info(f"Using existing generated code for {tool_name}")
        else:
          # Generate if not available
          self.logger.info(f"No generated code found for {tool_name}, generating new code")
          tool_result = self.tool_generator.generate_toolkit_from_json(tool)

          # Convert result to string
          if hasattr(tool_result, "code") and isinstance(tool_result.code, str):
            tool_code = tool_result.code
          elif hasattr(tool_result, "__str__"):
            tool_code = str(tool_result)
          else:
            tool_code = repr(tool_result)

        # Clean up tool code if needed
        if isinstance(tool_code, str) and tool_code.startswith("toolkit_code="):
          import re

          match = re.search(r"toolkit_code=['\"](.*)['\"]", tool_code, re.DOTALL)
          if match:
            tool_code = match.group(1).encode().decode("unicode_escape")

        # Handle API key configuration
        api_key_value = tool.get("api_key")
        if api_key_value:
          # Actual API key provided - use it directly in compose.yml
          api_key_env_var = api_key_value  # Store the actual key for compose.yml
          api_key_handling = f'os.environ.get("{tool_name.upper().replace(" ", "_")}_API_KEY")'  # Still use env var in code
        else:
          # No API key provided - use environment variable reference
          api_key_env_var = "${" + tool_name.upper().replace(" ", "_") + "_API_KEY}"
          api_key_handling = f'os.environ.get("{tool_name.upper().replace(" ", "_")}_API_KEY")'

        # Add processed tool
        generated_tools.append({
          "file_name": f"{tool_name.lower().replace(' ', '_')}_toolkit.py",
          "code": tool_code,
          "class_name": class_name,
          "api_key": api_key_value or api_key_env_var,
          "api_key_handling": api_key_handling,
          "version": tool["info"].get("version", "1"),
          "description": tool["info"].get("description", ""),
        })

      except Exception as e:
        self.logger.error(f"Failed to process tool {tool['info']['name']}: {str(e)}")
        raise

    self.logger.info(f"Processed {len(generated_tools)} tools")
    return generated_tools

  async def test_agent(self, agent_path: str, test_query: str) -> Dict[str, Any]:
    """Test a generated agent with a sample query."""
    self.logger.info(f"Testing agent at {agent_path} with query: {test_query}")

    try:
      import importlib.util
      import sys

      # Temporarily add agent directory to path
      agent_dir = os.path.dirname(agent_path)
      sys.path.insert(0, agent_dir)

      # Dynamically import the agent module
      spec = importlib.util.spec_from_file_location("agent_module", agent_path)
      if spec is None:
        raise ImportError(f"Could not create module spec from {agent_path}")
      if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {agent_path}")
      agent_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(agent_module)

      # Access the agent object and run the test query
      response = await agent_module.agent.arun(test_query)

      # Remove the added path
      sys.path.remove(agent_dir)

      self.logger.info("Agent test completed successfully")
      return {"success": True, "response": response.content, "tokens_used": response.usage.total_tokens if hasattr(response, "usage") else None}

    except Exception as e:
      self.logger.error(f"Error testing agent: {str(e)}")
      return {"success": False, "error": str(e)}

  async def deploy_agent(self, deployment_info: Dict[str, Any], environment: str = "development") -> Dict[str, Any]:
    """Deploy a generated agent to the specified environment."""
    self.logger.info(f"Deploying agent to {environment} environment")

    try:
      version_path = deployment_info["version_path"]

      # Execute deployment commands based on environment
      if environment == "development":
        # For development, just build the Docker container
        build_command = ["docker", "compose", "-f", os.path.join(version_path, "compose.yml"), "build"]

        import subprocess

        subprocess.run(build_command, check=True, cwd=version_path)

        self.logger.info(f"Agent built successfully in {environment} environment")
        return {"success": True, "environment": environment, "status": "built", "deployment_path": version_path}

      elif environment == "production":
        # For production, build and deploy
        deploy_commands = [
          ["docker", "compose", "-f", os.path.join(version_path, "compose.yml"), "build"],
          ["docker", "compose", "-f", os.path.join(version_path, "compose.yml"), "up", "-d"],
        ]

        import subprocess

        for cmd in deploy_commands:
          subprocess.run(cmd, check=True, cwd=version_path)

        self.logger.info(f"Agent deployed successfully to {environment} environment")
        return {"success": True, "environment": environment, "status": "deployed", "deployment_path": version_path}

      else:
        raise ValueError(f"Unsupported environment: {environment}")

    except Exception as e:
      self.logger.error(f"Error deploying agent: {str(e)}")
      return {"success": False, "environment": environment, "error": str(e)}

  def get_agent_status(self, agent_name: str, version: str = "latest") -> Dict[str, Any]:
    """Get the status of a deployed agent."""
    self.logger.info(f"Getting status for agent: {agent_name}, version: {version}")

    try:
      # Determine agent path
      base_dir = os.path.join(os.getcwd(), "src", "servers")

      if version == "latest":
        # Find latest version
        agent_dir = os.path.join(base_dir, agent_name)
        if not os.path.exists(agent_dir):
          raise FileNotFoundError(f"Agent {agent_name} not found")

        versions = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d)) and d.startswith("v")]
        if not versions:
          raise FileNotFoundError(f"No versions found for agent {agent_name}")

        version = max(versions, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

      agent_path = os.path.join(base_dir, agent_name, version)

      if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent {agent_name} version {version} not found")

      # Check if agent is running
      import subprocess

      container_name = f"{agent_name}_{version}"
      result = subprocess.run(["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"], capture_output=True, text=True)

      is_running = container_name in result.stdout

      return {
        "agent_name": agent_name,
        "version": version,
        "path": agent_path,
        "is_deployed": os.path.exists(os.path.join(agent_path, "compose.yml")),
        "is_running": is_running,
        "last_modified": os.path.getmtime(agent_path),
      }

    except Exception as e:
      self.logger.error(f"Error getting agent status: {str(e)}")
      return {"agent_name": agent_name, "version": version, "error": str(e), "is_deployed": False, "is_running": False}
