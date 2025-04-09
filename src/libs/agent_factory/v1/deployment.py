import os
import subprocess
from typing import Any, Dict, List


class DeploymentManager:
  """Manages deployment of generated agents."""

  def __init__(self, base_path=None):
    self.base_path = base_path or os.path.join(os.getcwd(), "src", "servers")

  def prepare_deployment(
    self,
    agent_name: str,
    agent_code: str,
    dockerfile: str,
    compose_yml: str,
    tools: List[Dict[str, Any]],
    version: str,
    processed_files: Dict[str, Any],
  ) -> Dict[str, Any]:
    """
    Prepare agent deployment by creating all necessary files.

    Args:
        agent_name: Name of the agent
        agent_code: Generated agent code
        dockerfile: Generated Dockerfile
        compose_yml: Generated docker-compose.yml
        tools: List of generated tools with their code
        version: Agent version (v1, v2, etc.)

    Returns:
        Dict with deployment information
    """
    # Create deployment directory structure in src/servers

    # Create agent directory inside servers folder
    deploy_dir = os.path.join(self.base_path, agent_name)
    os.makedirs(deploy_dir, exist_ok=True)

    version_dir = os.path.join(deploy_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    tools_dir = os.path.join(version_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)

    # Write agent file
    with open(os.path.join(version_dir, "agent.py"), "w") as f:
      f.write(agent_code)

    # Write tools
    for tool in tools:
      tool_code = tool["code"]

      # No need for the complex extraction if we're getting clean code
      # Just ensure it's a string
      if not isinstance(tool_code, str):
        tool_code = str(tool_code)

      # Write the clean code to file
      with open(os.path.join(tools_dir, tool["file_name"]), "w") as f:
        f.write(tool_code)

    # Write Dockerfile
    with open(os.path.join(version_dir, "Dockerfile"), "w") as f:
      f.write(dockerfile)

    # Write docker-compose.yml
    with open(os.path.join(version_dir, "compose.yml"), "w") as f:
      f.write(compose_yml)

    # Create requirements.txt
    with open(os.path.join(version_dir, "requirements.txt"), "w") as f:
      f.write("agno\nfastapi\nuvicorn\n")
      # Additional requirements could be determined from tools

    # Create __init__.py files
    with open(os.path.join(version_dir, "__init__.py"), "w") as f:
      f.write("")

    os.makedirs(os.path.join(version_dir, "tools"), exist_ok=True)
    with open(os.path.join(version_dir, "tools", "__init__.py"), "w") as f:
      f.write(processed_files["tools/__init__.py"])

    return {
      "deployment_path": deploy_dir,
      "version_path": version_dir,
      "tools_path": tools_dir,
      "files": {
        "agent": os.path.join(version_dir, "agent.py"),
        "dockerfile": os.path.join(version_dir, "Dockerfile"),
        "compose": os.path.join(version_dir, "compose.yml"),
      },
    }

  def build_and_deploy(self, deployment_info: Dict[str, Any], push_to_registry: bool = False) -> Dict[str, Any]:
    """
    Build and deploy the agent container.

    Args:
        deployment_info: Deployment information from prepare_deployment
        push_to_registry: Whether to push the image to a container registry

    Returns:
        Dict with build and deployment details
    """
    # This would handle the actual Docker build and deployment process
    # Implementation depends on your specific deployment strategy

    # Example simple local build:
    build_command = ["docker", "compose", "-f", deployment_info["files"]["compose"], "build"]

    try:
      subprocess.run(build_command, check=True, cwd=os.path.dirname(deployment_info["files"]["compose"]))

      # Start the container
      up_command = ["docker", "compose", "-f", deployment_info["files"]["compose"], "up", "-d"]
      subprocess.run(up_command, check=True, cwd=os.path.dirname(deployment_info["files"]["compose"]))

      return {"status": "success", "deployment_info": deployment_info, "container_running": True}
    except subprocess.CalledProcessError as e:
      return {"status": "error", "message": str(e), "deployment_info": deployment_info, "container_running": False}
