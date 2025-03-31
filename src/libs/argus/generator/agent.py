import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from src.common import logger

from .tools import ToolRegistry
from .requirements import RequirementsGenerator


class AgentGenerator:
    """Service to generate agent code based on configuration"""
    @classmethod
    def generate_agent_code(cls, config: Dict[str, Any], output_dir: str | None = None) -> Dict[str, Any]:
        """
        Generate code for a new agent based on the provided configuration

        Args:
            config: Dict containing agent configuration
            output_dir: Directory to save the generated agent code

        Returns:
            Dict with agent info including path to generated file
        """
        try:
            logger.debug(f"Starting agent generation with config: {config}")

            # Get the base directory of the application
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logger.debug(f"Base directory: {base_dir}")

            # Set output directory
            output_dir = output_dir or os.path.join(base_dir, "generated_agents")
            logger.debug(f"Output directory: {output_dir}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")

            # Validate tools configuration
            tools = []
            for tool_config in config.get("tools", []):
                tool_type = tool_config.get("tool_type")
                tool_meta = ToolRegistry.get_tool_metadata(tool_type)
                if not tool_meta:
                    raise ValueError(f"Unknown tool type: {tool_type}")

                # Validate the tool configuration
                ToolRegistry.validate_tool_config(tool_type, tool_config.get("parameters", {}))

                # Add tool metadata to the list
                tools.append({
                    "name": tool_config.get("name", tool_type),
                    "tool_type": tool_type,
                    "class_name": tool_meta.class_name,
                    "import_path": tool_meta.import_path,
                    "parameters": tool_config.get("parameters", {}),
                    "function_name_mapping": tool_meta.function_name_mapping
                })
            logger.debug(f"Validated tools: {tools}")

            # Generate requirements file in the agent directory
            requirements_path = os.path.join(output_dir, "requirements.txt")
            RequirementsGenerator.generate_requirements(config, requirements_path)
            logger.debug(f"Generated requirements file at: {requirements_path}")

            # Prepare the template context
            context = {
                "config": config,
                "tools": tools,
                "llm": config.get("llm", {})
            }
            logger.debug("Template context prepared")

            # Set up Jinja environment with absolute path
            templates_dir = os.path.join(base_dir, "templates")
            logger.debug(f"Templates directory (absolute): {os.path.abspath(templates_dir)}")

            if not os.path.exists(templates_dir):
                raise RuntimeError(f"Templates directory not found at: {templates_dir}")

            env = Environment(loader=FileSystemLoader(templates_dir))
            logger.debug("Created Jinja environment")

            try:
                template = env.get_template("agent.py.jinja")
                logger.debug("Successfully loaded template")
            except Exception as e:
                logger.error(f"Failed to load template: {str(e)}")
                available_templates = env.list_templates()
                logger.debug(f"Available templates: {available_templates}")
                raise

            # Render the template
            try:
                agent_code = template.render(**context)
                logger.debug("Successfully rendered template")
            except Exception as e:
                logger.error(f"Failed to render template: {str(e)}")
                raise

            # Save the generated agent code to a file in the same directory as requirements.txt
            agent_name = config["name"].lower()
            agent_filename = f"{agent_name}_agent.py"
            agent_path = os.path.join(output_dir, agent_filename)
            logger.debug(f"Agent file path: {agent_path}")

            # Ensure the file is written with UTF-8 encoding
            try:
                with open(agent_path, "w", encoding="utf-8") as f:
                    f.write(agent_code)
                logger.debug(f"Successfully wrote agent code to: {agent_path}")
            except Exception as e:
                logger.error(f"Failed to write agent file: {str(e)}")
                raise

            # Verify the file was created
            if not os.path.exists(agent_path):
                raise RuntimeError(f"Failed to create agent file at {agent_path}")
            logger.debug("Verified agent file exists")

            return {
                "name": config["name"],
                "file_path": agent_path,
                "requirements_path": requirements_path,
                "tools": tools
            }

        except Exception as e:
            logger.error(f"Error in generate_agent_code: {str(e)}", exc_info=True)
            raise
    @classmethod
    def save_agent_code(cls, agent_name: str, code: str, output_dir: Optional[str] = None) -> Path:
        """Save the generated agent code to a file"""
        # Get the base directory of the application
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set output directory
        output_dir = os.path.join(base_dir, "generated_agents")

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)

        # Save the generated agent code to a file
        agent_filename = f"{agent_name.lower()}_agent.py"
        agent_path = Path(output_dir) / agent_filename

        # Ensure the file is written with UTF-8 encoding
        with open(agent_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Verify the file was created
        if not agent_path.exists():
            raise RuntimeError(f"Failed to create agent file at {agent_path}")

        return agent_path