import os
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader
from src.common import logger

class RequirementsGenerator:
    """Service to generate requirements.txt files for agents"""

    @classmethod
    def generate_requirements(cls, config: Dict[str, Any], output_path: str) -> None:
        """
        Generate a requirements.txt file for the agent based on its configuration

        Args:
            config: Dict containing agent configuration
            output_path: Path where to save the requirements.txt file
        """
        try:
            logger.debug(f"Generating requirements for config: {config}")

            # Get the base directory of the application
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            templates_dir = os.path.join(base_dir, "templates")

            # Set up Jinja environment
            env = Environment(loader=FileSystemLoader(templates_dir))
            template = env.get_template("requirements.txt.jinja")

            # Prepare context
            context = {
                "config": config,
                "tools": config.get("tools", []),
                "llm": config.get("llm", {})
            }

            # Render template
            requirements_content = template.render(**context)

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(requirements_content)

            logger.debug(f"Generated requirements file at: {output_path}")

        except Exception as e:
            logger.error(f"Error generating requirements: {str(e)}", exc_info=True)
            raise

    @classmethod
    def get_missing_dependencies(cls, agent_config: Dict[str, Any]) -> List[str]:
        """
        Check which dependencies are missing for the given agent configuration

        Args:
            agent_config: The agent configuration containing tools and LLM provider info

        Returns:
            List of missing package names
        """
        # Create a temporary requirements file
        temp_requirements_path = "temp_requirements.txt"
        cls.generate_requirements(agent_config, temp_requirements_path)

        # Read the requirements
        with open(temp_requirements_path, "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#") and ">=" in line]

        # Extract package names (remove version info)
        packages = [req.split(">=")[0].strip() for req in requirements]

        # Check which packages are missing
        missing_packages = []
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        # Clean up
        os.remove(temp_requirements_path)

        return missing_packages