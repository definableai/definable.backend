#!/usr/bin/env python3
"""
CLI tool for creating new scripts with proper boilerplate code.
Generates scripts that follow the BaseScript architecture pattern.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import click

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from common.logger import log as logger


class ScriptGenerator:
  """Generator for creating new scripts with proper boilerplate."""

  def __init__(self):
    self.scripts_dir = Path(__file__).parent.parent / "executable"
    self.template_markers = {
      "SCRIPT_NAME": "Script name placeholder",
      "SCRIPT_DESCRIPTION": "Script description placeholder",
      "CLASS_NAME": "Class name placeholder",
      "SCRIPT_FILENAME": "Script filename placeholder",
    }

  def validate_script_name(self, name: str) -> tuple[bool, Optional[str]]:
    """
    Validate the script name follows Python naming conventions.

    Args:
        name: The proposed script name

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if name is empty
    if not name or not name.strip():
      return False, "Script name cannot be empty"

    name = name.strip()

    # Check Python identifier rules
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
      return False, "Script name must start with a letter and contain only lowercase letters, numbers, and underscores"

    # Check if name is too short
    if len(name) < 3:
      return False, "Script name must be at least 3 characters long"

    # Check if name is a Python keyword
    import keyword

    if keyword.iskeyword(name):
      return False, f"'{name}' is a Python keyword and cannot be used as a script name"

    # Check if script already exists
    script_path = self.scripts_dir / f"{name}.py"
    if script_path.exists():
      return False, f"Script '{name}.py' already exists in {self.scripts_dir}"

    return True, None

  def generate_class_name(self, script_name: str) -> str:
    """
    Generate a proper class name from script name.

    Args:
        script_name: The snake_case script name

    Returns:
        PascalCase class name
    """
    # Convert snake_case to PascalCase
    words = script_name.split("_")
    class_name = "".join(word.capitalize() for word in words)

    # Ensure it ends with 'Script'
    if not class_name.endswith("Script"):
      class_name += "Script"

    return class_name

  def get_script_template(self, script_name: str, description: str, class_name: str) -> str:
    """
    Generate the script template with proper substitutions.

    Args:
        script_name: The script name for tracking
        description: User-provided description
        class_name: The class name

    Returns:
        Complete script content
    """
    # Use default description if none provided
    if not description.strip():
      description = f"TODO: Add description for {script_name} script"

    template = f'''#!/usr/bin/env python3
"""
{description}
"""

import os
import sys

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sqlalchemy.ext.asyncio import AsyncSession

from scripts.core.base_script import BaseScript
from common.logger import log as logger


class {class_name}(BaseScript):
    """
    {description}
    """

    def __init__(self):
        super().__init__("{script_name}")

    async def execute(self, db: AsyncSession) -> None:
        """
        Main script execution logic.
        Implement your script's core functionality here.
        """
        logger.info("Starting {script_name} script execution...")

        # TODO: Implement your script logic here
        # Examples:
        # - Database operations: result = await db.execute(text("SELECT COUNT(*) FROM table"))
        # - External API calls
        # - File processing
        # - Data transformations

        logger.info("{script_name} script execution completed.")

    async def rollback(self, db: AsyncSession) -> None:
        """
        Rollback logic for the script.
        Implement this to undo changes made by the execute method.

        Note: This method is optional. Remove if not needed.
        """
        logger.info("Rolling back {script_name} script...")

        # TODO: Implement rollback logic here
        # Examples:
        # - Delete records created by execute()
        # - Revert configuration changes
        # - Clean up temporary files

        logger.info("{script_name} script rollback completed.")

    async def verify(self, db: AsyncSession) -> bool:
        """
        Verify script execution was successful.
        Return True if everything is as expected, False otherwise.

        Note: This method is optional. Remove if not needed.
        """
        logger.info("Verifying {script_name} script execution...")

        # TODO: Implement verification logic here
        # Examples:
        # - Check record counts: result = await db.execute(text("SELECT COUNT(*) FROM table WHERE condition"))
        # - Validate data integrity
        # - Confirm external service states

        return True  # Return True if verification passes


def main():
    """Entry point for backward compatibility."""
    script = {class_name}()
    script.main()


if __name__ == "__main__":
    script = {class_name}()
    script.run_cli()
'''
    return template

  def create_script(self, script_name: str, description: str, overwrite: bool = False) -> Path:
    """
    Create a new script file with the proper template.

    Args:
        script_name: The name of the script (snake_case)
        description: Description of what the script does
        overwrite: Whether to overwrite existing files

    Returns:
        Path to the created script file

    Raises:
        ValueError: If script name is invalid
        FileExistsError: If script exists and overwrite=False
    """
    # Validate script name
    is_valid, error_msg = self.validate_script_name(script_name)
    if not is_valid:
      raise ValueError(error_msg)

    script_path = self.scripts_dir / f"{script_name}.py"

    # Check if file exists and handle overwrite
    if script_path.exists() and not overwrite:
      raise FileExistsError(f"Script '{script_name}.py' already exists. Use --overwrite to replace it.")

    # Generate class name
    class_name = self.generate_class_name(script_name)

    # Generate template
    template_content = self.get_script_template(script_name, description, class_name)

    # Write the file
    try:
      with open(script_path, "w", encoding="utf-8") as f:
        f.write(template_content)

      # Make file executable on Unix-like systems
      if os.name != "nt":  # Not Windows
        os.chmod(script_path, 0o755)

      logger.info(f"Successfully created script: {script_path}")
      return script_path

    except Exception as e:
      raise RuntimeError(f"Failed to create script file: {e}")

  def list_scripts(self) -> List[Path]:
    """
    List all existing executable scripts in the executable directory.

    Returns:
        List of script file paths
    """
    script_files = []
    for file_path in self.scripts_dir.glob("*.py"):
      # Skip __init__.py
      if file_path.name != "__init__.py":
        script_files.append(file_path)

    return sorted(script_files)


# CLI Commands


@click.group(invoke_without_command=True)
@click.option("--list", "-l", "show_list", is_flag=True, help="List all existing scripts")
@click.pass_context
def cli(ctx, show_list):
  """Create and manage scripts with proper boilerplate code."""
  if show_list:
    list_all_scripts()
    sys.exit(0)
  elif ctx.invoked_subcommand is None:
    # If no subcommand and no flags, show help
    click.echo(ctx.get_help())


@cli.command()
@click.argument("script_name")
@click.option("--description", "-d", default="", help="Description of what the script does")
@click.option("--overwrite", "-i", is_flag=True, help="Overwrite existing script if it exists")
def init(script_name: str, description: str, overwrite: bool):
  """
  Initialize a new script with proper boilerplate code.

  SCRIPT_NAME should be in snake_case format (e.g., 'cleanup_old_data').
  The script will be created with all necessary boilerplate including:
  - BaseScript inheritance
  - Proper imports and path setup
  - Execute, rollback, and verify method stubs
  - CLI integration

  If no description is provided, a TODO placeholder will be added.
  """
  try:
    generator = ScriptGenerator()
    script_path = generator.create_script(script_name, description, overwrite)

    click.echo(f"Successfully created script: {script_path}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"1. Edit {script_path} and implement your logic in the execute() method")
    click.echo("2. Optionally implement rollback() and verify() methods")
    click.echo(f"3. Test your script: python {script_path}")
    click.echo(f"4. Run with CLI: python {script_path} run --force")

  except (ValueError, FileExistsError, RuntimeError) as e:
    click.echo(f"Error: {e}", err=True)
    sys.exit(1)
  except Exception as e:
    click.echo(f"Unexpected error: {e}", err=True)
    logger.error(f"Unexpected error creating script: {e}")
    sys.exit(1)


def list_all_scripts():
  """List all existing scripts in the scripts directory."""
  try:
    generator = ScriptGenerator()
    scripts = generator.list_scripts()

    if not scripts:
      click.echo("No scripts found in the scripts directory.")
      return

    click.echo(f"Found {len(scripts)} scripts in {generator.scripts_dir}:")
    click.echo("")

    for script_path in scripts:
      # Try to extract class name and description from the script
      try:
        with open(script_path, "r", encoding="utf-8") as f:
          content = f.read()

        # Extract description from docstring
        docstring_match = re.search(r'"""[\\n\\r]*(.+?)[\\n\\r]*"""', content, re.DOTALL)
        if docstring_match:
          description = docstring_match.group(1).strip()
          # Take only the first line if it's multiline
          description = description.split("\n")[0].strip()
        else:
          description = "No description"

        click.echo(f"{script_path.name}")
        click.echo(f"   Description: {description}")
        click.echo("")

      except Exception:
        click.echo(f"{script_path.name}")
        click.echo("   Description: Unable to read description")
        click.echo("")

  except Exception as e:
    click.echo(f"Error listing scripts: {e}", err=True)
    sys.exit(1)


@cli.command()
@click.argument("script_name")
def validate(script_name: str):
  """Validate a script name without creating the script."""
  try:
    generator = ScriptGenerator()
    is_valid, error_msg = generator.validate_script_name(script_name)

    if is_valid:
      class_name = generator.generate_class_name(script_name)
      click.echo(f"'{script_name}' is a valid script name")
      click.echo(f"   Class name would be: {class_name}")
      click.echo(f"   File path would be: {generator.scripts_dir / f'{script_name}.py'}")
    else:
      click.echo(f"'{script_name}' is not a valid script name: {error_msg}")
      sys.exit(1)

  except Exception as e:
    click.echo(f"Error validating script name: {e}", err=True)
    sys.exit(1)


@cli.command()
def info():
  """Show information about the script generation system."""
  generator = ScriptGenerator()
  scripts = generator.list_scripts()

  click.echo("Script Generation System Info")
  click.echo("=" * 40)
  click.echo(f"Scripts directory: {generator.scripts_dir}")
  click.echo(f"Total scripts: {len(scripts)}")
  click.echo("Base script architecture: BaseScript class")
  click.echo("")
  click.echo("Available commands:")
  click.echo("  init           Create a new script")
  click.echo("  validate       Validate a script name")
  click.echo("  info           Show this information")
  click.echo("")
  click.echo("Available options:")
  click.echo("  --list, -l     List all existing scripts")
  click.echo("")
  click.echo("Example usage:")
  click.echo("  python create_script.py init my_cleanup_task")
  click.echo("  python create_script.py init my_cleanup_task -d 'Clean up old data'")
  click.echo("  python create_script.py init my_cleanup_task -d 'Clean up old data' -i")
  click.echo("  python create_script.py --list")
  click.echo("  python create_script.py validate my_script_name")


if __name__ == "__main__":
  cli()
