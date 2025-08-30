# Scripts Architecture

This directory contains maintenance and setup scripts for the application. All scripts follow a standardized architecture using the `BaseScript` class to ensure consistency and maintainability.

## Directory Structure

```
src/scripts/
├── README.md                     # This documentation
├── __init__.py                   # Python package marker
├── core/                         # Core architecture and tools
│   ├── __init__.py
│   ├── base_script.py           # Base class for all scripts
│   ├── create_script.py         # CLI tool for generating new scripts
│   └── example_script.py        # Example/template script
└── executable/                   # Executable maintenance scripts
    ├── __init__.py
    ├── create_razorpay_plans.py  # Razorpay payment plan synchronization
    ├── create_stripe_plans.py    # Stripe payment plan synchronization
    ├── delete_all_test_stytch_users.py  # Test user cleanup (Stytch)
    └── ensure_model_props.py     # Model properties management
```

### Core vs Executable

- **`core/`**: Contains the script architecture, base classes, and development tools
  - `base_script.py`: The foundational BaseScript class that all scripts inherit from
  - `create_script.py`: CLI tool for generating new scripts with proper boilerplate
  - `example_script.py`: Reference implementation showing best practices

- **`executable/`**: Contains actual maintenance and operational scripts
  - Production scripts that perform specific tasks
  - All inherit from `BaseScript` and follow the established patterns
  - Can be run independently or as part of deployment processes

## Architecture Overview

The script architecture provides:
- **Execution tracking**: All scripts are tracked in the `script_run_tracker` database table
- **Rollback capabilities**: Scripts can implement rollback logic to undo changes
- **Status checking**: Check if a script has been executed and its current status
- **CLI interface**: Consistent command-line interface with run/rollback/status commands
- **Error handling**: Standardized error handling and logging
- **Force rerun**: Ability to force re-execution of previously successful scripts
- **Standardized path setup**: Each script includes a standard path setup block that ensures imports from the `src` directory work correctly when scripts are run directly

## Quick Start

### Option 1: Using the Script Generator (Recommended)

The fastest way to create a new script is using the built-in CLI generator:

```bash
# Create a new script with boilerplate
python src/scripts/core/create_script.py create my_script_name -d "Description of what the script does"

# Validate a script name before creating
python src/scripts/core/create_script.py validate my_script_name

# List all existing scripts
python src/scripts/core/create_script.py list

# Get help and system info
python src/scripts/core/create_script.py --help
python src/scripts/core/create_script.py info
```

### Option 2: Manual Creation

If you prefer to create scripts manually:

1. **Create a new Python file** in the `src/scripts/` directory
2. **Import the BaseScript class** from `scripts.base_script`
3. **Inherit from BaseScript** and implement required methods
4. **Add CLI entry point** in the `__main__` block

### Template

```python
#!/usr/bin/env python3
"""
Description of your script's purpose.
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


class YourScript(BaseScript):
    """Your script description."""
    
    def __init__(self):
        # Use a unique script name (usually the filename without extension)
        super().__init__("your_script_name")
    
    async def execute(self, db: AsyncSession) -> None:
        """Main script logic - REQUIRED"""
        logger.info("Starting your script...")
        # Add your implementation here
        logger.info("Script completed.")
    
    async def rollback(self, db: AsyncSession) -> None:
        """Rollback logic - OPTIONAL"""
        logger.info("Rolling back your script...")
        # Add rollback implementation here
        logger.info("Rollback completed.")
    
    async def verify(self, db: AsyncSession) -> bool:
        """Verification logic - OPTIONAL"""
        # Add verification logic here
        return True  # Return True if verification passes


def main():
    """Entry point for backward compatibility."""
    script = YourScript()
    script.main()


if __name__ == "__main__":
    script = YourScript()
    script.run_cli()
```

## Required Methods

### `execute(self, db: AsyncSession) -> None`
- **Purpose**: Contains the main logic of your script
- **Required**: Yes
- **Parameters**: Database session for performing operations
- **Example**: Database updates, API calls, file processing

### Optional Methods

### `rollback(self, db: AsyncSession) -> None`
- **Purpose**: Undo changes made by the execute method
- **Required**: No (default implementation logs a warning)
- **Parameters**: Database session for performing operations
- **Example**: Delete created records, revert configuration changes

### `verify(self, db: AsyncSession) -> bool`
- **Purpose**: Verify that the script executed successfully
- **Required**: No (default implementation returns True)
- **Parameters**: Database session for performing operations
- **Returns**: True if verification passes, False otherwise
- **Example**: Check record counts, validate data integrity

## Script Execution Tracking

All scripts are automatically tracked in the `script_run_tracker` table with the following statuses:
- `pending`: Script is currently running
- `success`: Script completed successfully
- `failed`: Script failed with an error
- `rolled_back`: Script was rolled back

## Command Line Interface

Every script automatically gets a CLI with these commands:

```bash
# Run the script
python your_script.py run

# Run with force (ignores previous successful execution)
python your_script.py run --force

# Rollback the script
python your_script.py rollback

# Check script status
python your_script.py status

# Run directly (backward compatibility)
python your_script.py

# Run with force (backward compatibility shorthand)
python your_script.py --force
```

## Best Practices

1. **Use descriptive script names**: Use clear, descriptive names that indicate the script's purpose
2. **Implement proper error handling**: Let exceptions bubble up to the base class for proper logging
3. **Add verification logic**: Implement the `verify` method to ensure your script completed successfully
4. **Implement rollback when possible**: Provide rollback functionality for scripts that make significant changes
5. **Use async/await**: All database operations should be asynchronous
6. **Log important steps**: Use the logger to track script progress
7. **Test thoroughly**: Test your script in a safe environment before deploying

## Environment Safety

For scripts that can be destructive (like user deletion), consider adding environment checks:

```python
async def execute(self, db: AsyncSession) -> None:
    # Check environment before proceeding
    if not self._is_safe_environment():
        raise Exception("This script should only run in test environments")
    
    # ... rest of your logic

def _is_safe_environment(self) -> bool:
    """Check if we're in a safe environment for this script."""
    environment = os.getenv("ENVIRONMENT", "production").lower()
    return environment in ["test", "development", "staging"]
```

## Server Lifecycle Integration

Scripts that need to run before application startup can be:
1. **Added to Docker startup scripts**
2. **Called from application initialization code**
3. **Scheduled as pre-deployment tasks**

Example of calling a script programmatically:
```python
from scripts.your_script import YourScript

# In your application startup
script = YourScript()
await script.run_script()
```

## Examples

See `example_script.py` for a complete example implementation.

Existing scripts in this directory demonstrate various patterns:
- `create_stripe_plans.py` - External API integration with rollback
- `create_razorpay_plans.py` - Similar payment integration pattern
- `ensure_model_props.py` - Database schema and data management
- `delete_all_test_stytch_users.py` - Environment-aware cleanup script

## Script Generator CLI Tool

The `create_script.py` tool provides a production-ready CLI for generating new scripts with proper boilerplate code.

### Features

- **Automatic Boilerplate Generation**: Creates scripts with all necessary imports, class structure, and CLI integration
- **Name Validation**: Ensures script names follow Python conventions and don't conflict with existing scripts
- **Smart Class Naming**: Converts snake_case script names to PascalCase class names
- **Template Customization**: Generates TODO comments and example code for common patterns
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Commands

#### `create`
Creates a new script with full boilerplate code.

```bash
python src/scripts/core/create_script.py create my_new_script -d "Clean up old user data"

# With overwrite protection
python src/scripts/core/create_script.py create existing_script -d "New description" --overwrite
```

**Generated script includes:**
- Proper BaseScript inheritance
- Path setup for imports
- Execute, rollback, and verify method stubs
- CLI integration with run/rollback/status commands
- TODO comments with implementation guidance
- Example code patterns

#### `validate`
Validates a script name without creating the file.

```bash
python src/scripts/core/create_script.py validate my_script_name
```

**Validation checks:**
- Python identifier rules (lowercase, underscores, numbers)
- Minimum length (3 characters)
- Not a Python keyword
- No file conflicts
- Proper naming conventions

#### `list`
Shows all existing scripts with descriptions.

```bash
python src/scripts/core/create_script.py list
```

**Output includes:**
- Script filenames
- Extracted descriptions from docstrings
- Total script count

#### `info`
Displays system information and usage examples.

```bash
python src/scripts/core/create_script.py info
```

### Generated Script Structure

When you create a script named `cleanup_old_data`, the generator creates:

```python
#!/usr/bin/env python3
"""
Clean up old data from the system
"""

import os
import sys

# Path setup (automatically included)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sqlalchemy.ext.asyncio import AsyncSession
from scripts.base_script import BaseScript
from common.logger import log as logger

class CleanupOldDataScript(BaseScript):
    """Clean up old data from the system"""
    
    def __init__(self):
        super().__init__("cleanup_old_data")
    
    async def execute(self, db: AsyncSession) -> None:
        """Main script execution logic."""
        logger.info("Starting cleanup_old_data script execution...")
        
        # TODO: Implement your script logic here
        # Examples provided in comments
        
        logger.info("cleanup_old_data script execution completed.")
    
    async def rollback(self, db: AsyncSession) -> None:
        """Rollback logic (optional)."""
        # TODO: Implement rollback logic
        pass
    
    async def verify(self, db: AsyncSession) -> bool:
        """Verification logic (optional)."""
        # TODO: Implement verification logic
        return True

def main():
    """Entry point for backward compatibility."""
    script = CleanupOldDataScript()
    script.main()

if __name__ == "__main__":
    script = CleanupOldDataScript()
    script.run_cli()
```

### Best Practices for Script Generator

1. **Use descriptive names**: `cleanup_old_users` instead of `cleanup`
2. **Include purpose in description**: Be specific about what the script does
3. **Follow snake_case**: The generator validates this automatically
4. **Implement incrementally**: Start with execute(), add rollback() and verify() as needed
5. **Test immediately**: Generated scripts work out of the box for testing