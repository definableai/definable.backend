# Scripts Architecture

This directory contains maintenance and setup scripts for the application. All scripts follow a standardized architecture using the `BaseScript` class to ensure consistency and maintainability.

## Architecture Overview

The script architecture provides:
- **Execution tracking**: All scripts are tracked in the `script_run_tracker` database table
- **Rollback capabilities**: Scripts can implement rollback logic to undo changes
- **Status checking**: Check if a script has been executed and its current status
- **CLI interface**: Consistent command-line interface with run/rollback/status commands
- **Error handling**: Standardized error handling and logging
- **Force rerun**: Ability to force re-execution of previously successful scripts
- **Standardized path setup**: Each script includes a standard path setup block that ensures imports from the `src` directory work correctly when scripts are run directly

## Creating a New Script

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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sqlalchemy.ext.asyncio import AsyncSession

from scripts.base_script import BaseScript
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