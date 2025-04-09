import logging
import re
from typing import Dict


class ResponseParser:
  """Parses responses from Claude to extract file contents."""

  def __init__(self, logger=None):
    self.logger = logger or logging.getLogger(__name__)

  def parse_response(self, response_text: str) -> Dict[str, str]:
    """Parse Claude's response to extract file contents."""
    self.logger.info("Parsing Claude response")
    files = {}

    # Define patterns to match different file sections - all possible formats
    patterns = {
      "agent.py": [
        r"---\s*agent\.py\s*---\s*```(?:python)?\s*(.*?)```",  # Format: --- agent.py --- ```python
        r"##\s*agent\.py\s*```python\s*(.*?)```",  # Format: ## agent.py ```python
        r"## File \d+: agent\.py\s*```python\s*(.*?)```",  # Format: ## File 1: agent.py ```python
        r"`agent.py`\s*```python\s*(.*?)```",  # Format: `agent.py` ```python
        r"agent\.py:?\s*```python\s*(.*?)```",  # Format: agent.py: ```python
      ],
      "Dockerfile": [
        r"---\s*Dockerfile\s*---\s*```(?:dockerfile)?\s*(.*?)```",
        r"##\s*Dockerfile\s*```dockerfile\s*(.*?)```",
        r"## File \d+: Dockerfile\s*```dockerfile\s*(.*?)```",
        r"`Dockerfile`\s*```dockerfile\s*(.*?)```",
        r"Dockerfile:?\s*```dockerfile\s*(.*?)```",
      ],
      "compose.yml": [
        r"---\s*compose\.yml\s*---\s*```(?:yaml|yml)?\s*(.*?)```",
        r"##\s*compose\.yml\s*```(?:yaml|yml)\s*(.*?)```",
        r"## File \d+: compose\.yml\s*```(?:yaml|yml)\s*(.*?)```",
        r"`compose\.yml`\s*```(?:yaml|yml)\s*(.*?)```",
        r"compose\.yml:?\s*```(?:yaml|yml)\s*(.*?)```",
      ],
      "requirements.txt": [
        r"---\s*requirements\.txt\s*---\s*```(?:txt|text)?\s*(.*?)```",
        r"##\s*requirements\.txt\s*```\s*(.*?)```",
        r"## File \d+: requirements\.txt\s*```(?:txt|text)?\s*(.*?)```",
        r"`requirements\.txt`\s*```(?:txt|text)?\s*(.*?)```",
        r"requirements\.txt:?\s*```(?:txt|text)?\s*(.*?)```",
      ],
    }

    # Extract each file's content using multiple patterns
    for file_name, pattern_list in patterns.items():
      found = False
      for pattern in pattern_list:
        matches = re.search(pattern, response_text, re.DOTALL)
        if matches:
          files[file_name] = matches.group(1).strip()
          found = True
          self.logger.info(f"Extracted {file_name} successfully using pattern")
          break

      if not found:
        self.logger.warning(f"Could not extract {file_name} using standard patterns, trying generic fallback")
        # Try a more generic pattern as fallback
        generic_pattern = rf"(?:^|\n).*{re.escape(file_name)}.*?```(?:python|dockerfile|yaml|yml|)?\s*(.*?)```"
        matches = re.search(generic_pattern, response_text, re.DOTALL)
        if matches:
          files[file_name] = matches.group(1).strip()
          self.logger.info(f"Extracted {file_name} using fallback pattern")
        else:
          self.logger.error(f"Failed to extract {file_name} even with fallback pattern")

    # Validate extracted files have minimum content
    for file_name, content in list(files.items()):
      if len(content.strip()) < 10:  # Arbitrary minimum size to catch empty/partial extractions
        self.logger.warning(f"Extracted {file_name} has suspiciously small content, may be incomplete")

    return files
