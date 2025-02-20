import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_unique_filename(original_filename: str, prefix: Optional[str] = None) -> str:
  """
  Generate a unique filename while preserving the original extension.

  Args:
      original_filename (str): Original filename
      prefix (str, optional): Optional prefix to add to the filename

  Returns:
      str: Unique sanitized filename

  Example outputs:
      Input: "my file.pdf" -> "2024-02-10_123456_my-file_a1b2c3d4.pdf"
      Input: "test.jpg", prefix="avatar" -> "2024-02-10_123456_avatar_test_a1b2c3d4.jpg"
  """
  # Get the file extension
  original_extension = Path(original_filename).suffix.lower()

  # Get the original filename without extension
  original_name = Path(original_filename).stem

  # Sanitize the filename:
  # 1. Convert to lowercase
  # 2. Replace spaces and special chars with hyphens
  # 3. Remove any non-alphanumeric chars (except hyphens)
  # 4. Remove multiple consecutive hyphens
  sanitized_name = original_name.lower()
  sanitized_name = re.sub(r"[^\w\s-]", "", sanitized_name)
  sanitized_name = re.sub(r"[-\s]+", "-", sanitized_name)
  sanitized_name = sanitized_name.strip("-")

  # Generate timestamp
  timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

  # Generate a short UUID (first 8 characters)
  unique_id = str(uuid.uuid4())[:8]

  # Construct the final filename
  if prefix:
    # Sanitize the prefix too
    prefix = re.sub(r"[^\w\s-]", "", prefix.lower())
    prefix = re.sub(r"[-\s]+", "-", prefix)
    prefix = prefix.strip("-")
    filename = f"{timestamp}_{prefix}_{sanitized_name}_{unique_id}{original_extension}"
  else:
    filename = f"{timestamp}_{sanitized_name}_{unique_id}{original_extension}"

  return filename
