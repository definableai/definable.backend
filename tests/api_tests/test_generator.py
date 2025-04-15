import json
import os

"""
Utility script to generate API tests based on a Postman collection.
This helps ensure all API endpoints have corresponding test coverage.
"""


def load_postman_collection(file_path):
  """Load a Postman collection from the specified file path."""
  with open(file_path, "r") as f:
    return json.load(f)


def extract_endpoints(collection):
  """Extract all API endpoints from a Postman collection."""
  endpoints = []

  def process_item(item, parent_name=None):
    if "item" in item:
      # This is a folder
      folder_name = item.get("name", "")
      for child_item in item["item"]:
        process_item(child_item, folder_name)
    else:
      # This is an endpoint
      endpoint = {
        "name": item.get("name", ""),
        "category": parent_name,
        "method": item.get("request", {}).get("method", ""),
        "url": item.get("request", {}).get("url", {}),
      }

      # Extract the path from the URL
      if isinstance(endpoint["url"], dict) and "path" in endpoint["url"]:
        endpoint["path"] = "/".join(endpoint["url"]["path"])
      elif isinstance(endpoint["url"], str):
        url_parts = endpoint["url"].split("?")[0].split("/")
        # Remove host part
        if url_parts and url_parts[0].startswith("{{"):
          url_parts = url_parts[1:]
        endpoint["path"] = "/".join(url_parts)
      else:
        endpoint["path"] = ""

      endpoints.append(endpoint)

  for item in collection.get("item", []):
    process_item(item)

  return endpoints


def generate_test_file_content(endpoints, category):
  """Generate test file content for a category of endpoints."""
  # Filter endpoints for the specified category
  category_endpoints = [e for e in endpoints if e["category"] and e["category"].lower() == category.lower()]

  if not category_endpoints:
    return None

  test_class_name = f"Test{category.title().replace(' ', '')}API"

  # Generate imports
  content = [
    "import pytest",
    "from fastapi.testclient import TestClient",
    "from unittest.mock import AsyncMock, MagicMock, patch",
    "import json",
    "import sys",
    "from uuid import uuid4",
    "",
    "@pytest.mark.asyncio",
    f"class {test_class_name}:",
    f'    """Test {category} API endpoints."""',
    "",
  ]

  # Generate test methods for each endpoint
  for endpoint in category_endpoints:
    method_name = endpoint["name"].lower().replace(" ", "_")
    http_method = endpoint["method"].lower()

    test_method = [
      f"    async def test_{method_name}(self, client, mock_db_session, auth_headers):",
      f'        """Test {endpoint["name"]} endpoint."""',
      "        # TODO: Implement test for this endpoint",
      f"        # {http_method.upper()} {endpoint['path']}",
      "",
    ]

    content.extend(test_method)

  return "\n".join(content)


def generate_api_tests(collection_path, output_dir):
  """Generate API test files based on a Postman collection."""
  # Load the collection
  collection = load_postman_collection(collection_path)

  # Extract endpoints
  endpoints = extract_endpoints(collection)

  # Group endpoints by category
  categories = set(e["category"] for e in endpoints if e["category"])

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Generate test files for each category
  for category in categories:
    content = generate_test_file_content(endpoints, category)
    if content:
      file_name = f"test_{category.lower().replace(' ', '_')}_template.py"
      file_path = os.path.join(output_dir, file_name)

      with open(file_path, "w") as f:
        f.write(content)

      print(f"Generated test file: {file_path}")


if __name__ == "__main__":
  # Path to your Postman collection
  collection_path = "api.json"

  # Output directory for generated test files
  output_dir = "tests/api_tests/generated"

  # Generate the test files
  generate_api_tests(collection_path, output_dir)
