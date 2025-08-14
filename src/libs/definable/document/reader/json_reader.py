import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import IO, Any, List, Union
from uuid import uuid4

from libs.definable.document.base import Document
from libs.definable.document.reader.base import Reader


class JSONReader(Reader):
  """Reader for JSON files"""

  chunk: bool = False

  def read(self, path: Union[Path, IO[Any]]) -> List[Document]:
    try:
      if isinstance(path, Path):
        if not path.exists():
          raise FileNotFoundError(f"Could not find file: {path}")

        json_name = path.name.split(".")[0]
        content = path.read_text("utf-8")

      elif isinstance(path, BytesIO):
        json_name = path.name.split(".")[0]
        path.seek(0)
        content = path.read().decode("utf-8")

      else:
        raise ValueError("Unsupported file type. Must be Path or BytesIO.")

      # Try to parse JSON with improved error handling
      json_contents = self._parse_json_content(content)

      if isinstance(json_contents, dict):
        json_contents = [json_contents]
      elif not isinstance(json_contents, list):
        # If it's neither dict nor list, wrap it in a list
        json_contents = [json_contents]

      documents = [
        Document(
          name=json_name,
          id=str(uuid4()),
          meta_data={"page": page_number},
          content=json.dumps(content),
        )
        for page_number, content in enumerate(json_contents, start=1)
      ]
      if self.chunk:
        chunked_documents = []
        for document in documents:
          chunked_documents.extend(self.chunk_document(document))
        return chunked_documents
      return documents
    except Exception as e:
      print(f"Error processing json file: {e}")
      raise ValueError("':' expected after '\"'")

  def _parse_json_content(self, content: str) -> Union[dict, list, Any]:
    """
    Parse JSON content with multiple fallback strategies.

    Args:
        content: Raw string content from the JSON file

    Returns:
        Parsed JSON content
    """
    content = content.strip()

    # Strategy 1: Try parsing as standard JSON
    try:
      return json.loads(content)
    except json.JSONDecodeError as e:
      print(f"Standard JSON parsing failed: {e}")

    # Strategy 2: Try parsing as JSONL (JSON Lines) format
    try:
      lines = content.split("\n")
      json_objects = []
      for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if line:  # Skip empty lines
          try:
            json_objects.append(json.loads(line))
          except json.JSONDecodeError:
            print(f"Failed to parse line {line_num}: {line[:50]}...")
            continue

      if json_objects:
        return json_objects
    except Exception as e:
      print(f"JSONL parsing failed: {e}")

    # Strategy 3: Try to extract the first valid JSON object
    try:
      decoder = json.JSONDecoder()
      obj, idx = decoder.raw_decode(content)
      print(f"Extracted first JSON object, ignoring content after index {idx}")
      return obj
    except json.JSONDecodeError as e:
      print(f"Partial JSON parsing failed: {e}")

    # Strategy 4: Try parsing as multiple concatenated JSON objects
    try:
      decoder = json.JSONDecoder()
      objects = []
      idx = 0
      while idx < len(content):
        content_slice = content[idx:].lstrip()
        if not content_slice:
          break
        try:
          obj, end_idx = decoder.raw_decode(content_slice)
          objects.append(obj)
          idx += len(content[idx:]) - len(content_slice) + end_idx
        except json.JSONDecodeError:
          break

      if objects:
        return objects
    except Exception as e:
      print(f"Multi-object JSON parsing failed: {e}")

    # If all strategies fail, raise the original error
    raise ValueError("Could not parse JSON content with any strategy")

  async def async_read(self, path: Union[Path, IO[Any]]) -> List[Document]:
    """Asynchronously read JSON files.

    Args:
        path (Union[Path, IO[Any]]): Path to a JSON file or a file-like object

    Returns:
        List[Document]: List of documents from the JSON file
    """
    return await asyncio.to_thread(self.read, path)
