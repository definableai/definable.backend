from pathlib import Path
from typing import Dict, Optional
import tempfile

from pydantic import BaseModel, Field

from libs.s3.v1 import s3_client
from models import KBDocumentModel

from ..loaders import DoclingFileLoader
from .base import BaseSourceHandler


class FileMetadata(BaseModel):
  """File metadata schema."""

  file_type: str = Field(..., description="File extension")
  original_filename: str = Field(..., description="Original file name")
  size: int = Field(..., description="File size in bytes")
  mime_type: str = Field(..., description="File MIME type")
  s3_key: Optional[str] = Field(None, description="S3 storage key")


class FileSourceHandler(BaseSourceHandler):
  """Handler for file-based sources."""

  ALLOWED_EXTENSIONS = {
    # Document formats
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html",
    "htm": "text/html",
    "md": "text/markdown",
    "asciidoc": "text/asciidoc",
    "adoc": "text/asciidoc",
    "csv": "text/csv",
    # Image formats
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    # XML formats
    "xml": "application/xml",
    "nxml": "application/xml",
    "uspto": "application/xml",
  }

  def __init__(self, config: Dict):
    super().__init__(config)
    self.max_file_size = config.get("max_file_size", 20 * 1024 * 1024)  # 20MB default
    self.temp_dir = Path(tempfile.gettempdir())

  async def validate_metadata(self, metadata: Dict, **kwargs) -> bool:
    """Validate file metadata."""
    try:
      file_metadata = FileMetadata(**metadata)

      # Check file extension
      if file_metadata.file_type not in self.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_metadata.file_type}")

      # Check file size
      if file_metadata.size > self.max_file_size:
        raise ValueError(f"File too large. Maximum size is {self.max_file_size} bytes")

      # Check MIME type
      expected_mime = self.ALLOWED_EXTENSIONS[file_metadata.file_type]
      if file_metadata.mime_type != expected_mime:
        raise ValueError(f"Invalid MIME type. Expected {expected_mime}")

      return True

    except Exception as e:
      raise ValueError(f"Invalid file metadata: {str(e)}")

  async def preprocess(self, document: KBDocumentModel, **kwargs) -> None:
    """Preprocess the file (upload to S3)."""
    pass

  async def extract_content(self, document: KBDocumentModel, **kwargs) -> str:
    """Extract content from the file."""
    temp_path = None
    try:
      metadata = document.source_metadata
      s3_key = metadata.get("s3_key")

      if not s3_key:
        raise ValueError("S3 key not found in metadata")

      # Download file from S3 asynchronously
      file_content = await s3_client.download_file(s3_key)
      temp_path = self.temp_dir / f"{document.id}"

      # Write file asynchronously
      import aiofiles  # type: ignore

      async with aiofiles.open(temp_path, "wb") as f:
        await f.write(file_content.read())

      # Use Docling loader with required arguments
      loader = DoclingFileLoader(kb_id=document.kb_id, document=document)
      docs = []
      async for doc in loader.load():
        docs.append(doc)
      content = "\n\n".join(doc.page_content for doc in docs)

      return content

    except Exception as e:
      raise ValueError(f"Failed to extract content: {str(e)}")

    finally:
      # Cleanup temp file
      if temp_path and temp_path.exists():
        temp_path.unlink()

  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    try:
      temp_path = self.temp_dir / f"{document.id}"
      if temp_path.exists():
        temp_path.unlink()
    except Exception as e:
      print(f"Cleanup error: {str(e)}")
