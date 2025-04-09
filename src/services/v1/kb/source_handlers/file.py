from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from libs.s3.v1 import s3_client
from models import KBDocumentModel

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
    # Documents
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "md": "text/markdown",
    "mdx": "text/markdown",
    # Spreadsheets
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    # Presentations
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Emails
    "eml": "message/rfc822",
    "msg": "application/vnd.ms-outlook",
    # Web
    "html": "text/html",
    "htm": "text/html",
    "xml": "application/xml",
    # Others
    "epub": "application/epub+zip",
  }

  def __init__(self, config: Dict):
    super().__init__(config)
    self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB default
    self.temp_dir = Path("/tmp")

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
    try:
      metadata = document.source_metadata
      s3_key = metadata.get("s3_key")

      if not s3_key:
        raise ValueError("S3 key not found in metadata")

      # Download file from S3
      file_content = await s3_client.download_file(s3_key)
      temp_path = self.temp_dir / f"{document.id}"

      with open(temp_path, "wb") as f:
        f.write(file_content.read())

      # Get appropriate loader based on file type
      loader = self._get_loader(temp_path, metadata["file_type"])
      if not loader:
        raise ValueError(f"No loader available for file type: {metadata['file_type']}")

      # Extract content
      docs = await loader.aload()
      content = "\n\n".join(doc.page_content for doc in docs)

      return content

    except Exception as e:
      raise ValueError(f"Failed to extract content: {str(e)}")

    finally:
      # Cleanup temp file
      if temp_path.exists():
        temp_path.unlink()

  async def cleanup(self, document: KBDocumentModel, **kwargs) -> None:
    """Cleanup any temporary resources."""
    try:
      temp_path = self.temp_dir / f"{document.id}"
      if temp_path.exists():
        temp_path.unlink()
    except Exception as e:
      print(f"Cleanup error: {str(e)}")

  def _get_loader(self, file_path: Path, file_type: str):
    """Get appropriate loader based on file type."""
    from langchain_community.document_loaders import (
      CSVLoader,
      PyPDFLoader,
      TextLoader,
      UnstructuredEmailLoader,
      UnstructuredEPubLoader,
      UnstructuredExcelLoader,
      UnstructuredHTMLLoader,
      UnstructuredMarkdownLoader,
      UnstructuredPowerPointLoader,
      UnstructuredWordDocumentLoader,
      UnstructuredXMLLoader,
    )

    loaders = {
      "pdf": PyPDFLoader,
      "docx": UnstructuredWordDocumentLoader,
      "doc": UnstructuredWordDocumentLoader,
      "html": UnstructuredHTMLLoader,
      "htm": UnstructuredHTMLLoader,
      "md": UnstructuredMarkdownLoader,
      "mdx": UnstructuredMarkdownLoader,
      "xml": UnstructuredXMLLoader,
      "epub": UnstructuredEPubLoader,
      "csv": CSVLoader,
      "eml": UnstructuredEmailLoader,
      "msg": UnstructuredEmailLoader,
      "pptx": UnstructuredPowerPointLoader,
      "ppt": UnstructuredPowerPointLoader,
      "xlsx": UnstructuredExcelLoader,
      "xls": UnstructuredExcelLoader,
      "txt": TextLoader,
    }

    loader_class = loaders.get(file_type)
    return loader_class(str(file_path)) if loader_class else None
