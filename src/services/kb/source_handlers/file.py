from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from libs.agno.document.reader.base import Reader
from libs.agno.document.reader.csv_reader import CSVReader
from libs.agno.document.reader.docx_reader import DocxReader
from libs.agno.document.reader.json_reader import JSONReader
from libs.agno.document.reader.markdown_reader import MarkdownReader

# Import all available readers
from libs.agno.document.reader.pdf_reader import PDFReader
from libs.agno.document.reader.text_reader import TextReader
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


class FileReaderFactory:
  """Factory class for selecting appropriate file readers."""

  # Mapping of file extensions to reader classes
  READER_MAPPING = {
    "pdf": PDFReader,
    "docx": DocxReader,
    "csv": CSVReader,
    "md": MarkdownReader,
    "markdown": MarkdownReader,
    "txt": TextReader,
    "text": TextReader,
    "json": JSONReader,
  }

  @classmethod
  def get_reader(cls, file_type: str, **kwargs) -> Reader:
    """Get appropriate reader for file type."""
    file_ext = file_type.lower()

    if file_ext not in cls.READER_MAPPING:
      raise ValueError(f"Unsupported file type: {file_type}")

    reader_class = cls.READER_MAPPING[file_ext]
    return reader_class(chunk=False, **kwargs)

  @classmethod
  def supports_file_type(cls, file_type: str) -> bool:
    """Check if file type is supported."""
    return file_type.lower() in cls.READER_MAPPING


class FileSourceHandler(BaseSourceHandler):
  """Handler for file-based sources."""

  ALLOWED_EXTENSIONS = {
    # Document formats
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "csv": "text/csv",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "txt": "text/plain",
    "text": "text/plain",
    "json": "application/json",
    # Add more as needed
  }

  def __init__(self, config: Dict):
    super().__init__(config)
    self.max_file_size = config.get("max_file_size", 20 * 1024 * 1024)  # 20MB default
    self.temp_dir = Path("/tmp")

  async def validate_metadata(self, metadata: Dict, **kwargs) -> bool:
    """Validate file metadata."""
    try:
      file_metadata = FileMetadata(**metadata)

      # Check if file type is supported by our readers
      if not FileReaderFactory.supports_file_type(file_metadata.file_type):
        raise ValueError(f"Unsupported file type: {file_metadata.file_type}")

      # Check file size
      if file_metadata.size > self.max_file_size:
        raise ValueError(f"File too large. Maximum size is {self.max_file_size} bytes")

      # Check MIME type
      expected_mime = self.ALLOWED_EXTENSIONS.get(file_metadata.file_type)
      if expected_mime and file_metadata.mime_type != expected_mime:
        raise ValueError(f"Invalid MIME type. Expected {expected_mime}")

      return True

    except Exception as e:
      raise ValueError(f"Invalid file metadata: {str(e)}")

  async def preprocess(self, document: KBDocumentModel, **kwargs) -> None:
    """Preprocess the file (upload to S3)."""
    pass

  async def extract_content(self, document: KBDocumentModel, **kwargs) -> str:
    """Extract content from the file using appropriate reader."""
    temp_path = None
    try:
      metadata = document.source_metadata
      s3_key = metadata.get("s3_key")
      file_type = metadata.get("file_type")

      if not s3_key:
        raise ValueError("S3 key not found in metadata")

      if not file_type:
        raise ValueError("File type not found in metadata")

      # Download file from S3
      file_content = await s3_client.download_file(s3_key)
      temp_path = self.temp_dir / f"{document.id}"

      with open(temp_path, "wb") as f:
        f.write(file_content.read())

      # Get appropriate reader for file type
      reader = FileReaderFactory.get_reader(file_type)

      # Extract content using the reader
      documents = await reader.async_read(temp_path)

      # Combine all document content
      content = "\n\n".join(doc.content for doc in documents)

      print("content : ", content)

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
