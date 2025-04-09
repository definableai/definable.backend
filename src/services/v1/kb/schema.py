from datetime import datetime
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional
from uuid import UUID

from fastapi import File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from models import DocumentStatus


class AllowedFileExtension(str, Enum):
  """Allowed file extensions for knowledge base documents."""

  TXT = "txt"
  MARKDOWN = "md"
  MDX = "mdx"
  PDF = "pdf"
  HTML = "html"
  HTM = "htm"
  XLSX = "xlsx"
  XLS = "xls"
  DOCX = "docx"
  CSV = "csv"
  EML = "eml"
  MSG = "msg"
  PPTX = "pptx"
  PPT = "ppt"
  XML = "xml"
  EPUB = "epub"


def validate_file_extension(filename: str) -> str:
  """Validate file extension."""
  # Get the file extension (lowercase)
  ext = f"{filename.split('.')[-1].lower()}" if "." in filename else ""

  # Check if extension is allowed
  if ext not in [e.value for e in AllowedFileExtension]:
    allowed_extensions = ", ".join(sorted(e.value for e in AllowedFileExtension))
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed file types are: {allowed_extensions}")

  return ext


class DocumentChunkCreate(BaseModel):
  """Schema for creating a new document chunk."""

  content: str = Field(..., description="Content of the chunk", min_length=1)


class KnowledgeBaseSettings(BaseModel):
  """Knowledge base settings schema."""

  embedding_model: str = Field(..., max_length=50)
  max_chunk_size: int = Field(default=1000, gt=0)
  chunk_overlap: int = Field(default=100, gt=0)
  separator: str = Field(default="\n\n", max_length=10)
  version: int = Field(default=1, gt=0)


class KnowledgeBaseCreate(BaseModel):
  """Knowledge base create schema."""

  name: str = Field(..., min_length=1, max_length=100)
  settings: KnowledgeBaseSettings


class KnowledgeBaseUpdate(BaseModel):
  """Knowledge base update schema."""

  name: Optional[str] = Field(None, min_length=1, max_length=100)


class KnowledgeBaseResponse(BaseModel):
  """Knowledge base response schema."""

  id: UUID
  name: str
  collection_id: UUID
  organization_id: UUID
  user_id: UUID
  created_at: datetime

  class Config:
    from_attributes = True


# Base Models
class DocumentBase(BaseModel):
  """Base document fields."""

  title: str = Field(..., min_length=1, max_length=200)
  description: str = Field(default="")


# File Document
class FileDocumentData(BaseModel):
  """Form data for file uploads."""

  title: Annotated[str, Form(..., description="Title of the document")]
  description: Annotated[str, Form(..., description="Description of the document")]
  file: Annotated[UploadFile, File(..., description="File to upload")]
  source_id: Annotated[Optional[UUID], Form(..., description="Source ID")] = None

  def get_metadata(self) -> Dict:
    """Generate metadata for file document."""
    if not self.file.filename:
      raise ValueError("File must have a name")

    return {
      "file_type": self.file.filename.split(".")[-1].lower(),
      "original_filename": self.file.filename,
      "size": self.file.size,
      "mime_type": self.file.content_type,
    }


async def validate_file_document_data(
  title: Annotated[str, Form(min_length=1, max_length=200)],
  file: Annotated[UploadFile, File(description="File to upload")],
  description: Annotated[str, Form()] = "",
  source_id: Annotated[Optional[UUID], Form(..., description="Source ID")] = None,
) -> FileDocumentData:
  """Validate file document form data."""
  return FileDocumentData(title=title, description=description, file=file, source_id=source_id)


class ScrapeOptions(BaseModel):
  """Scrape options schema."""

  # waitFor: Optional[int] = Field(default=0, ge=0, description="Wait time in milliseconds")
  excludeTags: Optional[List[str]] = Field(default=[""], description="Tags to exclude")
  includeTags: Optional[List[str]] = Field(default=[""], description="Tags to include only")
  onlyMainContent: bool = Field(description="Extract only main content")
  formats: List[str] = Field(description="Formats to extract")  # Required field

  # don't allow extra fields
  # class Config:
  #   extra = "forbid"


class CrawlerOptions(BaseModel):
  """Crawler options schema."""

  maxDepth: int = Field(default=0, ge=0, description="Maximum crawl depth")
  limit: int = Field(default=10, ge=1, le=1000, description="Maximum pages to crawl")
  includePaths: Optional[List[str]] = Field(default=[], description="Paths to include")
  excludePaths: Optional[List[str]] = Field(default=[], description="Paths to exclude")
  ignoreSitemap: bool = Field(default=False, description="Ignore sitemaps")
  allowBackwardLinks: bool = Field(default=False, description="Include all backlinks")
  scrapeOptions: ScrapeOptions = Field(..., description="Scrape options")  # Required field

  # class Config:
  #   extra = "forbid"


class MapOptions(BaseModel):
  """Map options schema."""

  includeSubdomains: bool = Field(default=True, description="Include subdomains")
  ignoreSitemap: bool = Field(default=False, description="Ignore sitemaps")


# URL Document
class URLDocumentData(DocumentBase):
  """JSON data for URL processing."""

  url: str = Field(..., description="Single URL to scrape")
  operation: Literal["scrape", "crawl", "map"] = Field(..., description="Operation to perform")
  source_id: Optional[UUID] = Field(default=None, description="Source ID")
  settings: ScrapeOptions | CrawlerOptions | MapOptions = Field(..., description="Settings for the operation")

  @model_validator(mode="after")
  def validate_operation_settings(self) -> "URLDocumentData":
    """Validate and convert settings based on operation."""
    try:
      if self.operation == "scrape":
        self.settings = ScrapeOptions.model_validate(self.settings)
      elif self.operation == "crawl":
        self.settings = CrawlerOptions.model_validate(self.settings)
      elif self.operation == "map":
        self.settings = MapOptions.model_validate(self.settings)
      else:
        raise ValueError(f"Invalid operation: {self.operation}")
    except Exception as e:
      raise ValueError(f"Invalid settings for operation {self.operation}: {str(e)}")
    return self

  def get_metadata(self) -> Dict:
    """Generate metadata for URL document."""
    return {"url": self.url, "operation": self.operation, "settings": self.settings.model_dump()}


class KBDocumentUpdate(BaseModel):
  """Knowledge base document update schema."""

  title: Optional[str] = Field(None, min_length=1, max_length=200)
  description: Optional[str] = None
  processing_status: Optional[DocumentStatus] = None


class KBDocumentResponse(BaseModel):
  """Knowledge base document response schema."""

  id: UUID
  title: str
  description: Optional[str]
  kb_id: UUID
  source_type_id: int
  source_id: Optional[UUID]
  source_metadata: Dict = Field(..., description="Source-specific metadata")
  content: Optional[str]
  extraction_status: DocumentStatus
  indexing_status: DocumentStatus
  error_message: Optional[str]
  extraction_completed_at: Optional[datetime]
  indexing_completed_at: Optional[datetime]

  class Config:
    from_attributes = True


class KnowledgeBaseDetailResponse(KnowledgeBaseResponse):
  """Knowledge base detail response schema."""

  documents: List[KBDocumentResponse]

  class Config:
    from_attributes = True


class DocumentChunk(BaseModel):
  """Document chunk schema."""

  id: Optional[UUID] = None
  chunk_id: int
  content: str
  metadata: dict
  score: Optional[float] = None


class KBDocumentChunksResponse(BaseModel):
  """Knowledge base document chunks response schema."""

  document_id: UUID
  title: str
  chunks: List[DocumentChunk]
  total_chunks: int

  class Config:
    from_attributes = True


class DocumentChunkUpdate(BaseModel):
  """Schema for updating a document chunk."""

  chunk_id: str = Field(..., description="Chunk ID to update")
  content: str = Field(..., description="New content for the chunk", min_length=1)


class DocumentChunkDelete(BaseModel):
  """Schema for deleting document chunks."""

  chunk_ids: List[str]
