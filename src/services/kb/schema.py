from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .model import DocumentType, ProcessingStatus


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


class KBDocumentCreate(BaseModel):
  """Knowledge base document create schema."""

  title: str = Field(..., min_length=1, max_length=200)
  description: Optional[str] = None
  doc_type: DocumentType
  file_type: str = Field(..., max_length=20)
  embedding_model: str = Field(..., max_length=50)


class KBDocumentUpdate(BaseModel):
  """Knowledge base document update schema."""

  title: Optional[str] = Field(None, min_length=1, max_length=200)
  description: Optional[str] = None
  processing_status: Optional[ProcessingStatus] = None


class KBDocumentResponse(BaseModel):
  """Knowledge base document response schema."""

  id: UUID
  title: str
  description: Optional[str]
  doc_type: DocumentType
  s3_key: Optional[str]
  download_url: Optional[str] = None
  original_filename: Optional[str]
  file_type: str
  file_size: Optional[int]
  kb_id: UUID
  last_processed_at: Optional[datetime]
  processing_status: ProcessingStatus

  class Config:
    from_attributes = True


class KnowledgeBaseDetailResponse(KnowledgeBaseResponse):
  """Knowledge base detail response schema."""

  documents: List[KBDocumentResponse]

  class Config:
    from_attributes = True


class DocumentChunk(BaseModel):
  """Document chunk schema."""

  chunk_id: str
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
