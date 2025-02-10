from datetime import datetime
from enum import IntEnum
from typing import Optional
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class DocumentType(IntEnum):
  """Document type enum."""

  FILE = 0
  URL = 1


class ProcessingStatus(IntEnum):
  """Processing status enum."""

  PENDING = 0
  PROCESSING = 1
  COMPLETED = 2
  FAILED = 3


class KnowledgeBaseModel(CRUD):
  """Knowledge base model."""

  __tablename__ = "knowledge_base"

  name: Mapped[str] = mapped_column(String(100), nullable=False)
  collection_id: Mapped[UUID] = mapped_column(PostgresUUID, nullable=False, unique=True)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  embedding_model: Mapped[str] = mapped_column(String(50), nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)


class KBDocumentModel(CRUD):
  """Knowledge base document model."""

  __tablename__ = "kb_documents"

  title: Mapped[str] = mapped_column(String(200), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  doc_type: Mapped[DocumentType] = mapped_column(SmallInteger, nullable=False)
  s3_key: Mapped[str] = mapped_column(String(500), nullable=True)
  original_filename: Mapped[str] = mapped_column(String(255), nullable=True)
  file_type: Mapped[str] = mapped_column(String(20), nullable=False)
  file_size: Mapped[int] = mapped_column(BigInteger, nullable=True)
  kb_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False)
  last_processed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
  processing_status: Mapped[int] = mapped_column(SmallInteger, server_default="0", nullable=False)
