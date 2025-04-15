from datetime import datetime
from enum import IntEnum
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class DocumentStatus(IntEnum):
  """Document processing status enum."""

  PENDING = 0
  PROCESSING = 1
  COMPLETED = 2
  FAILED = 3


class KBFolder(CRUD):
  """Knowledge base folder model."""

  __tablename__ = "kb_folders"

  name: Mapped[str] = mapped_column(String(100), nullable=False)
  parent_id: Mapped[UUID] = mapped_column(ForeignKey("kb_folders.id", ondelete="CASCADE"), nullable=True)
  folder_info: Mapped[dict] = mapped_column(JSONB, nullable=False)
  kb_id: Mapped[UUID] = mapped_column(PostgresUUID, ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False)
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)


class SourceTypeModel(CRUD):
  """Source type model."""

  __tablename__ = "source_types"

  id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
  name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
  handler_class: Mapped[str] = mapped_column(String(100), nullable=False)
  config_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
  metadata_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
  created_at: Mapped[datetime] = mapped_column(nullable=False)


class KnowledgeBaseModel(CRUD):
  """Knowledge base model."""

  __tablename__ = "knowledge_base"

  name: Mapped[str] = mapped_column(String(100), nullable=False)
  collection_id: Mapped[UUID] = mapped_column(PostgresUUID, nullable=False, unique=True)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  embedding_model: Mapped[str] = mapped_column(String(50), nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)


class KBDocumentModel(CRUD):
  """Knowledge base document model."""

  __tablename__ = "kb_documents"

  title: Mapped[str] = mapped_column(String(200), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  kb_id: Mapped[UUID] = mapped_column(ForeignKey("knowledge_base.id", ondelete="CASCADE"), nullable=False)
  folder_id: Mapped[UUID] = mapped_column(ForeignKey("kb_folders.id", ondelete="CASCADE"), nullable=True)
  source_type_id: Mapped[int] = mapped_column(ForeignKey("source_types.id", ondelete="CASCADE"), nullable=False)
  source_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False)
  content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  extraction_status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=DocumentStatus.PENDING)
  indexing_status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=DocumentStatus.PENDING)
  error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  extraction_completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
  indexing_completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)
