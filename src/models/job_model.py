from datetime import datetime
from enum import IntEnum
from typing import Dict, Optional, Any
from uuid import UUID

from sqlalchemy import ForeignKey, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class JobStatus(IntEnum):
  """Job status enum."""

  PENDING = 0
  PROCESSING = 1
  COMPLETED = 2
  FAILED = 3
  CANCELLED = 4


class JobModel(CRUD):
  """Job model for tracking background tasks."""

  __tablename__ = "jobs"

  name: Mapped[str] = mapped_column(String(100), nullable=False)
  description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  created_by: Mapped[UUID] = mapped_column(PostgresUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=JobStatus.PENDING)
  message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
  parent_job_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  parent_job: Mapped[Optional["JobModel"]] = relationship("JobModel", remote_side="JobModel.id", back_populates="child_jobs")
  child_jobs: Mapped[list["JobModel"]] = relationship("JobModel", back_populates="parent_job")
