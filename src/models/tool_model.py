from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class ToolCategoryModel(CRUD):
  """Tool category model."""

  __tablename__ = "tools_category"
  __table_args__ = (UniqueConstraint("name", name="uq_tool_category_name"),)

  name: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)


class ToolModel(CRUD):
  """Tool model."""

  __tablename__ = "tools"
  __table_args__ = (UniqueConstraint("name", "version", name="uq_tool_name_version"),)

  name: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  category_id: Mapped[UUID] = mapped_column(ForeignKey("tools_category.id", ondelete="CASCADE"), nullable=False)
  logo_url: Mapped[str] = mapped_column(String(255), nullable=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"), nullable=False)
  version: Mapped[str] = mapped_column(String(50), nullable=False)
  is_public: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
  is_verified: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
  inputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
  outputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
  configuration: Mapped[dict] = mapped_column(JSONB, nullable=True)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
  generated_code: Mapped[str] = mapped_column(Text, nullable=True)
  updated_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=datetime.now(timezone.utc),
    onupdate=datetime.now(timezone.utc),
    server_default=text("CURRENT_TIMESTAMP"),
  )
