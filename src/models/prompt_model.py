from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class PromptCategoryModel(CRUD):
  """Prompt category model."""

  __tablename__ = "prompt_categories"

  name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  icon_url: Mapped[str] = mapped_column(String(255), nullable=True)
  display_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)

  # Relationships
  prompts = relationship("PromptModel", back_populates="category", cascade="all, delete-orphan")


class PromptModel(CRUD):
  """Prompt model."""

  __tablename__ = "prompts"

  category_id: Mapped[UUID] = mapped_column(ForeignKey("prompt_categories.id", ondelete="CASCADE"), nullable=False)
  creator_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  title: Mapped[str] = mapped_column(String(200), nullable=False)
  content: Mapped[str] = mapped_column(Text, nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  is_public: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
  is_featured: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)

  # Relationships
  category = relationship("PromptCategoryModel", back_populates="prompts")
  creator = relationship("UserModel", lazy="select")
  organization = relationship("OrganizationModel", lazy="select")
