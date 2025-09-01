from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD, Base


class LLMCategoryModel(CRUD):
  """LLM category model."""

  __tablename__ = "llm_category"

  name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
  description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  display_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  model_categories: Mapped[list["LLMModelCategoryModel"]] = relationship("LLMModelCategoryModel", back_populates="category")

  def __repr__(self) -> str:
    return f"<LLMCategory(id={self.id}, name={self.name}, description={self.description}, is_active={self.is_active})>"


class LLMModelCategoryModel(Base):
  """LLM model categories junction table (many-to-many)."""

  __tablename__ = "llm_model_categories"
  __table_args__ = (UniqueConstraint("model_id", "category_id", name="uq_model_category"),)

  id: Mapped[UUID] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))
  model_id: Mapped[UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
  category_id: Mapped[UUID] = mapped_column(ForeignKey("llm_category.id", ondelete="CASCADE"), nullable=False)
  is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
  created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

  # Relationships
  category: Mapped["LLMCategoryModel"] = relationship("LLMCategoryModel", back_populates="model_categories")

  def __repr__(self) -> str:
    return f"<LLMModelCategory(id={self.id}, model_id={self.model_id}, category_id={self.category_id}, is_primary={self.is_primary})>"
