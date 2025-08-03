from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD, Base


class AgentToolModel(Base):
  """Agent tool association model."""

  __tablename__ = "agent_tools"
  __table_args__ = (UniqueConstraint("agent_id", "tool_id", name="uq_agent_tool"),)

  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True)
  tool_id: Mapped[UUID] = mapped_column(ForeignKey("tools.id", ondelete="CASCADE"), primary_key=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  added_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentCategoryModel(Base):
  """Agent category model."""

  __tablename__ = "agents_category"

  id: Mapped[UUID] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))
  name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
  updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

  agents: Mapped[list["AgentModel"]] = relationship("AgentModel", back_populates="category")

  def __repr__(self) -> str:
    return f"<AgentCategory(id={self.id}, name={self.name}, description={self.description}, is_active={self.is_active})>"


class AgentModel(CRUD):
  """Agent model."""

  __tablename__ = "agents"

  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  name: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  model_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=True)
  category_id: Mapped[UUID] = mapped_column(ForeignKey("agents_category.id", ondelete="CASCADE"), nullable=True)
  properties: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
  version: Mapped[str] = mapped_column(String(255), nullable=False)
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  category: Mapped["AgentCategoryModel"] = relationship("AgentCategoryModel", back_populates="agents")

  def __repr__(self) -> str:
    return (
      f"<Agent(id={self.id}, name={self.name}, model_id={self.model_id}), "
      f"organization_id={self.organization_id}, user_id={self.user_id}, "
      f"description={self.description}, is_active={self.is_active}, "
      f"settings={self.settings}, version={self.version}, "
      f"updated_at={self.updated_at})>"
    )
