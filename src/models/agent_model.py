from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD, Base


class AgentToolModel(Base):
  """Agent tool association model."""

  __tablename__ = "agent_tools"
  __table_args__ = (UniqueConstraint("agent_id", "tool_id", name="uq_agent_tool"),)

  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True)
  tool_id: Mapped[UUID] = mapped_column(ForeignKey("tools.id", ondelete="CASCADE"), primary_key=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  added_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentModel(CRUD):
  """Agent model."""

  __tablename__ = "agents"

  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  name: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  model_id: Mapped[UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
  version: Mapped[str] = mapped_column(String(255), nullable=False)
