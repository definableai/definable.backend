from datetime import UTC, datetime
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


class AgentAnalyticsModel(CRUD):
  """Model for storing analytics data related to agent interactions."""

  __tablename__ = "agent_analytics"

  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, comment="Foreign key referencing the agent")
  session_id: Mapped[UUID] = mapped_column(
    ForeignKey("messages.id", ondelete="SET NULL"), nullable=False, comment="Foreign key referencing the chat session (messages.id)"
  )
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, comment="Foreign key referencing the user")
  org_id: Mapped[UUID] = mapped_column(
    ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, comment="Foreign key referencing the organization"
  )
  memory: Mapped[dict] = mapped_column(JSONB, nullable=True, comment="Memory data associated with the agent")
  agent_data: Mapped[dict] = mapped_column(JSONB, nullable=True, comment="Data specific to the agent")
  session_data: Mapped[dict] = mapped_column(JSONB, nullable=True, comment="Data specific to the session")

  # Update these to ensure timestamps are stored without timezone information
  created_at: Mapped[datetime] = mapped_column(
    default=lambda: datetime.now(UTC).replace(tzinfo=None), nullable=False, comment="Timestamp when the record was created"
  )
  updated_at: Mapped[datetime] = mapped_column(
    default=lambda: datetime.now(UTC).replace(tzinfo=None),
    onupdate=lambda: datetime.now(UTC).replace(tzinfo=None),
    nullable=False,
    comment="Timestamp when the record was last updated",
  )

  def __repr__(self) -> str:
    return f"<AgentAnalytics(id={self.id}, agent_id={self.agent_id}, session_id={self.session_id})>"
