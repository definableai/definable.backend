from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class LogType(str, Enum):
  """Agent deployment log types."""

  INIT = "INIT"
  BUILD = "BUILD"
  DEPLOY = "DEPLOY"
  FINAL = "FINAL"


class LogLevel(str, Enum):
  """Log levels for deployment logs."""

  INFO = "INFO"
  WARN = "WARN"
  ERROR = "ERROR"
  DEBUG = "DEBUG"


class TraceStatus(str, Enum):
  """Status for deployment traces."""

  STARTED = "STARTED"
  SUCCESS = "SUCCESS"
  ERROR = "ERROR"
  TIMEOUT = "TIMEOUT"


class AgentDeploymentLogModel(CRUD):
  """Agent deployment log model."""

  __tablename__ = "agent_deployment_logs"

  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  api_key_id: Mapped[UUID] = mapped_column(ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False)
  deployment_id: Mapped[str] = mapped_column(String(255), nullable=False)
  log_type: Mapped[str] = mapped_column(String(20), nullable=False)
  log_level: Mapped[str] = mapped_column(String(10), nullable=False)
  message: Mapped[str] = mapped_column(Text, nullable=False)
  log_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
  timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
  source: Mapped[str] = mapped_column(String(100), nullable=False)

  def __repr__(self) -> str:
    return (
      f"<AgentDeploymentLog(id={self.id}, agent_id={self.agent_id}, "
      f"user_id={self.user_id}, api_key_id={self.api_key_id}, deployment_id={self.deployment_id}, "
      f"log_type={self.log_type}, log_level={self.log_level}, timestamp={self.timestamp})>"
    )


class AgentDeploymentTraceModel(CRUD):
  """Agent deployment trace model."""

  __tablename__ = "agent_deployment_traces"

  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  api_key_id: Mapped[UUID] = mapped_column(ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False)
  deployment_id: Mapped[str] = mapped_column(String(255), nullable=False)
  trace_id: Mapped[str] = mapped_column(String(255), nullable=False)
  span_id: Mapped[str] = mapped_column(String(255), nullable=False)
  parent_span_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
  operation_name: Mapped[str] = mapped_column(String(255), nullable=False)
  start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
  end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
  duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
  status: Mapped[str] = mapped_column(String(20), nullable=False)
  tags: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
  trace_logs: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})

  def __repr__(self) -> str:
    return (
      f"<AgentDeploymentTrace(id={self.id}, agent_id={self.agent_id}, "
      f"user_id={self.user_id}, api_key_id={self.api_key_id}, deployment_id={self.deployment_id}, "
      f"trace_id={self.trace_id}, span_id={self.span_id}, operation_name={self.operation_name}, "
      f"status={self.status})>"
    )
