import uuid as uuid_pkg
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models import Base


class MCPServerModel(Base):
  __tablename__ = "mcp_servers"
  id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  name: Mapped[str] = mapped_column(String, nullable=False)
  mcp_url: Mapped[str] = mapped_column(String, nullable=False)
  auth_config_ids: Mapped[list] = mapped_column(JSONB, nullable=True)
  toolkits: Mapped[list] = mapped_column(JSONB, nullable=True)
  allowed_tools: Mapped[list] = mapped_column(JSONB, nullable=True)
  server_instance_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class MCPSessionModel(Base):
  __tablename__ = "mcp_sessions"
  id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  instance_id: Mapped[str] = mapped_column(String, nullable=False)
  mcp_server_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"), nullable=False)
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
