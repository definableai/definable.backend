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
  toolkits: Mapped[list] = mapped_column(JSONB, nullable=True)
  auth_config_ids: Mapped[list] = mapped_column(JSONB, nullable=True)
  auth_scheme: Mapped[str] = mapped_column(String, nullable=True)
  expected_input_fields: Mapped[list] = mapped_column(JSONB, nullable=True)
  allowed_tools: Mapped[list] = mapped_column(JSONB, nullable=True)
  server_instance_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class MCPToolkitModel(Base):
  __tablename__ = "mcp_toolkits"
  id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
  slug: Mapped[str] = mapped_column(String, nullable=False, unique=True)
  logo: Mapped[str] = mapped_column(String, nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class MCPSessionModel(Base):
  __tablename__ = "mcp_sessions"
  id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  instance_id: Mapped[str] = mapped_column(String, nullable=False)
  mcp_server_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"), nullable=False)
  user_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
  org_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class MCPUserModel(Base):
  __tablename__ = "mcp_users"
  id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  user_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
  server_id: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"), nullable=False)
  composio_user_id: Mapped[str] = mapped_column(String, nullable=False)
  connected_account_id: Mapped[str] = mapped_column(String, nullable=True)
  connection_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
  created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
  updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now(timezone.utc))
