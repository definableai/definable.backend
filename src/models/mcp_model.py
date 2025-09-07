from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class MCPServerModel(CRUD):
  __tablename__ = "mcp_servers"
  name: Mapped[str] = mapped_column(String, nullable=False)
  toolkits: Mapped[list] = mapped_column(JSONB, nullable=True)
  auth_config_ids: Mapped[list] = mapped_column(JSONB, nullable=True)
  auth_scheme: Mapped[str] = mapped_column(String, nullable=True)
  expected_input_fields: Mapped[list] = mapped_column(JSONB, nullable=True)
  allowed_tools: Mapped[list] = mapped_column(JSONB, nullable=True)
  server_instance_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MCPToolkitModel(CRUD):
  __tablename__ = "mcp_toolkits"
  name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
  slug: Mapped[str] = mapped_column(String, nullable=False, unique=True)
  logo: Mapped[str] = mapped_column(String, nullable=True)


class MCPSessionModel(CRUD):
  __tablename__ = "mcp_sessions"
  instance_id: Mapped[str] = mapped_column(String, nullable=False)
  mcp_server_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
  org_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
  connected_account_id: Mapped[str] = mapped_column(String, nullable=False)
  status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
