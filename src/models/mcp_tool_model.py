from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class MCPToolModel(CRUD):
  __tablename__ = "mcp_tools"
  name: Mapped[str] = mapped_column(String, nullable=False)
  slug: Mapped[str] = mapped_column(String, nullable=False)
  description: Mapped[str] = mapped_column(String, nullable=True)
  mcp_server_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_servers.id"), nullable=False)
