from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class MCPToolModel(CRUD):
  __tablename__ = "mcp_tools"
  name: Mapped[str] = mapped_column(String, nullable=False)
  slug: Mapped[str] = mapped_column(String, nullable=False)
  description: Mapped[str] = mapped_column(String, nullable=True)
  toolkit_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("mcp_toolkits.id"), nullable=False)
  toolkit_rel = relationship("MCPToolkitModel", backref="tools")
