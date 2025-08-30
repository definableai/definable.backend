from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.models import Base


class MCPToolModel(Base):
  __tablename__ = "mcp_tools"
  id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
  name: Mapped[str] = mapped_column(String, nullable=False)
  slug: Mapped[str] = mapped_column(String, nullable=False)
  description: Mapped[str] = mapped_column(String, nullable=True)
  toolkit: Mapped[str] = mapped_column(String, nullable=False)
