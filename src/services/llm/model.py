from sqlalchemy import UUID, Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class LLMModel(CRUD):
  """LLM model."""

  __tablename__ = "models"

  name: Mapped[str] = mapped_column(String(255), nullable=False)
  provider: Mapped[str] = mapped_column(String(100), nullable=False)
  version: Mapped[str] = mapped_column(String(50), nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  config: Mapped[dict] = mapped_column(JSONB, nullable=False)


class AgentsModel(CRUD):
  """Agents model."""

  __tablename__ = "agents"

  name: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  model_id: Mapped[UUID] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
