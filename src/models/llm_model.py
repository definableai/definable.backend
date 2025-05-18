from sqlalchemy import Boolean, String
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
  props: Mapped[dict] = mapped_column(JSONB, nullable=False)
