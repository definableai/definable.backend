from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class PublicUploadModel(CRUD):
  """Model for public uploads."""

  __tablename__ = "public_uploads"

  filename: Mapped[str] = mapped_column(String(255), nullable=False)
  content_type: Mapped[str] = mapped_column(String(100), nullable=False)
  url: Mapped[str] = mapped_column(String(1024), nullable=False)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)
