from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class APIKeyModel(CRUD):
  """API Key model for authentication."""

  __tablename__ = "api_keys"

  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
  agent_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=True, index=True)
  token_type: Mapped[str] = mapped_column(String(20), nullable=False, default="api", index=True)  # "auth" or "api"
  api_key_token: Mapped[str] = mapped_column(String(500), nullable=False, unique=True, index=True)  # Plain JWT
  api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)  # Hash for indexing
  name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
  permissions: Mapped[dict] = mapped_column(JSONB, nullable=False, default={})
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
  expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
  last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # No relationships needed for the simple verification use case

  # Indexes for performance
  __table_args__ = (
    Index("idx_api_keys_user_active", "user_id", "is_active"),
    Index("idx_api_keys_agent_active", "agent_id", "is_active"),
    Index("idx_api_keys_hash_active", "api_key_hash", "is_active"),
  )

  def __repr__(self) -> str:
    """String representation."""
    return (
      f"<APIKey(id={self.id}, user_id={self.user_id}, agent_id={self.agent_id}, "
      f"name={self.name}, is_active={self.is_active}, expires_at={self.expires_at})>"
    )

  @property
  def is_expired(self) -> bool:
    """Check if the API key is expired."""
    if self.expires_at is None:
      return False
    return datetime.utcnow() > self.expires_at

  def update_last_used(self) -> None:
    """Update the last used timestamp."""
    self.last_used_at = datetime.utcnow()
