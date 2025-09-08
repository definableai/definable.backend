from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class UserModel(CRUD):
  """User model."""

  __tablename__ = "users"

  stytch_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
  email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
  password: Mapped[str] = mapped_column(String(64), nullable=True)
  first_name: Mapped[str] = mapped_column(String(50), nullable=True)
  last_name: Mapped[str] = mapped_column(String(50), nullable=True)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)

  @property
  def full_name(self) -> str:
    """Get full name."""
    return f"{self.first_name} {self.last_name}"

  def __repr__(self) -> str:
    """String representation."""
    return f"<User {self.email}>"
