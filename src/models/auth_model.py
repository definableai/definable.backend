from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class UserModel(CRUD):
  """User model."""

  __tablename__ = "users"

  stytch_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
  email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
  password: Mapped[str] = mapped_column(String(64), nullable=True)
  first_name: Mapped[str] = mapped_column(String(50), nullable=True)
  last_name: Mapped[str] = mapped_column(String(50), nullable=True)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)

  # Relationships
  sent_invitations = relationship("InvitationModel", foreign_keys="InvitationModel.invited_by", back_populates="inviter")

  @property
  def full_name(self) -> str:
    """Get full name."""
    return f"{self.first_name} {self.last_name}"

  def __repr__(self) -> str:
    """String representation."""
    return f"<User {self.email}>"
