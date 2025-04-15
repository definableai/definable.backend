from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class UserModel(CRUD):
  """User model."""

  __tablename__ = "users"

  email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
  password: Mapped[str] = mapped_column(String(64), nullable=False)
  first_name: Mapped[str] = mapped_column(String(50), nullable=False)
  last_name: Mapped[str] = mapped_column(String(50), nullable=False)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)

  # Relationships
  sent_invitations = relationship("InvitationModel", foreign_keys="InvitationModel.invited_by", back_populates="inviter")

  @property
  def full_name(self) -> str:
    """Get full name."""
    return f"{self.first_name} {self.last_name}"

  def __repr__(self) -> str:
    """String representation."""
    return f"<User {self.email}>"
