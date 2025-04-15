from datetime import datetime
from enum import IntEnum
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class InvitationStatus(IntEnum):
  """Invitation status enum."""

  PENDING = 0
  ACCEPTED = 1
  REJECTED = 2
  EXPIRED = 3


class InvitationModel(CRUD):
  """Invitation model for managing organization invitations."""

  __tablename__ = "invites"
  __table_args__ = {"extend_existing": True}
  # Removing the unique constraint to allow multiple invitations
  # __table_args__ = (UniqueConstraint("organization_id", "invitee_email", name="uq_invitation_org_email"),)

  organization_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  role_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
  invitee_email: Mapped[str] = mapped_column(String(255), nullable=False)
  invited_by: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=InvitationStatus.PENDING)
  expiry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

  # Relationships
  organization = relationship("OrganizationModel", back_populates="invitations")
  role = relationship("RoleModel", back_populates="invitations")
  inviter = relationship("UserModel", foreign_keys=[invited_by], back_populates="sent_invitations")

  def __repr__(self) -> str:
    return f"<InvitationModel(id={self.id}, email={self.invitee_email}, status={self.status})>"
