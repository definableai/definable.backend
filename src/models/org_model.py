from uuid import UUID

from sqlalchemy import Boolean, Enum, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class OrganizationModel(CRUD):
  """Organization model."""

  __tablename__ = "organizations"

  name: Mapped[str] = mapped_column(String(100), nullable=False)
  slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=True, server_default="{}")
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)

  # Relationships
  invitations = relationship("InvitationModel", back_populates="organization", cascade="all, delete-orphan")


class OrganizationMemberModel(CRUD):
  """Organization member model."""

  __tablename__ = "organization_members"
  __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_org_member"),)

  organization_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  role_id: Mapped[UUID | None] = mapped_column(PGUUID, ForeignKey("roles.id", ondelete="CASCADE"), nullable=True)
  invited_by: Mapped[UUID | None] = mapped_column(PGUUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
  invite_id: Mapped[UUID | None] = mapped_column(PGUUID, ForeignKey("invites.id", ondelete="SET NULL"), nullable=True, index=True)
  deleted_by: Mapped[UUID | None] = mapped_column(PGUUID, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
  status: Mapped[str] = mapped_column(
    Enum("invited", "active", "suspended", "deleted", name="member_status"),
    nullable=False,
  )

  # Relationships
  invitation_used = relationship("InvitationModel", foreign_keys=[invite_id])
