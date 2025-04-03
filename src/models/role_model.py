from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class RoleModel(CRUD):
  """Role model."""

  __tablename__ = "roles"
  __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_role_org_name"),)

  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  name: Mapped[str] = mapped_column(String(50), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  is_system_role: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
  hierarchy_level: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

  # Relationships
  permissions = relationship("PermissionModel", secondary="role_permissions", lazy="joined")
  organization = relationship("OrganizationModel", lazy="select")
  invitations = relationship("InvitationModel", back_populates="role", cascade="all, delete-orphan")


class PermissionModel(CRUD):
  """Permission model."""

  __tablename__ = "permissions"
  __table_args__ = (UniqueConstraint("resource", "action", name="uq_permission_resource_action"),)

  name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  resource: Mapped[str] = mapped_column(String(30), nullable=False)
  action: Mapped[str] = mapped_column(String(20), nullable=False)


class RolePermissionModel(CRUD):
  """Role permission model."""

  __tablename__ = "role_permissions"
  __table_args__ = (UniqueConstraint("role_id", "permission_id", name="uq_role_permission"),)

  role_id: Mapped[UUID] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
  permission_id: Mapped[UUID] = mapped_column(ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False)
