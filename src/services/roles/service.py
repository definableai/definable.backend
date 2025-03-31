from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, delete, func, insert, not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from services.__base.acquire import Acquire

from .model import PermissionModel, RoleModel, RolePermissionModel
from .schema import PermissionCreate, PermissionResponse, RoleCreate, RoleResponse, RoleUpdate


class RoleService:
  """Role service."""

  http_exposed = ["post=permission", "delete=permission", "post=create", "put=update", "delete=remove", "get=list_roles", "get=list_permissions"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.models = acquire.models

  async def post_permission(
    self,
    permission: PermissionCreate,
    session: AsyncSession = Depends(get_db),
  ) -> PermissionResponse:
    """Create a new permission and update default role permissions."""
    # Create permission
    db_permission = PermissionModel(**permission.model_dump())
    session.add(db_permission)
    await session.flush()

    await session.commit()
    return PermissionResponse.model_validate(db_permission)

  async def delete_permission(
    self,
    permission_id: UUID,
    session: AsyncSession = Depends(get_db),
  ) -> None:
    """Delete a permission and its associations."""
    # Delete from role_permissions and default_role_permissions first
    await session.execute(delete(RolePermissionModel).where(RolePermissionModel.permission_id == permission_id))

    # Delete the permission
    await session.execute(delete(PermissionModel).where(PermissionModel.id == permission_id))
    await session.commit()

  async def post_create(
    self,
    org_id: UUID,
    role_data: RoleCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("roles", "write")),
  ) -> RoleResponse:
    """Create a user-defined role."""
    # check if role name is unique
    role_name = role_data.name
    role_exists = await session.execute(select(RoleModel).where(RoleModel.name == role_name, RoleModel.organization_id == org_id))
    if role_exists.unique().scalar_one_or_none():
      raise HTTPException(status_code=400, detail="Role name already exists")

    # Validate hierarchy level
    await self._validate_hierarchy_level(org_id, role_data.hierarchy_level, session)

    # Create role
    db_role = RoleModel(organization_id=org_id, is_system_role=False, **role_data.model_dump(exclude={"permission_ids"}))
    session.add(db_role)
    await session.flush()

    # Add permissions
    await self._add_role_permissions(db_role.id, role_data.permission_ids, session)
    await session.commit()

    return await self._get_role_with_permissions(db_role.id, session)

  async def put_update(
    self,
    organization_id: UUID,
    role_id: UUID,
    role_data: RoleUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("roles", "write")),
  ) -> RoleResponse:
    """Update a user-defined role."""
    # Get existing role
    db_role = await self._get_role(role_id, organization_id, session)
    if not db_role:
      raise HTTPException(status_code=404, detail="Role not found")
    if db_role.is_system_role:
      raise HTTPException(status_code=400, detail="Cannot update system role")

    # Update fields
    update_data = role_data.model_dump(exclude_unset=True)
    if "hierarchy_level" in update_data:
      await self._validate_hierarchy_level(organization_id, update_data["hierarchy_level"], session)

    # Update permissions if provided
    if "permission_ids" in update_data:
      await session.execute(delete(RolePermissionModel).where(RolePermissionModel.role_id == role_id))
      await self._add_role_permissions(role_id, update_data.pop("permission_ids"), session)

    # Update role attributes
    for field, value in update_data.items():
      setattr(db_role, field, value)

    await session.commit()
    return await self._get_role_with_permissions(role_id, session)

  async def delete_remove(
    self,
    org_id: UUID,
    role_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("roles", "delete")),
  ) -> dict:
    """
    Delete a user-defined role.

    Args:
        organization_id: Organization ID
        role_id: Role ID to delete
        session: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If role cannot be deleted
    """
    # Get role with member count
    query = (
      select(RoleModel, func.count(self.models["OrganizationMemberModel"].id).label("member_count"))
      .outerjoin(self.models["OrganizationMemberModel"], RoleModel.id == self.models["OrganizationMemberModel"].role_id)
      .where(and_(RoleModel.id == role_id, RoleModel.organization_id == org_id))
      .group_by(RoleModel.id)
    )
    result = await session.execute(query)
    role_data = result.first()
    if not role_data:
      raise HTTPException(status_code=404, detail="Role not found")
    role, member_count = role_data
    # Check if it's a system role
    if role.is_system_role:
      raise HTTPException(status_code=400, detail="Cannot delete system role")
    # Check if role is assigned to any members
    if member_count > 0:
      raise HTTPException(status_code=400, detail="Cannot delete role that is assigned to members")
    try:
      # Delete role permissions first
      await session.execute(delete(RolePermissionModel).where(RolePermissionModel.role_id == role_id))
      await session.execute(
        delete(RoleModel).where(and_(RoleModel.id == role_id, RoleModel.organization_id == org_id, not_(RoleModel.is_system_role)))
      )
      await session.commit()
      return {"message": "Role deleted successfully"}

    except Exception as e:
      await session.rollback()
      raise HTTPException(status_code=500, detail=f"Failed to delete role: {str(e)}")

  async def get_list_roles(
    self,
    organization_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("roles", "read")),
  ) -> List[RoleResponse]:
    query = select(RoleModel).where(RoleModel.organization_id == organization_id)
    result = await session.execute(query)
    return list(result.unique().scalars().all())

  async def get_list_permissions(
    self,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> List[PermissionResponse]:
    query = select(PermissionModel)
    result = await session.execute(query)
    return list(result.unique().scalars().all())

  ### PRIVATE METHODS ###
  async def _get_default_roles(
    self,
    session: AsyncSession,
  ) -> List[RoleModel]:
    """Get all default roles."""
    query = select(RoleModel).where(RoleModel.is_system_role)
    result = await session.execute(query)
    return list(result.scalars().all())

  async def _add_role_permission(
    self,
    role_id: UUID,
    permission_id: UUID,
    session: AsyncSession,
  ) -> None:
    """Add permission to default role."""
    await session.execute(insert(RolePermissionModel).values({"role_id": role_id, "permission_id": permission_id}))

  async def _validate_hierarchy_level(
    self,
    organization_id: UUID,
    hierarchy_level: int,
    session: AsyncSession,
  ) -> None:
    """Validate hierarchy level."""
    if hierarchy_level >= 90:  # Reserved for system roles
      raise HTTPException(status_code=400, detail="Hierarchy level must be less than 90")

  async def _add_role_permissions(
    self,
    role_id: UUID,
    permission_ids: List[UUID],
    session: AsyncSession,
  ) -> None:
    """Add permissions to role."""
    role_permissions_data = [{"role_id": role_id, "permission_id": permission_id} for permission_id in permission_ids]

    await session.execute(insert(RolePermissionModel).values(role_permissions_data))

  async def _get_role(
    self,
    role_id: UUID,
    organization_id: UUID,
    session: AsyncSession,
  ) -> Optional[RoleModel]:
    """Get role by ID and organization ID."""
    query = select(RoleModel).where(
      and_(
        RoleModel.id == role_id,
        RoleModel.organization_id == organization_id,
      )
    )
    result = await session.execute(query)
    return result.unique().scalar_one_or_none()

  async def _get_role_with_permissions(
    self,
    role_id: UUID,
    session: AsyncSession,
  ) -> RoleResponse:
    """Get role with permissions."""
    # Query role with permissions using join
    query = (
      select(RoleModel)
      .join(RolePermissionModel, RoleModel.id == RolePermissionModel.role_id)
      .join(PermissionModel, RolePermissionModel.permission_id == PermissionModel.id)
      .where(RoleModel.id == role_id)
    )

    result = await session.execute(query)
    role = result.unique().scalar_one_or_none()
    if not role:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")

    # Convert to response model
    return RoleResponse(
      id=role.id,
      organization_id=role.organization_id,
      name=role.name,
      description=role.description,
      is_system_role=role.is_system_role,
      hierarchy_level=role.hierarchy_level,
      created_at=role.created_at,
      permissions=[
        PermissionResponse(id=perm.id, name=perm.name, description=perm.description, resource=perm.resource, action=perm.action)
        for perm in role.permissions
      ],
    )
