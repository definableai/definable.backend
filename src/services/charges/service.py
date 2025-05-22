from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from models import ChargeModel
from services.__base.acquire import Acquire

from .schema import ChargeCreateSchema, ChargeResponseSchema, ChargeUpdateSchema


class ChargesService:
  """Service for managing charge definitions."""

  http_exposed = [
    "get=charge",
    "get=charges",
    "post=charge",
    "patch=charge",
    "delete=charge",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def get_charge(
    self,
    charge_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> ChargeResponseSchema:
    """Get a charge by ID."""
    charge = await session.get(ChargeModel, charge_id)
    if not charge:
      raise HTTPException(status_code=404, detail="Charge not found")
    return ChargeResponseSchema.from_charge(charge)

  async def get_charges(
    self,
    is_active: Optional[bool] = None,
    service: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """Get all charges with optional filtering."""
    query = select(ChargeModel)

    # Apply filters
    conditions = []
    if is_active is not None:
      conditions.append(ChargeModel.is_active == is_active)
    if service:
      conditions.append(ChargeModel.service == service)

    if conditions:
      query = query.where(and_(*conditions))

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Execute query
    result = await session.execute(query)
    charges = result.scalars().all()

    # Count total results for pagination
    count_query = select(ChargeModel)
    if conditions:
      count_query = count_query.where(and_(*conditions))
    count_result = await session.execute(count_query)
    total_count = len(count_result.scalars().all())

    # Convert charges to response schema using the from_charge method
    charge_list = [ChargeResponseSchema.from_charge(charge) for charge in charges]

    return {
      "items": charge_list,
      "pagination": {
        "total": total_count,
        "offset": offset,
        "limit": limit,
      },
    }

  async def post_charge(
    self,
    charge_data: ChargeCreateSchema,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> ChargeResponseSchema:
    """Create a new charge definition."""
    # Check if a charge with this name already exists
    existing_query = select(ChargeModel).where(ChargeModel.name == charge_data.name)
    existing_result = await session.execute(existing_query)
    existing_charge = existing_result.scalars().first()

    if existing_charge:
      raise HTTPException(status_code=400, detail=f"Charge with name '{charge_data.name}' already exists")

    # Create new charge with UUID
    from uuid import uuid4

    new_charge = ChargeModel(id=uuid4(), **charge_data.model_dump())
    session.add(new_charge)
    await session.commit()
    await session.refresh(new_charge)

    return ChargeResponseSchema.from_charge(new_charge)

  async def patch_charge(
    self,
    charge_id: UUID,
    charge_data: ChargeUpdateSchema,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> ChargeResponseSchema:
    """Update an existing charge."""
    # Get existing charge
    charge = await session.get(ChargeModel, charge_id)
    if not charge:
      raise HTTPException(status_code=404, detail="Charge not found")

    # If name is being changed, check for duplicates
    if charge_data.name and charge_data.name != charge.name:
      existing_query = select(ChargeModel).where(ChargeModel.name == charge_data.name)
      existing_result = await session.execute(existing_query)
      existing_charge = existing_result.scalars().first()

      if existing_charge:
        raise HTTPException(status_code=400, detail=f"Charge with name '{charge_data.name}' already exists")

    # Update charge with provided data
    update_data = charge_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
      setattr(charge, key, value)

    await session.commit()
    await session.refresh(charge)

    return ChargeResponseSchema.from_charge(charge)

  async def delete_charge(
    self,
    charge_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "write")),
  ) -> Dict[str, str]:
    """Delete a charge."""
    charge = await session.get(ChargeModel, charge_id)
    if not charge:
      raise HTTPException(status_code=404, detail="Charge not found")

    # Store charge name for logging
    charge_name = charge.name

    # Delete the charge
    await session.delete(charge)
    await session.commit()

    self.logger.info(f"Deleted charge: {charge_name} (ID: {charge_id})")
    return {"message": f"Charge '{charge_name}' deleted successfully"}
