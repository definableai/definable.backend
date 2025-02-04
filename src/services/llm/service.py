from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from services.__base.acquire import Acquire

from .model import LLMModel
from .schema import LLMCreate, LLMResponse, LLMUpdate


class LLMService:
  """LLM service for managing language models."""

  http_exposed = ["post=add", "post=update", "delete=remove", "get=list"]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire

  async def post_add(self, model_data: LLMCreate, session: AsyncSession = Depends(get_db)) -> LLMResponse:
    """Add a new language model."""
    # Check if model exists
    query = select(LLMModel).where(LLMModel.name == model_data.name, LLMModel.provider == model_data.provider, LLMModel.version == model_data.version)
    result = await session.execute(query)
    if result.scalar_one_or_none():
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model already exists")

    # Create model
    db_model = LLMModel(**model_data.model_dump())
    session.add(db_model)
    await session.commit()
    await session.refresh(db_model)

    return LLMResponse.model_validate(db_model)

  async def post_update(self, model_id: UUID, model_data: LLMUpdate, session: AsyncSession = Depends(get_db)) -> LLMResponse:
    """Update an existing language model."""
    # Get model
    query = select(LLMModel).where(LLMModel.id == model_id)
    result = await session.execute(query)
    db_model = result.scalar_one_or_none()

    if not db_model:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    # Update fields
    update_data = model_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(db_model, field, value)

    await session.commit()
    await session.refresh(db_model)

    return LLMResponse.model_validate(db_model)

  async def delete_remove(self, model_id: UUID, session: AsyncSession = Depends(get_db)) -> dict:
    """Remove a language model."""
    # Get model
    query = select(LLMModel).where(LLMModel.id == model_id)
    result = await session.execute(query)
    db_model = result.scalar_one_or_none()

    if not db_model:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    await session.delete(db_model)
    await session.commit()

    return {"message": "Model deleted successfully"}

  async def get_list(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("models", "read")),
  ) -> List[LLMResponse]:
    """Get list of all language models."""
    query = select(LLMModel)
    result = await session.execute(query)
    models = result.scalars().all()

    return [LLMResponse.model_validate(model) for model in models]
