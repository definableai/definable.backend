from typing import Dict, List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from database import get_db
from dependencies.security import RBAC, JWTBearer
from models import PromptCategoryModel, PromptModel
from services.__base.acquire import Acquire

from .schema import (
  PaginatedPromptResponse,
  PromptCategoryCreate,
  PromptCategoryResponse,
  PromptCategoryUpdate,
  PromptCreate,
  PromptResponse,
  PromptUpdate,
)


class PromptService:
  """Prompt service."""

  http_exposed = [
    "get=list_categories",
    "post=create_category",
    "put=update_category",
    "delete=delete_category",
    "get=get_category",
    "get=list_prompts",
    "post=create_prompt",
    "put=update_prompt",
    "delete=delete_prompt",
    "get=get_prompt",
    "get=list_all_prompts",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger
    self.logger.info("PromptService initialized")

  async def get_list_categories(
    self, active_only: bool = True, session: AsyncSession = Depends(get_db), user: dict = Depends(JWTBearer())
  ) -> List[PromptCategoryResponse]:
    """Get all prompt categories."""
    self.logger.info(f"Listing categories with active_only={active_only}")
    query = select(PromptCategoryModel)
    if active_only:
      query = query.where(PromptCategoryModel.is_active)
    query = query.order_by(PromptCategoryModel.display_order)

    result = await session.execute(query)
    categories = result.scalars().all()
    self.logger.debug(f"Found {len(categories)} categories")

    return [PromptCategoryResponse.model_validate(category) for category in categories]

  async def post_create_category(
    self, category_data: PromptCategoryCreate, session: AsyncSession = Depends(get_db), user: dict = Depends(JWTBearer())
  ) -> PromptCategoryResponse:
    """Create a new prompt category."""
    self.logger.info(f"Creating new prompt category: {category_data.name}")
    # Check if category with the same name already exists
    query = select(PromptCategoryModel).where(PromptCategoryModel.name == category_data.name)
    result = await session.execute(query)
    if result.scalars().first():
      self.logger.warning(f"Attempted to create duplicate category with name: {category_data.name}")
      raise HTTPException(status_code=400, detail="Category with this name already exists")

    # Create new category
    db_category = PromptCategoryModel(**category_data.model_dump())
    session.add(db_category)
    await session.commit()
    await session.refresh(db_category)
    self.logger.info(f"Created new prompt category id {db_category.id} and name '{db_category.name}'")
    return PromptCategoryResponse(
      id=db_category.id,
      name=db_category.name,
      description=db_category.description,
      icon_url=db_category.icon_url,
      display_order=db_category.display_order,
    )

  # TODO: we would be a superuser access becuase some resources are only managed by us
  async def put_update_category(
    self,
    category_id: UUID,
    category_data: PromptCategoryUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> PromptCategoryResponse:
    """Update a prompt category."""
    self.logger.info(f"Updating category with id: {category_id}")
    # Get the category
    query = select(PromptCategoryModel).where(PromptCategoryModel.id == category_id)
    result = await session.execute(query)
    category = result.scalars().first()

    if not category:
      self.logger.warning(f"Category not found with id: {category_id}")
      raise HTTPException(status_code=404, detail="Category not found")

    # Check for name uniqueness if name is being updated
    if category_data.name and category_data.name != category.name:
      self.logger.debug(f"Checking name uniqueness for: {category_data.name}")
      name_check = select(PromptCategoryModel).where(PromptCategoryModel.name == category_data.name, PromptCategoryModel.id != category_id)
      result = await session.execute(name_check)
      if result.scalars().first():
        self.logger.warning(f"Category name conflict: {category_data.name}")
        raise HTTPException(status_code=400, detail="Category with this name already exists")

    # Update category fields
    update_data = category_data.model_dump(exclude_unset=True)
    self.logger.debug(f"Updating fields: {', '.join(update_data.keys())}")
    for field, value in update_data.items():
      setattr(category, field, value)

    await session.commit()
    await session.refresh(category)
    self.logger.info(f"Category updated successfully: {category.id}")

    return PromptCategoryResponse.model_validate(category)

  async def delete_delete_category(
    self,
    category_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> Dict:
    """Delete a prompt category."""
    self.logger.info(f"Attempting to delete category: {category_id}")
    # Check if category exists
    query = select(PromptCategoryModel).where(PromptCategoryModel.id == category_id)
    result = await session.execute(query)
    category = result.scalars().first()

    if not category:
      self.logger.warning(f"Category not found for deletion: {category_id}")
      raise HTTPException(status_code=404, detail="Category not found")

    # Check if category has prompts
    prompt_count = await session.execute(select(func.count(PromptModel.id)).where(PromptModel.category_id == category_id))
    count = prompt_count.scalar_one()

    if count > 0:
      self.logger.warning(f"Cannot delete category {category_id} with {count} prompts")
      raise HTTPException(status_code=400, detail=f"Cannot delete category that has {count} prompts. Move or delete the prompts first.")

    # Delete the category
    await session.delete(category)
    await session.commit()
    self.logger.info(f"Category {category_id} deleted successfully")

    return {"message": "Category deleted successfully"}

  async def get_get_category(
    self,
    category_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> PromptCategoryResponse:
    """Get a prompt category by ID."""
    self.logger.info(f"Fetching category with id: {category_id}")
    query = select(PromptCategoryModel).where(PromptCategoryModel.id == category_id)
    result = await session.execute(query)
    category = result.scalars().first()

    if not category:
      self.logger.warning(f"Category not found: {category_id}")
      raise HTTPException(status_code=404, detail="Category not found")

    self.logger.debug(f"Retrieved category: {category.name}")
    return PromptCategoryResponse.model_validate(category)

  async def get_list_prompts(
    self,
    org_id: UUID,
    category_id: Optional[UUID] = None,
    include_public: bool = True,
    is_featured: Optional[bool] = None,
    offset: int = 0,
    limit: int = 20,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("prompts", "read")),
  ) -> PaginatedPromptResponse:
    """Get prompts with pagination."""
    self.logger.info(f"Listing prompts for org: {org_id}, category: {category_id}, include_public: {include_public}, is_featured: {is_featured}")
    # Base query for prompts
    base_query = select(PromptModel).options(joinedload(PromptModel.category))

    # Add conditions based on the include_public flag
    if include_public:
      query = base_query.where(or_(PromptModel.organization_id == org_id, PromptModel.is_public))
    else:
      query = base_query.where(PromptModel.organization_id == org_id)

    # Apply filters
    if category_id:
      query = query.where(PromptModel.category_id == category_id)

    if is_featured is not None:
      query = query.where(PromptModel.is_featured == is_featured)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.scalar(count_query) or 0
    self.logger.debug(f"Total matching prompts: {total}")

    # Apply pagination
    query = query.order_by(PromptModel.created_at.desc())
    query = query.offset(offset * limit).limit(limit + 1)

    # Execute query
    result = await session.execute(query)
    prompts = result.unique().scalars().all()

    # Check if there are more prompts
    has_more = len(prompts) > limit
    prompts = prompts[:limit]  # Remove the extra item used to check for more
    self.logger.debug(f"Retrieved {len(prompts)} prompts, has_more: {has_more}")

    # Convert to response models
    prompt_responses = [
      PromptResponse(**{
        **{k: v for k, v in prompt.__dict__.items() if k not in ["_sa_instance_state", "category"]},
        "category": PromptCategoryResponse.model_validate(prompt.category),
      })
      for prompt in prompts
    ]

    return PaginatedPromptResponse(prompts=prompt_responses, total=total, has_more=has_more)

  async def post_create_prompt(
    self,
    org_id: UUID,
    category_id: UUID,
    prompt_data: PromptCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> PromptResponse:
    """Create a new prompt."""
    self.logger.info(f"Creating new prompt for org: {org_id} in category: {category_id}")
    # Verify that category exists and acrive
    category_query = select(PromptCategoryModel).where(PromptCategoryModel.id == category_id, PromptCategoryModel.is_active)
    result = await session.execute(category_query)
    category = result.scalars().first()

    if not category:
      self.logger.warning(f"Category not found or inactive: {category_id}")
      raise HTTPException(status_code=404, detail="Category not found")

    # Create the prompt
    new_prompt = PromptModel(**prompt_data.model_dump(), creator_id=user["id"], organization_id=org_id, category_id=category_id)

    session.add(new_prompt)
    await session.commit()
    await session.refresh(new_prompt)
    self.logger.info(f"Created new prompt with id: {new_prompt.id}")

    # Get the prompt with category joined
    query = select(PromptModel).options(joinedload(PromptModel.category)).where(PromptModel.id == new_prompt.id)
    result = await session.execute(query)
    prompt_with_category = result.unique().scalar_one()

    # Convert to response model
    return PromptResponse(**{
      **{k: v for k, v in prompt_with_category.__dict__.items() if k not in ["_sa_instance_state", "category"]},
      "category": PromptCategoryResponse.model_validate(prompt_with_category.category),
    })

  async def put_update_prompt(
    self,
    org_id: UUID,
    category_id: UUID,
    prompt_id: UUID,
    prompt_data: PromptUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("prompts", "write")),
  ) -> PromptResponse:
    """Update a prompt."""
    self.logger.info(f"Updating prompt: {prompt_id} in org: {org_id}, category: {category_id}")

    # check if category exists
    category_query = select(PromptCategoryModel).where(PromptCategoryModel.id == category_id, PromptCategoryModel.is_active)
    result = await session.execute(category_query)
    category = result.scalars().first()

    if not category:
      self.logger.warning(f"Category not found: {category_id}")
      raise HTTPException(status_code=404, detail="Category not found")

    # Get the prompt and verify ownership
    query = (
      select(PromptModel)
      .options(joinedload(PromptModel.category))
      .where(PromptModel.id == prompt_id, PromptModel.organization_id == org_id, PromptModel.category_id == category_id)
    )
    result = await session.execute(query)
    prompt = result.unique().scalar_one_or_none()

    if not prompt:
      self.logger.warning(f"Prompt not found or access denied: {prompt_id}")
      raise HTTPException(status_code=404, detail="Prompt not found or you don't have access")

    # Update prompt fields
    update_data = prompt_data.model_dump(exclude_unset=True)
    self.logger.debug(f"Updating prompt fields: {', '.join(update_data.keys())}")
    for field, value in update_data.items():
      setattr(prompt, field, value)

    await session.commit()
    await session.refresh(prompt)
    self.logger.info(f"Prompt updated successfully: {prompt_id}")

    # Get the updated prompt with category
    query = select(PromptModel).options(joinedload(PromptModel.category)).where(PromptModel.id == prompt_id)
    result = await session.execute(query)
    updated_prompt = result.unique().scalar_one()

    # Convert to response model
    return PromptResponse(**{
      **{k: v for k, v in updated_prompt.__dict__.items() if k not in ["_sa_instance_state", "category"]},
      "category": PromptCategoryResponse.model_validate(updated_prompt.category),
    })

  async def delete_delete_prompt(
    self,
    org_id: UUID,
    prompt_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("prompts", "delete")),
  ) -> Dict:
    """Delete a prompt."""
    self.logger.info(f"Attempting to delete prompt: {prompt_id} in org: {org_id}")
    # Get the prompt and verify ownership
    query = select(PromptModel).where(PromptModel.id == prompt_id, PromptModel.organization_id == org_id)
    result = await session.execute(query)
    prompt = result.scalars().first()

    if not prompt:
      self.logger.warning(f"Prompt not found or access denied for deletion: {prompt_id}")
      raise HTTPException(status_code=404, detail="Prompt not found or you don't have access")

    # Delete the prompt
    await session.delete(prompt)
    await session.commit()
    self.logger.info(f"Prompt {prompt_id} deleted successfully")

    return {"message": "Prompt deleted successfully"}

  async def get_list_all_prompts(
    self,
    category_id: Optional[UUID] = None,
    is_featured: Optional[bool] = None,
    offset: int = 0,
    limit: int = 20,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> PaginatedPromptResponse:
    """Get all public prompts with pagination."""
    self.logger.info(f"Listing all public prompts, category: {category_id}, is_featured: {is_featured}")
    # Base query for public prompts
    base_query = select(PromptModel).options(joinedload(PromptModel.category)).where(PromptModel.is_public)

    # Apply filters
    if category_id:
      base_query = base_query.where(PromptModel.category_id == category_id)

    if is_featured is not None:
      base_query = base_query.where(PromptModel.is_featured == is_featured)

    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total = await session.scalar(count_query) or 0
    self.logger.debug(f"Total matching public prompts: {total}")

    # Apply pagination
    query = base_query.order_by(PromptModel.created_at.desc())
    query = query.offset(offset * limit).limit(limit + 1)

    # Execute query
    result = await session.execute(query)
    prompts = result.unique().scalars().all()

    # Check if there are more prompts
    has_more = len(prompts) > limit
    prompts = prompts[:limit]  # Remove the extra item used to check for more
    self.logger.debug(f"Retrieved {len(prompts)} public prompts, has_more: {has_more}")

    # Convert to response models
    prompt_responses = [
      PromptResponse(**{
        **{k: v for k, v in prompt.__dict__.items() if k not in ["_sa_instance_state", "category"]},
        "category": PromptCategoryResponse.model_validate(prompt.category),
      })
      for prompt in prompts
    ]

    return PaginatedPromptResponse(prompts=prompt_responses, total=total, has_more=has_more)
