from typing import Optional
from uuid import UUID

import aiohttp
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.tools_factory.v1 import ToolGenerator, generate_boilerplate
from models import ToolCategoryModel, ToolModel
from services.__base.acquire import Acquire

from .schema import (
  PaginatedToolResponse,
  ToolCategoryCreate,
  ToolCategoryResponse,
  ToolCategoryUpdate,
  ToolCreate,
  ToolResponse,
  ToolTestRequest,
  ToolUpdate,
)


class ToolService:
  """Tool service."""

  http_exposed = [
    "get=list",
    "get=list_all",
    "get=list_categories",
    "post=create_category",
    "put=update_category",
    "delete=delete_category",
    "post=test_tool",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.settings = acquire.settings
    self.generator = ToolGenerator()

  async def post(
    self,
    org_id: UUID,
    tool_data: ToolCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "write")),
  ) -> JSONResponse:
    """Create a new tool."""
    # Check if tool name and version combination exists
    query = select(ToolModel).where(ToolModel.name == tool_data.name, ToolModel.version == tool_data.version)
    result = await session.execute(query)
    if result.scalar_one_or_none():
      raise HTTPException(status_code=400, detail="Tool with this name and version already exists")

    # check if category exists
    query = select(ToolCategoryModel).where(ToolCategoryModel.id == tool_data.category_id)
    result = await session.execute(query)
    if not result.scalar_one_or_none():
      raise HTTPException(status_code=400, detail="Category not found")

    dump = tool_data.model_dump()
    raw_json = {
      "info": {
        "name": dump["name"],
        "version": dump["version"],
        "description": dump["description"],
      },
      "function_info": dump["settings"]["function_info"],
      "configuration": dump["configuration"],
    }
    # Generate tool
    try:
      response = await self.generator.generate_toolkit_from_json(raw_json)
    except Exception as e:
      raise HTTPException(status_code=400, detail=f"Error generating tool: {e}")

    # Create tool
    db_tool = ToolModel(
      name=tool_data.name,
      description=tool_data.description,
      organization_id=org_id,
      user_id=user["id"],
      category_id=tool_data.category_id,
      logo_url=tool_data.logo_url,
      is_active=tool_data.is_active,
      version=tool_data.version,
      is_public=tool_data.is_public,
      is_verified=tool_data.is_verified,
      inputs=dump["inputs"],  # because pydantic doesn't retun nested class as dicts
      outputs=dump["outputs"],
      configuration=dump["configuration"],
      settings=dump["settings"],
      generated_code=response,
    )
    session.add(db_tool)
    await session.commit()
    await session.refresh(db_tool)

    return JSONResponse(status_code=201, content={"message": "Tool created successfully", "id": str(db_tool.id)})

  async def put(
    self,
    org_id: UUID,
    tool_id: UUID,
    tool_data: ToolUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "write")),
  ) -> JSONResponse:
    # get tool
    query = select(ToolModel).where(ToolModel.id == tool_id)
    result = await session.execute(query)
    db_tool = result.scalar_one_or_none()

    if not db_tool:
      raise HTTPException(status_code=404, detail="Tool not found")

    # update tool
    for field, value in tool_data.model_dump(exclude_unset=True).items():
      setattr(db_tool, field, value)

    await session.commit()
    await session.refresh(db_tool)

    return JSONResponse(status_code=200, content={"message": "Tool updated successfully"})

  async def post_test_tool(
    self,
    tool_id: UUID,
    tool_test_request: ToolTestRequest,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> JSONResponse:
    # get tool
    query = select(ToolModel).where(ToolModel.id == tool_id)
    result = await session.execute(query)
    db_tool = result.scalar_one_or_none()

    if not db_tool:
      raise HTTPException(status_code=404, detail="Tool not found")

    # TODO: what is the benefit of using pydantic here if I have to dump it? check it later!
    raw_json = tool_test_request.model_dump()
    class_name = "".join(word.capitalize() for word in db_tool.name.split()) + "Toolkit"
    # generate the boilerplate
    boilerplate = generate_boilerplate(
      class_name=class_name,
      input_prompt=raw_json["input_prompt"],
      model_name=raw_json["model_name"],
      provider=raw_json["provider"],
      api_key=raw_json["api_key"],
      config_items=raw_json["config_items"],
      instructions=raw_json["instructions"],
    )
    print(boilerplate)
    test_response = None
    requirements = db_tool.settings.get("requirements", [])
    # TODO: it must be done based on the agent framework, and the model name
    requirements += ["agno", "openai", "anthropic"]
    # send the boilerplate to the python sandbox testing url
    async with aiohttp.ClientSession() as http_session:
      async with http_session.post(
        self.settings.python_sandbox_testing_url,
        json={
          "agent_code": boilerplate,
          "tool_code": db_tool.generated_code,
          "requirements": requirements,
          "version": db_tool.version,
          "tool_name": class_name,
        },
      ) as response:
        test_response = await response.json()
    print(test_response)
    return JSONResponse(status_code=200, content={"message": "Tool tested successfully", "test_response": test_response})

  # to fetch tools for an organization
  async def get_list(
    self,
    org_id: UUID,
    offset: int = 0,
    limit: int = 10,
    category_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "read")),
  ) -> PaginatedToolResponse:
    """Get paginated list of tools for an organization."""
    # Base query for total count
    count_query = select(func.count(ToolModel.id))
    if category_id:
      count_query = count_query.where(ToolModel.category_id == category_id)
    total = await session.scalar(count_query)

    # Main query with joins for details
    query = select(ToolModel).join(ToolCategoryModel).where(ToolModel.organization_id == org_id).order_by(ToolModel.created_at.desc())

    if category_id:
      query = query.where(ToolModel.category_id == category_id)

    query = query.offset(offset * limit).limit(limit + 1)

    result = await session.execute(query)
    tools = result.scalars().all()

    # Process results
    tool_list = [ToolResponse.model_validate(tool) for tool in tools[:limit]]
    has_more = len(tools) > limit

    return PaginatedToolResponse(tools=tool_list, total=total or 0, has_more=has_more)

  # to fetch all tools across organizations
  async def get_list_all(
    self,
    offset: int = 0,
    limit: int = 10,
    category_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),  # Require admin access
  ) -> PaginatedToolResponse:
    """Get paginated list of all tools across organizations."""
    # Base query for total count
    count_query = select(func.count(ToolModel.id))
    if category_id:
      count_query = count_query.where(ToolModel.category_id == category_id)
    total = await session.scalar(count_query)

    # Main query with joins for details
    query = select(ToolModel).join(ToolCategoryModel).order_by(ToolModel.created_at.desc())

    if category_id:
      query = query.where(ToolModel.category_id == category_id)

    query = query.offset(offset * limit).limit(limit + 1)

    result = await session.execute(query)
    tools = result.scalars().all()

    # Process results
    tool_list = [ToolResponse.model_validate(tool) for tool in tools[:limit]]
    has_more = len(tools) > limit

    return PaginatedToolResponse(tools=tool_list, total=total or 0, has_more=has_more)

  # TODO: need to think who can manipulate categories
  async def post_create_category(
    self,
    org_id: UUID,
    category_data: ToolCategoryCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "write")),
  ) -> ToolCategoryResponse:
    """Create a new tool category."""
    # Check if category name exists
    query = select(ToolCategoryModel).where(ToolCategoryModel.name == category_data.name)
    result = await session.execute(query)
    if result.scalar_one_or_none():
      raise HTTPException(status_code=400, detail="Category name already exists")

    # Create category
    db_category = ToolCategoryModel(**category_data.model_dump())
    session.add(db_category)
    await session.commit()
    await session.refresh(db_category)

    return ToolCategoryResponse.model_validate(db_category)

  async def put_update_category(
    self,
    org_id: UUID,
    category_id: UUID,
    category_data: ToolCategoryUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "write")),
  ) -> ToolCategoryResponse:
    """Update a tool category."""
    # Get category
    query = select(ToolCategoryModel).where(ToolCategoryModel.id == category_id)
    result = await session.execute(query)
    db_category = result.scalar_one_or_none()

    if not db_category:
      raise HTTPException(status_code=404, detail="Category not found")

    # Update fields
    update_data = category_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(db_category, field, value)

    await session.commit()
    await session.refresh(db_category)

    return ToolCategoryResponse.model_validate(db_category)

  async def delete_delete_category(
    self,
    category_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("tools", "delete")),
  ) -> dict:
    """Delete a tool category."""
    # Get category
    query = select(ToolCategoryModel).where(ToolCategoryModel.id == category_id)
    result = await session.execute(query)
    db_category = result.scalar_one_or_none()

    if not db_category:
      raise HTTPException(status_code=404, detail="Category not found")

    # Check if category has tools
    tools_query = select(ToolModel).where(ToolModel.category_id == category_id)
    tools_result = await session.execute(tools_query)
    if tools_result.scalar_one_or_none():
      raise HTTPException(status_code=400, detail="Cannot delete category with tools")

    await session.delete(db_category)
    await session.commit()

    return {"message": "Category deleted successfully"}

  async def get_list_categories(
    self,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> list[ToolCategoryResponse]:
    """Get list of all tool categories."""
    query = select(ToolCategoryModel).order_by(ToolCategoryModel.name)
    result = await session.execute(query)
    categories = result.scalars().all()
    print(categories)

    return [ToolCategoryResponse.model_validate(category) for category in categories]
