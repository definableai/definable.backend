# from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Type, TypeVar

# from fastapi import HTTPException
# from pydantic import BaseModel
# from sqlalchemy import and_, select
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.sql import Select

# from database.models import Base

# ModelType = TypeVar("ModelType", bound=Base)
# CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
# UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
# ResponseSchemaType = TypeVar("ResponseSchemaType", bound=BaseModel)
# FilterSchemaType = TypeVar("FilterSchemaType", bound=BaseModel)


# class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ResponseSchemaType]):
#   """Base service with common CRUD operations and extensible features."""

#   def __init__(self, model: Type[ModelType]):
#     self.model = model
#     self._pre_create_hooks: List[Callable] = []
#     self._post_create_hooks: List[Callable] = []
#     self._pre_update_hooks: List[Callable] = []
#     self._post_update_hooks: List[Callable] = []
#     self._pre_delete_hooks: List[Callable] = []
#     self._post_delete_hooks: List[Callable] = []

#   def add_hook(self, hook_type: str, func: Callable) -> None:
#     """Add a hook to be executed before/after operations."""
#     hook_list = getattr(self, f"_pre_{hook_type}_hooks" if hook_type.startswith("pre_") else f"_post_{hook_type}_hooks")
#     hook_list.append(func)

#   async def _execute_hooks(self, hooks: List[Callable], *args, **kwargs) -> None:
#     """Execute hooks in order."""
#     for hook in hooks:
#       await hook(*args, **kwargs)

#   def build_query(self, filters: Optional[Dict[str, Any]] = None) -> Select:
#     """Build base query with filters."""
#     query = select(self.model)
#     if filters:
#       conditions = [getattr(self.model, k) == v for k, v in filters.items()]
#       query = query.where(and_(*conditions))
#     return query

#   async def post(self, session: AsyncSession, data: CreateSchemaType, **kwargs: Any) -> ResponseSchemaType:
#     """Create a new record with hooks."""
#     await self._execute_hooks(self._pre_create_hooks, data, session)

#     db_item = self.model(**data.model_dump(), **kwargs)
#     session.add(db_item)
#     await session.commit()
#     await session.refresh(db_item)

#     await self._execute_hooks(self._post_create_hooks, db_item, session)
#     return db_item

#   async def get(self, session: AsyncSession, _id: Any, raise_if_not_found: bool = True) -> Optional[ModelType]:
#     """Get a record by id with optional error handling."""
#     db_item = await session.get(self.model, _id)
#     if not db_item and raise_if_not_found:
#       raise HTTPException(status_code=404, detail=f"{self.model.__name__} not found")
#     return db_item

#   async def list(
#     self,
#     session: AsyncSession,
#     filters: Optional[Dict[str, Any]] = None,
#     skip: int = 0,
#     limit: int = 100,
#   ) -> Sequence[ModelType]:
#     """Get multiple records with filtering and pagination."""
#     query = self.build_query(filters)
#     query = query.offset(skip).limit(limit)
#     result = await session.execute(query)
#     return result.scalars().all()

#   async def update(self, session: AsyncSession, _id: Any, data: UpdateSchemaType) -> ResponseSchemaType | None:
#     """Update a record with hooks."""
#     db_item = await self.get(session, _id)
#     await self._execute_hooks(self._pre_update_hooks, db_item, data, session)

#     update_data = data.model_dump(exclude_unset=True)
#     for field, value in update_data.items():
#       setattr(db_item, field, value)

#     await session.commit()
#     await session.refresh(db_item)

#     await self._execute_hooks(self._post_update_hooks, db_item, session)
#     return db_item

#   async def delete(self, session: AsyncSession, _id: Any) -> None:
#     """Delete a record with hooks."""
#     db_item = await self.get(session, _id)
#     await self._execute_hooks(self._pre_delete_hooks, db_item, session)

#     await session.delete(db_item)
#     await session.commit()

#     await self._execute_hooks(self._post_delete_hooks, _id, session)

#   async def exists(self, session: AsyncSession, filters: Dict[str, Any]) -> bool:
#     """Check if records exist with given filters."""
#     query = self.build_query(filters).exists()
#     result = await session.execute(select(query))
#     return bool(result.scalar())
