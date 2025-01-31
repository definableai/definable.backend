import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import Boolean, DateTime, MetaData, select, text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import AsyncAdaptedQueuePool

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for models
T = TypeVar("T", bound="BaseModel")

# Create async engine with connection pooling
async_engine = create_async_engine(
  settings.database_url,
  poolclass=AsyncAdaptedQueuePool,
  pool_size=20,  # Adjust based on your needs
  max_overflow=10,
  pool_timeout=30,  # Connection timeout
  pool_pre_ping=True,  # Enable connection health checks
  echo=settings.debug_mode,  # SQL query logging
)

# Create sessionmaker
AsyncSessionLocal = async_sessionmaker(
  async_engine,
  expire_on_commit=False,
  autoflush=False,
)

# Metadata for schema management
convention = {
  "ix": "ix_%(column_0_label)s",
  "uq": "uq_%(table_name)s_%(column_0_name)s",
  "ck": "ck_%(table_name)s_%(constraint_name)s",
  "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
  "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=convention)


class UUIDMixin:
  """UUID primary key mixin."""

  id: Mapped[UUID] = mapped_column(
    PgUUID(as_uuid=True),
    primary_key=True,
    server_default=text("gen_random_uuid()"),
    index=True,
  )


class TimestampMixin:
  """Timestamp mixin for created_at and updated_at."""

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    server_default=text("CURRENT_TIMESTAMP"),
    nullable=False,
    index=True,
  )
  updated_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    onupdate=lambda: datetime.now(timezone.utc),
    server_default=text("CURRENT_TIMESTAMP"),
    nullable=False,
  )


class SoftDeleteMixin:
  """Soft delete mixin."""

  deleted_at: Mapped[Optional[datetime]] = mapped_column(
    DateTime(timezone=True),
    nullable=True,
    index=True,
  )
  is_deleted: Mapped[bool] = mapped_column(
    Boolean,
    default=False,
    server_default=text("false"),
    nullable=False,
    index=True,
  )


class BaseModel(DeclarativeBase, UUIDMixin, TimestampMixin, SoftDeleteMixin):
  """Base model with CRUD operations."""

  __abstract__ = True
  metadata = metadata

  @classmethod
  async def _get_session(cls) -> AsyncSession:
    """Get a database session."""
    return AsyncSessionLocal()

  @classmethod
  async def create(cls: Type[T], **kwargs) -> T:
    """Create a new record with error handling and logging."""
    try:
      instance = cls(**kwargs)
      async with await cls._get_session() as session:
        async with session.begin():
          session.add(instance)
          await session.flush()
          await session.refresh(instance)
          return instance
    except SQLAlchemyError as e:
      logger.error(f"Error creating {cls.__name__}: {str(e)}")
      raise

  @classmethod
  async def read_by_id(cls: Type[T], _id: UUID) -> Optional[T]:
    """Read a record by ID with error handling."""
    try:
      async with await cls._get_session() as session:
        result = await session.execute(select(cls).where(cls.id == _id).where(cls.is_deleted.is_(False)))
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
      logger.error(f"Error reading {cls.__name__} by ID: {str(e)}")
      raise

  @classmethod
  async def read(cls: Type[T], **filters) -> List[T]:
    """Read records with filters and error handling."""
    try:
      async with await cls._get_session() as session:
        query = select(cls).where(cls.is_deleted.is_(False))
        if filters:
          conditions = [getattr(cls, k) == v for k, v in filters.items()]
          query = query.where(*conditions)
        result = await session.execute(query)
        return list(result.scalars().all())
    except SQLAlchemyError as e:
      logger.error(f"Error reading {cls.__name__} with filters: {str(e)}")
      raise

  async def update(self: T, **kwargs) -> T:
    """Update a record with error handling."""
    try:
      async with await self._get_session() as session:
        async with session.begin():
          for key, value in kwargs.items():
            setattr(self, key, value)
          session.add(self)
          await session.flush()
          await session.refresh(self)
          return self
    except SQLAlchemyError as e:
      logger.error(f"Error updating {self.__class__.__name__}: {str(e)}")
      raise

  async def delete(self, hard: bool = False) -> None:
    """Delete a record (soft or hard) with error handling."""
    try:
      async with await self._get_session() as session:
        async with session.begin():
          if hard:
            await session.delete(self)
          else:
            self.is_deleted = True
            self.deleted_at = datetime.now(timezone.utc)
            session.add(self)
    except SQLAlchemyError as e:
      logger.error(f"Error deleting {self.__class__.__name__}: {str(e)}")
      raise

  @classmethod
  async def bulk_create(cls: Type[T], records: List[Dict[str, Any]]) -> List[T]:
    """Bulk create records with error handling."""
    try:
      instances = [cls(**record) for record in records]
      async with await cls._get_session() as session:
        async with session.begin():
          session.add_all(instances)
          await session.flush()
          return instances
    except SQLAlchemyError as e:
      logger.error(f"Error bulk creating {cls.__name__}: {str(e)}")
      raise

  @classmethod
  async def bulk_update(cls: Type[T], records: List[Dict[str, Any]]) -> List[T]:
    """Bulk update records with error handling."""
    try:
      async with await cls._get_session() as session:
        async with session.begin():
          instances = []
          for record in records:
            if "id" not in record:
              raise ValueError("ID is required for bulk update")
            instance = await session.get(cls, record["id"])
            if instance:
              for key, value in record.items():
                if key != "id":
                  setattr(instance, key, value)
              instances.append(instance)
          await session.flush()
          return instances
    except SQLAlchemyError as e:
      logger.error(f"Error bulk updating {cls.__name__}: {str(e)}")
      raise
