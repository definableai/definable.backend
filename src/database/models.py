import logging
import uuid as uuid_pkg
from datetime import datetime, timezone

from sqlalchemy import DateTime, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .postgres import async_session


class UUIDMixin:
  id: Mapped[uuid_pkg.UUID] = mapped_column(
    UUID(as_uuid=True),
    primary_key=True,
    server_default=text("gen_random_uuid()"),
  )


class TimestampMixin:
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=datetime.now(timezone.utc),
    server_default=text("CURRENT_TIMESTAMP"),
  )


# class SoftDeleteMixin:
#   deleted_at: Mapped[datetime] = mapped_column(
#     DateTime,
#     nullable=True,
#     onupdate=datetime.now(timezone.utc).replace(tzinfo=None),
#   )
#   is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class Base(DeclarativeBase):
  pass


class CRUD(Base, UUIDMixin, TimestampMixin):
  """CRUD mixin."""

  __abstract__ = True

  async def create(self):
    async with async_session() as session:
      session.add(self)
      await session.commit()
      return self

  @classmethod
  async def read(cls, _id: uuid_pkg.UUID):
    """Read a record from the database."""
    async with async_session() as session:
      try:
        # Ensure the query is correctly formed
        query = select(cls).where(cls.id == _id)
        result = await session.execute(query)

        # result = await session.execute(query)
        instance = result.scalars().first()

        if instance:
          logging.info(f"Retrieved {instance} from the database.")
        else:
          logging.info(f"No record found with ID: {_id}")

        return instance
      except SQLAlchemyError as e:
        # Log the error and re-raise it
        logging.error(f"An error occurred: {e}")
        raise

  @classmethod
  async def read_all(cls, **kwargs):
    """Read all records from the database."""
    async with async_session() as session:
      try:
        if kwargs:
          filters = [getattr(cls, key) == value for key, value in kwargs.items()]
          query = select(cls).where(*filters)
        else:
          query = select(cls)
        result = await session.execute(query)
        instances = result.scalars().all()
        return instances
      except SQLAlchemyError as e:
        logging.error(f"An error occurred while fetching records: {e}")
        raise

  async def update(self, **kwargs):
    """Update a record in the database."""
    async with async_session() as session:
      for attr, value in kwargs.items():
        setattr(self, attr, value)
      session.add(self)
      await session.commit()
      return self

  async def delete(self):
    """Delete a record from the database."""
    async with async_session() as session:
      await session.delete(self)
      await session.commit()

  async def replace(self, **kwargs):
    """Replace a record in the database."""
    async with async_session() as session:
      await session.delete(self)
      await session.commit()
      for attr, value in kwargs.items():
        setattr(self, attr, value)
      session.add(self)
      await session.commit()
      return self
