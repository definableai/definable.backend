# """Module for database schema."""

# import uuid as uuid_pkg
# from datetime import datetime
# from typing import Any, Optional

# from pydantic import BaseModel, Field, field_serializer


# class HealthCheck(BaseModel):
#   """Health check schema."""

#   name: str
#   version: str
#   description: str


# class UUIDSchema(BaseModel):
#   """UUID schema."""

#   uuid: uuid_pkg.UUID = Field(default_factory=uuid_pkg.uuid4)


# class TimestampSchema(BaseModel):
#   """Timestamp schema."""

#   created_at: datetime = Field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))
#   updated_at: Optional[datetime] = Field(default=None)

#   @field_serializer("created_at")
#   def serialize_dt(self, created_at: datetime | None, _info: Any) -> str | None:
#     """Serialize created_at."""
#     if created_at is not None:
#       return created_at.isoformat()

#     return None

#   @field_serializer("updated_at")
#   def serialize_updated_at(self, updated_at: datetime | None, _info: Any) -> str | None:
#     """Serialize updated_at."""
#     if updated_at is not None:
#       return updated_at.isoformat()
#     return None


# class PersistentDeletion(BaseModel):
#   """Persistent deletion schema."""

#   deleted_at: datetime | None = Field(default=None)
#   is_deleted: bool = False

#   @field_serializer("deleted_at")
#   def serialize_dates(self, deleted_at: datetime | None, _info: Any) -> str | None:
#     """Serialize deleted_at."""
#     if deleted_at is not None:
#       return deleted_at.isoformat()

#     return None
