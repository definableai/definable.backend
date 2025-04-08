from uuid import UUID

from pydantic import BaseModel


class OrganizationCreate(BaseModel):
  """Organization create schema."""

  name: str


class OrganizationResponse(BaseModel):
  """Organization response schema."""

  id: UUID
  name: str
  slug: str

  class Config:
    from_attributes = True
