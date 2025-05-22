from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from models import ChargeModel


class ChargeBaseSchema(BaseModel):
  """Base schema for charge operations."""

  name: str = Field(..., description="Unique name for the charge")
  amount: int = Field(..., description="Amount of credits to charge")
  unit: str = Field(default="credit", description="Unit of measurement (e.g., credit)")
  measure: str = Field(..., description="Measurement type (e.g., tokens, images, seconds)")
  service: str = Field(..., description="Service this charge applies to")
  action: str = Field(..., description="Specific action being charged")
  description: Optional[str] = Field(None, description="Optional description of the charge")


class ChargeCreateSchema(ChargeBaseSchema):
  """Schema for creating a new charge."""

  is_active: bool = Field(default=True, description="Whether this charge is active")


class ChargeUpdateSchema(BaseModel):
  """Schema for updating an existing charge."""

  name: Optional[str] = Field(None, description="Unique name for the charge")
  amount: Optional[int] = Field(None, description="Amount of credits to charge")
  unit: Optional[str] = Field(None, description="Unit of measurement (e.g., credit)")
  measure: Optional[str] = Field(None, description="Measurement type (e.g., tokens, images, seconds)")
  service: Optional[str] = Field(None, description="Service this charge applies to")
  action: Optional[str] = Field(None, description="Specific action being charged")
  description: Optional[str] = Field(None, description="Optional description of the charge")
  is_active: Optional[bool] = Field(None, description="Whether this charge is active")

  class Config:
    json_schema_extra = {
      "example": {
        "name": "text_generation",
        "amount": 50,
        "service": "llm",
        "action": "completion",
        "description": "Credits per 1000 tokens for text generation",
      }
    }


class ChargeResponseSchema(ChargeBaseSchema):
  """Schema for charge responses."""

  id: UUID
  is_active: bool
  created_at: Optional[str]
  updated_at: Optional[str]

  class Config:
    from_attributes = True

  @classmethod
  def from_charge(cls, charge: ChargeModel):
    """Convert a ChargeModel to ChargeResponseSchema."""
    return cls(
      id=charge.id,
      name=charge.name,
      amount=charge.amount,
      unit=charge.unit,
      measure=charge.measure,
      service=charge.service,
      action=charge.action,
      description=charge.description,
      is_active=charge.is_active,
      created_at=charge.created_at.isoformat() if hasattr(charge, "created_at") else None,
      updated_at=getattr(charge, "updated_at", charge.created_at).isoformat()
      if hasattr(charge, "updated_at") or hasattr(charge, "created_at")
      else None,
    )
