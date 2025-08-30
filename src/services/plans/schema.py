from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from models import BillingPlanModel


class CurrencyType(str, Enum):
  USD = "USD"
  INR = "INR"


class BillingPlanBaseSchema(BaseModel):
  """Base schema for billing plan operations."""

  name: str = Field(..., description="Name of the billing plan")
  amount: float = Field(..., description="Cost of the plan in currency units")
  credits: int = Field(..., description="Number of credits provided by this plan")
  discount_percentage: float = Field(default=0.0, description="Discount percentage (0-100)")
  currency: CurrencyType = Field(default=CurrencyType.USD, description="Currency code (e.g., USD, EUR, INR)")

  @field_validator("discount_percentage")
  def validate_discount_percentage(cls, v):
    if v < 0 or v > 100:
      raise ValueError("Discount percentage must be between 0 and 100")
    return v


class BillingPlanCreateSchema(BillingPlanBaseSchema):
  """Schema for creating a new billing plan."""

  is_active: bool = Field(default=True, description="Whether this plan is active")


class BillingPlanUpdateSchema(BaseModel):
  """Schema for updating an existing billing plan."""

  name: Optional[str] = Field(None, description="Name of the billing plan")
  amount: Optional[float] = Field(None, description="Cost of the plan in currency units")
  credits: Optional[int] = Field(None, description="Number of credits provided by this plan")
  discount_percentage: Optional[float] = Field(None, description="Discount percentage (0-100)")
  is_active: Optional[bool] = Field(None, description="Whether this plan is active")
  currency: Optional[CurrencyType] = Field(None, description="Currency code (e.g., USD, EUR, INR)")

  @field_validator("discount_percentage")
  def validate_discount_percentage(cls, v):
    if v is not None and (v < 0 or v > 100):
      raise ValueError("Discount percentage must be between 0 and 100")
    return v

  class Config:
    json_schema_extra = {"example": {"name": "Pro Plan", "amount": 29.99, "credits": 30000, "discount_percentage": 0.0, "is_active": True}}


class BillingPlanResponseSchema(BillingPlanBaseSchema):
  """Schema for billing plan responses."""

  id: UUID
  is_active: bool
  created_at: datetime
  updated_at: Optional[datetime] = None

  class Config:
    from_attributes = True
    json_schema_extra = {
      "example": {
        "id": "123e4567-e89b-12d3-a456-426655440000",
        "name": "Pro Plan",
        "amount": 29.99,
        "credits": 30000,
        "discount_percentage": 0.0,
        "is_active": True,
        "currency": CurrencyType.USD,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
      }
    }

  @classmethod
  def from_plan(cls, plan: BillingPlanModel):
    """Convert a BillingPlanModel to BillingPlanResponseSchema."""
    updated_at = getattr(plan, "updated_at", None) or plan.created_at

    return cls(
      id=plan.id,
      name=plan.name,
      amount=plan.amount,
      credits=plan.credits,
      discount_percentage=plan.discount_percentage,
      is_active=plan.is_active,
      currency=plan.currency,
      created_at=plan.created_at,
      updated_at=updated_at,
    )
