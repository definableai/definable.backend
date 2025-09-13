from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from models import BillingPlanModel, FeatureType


class CurrencyType(str, Enum):
  USD = "USD"
  INR = "INR"


class BillingCycleType(str, Enum):
  MONTHLY = "MONTHLY"
  YEARLY = "YEARLY"


class PlanFeatureCategorySchema(BaseModel):
  """Schema for plan feature categories."""

  id: UUID
  name: str
  display_name: str
  description: Optional[str] = None
  sort_order: int

  class Config:
    from_attributes = True


class PlanFeatureSchema(BaseModel):
  """Schema for individual plan features."""

  id: UUID
  name: str
  display_name: str
  feature_type: FeatureType
  measurement_unit: Optional[str] = None
  description: Optional[str] = None
  sort_order: int
  category: PlanFeatureCategorySchema

  class Config:
    from_attributes = True


class PlanFeatureLimitSchema(BaseModel):
  """Schema for plan feature limits."""

  id: UUID
  is_available: bool
  limit_value: Optional[int] = None
  limit_metadata: Optional[Dict] = None
  reset_period: Optional[str] = None  # Use string to avoid enum conversion issues
  formatted_limit: str = Field(..., description="Human-readable limit representation")
  feature: PlanFeatureSchema

  class Config:
    from_attributes = True


class PlanFeaturesGroupedSchema(BaseModel):
  """Schema for grouped plan features by category."""

  category: PlanFeatureCategorySchema
  features: List[PlanFeatureLimitSchema]


class BillingPlanBaseSchema(BaseModel):
  """Base schema for billing plan operations."""

  name: str = Field(..., description="Name of the billing plan")
  description: Optional[str] = Field(None, description="Description of the billing plan")
  amount: float = Field(..., description="Cost of the plan in currency units")
  credits: int = Field(..., description="Number of credits provided by this plan")
  discount_percentage: float = Field(default=0.0, description="Discount percentage (0-100)")
  currency: CurrencyType = Field(default=CurrencyType.USD, description="Currency code (e.g., USD, EUR, INR)")
  cycle: BillingCycleType = Field(default=BillingCycleType.MONTHLY, description="Billing cycle (MONTHLY, YEARLY)")

  @field_validator("discount_percentage")
  def validate_discount_percentage(cls, v):
    if v < 0 or v > 100:
      raise ValueError("Discount percentage must be between 0 and 100")
    return v

  @field_validator("amount")
  def validate_amount(cls, v):
    if v < 0:
      raise ValueError("Amount must be non-negative")
    return v

  @field_validator("credits")
  def validate_credits(cls, v):
    if v < 0:
      raise ValueError("Credits must be non-negative")
    return v


class BillingPlanCreateSchema(BillingPlanBaseSchema):
  """Schema for creating a new billing plan."""

  is_active: bool = Field(default=True, description="Whether this plan is active")


class BillingPlanUpdateSchema(BaseModel):
  """Schema for updating an existing billing plan."""

  name: Optional[str] = Field(None, description="Name of the billing plan")
  description: Optional[str] = Field(None, description="Description of the billing plan")
  amount: Optional[float] = Field(None, description="Cost of the plan in currency units")
  credits: Optional[int] = Field(None, description="Number of credits provided by this plan")
  discount_percentage: Optional[float] = Field(None, description="Discount percentage (0-100)")
  is_active: Optional[bool] = Field(None, description="Whether this plan is active")
  currency: Optional[CurrencyType] = Field(None, description="Currency code (e.g., USD, EUR, INR)")
  cycle: Optional[BillingCycleType] = Field(None, description="Billing cycle (MONTHLY, YEARLY)")

  @field_validator("discount_percentage")
  def validate_discount_percentage(cls, v):
    if v is not None and (v < 0 or v > 100):
      raise ValueError("Discount percentage must be between 0 and 100")
    return v

  @field_validator("amount")
  def validate_amount(cls, v):
    if v is not None and v < 0:
      raise ValueError("Amount must be non-negative")
    return v

  @field_validator("credits")
  def validate_credits(cls, v):
    if v is not None and v < 0:
      raise ValueError("Credits must be non-negative")
    return v

  class Config:
    json_schema_extra = {
      "example": {
        "name": "Pro Plan",
        "description": "Best for growing teams and businesses",
        "amount": 29.99,
        "credits": 30000,
        "discount_percentage": 0.0,
        "is_active": True,
        "cycle": "MONTHLY",
      }
    }


class BillingPlanResponseSchema(BillingPlanBaseSchema):
  """Schema for billing plan responses."""

  id: UUID
  is_active: bool
  created_at: datetime
  updated_at: Optional[datetime] = None
  plan_id: Optional[str] = Field(None, description="External payment provider plan ID")

  # Computed fields from model properties
  is_yearly: bool = Field(..., description="Whether this is a yearly billing cycle")
  is_monthly: bool = Field(..., description="Whether this is a monthly billing cycle")
  effective_amount: float = Field(..., description="Effective amount after applying discount")
  monthly_equivalent: float = Field(..., description="Monthly equivalent amount")

  # Features and settings
  features: List[PlanFeaturesGroupedSchema] = Field(default_factory=list, description="Grouped plan features by category")
  feature_count: int = Field(0, description="Total number of features available in this plan")
  available_features_count: int = Field(0, description="Number of available features in this plan")

  class Config:
    from_attributes = True
    json_schema_extra = {
      "example": {
        "id": "123e4567-e89b-12d3-a456-426655440000",
        "name": "Pro Plan",
        "description": "Best for growing teams and businesses",
        "amount": 279.0,
        "credits": 30000,
        "discount_percentage": 20.0,
        "is_active": True,
        "currency": CurrencyType.USD,
        "cycle": BillingCycleType.YEARLY,
        "plan_id": "plan_123abc",
        "is_yearly": True,
        "is_monthly": False,
        "effective_amount": 223.2,
        "monthly_equivalent": 23.25,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
      }
    }

  @classmethod
  def from_plan(cls, plan: BillingPlanModel, include_features: bool = True, manual_features: Optional[dict] = None):
    """Convert a BillingPlanModel to BillingPlanResponseSchema."""
    if manual_features is None:
      manual_features = {}
    updated_at = getattr(plan, "updated_at", None) or plan.created_at

    # Group features by category
    features_grouped = []
    feature_count = 0
    available_features_count = 0

    # Get feature limits from manual features only (to avoid lazy loading in async context)
    feature_limits_to_use = []
    if include_features:
      plan_id = str(plan.id)
      if plan_id in manual_features and manual_features[plan_id]:
        # Use manually provided features (from join queries)
        feature_limits_to_use = manual_features[plan_id]
      # Note: We don't use SQLAlchemy relationship here to avoid MissingGreenlet errors
      # All feature data should be provided via manual_features parameter

    if feature_limits_to_use:
      # Group limits by category - using simpler approach to avoid type issues
      categories_dict = {}

      for limit in feature_limits_to_use:
        if limit.feature and limit.feature.category:
          category = limit.feature.category
          category_id = str(category.id)

          if category_id not in categories_dict:
            categories_dict[category_id] = {"category_obj": category, "limit_list": []}

          categories_dict[category_id]["limit_list"].append(limit)
          feature_count += 1
          if limit.is_available:
            available_features_count += 1

      # Convert to schema format
      for category_data in categories_dict.values():
        category_obj = category_data["category_obj"]
        limit_list = category_data["limit_list"]

        # Convert SQLAlchemy category object to dict for Pydantic
        category_dict = {
          "id": category_obj.id,
          "name": category_obj.name,
          "display_name": category_obj.display_name,
          "description": category_obj.description,
          "sort_order": category_obj.sort_order,
        }
        category_schema = PlanFeatureCategorySchema(**category_dict)
        feature_schemas = []

        # Sort features by sort_order within category
        sorted_limits = sorted(limit_list, key=lambda x: x.feature.sort_order)
        for limit in sorted_limits:
          # Convert SQLAlchemy feature object to dict for Pydantic
          feature_dict = {
            "id": limit.feature.id,
            "name": limit.feature.name,
            "display_name": limit.feature.display_name,
            "feature_type": limit.feature.feature_type,
            "measurement_unit": limit.feature.measurement_unit,
            "description": limit.feature.description,
            "sort_order": limit.feature.sort_order,
            "category": category_schema,
          }
          feature_schema = PlanFeatureSchema(**feature_dict)

          # Handle reset_period enum conversion safely
          reset_period_str = None
          if limit.reset_period:
            # Convert enum to string, handling both enum objects and string values
            if hasattr(limit.reset_period, "value"):
              reset_period_str = limit.reset_period.value
            else:
              reset_period_str = str(limit.reset_period)

          limit_schema = PlanFeatureLimitSchema(
            id=limit.id,
            is_available=limit.is_available,
            limit_value=limit.limit_value,
            limit_metadata=limit.limit_metadata,
            reset_period=reset_period_str,
            formatted_limit=limit.formatted_limit,
            feature=feature_schema,
          )
          feature_schemas.append(limit_schema)

        features_grouped.append(PlanFeaturesGroupedSchema(category=category_schema, features=feature_schemas))

      # Sort categories by sort_order
      features_grouped.sort(key=lambda x: x.category.sort_order)

    return cls(
      id=plan.id,
      name=plan.name,
      description=plan.description,
      amount=plan.amount,
      credits=plan.credits,
      discount_percentage=plan.discount_percentage,
      is_active=plan.is_active,
      currency=plan.currency,
      cycle=plan.cycle,
      plan_id=plan.plan_id,
      is_yearly=plan.is_yearly,
      is_monthly=plan.is_monthly,
      effective_amount=round(plan.effective_amount, 2),
      monthly_equivalent=round(plan.monthly_equivalent, 2),
      created_at=plan.created_at,
      updated_at=updated_at,
      features=features_grouped,
      feature_count=feature_count,
      available_features_count=available_features_count,
    )
