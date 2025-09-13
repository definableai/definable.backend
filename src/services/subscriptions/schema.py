from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CreateSubscriptionWithPlanIdRequestSchema(BaseModel):
  """Schema for creating a subscription with an existing Razorpay plan_id."""

  plan_id: str = Field(..., description="Razorpay plan ID")
  total_count: int = Field(default=12, description="Total billing cycles")
  quantity: int = Field(default=1, description="Quantity of the subscription")
  start_at: Optional[int] = Field(default=None, description="Subscription start timestamp")
  expire_by: Optional[int] = Field(default=None, description="Subscription expiry timestamp")
  customer_notify: bool = Field(default=True, description="Whether to notify customer")
  addons: Optional[List[Dict[str, Any]]] = Field(default=None, description="Additional items")
  notes: Optional[Dict[str, Any]] = Field(default=None, description="Additional notes")

  class Config:
    json_schema_extra = {
      "example": {
        "plan_id": "plan_RBJeQD2aBZALra",
        "total_count": 12,
        "quantity": 1,
        "start_at": 1935689600,
        "expire_by": 1893456000,
        "customer_notify": True,
        "addons": [{"item": {"name": "Delivery charges", "amount": 9000, "currency": "INR"}}],
        "notes": {"notes_key_1": "Tea, Earl Grey, Hot", "notes_key_2": "Custom subscription"},
      }
    }


class RazorpaySubscriptionSchema(BaseModel):
  """Schema for Razorpay subscription details."""

  id: str = Field(..., description="Subscription ID")
  entity: Optional[str] = Field(default="subscription", description="Entity type")
  plan_id: Optional[str] = Field(default=None, description="Plan ID")
  status: Optional[str] = Field(default=None, description="Subscription status")
  current_start: Optional[int] = Field(default=None, description="Current cycle start")
  current_end: Optional[int] = Field(default=None, description="Current cycle end")
  ended_at: Optional[int] = Field(default=None, description="Subscription end time")
  quantity: Optional[int] = Field(default=1, description="Subscription quantity")
  notes: Optional[Dict[str, Any]] = Field(default=None, description="Additional notes")
  charge_at: Optional[int] = Field(default=None, description="Next charge time")
  start_at: Optional[int] = Field(default=None, description="Subscription start time")
  end_at: Optional[int] = Field(default=None, description="Subscription end time")
  auth_attempts: Optional[int] = Field(default=0, description="Authorization attempts")
  total_count: Optional[int] = Field(default=None, description="Total billing cycles")
  paid_count: Optional[int] = Field(default=0, description="Paid cycles count")
  customer_notify: Optional[bool] = Field(default=True, description="Customer notification status")
  created_at: Optional[int] = Field(default=None, description="Creation timestamp")
  expire_by: Optional[int] = Field(default=None, description="Expiry timestamp")
  short_url: Optional[str] = Field(default=None, description="Razorpay checkout URL")
  has_scheduled_changes: Optional[bool] = Field(default=False, description="Has scheduled changes")
  change_scheduled_at: Optional[int] = Field(default=None, description="Change scheduled time")
  source: Optional[str] = Field(default="api", description="Subscription source")
  remaining_count: Optional[int] = Field(default=None, description="Remaining billing cycles")

  class Config:
    json_schema_extra = {
      "example": {
        "id": "sub_RBgGuOS1lsgGrm",
        "entity": "subscription",
        "plan_id": "plan_RBJeQD2aBZALra",
        "status": "created",
        "current_start": None,
        "current_end": None,
        "ended_at": None,
        "quantity": 1,
        "notes": {"notes_key_1": "Tea, Earl Grey, Hot", "notes_key_2": "Tea, Earl Grey… decaf."},
        "charge_at": 1935689600,
        "start_at": 1935689600,
        "end_at": 1964716200,
        "auth_attempts": 0,
        "total_count": 12,
        "paid_count": 0,
        "customer_notify": True,
        "created_at": 1756586346,
        "expire_by": 1893456000,
        "short_url": "https://rzp.io/rzp/69K4tDO",
        "has_scheduled_changes": False,
        "change_scheduled_at": None,
        "source": "api",
        "remaining_count": 12,
      }
    }


class CreateSubscriptionWithPlanIdResponseSchema(BaseModel):
  """Schema for subscription creation response."""

  success: bool = Field(..., description="Operation success status")
  provider: str = Field(default="razorpay", description="Payment provider")
  subscription: RazorpaySubscriptionSchema = Field(..., description="Subscription details")
  customer_id: str = Field(..., description="Customer ID in payment provider")
  db_subscription_id: Optional[str] = Field(None, description="Database subscription ID")

  class Config:
    json_schema_extra = {
      "example": {
        "success": True,
        "provider": "razorpay",
        "subscription": {
          "id": "sub_RBgGuOS1lsgGrm",
          "entity": "subscription",
          "plan_id": "plan_RBJeQD2aBZALra",
          "status": "created",
          "short_url": "https://rzp.io/rzp/69K4tDO",
          "total_count": 12,
          "remaining_count": 12,
          "start_at": 1935689600,
          "expire_by": 1893456000,
          "customer_notify": True,
          "quantity": 1,
        },
        "customer_id": "cust_xyz123",
      }
    }


class SubscriptionResponse(BaseModel):
  subscription_id: str
  subscription_url: str
  created_at: datetime
  expire_by: datetime


class SubscriptionDetailResponse(BaseModel):
  """Schema for detailed subscription information."""

  id: UUID = Field(..., description="Internal subscription ID")
  organization_id: UUID = Field(..., description="Organization ID")
  user_id: UUID = Field(..., description="User ID")
  provider_id: UUID = Field(..., description="Payment provider ID")
  subscription_id: Optional[str] = Field(None, description="Provider subscription ID")
  is_active: bool = Field(..., description="Whether subscription is active")
  settings: Optional[Dict[str, Any]] = Field(None, description="Subscription settings")
  created_at: datetime = Field(..., description="Creation timestamp")
  provider_name: Optional[str] = Field(None, description="Payment provider name")

  @classmethod
  def from_model(cls, subscription):
    """Create response from SubscriptionModel."""
    return cls(
      id=subscription.id,
      organization_id=subscription.organization_id,
      user_id=subscription.user_id,
      provider_id=subscription.provider_id,
      subscription_id=subscription.subscription_id,
      is_active=subscription.is_active,
      settings=subscription.settings,
      created_at=subscription.created_at,
      provider_name=subscription.provider.name if subscription.provider else None,
    )

  class Config:
    from_attributes = True
    json_schema_extra = {
      "example": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "organization_id": "123e4567-e89b-12d3-a456-426614174001",
        "user_id": "123e4567-e89b-12d3-a456-426614174002",
        "provider_id": "123e4567-e89b-12d3-a456-426614174003",
        "subscription_id": "sub_RBgGuOS1lsgGrm",
        "is_active": True,
        "settings": {"plan_id": "plan_RBJeQD2aBZALra", "status": "active", "currency": "INR", "amount": 2499.00, "credits": 29988},
        "created_at": "2024-01-15T10:30:00Z",
        "provider_name": "razorpay",
      }
    }
