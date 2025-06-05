from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import UUID4, BaseModel

from models import TransactionModel


class TransactionType(str, Enum):
  CREDIT_PURCHASE = "credit_purchase"
  CREDIT_USAGE = "credit_usage"
  REFUND = "refund"
  DISCOUNT = "discount"


class TransactionStatus(str, Enum):
  PENDING = "pending"
  COMPLETED = "completed"
  FAILED = "failed"
  REFUNDED = "refunded"


class RegionEnum(str, Enum):
  INTERNATIONAL = "international"
  INDIAN = "indian"


class PaymentProviderEnum(str, Enum):
  STRIPE = "stripe"
  RAZORPAY = "razorpay"


class BillingPlanSchema(BaseModel):
  id: UUID4
  name: str
  description: Optional[str] = None
  amount: float
  credits: int
  discount_percentage: float = 0.0
  is_active: bool = True
  created_at: datetime
  updated_at: datetime

  class Config:
    from_attributes = True


class WalletResponseSchema(BaseModel):
  id: UUID4
  balance: int
  hold: int
  credits_spent: int
  low_balance: bool = False
  error: Optional[str] = None

  class Config:
    from_attributes = True


class CreditBalanceResponseExtendedSchema(BaseModel):
  balance: int
  balance_usd: float
  credits_spent: int
  last_reset_date: Optional[datetime] = None
  low_balance: bool = False

  class Config:
    from_attributes = True


class TransactionSchema(BaseModel):
  id: UUID4
  user_id: UUID4
  organization_id: UUID4
  type: TransactionType
  status: TransactionStatus
  amount: float
  credits: int
  description: Optional[str]
  created_at: datetime

  class Config:
    from_attributes = True


class CheckoutSessionCreateSchema(BaseModel):
  customer_email: str
  plan_id: Optional[UUID4] = None
  amount: Optional[float] = None
  region: RegionEnum = RegionEnum.INTERNATIONAL

  def validate_payment_source(cls, v, values):
    if "amount" not in values and "plan_id" not in values:
      raise ValueError("Either amount or plan_id must be provided")
    return v

  class Config:
    use_enum_values = True


# Response schemas
class TransactionResponseSchema(BaseModel):
  id: UUID4
  user_id: UUID4
  organization_id: UUID4
  type: str
  status: str
  amount: Optional[float] = 0.0
  credits: int
  description: Optional[str]
  created_at: datetime

  class Config:
    from_attributes = True


class TransactionWithInvoiceSchema(TransactionResponseSchema):
  has_invoice: bool = False
  amount: Optional[float] = 0.0
  currency: str = "USD"

  @classmethod
  def from_transaction(cls, tx: TransactionModel):
    # Determine currency based on payment provider
    currency = "USD"
    if tx.payment_provider == "razorpay":
      currency = "INR"

    # Create the basic schema with all fields
    schema = cls(
      id=tx.id,
      user_id=tx.user_id,
      organization_id=tx.organization_id,
      type=tx.type,
      status=tx.status,
      amount=tx.amount or 0.0,
      credits=tx.credits,
      description=tx.description,
      created_at=tx.created_at,
      has_invoice=(tx.stripe_invoice_id is not None or tx.razorpay_invoice_id is not None),
      currency=currency,
    )
    return schema


class CreditBalanceResponseSchema(BaseModel):
  id: UUID4
  organization_id: UUID4
  balance: int
  credits_spent: int
  last_reset_date: Optional[datetime] = None
  created_at: datetime
  updated_at: datetime
  low_balance: bool = False

  class Config:
    from_attributes = True


class BillingPlanCreateSchema(BaseModel):
  name: str
  amount: float
  credits: int
  discount_percentage: float = 0.0
  is_active: bool = True


class BillingPlanResponseSchema(BaseModel):
  id: UUID4
  name: str
  amount: float
  credits: int
  discount_percentage: float
  is_active: bool
  currency: str

  class Config:
    from_attributes = True  # Change this line


class TransactionFilterParamsSchema(BaseModel):
  transaction_type: Optional[TransactionType] = None
  date_from: Optional[datetime] = None
  date_to: Optional[datetime] = None


class TransactionResponseExtendedSchema(TransactionResponseSchema):
  stripe_invoice_id: Optional[str] = None

  class Config:
    from_attributes = True


class CreditCalculationRequestSchema(BaseModel):
  amount: float


class CreditCalculationResponseSchema(BaseModel):
  amount: float
  credits: int
  currency: str = "USD"

  class Config:
    from_orm = True


class ServiceCreateSchema(BaseModel):
  name: str
  credit_cost: int
  description: Optional[str] = None


class ServiceResponseSchema(BaseModel):
  id: UUID4
  name: str
  credit_cost: int
  description: Optional[str] = None
  org_id: Optional[UUID4] = None
  created_at: datetime
  updated_at: datetime

  class Config:
    from_attributes = True


class TransactionUserSchema(BaseModel):
  id: UUID4
  email: str
  name: str

  class Config:
    from_attributes = True


class UsageHistoryItemSchema(BaseModel):
  id: UUID4
  timestamp: datetime
  description: str
  charge_name: str
  service: str
  credits_used: int
  transaction_type: str
  status: str
  user: TransactionUserSchema
  action: str
  qty: Optional[int] = None

  class Config:
    from_attributes = True


class UsageHistoryResponseSchema(BaseModel):
  usage_history: List[UsageHistoryItemSchema]
  total_credits_used: int
  total_cost_usd: float
  pagination: Dict[str, Any]

  class Config:
    from_attributes = True
