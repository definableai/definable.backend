from pydantic import BaseModel, UUID4, conint, constr, HttpUrl
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from sqlalchemy import func


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


class BillingPlanSchema(BaseModel):
    id: UUID4
    name: str
    amount_usd: float
    credits: int
    discount_percentage: float = 0.0
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

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
    type: TransactionType
    status: TransactionStatus
    amount_usd: float
    credits: int
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class CheckoutSessionCreateSchema(BaseModel):
    plan_id: Optional[UUID4] = None
    amount_usd: Optional[float] = None
    success_url: HttpUrl
    cancel_url: HttpUrl
    customer_email: Optional[str] = None

    def validate_payment_source(cls, v, values):
        if "amount_usd" not in values and "plan_id" not in values:
            raise ValueError("Either amount_usd or plan_id must be provided")
        return v


# Response schemas
class TransactionResponseSchema(BaseModel):
    id: UUID4
    type: str
    status: str
    amount_usd: float
    credits: int
    description: Optional[str]
    created_at: datetime

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


class CreditBalanceResponseSchema(BaseModel):
    id: UUID4
    user_id: UUID4
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
    amount_usd: float
    credits: int
    discount_percentage: float = 0.0
    is_active: bool = True


class BillingPlanResponseSchema(BillingPlanCreateSchema):
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


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
    amount_usd: float  # Changed from amount to amount_usd to match service
    credits: int

    class Config:
        from_attributes = True


class ServiceCreateSchema(BaseModel):
    name: str
    credit_cost: int
    description: Optional[str] = None


class ServiceResponseSchema(BaseModel):
    id: UUID4
    name: str
    credit_cost: int
    description: Optional[str] = None
    org_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
