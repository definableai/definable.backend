# src/services/billing/schema.py
from pydantic import BaseModel, UUID4, conint, constr, HttpUrl
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

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

class CreditBalanceSchema(BaseModel):
    balance: int
    balance_usd: float

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

class CheckoutSessionCreate(BaseModel):
    amount_usd: float
    success_url: HttpUrl
    cancel_url: HttpUrl
    customer_email: Optional[str] = None

class StripeCustomerPortalCreate(BaseModel):
    return_url: HttpUrl

# Billable Service and Function schemas
class BillableServiceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class BillableFunctionCreate(BaseModel):
    service_id: UUID4
    name: str
    path: str
    version: str
    pricing_type: str
    base_price_credits: int
    pricing_config: Optional[Dict[str, Any]] = None

# Billing Transaction schemas
class BillingTransactionCreate(BaseModel):
    function_id: UUID4
    parent_transaction_id: Optional[UUID4] = None
    credits_used: int
    metadata: Optional[Dict[str, Any]] = None

class BillingTransactionResponse(BaseModel):
    id: UUID4
    function_id: UUID4
    function_version: str
    type: str
    status: str
    credits_used: int
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    children: List['BillingTransactionResponse'] = []

    class Config:
        from_attributes = True

# Response schemas
class TransactionResponse(BaseModel):
    id: UUID4
    type: str
    status: str
    amount_usd: float
    credits: int
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class CreditBalanceResponse(BaseModel):
    balance: int
    balance_usd: float

    class Config:
        from_attributes = True