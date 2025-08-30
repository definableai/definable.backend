from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from sqlalchemy import (
  JSON,
  Boolean,
  Column,
  DateTime,
  Float,
  ForeignKey,
  Integer,
  String,
  UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class TransactionType(str, Enum):
  CREDIT = "CREDIT"
  DEBIT = "DEBIT"
  HOLD = "HOLD"
  RELEASE = "RELEASE"


class TransactionStatus(str, Enum):
  PENDING = "PENDING"
  COMPLETED = "COMPLETED"
  FAILED = "FAILED"
  CANCELLED = "CANCELLED"


class CustomerModel(CRUD):
  """Customer accounts for payment providers."""

  __tablename__ = "customer"
  __table_args__ = (
    UniqueConstraint("user_id", "payment_provider", name="uq_customer_user_provider"),
    UniqueConstraint("customer_id", "payment_provider", name="uq_customer_external_provider"),
  )

  user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  payment_provider: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
  customer_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False, index=True)
  provider_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

  # Relationships
  user = relationship("UserModel", lazy="select")

  def __repr__(self) -> str:
    return f"<Customer {self.user_id}@{self.payment_provider}:{self.customer_id}>"


# Add BillingPlan model
class BillingPlanModel(CRUD):
  __tablename__ = "billing_plans"

  name = Column(String, nullable=False)
  amount = Column(Float, nullable=False)
  credits = Column(Integer, nullable=False)
  discount_percentage = Column(Float, default=0.0)
  is_active = Column(Boolean, default=True)
  currency = Column(String(3), nullable=False, default="USD")
  plan_id = Column(String, nullable=True)


class ChargeModel(CRUD):
  """Model for storing charge definitions."""

  __tablename__ = "charges"

  name = Column(String, unique=True, nullable=False)
  amount = Column(Integer, nullable=False)
  unit = Column(String, nullable=False, default="credit")
  measure = Column(String, nullable=False)
  service = Column(String, nullable=False)
  action = Column(String, nullable=False)
  description = Column(String, nullable=True)
  is_active = Column(Boolean, default=True)


# Renamed from CreditBalanceModel to WalletModel
class WalletModel(CRUD):
  __tablename__ = "wallets"

  organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
  balance: Mapped[int] = mapped_column(Integer, default=0)
  hold: Mapped[int] = mapped_column(Integer, default=0)
  credits_spent: Mapped[int] = mapped_column(Integer, default=0)
  last_reset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class TransactionModel(CRUD):
  __tablename__ = "transactions"

  user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
  organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
  type: Mapped[str] = mapped_column(String, nullable=False)
  status: Mapped[str] = mapped_column(String, nullable=False)
  credits: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
  amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
  description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  transaction_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

  # Payment Provider Information
  payment_provider = Column(String(20), nullable=True)

  # Payment Provider Data (consolidated into JSON)
  payment_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
  # Structure: {
  #   "customer_id": "cus_...",       # Customer ID from payment provider
  #   "invoice_id": "inv_...",        # Invoice ID from payment provider
  #   "payment_intent_id": "pi_...",  # Stripe payment intent ID
  #   "payment_id": "pay_...",        # Razorpay payment ID
  #   "session_id": "cs_...",         # Checkout session ID
  # }

  # Relationships
  user = relationship("UserModel", lazy="select")
  organization = relationship("OrganizationModel", lazy="select")

  def __repr__(self) -> str:
    return f"<Transaction {self.id}: {self.type} {self.credits} credits>"

  @property
  def customer_id(self) -> Optional[str]:
    """Get customer ID from payment_metadata."""
    return self.payment_metadata.get("customer_id") if self.payment_metadata else None

  @property
  def invoice_id(self) -> Optional[str]:
    """Get invoice ID from payment_metadata."""
    return self.payment_metadata.get("invoice_id") if self.payment_metadata else None

  @property
  def payment_id(self) -> Optional[str]:
    """Get payment ID from payment_metadata (Stripe: payment_intent_id, Razorpay: payment_id)."""
    if not self.payment_metadata:
      return None
    return self.payment_metadata.get("payment_intent_id") or self.payment_metadata.get("payment_id")
