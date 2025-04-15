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
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

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


# Add BillingPlan model
class BillingPlanModel(CRUD):
  __tablename__ = "billing_plans"

  name = Column(String, nullable=False)
  amount_usd = Column(Float, nullable=False)
  credits = Column(Integer, nullable=False)
  discount_percentage = Column(Float, default=0.0)
  is_active = Column(Boolean, default=True)


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
  amount_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
  stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  transaction_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
  stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
