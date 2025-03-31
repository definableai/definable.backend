import uuid
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
  func,
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


# Add BillingPlan model
class BillingPlanModel(CRUD):
  __tablename__ = "billing_plans"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  name = Column(String, nullable=False)
  amount_usd = Column(Float, nullable=False)
  credits = Column(Integer, nullable=False)
  discount_percentage = Column(Float, default=0.0)
  is_active = Column(Boolean, default=True)
  created_at = Column(DateTime, default=datetime.utcnow)
  updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChargeModel(CRUD):
  """Model for storing charge definitions."""

  __tablename__ = "charges"

  id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  name = Column(String, unique=True, nullable=False)
  amount = Column(Integer, nullable=False)
  unit = Column(String, nullable=False, default="credit")
  measure = Column(String, nullable=False)
  service = Column(String, nullable=False)
  action = Column(String, nullable=False)
  description = Column(String, nullable=True)
  is_active = Column(Boolean, default=True)
  created_at = Column(DateTime, server_default=func.now())
  updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Renamed from CreditBalanceModel to WalletModel
class WalletModel(CRUD):
  __tablename__ = "wallets"

  id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  user_id: Mapped[UUID] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("users.id", ondelete="CASCADE"),
    unique=True,
    nullable=False,
  )
  balance: Mapped[int] = mapped_column(Integer, default=0)
  hold: Mapped[int] = mapped_column(Integer, default=0)
  credits_spent: Mapped[int] = mapped_column(Integer, default=0)
  last_reset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

  # Add relationship
  user = relationship("UserModel", back_populates="wallets")
  transactions = relationship("TransactionModel", primaryjoin="WalletModel.user_id==foreign(TransactionModel.user_id)")


class TransactionModel(CRUD):
  __tablename__ = "transactions"

  id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
  user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
  org_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
  type: Mapped[str] = mapped_column(String, nullable=False)
  status: Mapped[str] = mapped_column(String, nullable=False)
  credits: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
  amount_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
  stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  transaction_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
  stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

  # Relationships
  user = relationship("UserModel", back_populates="transactions")
  wallet = relationship("WalletModel", primaryjoin="foreign(WalletModel.user_id)==TransactionModel.user_id", uselist=False)
