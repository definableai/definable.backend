from datetime import datetime
import uuid
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from typing import Optional, Dict
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from src.database import Base
from enum import Enum
from database import CRUD


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


# Add BillingPlan model
class BillingPlanModel(Base):
    __tablename__ = "billing_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    amount_usd = Column(Float, nullable=False)
    credits = Column(Integer, nullable=False)
    discount_percentage = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Extend CreditBalance model with spent credits and last reset
class CreditBalanceModel(CRUD):  # Change from Base to CRUD
    __tablename__ = "credit_balances"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    balance: Mapped[int] = mapped_column(Integer, default=0)
    credits_spent: Mapped[int] = mapped_column(Integer, default=0)
    last_reset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Add relationship
    user = relationship("UserModel", back_populates="credit_balances")


class TransactionModel(CRUD):  # Change Base to CRUD
    __tablename__ = "transactions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    type: Mapped[str] = mapped_column(String, nullable=False)  # Changed from SQLEnum
    status: Mapped[str] = mapped_column(String, nullable=False)  # Changed from SQLEnum
    amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    credits: Mapped[int] = mapped_column(Integer, nullable=False)
    stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    transaction_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("UserModel", back_populates="transactions")


class LLMModelPricingModel(Base):
    __tablename__ = "llm_model_pricing"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, unique=True, nullable=False)
    credits_per_token = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ServiceModel(Base):
    """Model for storing service definitions and their credit costs."""

    __tablename__ = "billing_services"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    credit_cost = Column(Integer, nullable=False)
    description = Column(String, nullable=True)
    org_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class UserUsageModel(Base):
    """Model for tracking monthly service usage per user."""

    __tablename__ = "billing_user_usage"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    org_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    service_id = Column(
        UUID(as_uuid=True),
        ForeignKey("billing_services.id", ondelete="CASCADE"),
        nullable=False,
    )
    month = Column(DateTime, nullable=False)
    total_requests = Column(Integer, server_default="0")
    total_credits = Column(Integer, server_default="0")
    created_at = Column(DateTime, server_default=func.now())


class ApiRequestModel(Base):
    """Model for tracking individual API requests."""

    __tablename__ = "billing_api_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    org_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    service_id = Column(
        UUID(as_uuid=True), ForeignKey("billing_services.id"), nullable=False
    )
    credits_used = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    request_time = Column(DateTime, nullable=False)
    response_time = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
