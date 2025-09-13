from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import (
  JSON,
  Boolean,
  Column,
  DateTime,
  Float,
  ForeignKey,
  Integer,
  Numeric,
  String,
  Text,
  UniqueConstraint,
  func,
)
from sqlalchemy import (
  Enum as SQLAlchemyEnum,
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


class BillingCycle(str, Enum):
  MONTHLY = "MONTHLY"
  YEARLY = "YEARLY"


class ProcessingStatus(str, Enum):
  PENDING = "pending"
  PROCESSED = "processed"
  FAILED = "failed"
  SKIPPED = "skipped"
  DUPLICATE = "duplicate"


class FeatureType(str, Enum):
  MODEL_ACCESS = "model_access"
  USAGE_LIMIT = "usage_limit"
  STORAGE_LIMIT = "storage_limit"
  FEATURE_TOGGLE = "feature_toggle"
  FILE_LIMIT = "file_limit"
  TIME_LIMIT = "time_limit"


class ResetPeriod(str, Enum):
  DAILY = "daily"
  MONTHLY = "monthly"
  NEVER = "never"


class CustomerModel(CRUD):
  """Customer accounts for payment providers."""

  __tablename__ = "customer"
  __table_args__ = (
    UniqueConstraint("user_id", "provider_id", name="uq_customer_user_provider"),
    UniqueConstraint("customer_id", "provider_id", name="uq_customer_external_provider"),
  )

  user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  provider_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=False, index=True)
  customer_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False, index=True)
  provider_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

  # Relationships
  user = relationship("UserModel", lazy="select")
  payment_provider = relationship("PaymentProviderModel", lazy="select")

  def __repr__(self) -> str:
    provider_name = self.payment_provider.name if self.payment_provider else "Unknown"
    return f"<Customer {self.user_id}@{provider_name}:{self.customer_id}>"


class BillingPlanModel(CRUD):
  """Billing plans with cycle support (monthly/yearly) and multi-currency."""

  __tablename__ = "billing_plans"

  name = Column(String, nullable=False)
  description = Column(String, nullable=True)
  amount = Column(Float, nullable=False)
  credits = Column(Integer, nullable=False)
  discount_percentage = Column(Float, default=0.0)
  is_active = Column(Boolean, default=True)
  currency = Column(String(3), nullable=False, default="USD", index=True)
  cycle: Mapped[str] = mapped_column(
    SQLAlchemyEnum(BillingCycle, name="billing_cycle_enum", create_type=False),
    nullable=False,
    default=BillingCycle.MONTHLY,
    server_default="monthly",
    index=True,
  )
  plan_id = Column(String, nullable=True, index=True)
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  subscriptions = relationship("SubscriptionModel", back_populates="plan", lazy="select")
  feature_limits: Mapped[List["PlanFeatureLimitModel"]] = relationship("PlanFeatureLimitModel", back_populates="billing_plan", lazy="select")

  def __repr__(self) -> str:
    return f"<BillingPlan {self.name} {self.cycle} {self.currency}: {self.credits} credits for {self.amount}>"

  @property
  def is_yearly(self) -> bool:
    """Check if this is a yearly billing cycle."""
    return self.cycle == BillingCycle.YEARLY

  @property
  def is_monthly(self) -> bool:
    """Check if this is a monthly billing cycle."""
    return self.cycle == BillingCycle.MONTHLY

  @property
  def effective_amount(self) -> float:
    """Get the effective amount after applying discount."""
    if self.discount_percentage > 0:
      return float(self.amount) * (1 - float(self.discount_percentage) / 100)
    return float(self.amount)

  @property
  def monthly_equivalent(self) -> float:
    """Get the monthly equivalent amount."""
    if self.is_yearly:
      return float(self.amount) / 12
    return float(self.amount)


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
  payment_provider = Column(String(20), nullable=True)  # Keep for backward compatibility
  provider_id: Mapped[Optional[str]] = mapped_column(
    UUID(as_uuid=True), ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=True, index=True
  )

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
  provider = relationship("PaymentProviderModel", lazy="select")

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


class PaymentProviderModel(CRUD):
  """Payment provider model for managing different payment processors."""

  __tablename__ = "payment_providers"

  name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False, index=True)

  # Relationships
  subscriptions = relationship("SubscriptionModel", back_populates="provider", lazy="select")

  def __repr__(self) -> str:
    return f"<PaymentProvider {self.name} (active={self.is_active})>"

  @classmethod
  async def get_by_name(cls, name: str, session):
    """Get payment provider by name."""
    from sqlalchemy import select

    query = select(cls).where(cls.name == name, cls.is_active)
    result = await session.execute(query)
    return result.scalar_one_or_none()


class StatusCodeModel(CRUD):
  """Status codes for webhook events across payment providers."""

  __tablename__ = "status_codes"

  code: Mapped[str] = mapped_column(String(10), primary_key=True)
  name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
  category: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

  # Relationships
  transaction_logs = relationship("TransactionLogModel", back_populates="status_code_ref", lazy="select")

  def __repr__(self) -> str:
    return f"<StatusCode {self.code}: {self.name} ({self.category})>"

  @classmethod
  async def get_by_code(cls, code: str, session):
    """Get status code by code."""
    from sqlalchemy import select

    query = select(cls).where(cls.code == code)
    result = await session.execute(query)
    return result.scalar_one_or_none()


class TransactionLogModel(CRUD):
  """Transaction logs for webhook events from payment providers."""

  __tablename__ = "transaction_logs"
  __table_args__ = (UniqueConstraint("event_id", "provider_id", "event_type", name="uq_transaction_logs_event_provider"),)

  event_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Provider's webhook event ID
  provider_id: Mapped[str] = mapped_column(
    UUID(as_uuid=True), ForeignKey("payment_providers.id"), nullable=False, index=True
  )  # Reference to payment provider
  event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # Raw event name from provider
  status_code: Mapped[Optional[str]] = mapped_column(String(10), ForeignKey("status_codes.code"), nullable=True, index=True)
  entity_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # subscription, payment, order, invoice
  entity_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)  # Provider's entity ID
  user_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
  organization_id: Mapped[Optional[str]] = mapped_column(
    UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
  )
  customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Provider's customer ID
  amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)  # Amount in original currency
  currency: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)  # Currency code (USD, INR, etc.)
  processing_status: Mapped[ProcessingStatus] = mapped_column(
    SQLAlchemyEnum(ProcessingStatus, name="processing_status_enum", create_type=False, values_callable=lambda obj: [e.value for e in obj]),
    nullable=False,
    default=ProcessingStatus.PENDING,
    server_default="pending",
    index=True,
  )
  processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
  error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
  payload: Mapped[Dict] = mapped_column(JSON, nullable=False)  # Full webhook payload
  extracted_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)  # Extracted/processed metadata
  signature: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Webhook signature

  # Relationships
  status_code_ref = relationship("StatusCodeModel", back_populates="transaction_logs", lazy="select")
  provider = relationship("PaymentProviderModel", lazy="select")
  user = relationship("UserModel", lazy="select")
  organization = relationship("OrganizationModel", lazy="select")

  def __repr__(self) -> str:
    return f"<TransactionLog {self.provider.name if self.provider else 'Unknown'}:{self.event_type} ({self.processing_status})>"

  @property
  def is_processed(self) -> bool:
    """Check if the webhook has been successfully processed."""
    return self.processing_status == ProcessingStatus.PROCESSED

  @property
  def is_failed(self) -> bool:
    """Check if the webhook processing failed."""
    return self.processing_status == ProcessingStatus.FAILED

  @property
  def needs_retry(self) -> bool:
    """Check if the webhook needs to be retried."""
    return self.processing_status in [ProcessingStatus.PENDING, ProcessingStatus.FAILED] and self.retry_count < 3

  @classmethod
  async def get_by_event_id(cls, event_id: str, provider_id: str, event_type: str, session):
    """Get transaction log by event ID, provider ID, and event type."""
    from sqlalchemy import select

    query = select(cls).where(cls.event_id == event_id, cls.provider_id == provider_id, cls.event_type == event_type)
    result = await session.execute(query)
    return result.scalar_one_or_none()

  @classmethod
  async def get_by_event_id_and_provider_name(cls, event_id: str, provider_name: str, session):
    """Get transaction log by event ID and provider name."""
    from sqlalchemy import select

    query = select(cls).join(PaymentProviderModel).where(cls.event_id == event_id, PaymentProviderModel.name == provider_name)
    result = await session.execute(query)
    return result.scalar_one_or_none()


class PlanFeatureCategoryModel(CRUD):
  """Categories for organizing plan features (e.g., AI Models, Storage, Limits)."""

  __tablename__ = "plan_feature_categories"

  name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
  display_name: Mapped[str] = mapped_column(String(100), nullable=False)
  description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  features: Mapped[List["PlanFeatureModel"]] = relationship("PlanFeatureModel", back_populates="category", lazy="select")

  def __repr__(self) -> str:
    return f"<PlanFeatureCategory {self.name}: {self.display_name}>"


class PlanFeatureModel(CRUD):
  """Individual features that can be assigned to billing plans."""

  __tablename__ = "plan_features"

  category_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("plan_feature_categories.id", ondelete="CASCADE"), nullable=False)
  name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
  display_name: Mapped[str] = mapped_column(String(200), nullable=False)
  feature_type: Mapped[FeatureType] = mapped_column(
    SQLAlchemyEnum(FeatureType, name="feature_type_enum", create_type=False, values_callable=lambda obj: [e.value for e in obj]),
    nullable=False,
    index=True,
  )
  measurement_unit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
  description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  category: Mapped["PlanFeatureCategoryModel"] = relationship("PlanFeatureCategoryModel", back_populates="features", lazy="select")
  limits: Mapped[List["PlanFeatureLimitModel"]] = relationship("PlanFeatureLimitModel", back_populates="feature", lazy="select")

  def __repr__(self) -> str:
    return f"<PlanFeature {self.name}: {self.display_name} ({self.feature_type.value})>"


class PlanFeatureLimitModel(CRUD):
  """Specific feature limits and availability for each billing plan."""

  __tablename__ = "plan_feature_limits"
  __table_args__ = (UniqueConstraint("billing_plan_id", "feature_id", name="uq_plan_feature_limits_plan_feature"),)

  billing_plan_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("billing_plans.id", ondelete="CASCADE"), nullable=False)
  feature_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("plan_features.id", ondelete="CASCADE"), nullable=False)
  is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
  limit_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
  limit_metadata: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
  reset_period: Mapped[Optional[ResetPeriod]] = mapped_column(
    SQLAlchemyEnum(ResetPeriod, name="reset_period_enum", create_type=False, values_callable=lambda obj: [e.value for e in obj]), nullable=True
  )
  updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  billing_plan: Mapped["BillingPlanModel"] = relationship("BillingPlanModel", back_populates="feature_limits", lazy="select")
  feature: Mapped["PlanFeatureModel"] = relationship("PlanFeatureModel", back_populates="limits", lazy="select")

  def __repr__(self) -> str:
    return f"<PlanFeatureLimit {self.billing_plan.name if self.billing_plan else 'Unknown'}:{self.feature.name if self.feature else 'Unknown'} available={self.is_available}>"  # noqa: E501

  @property
  def formatted_limit(self) -> str:
    """Get a human-readable representation of the limit."""
    if not self.is_available:
      return "Not Available"
    if self.limit_value is None:
      return "Unlimited"
    unit = self.feature.measurement_unit if self.feature else ""
    return f"{self.limit_value:,}{' ' + unit if unit else ''}"
