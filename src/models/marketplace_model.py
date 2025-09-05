import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class AssistantType(enum.Enum):
  """Assistant type enumeration."""

  LLM_MODEL = "llm_model"
  AGENT = "agent"


class PricingType(enum.Enum):
  """Pricing type enumeration."""

  FREE = "free"
  PAID = "paid"


class MarketplaceAssistantModel(CRUD):
  """Marketplace assistant model."""

  __tablename__ = "marketplace_assistants"
  __table_args__ = (
    UniqueConstraint("assistant_type", "assistant_id", name="uq_marketplace_assistant_type_id"),
    CheckConstraint("assistant_type IN ('llm_model', 'agent')", name="ck_assistant_type"),
    CheckConstraint("pricing_type IN ('free', 'paid')", name="ck_pricing_type"),
    CheckConstraint("rating_avg >= 0 AND rating_avg <= 5", name="ck_rating_avg_range"),
    CheckConstraint("rating_count >= 0", name="ck_rating_count_positive"),
    CheckConstraint("conversation_count >= 0", name="ck_conversation_count_positive"),
  )

  assistant_type: Mapped[str] = mapped_column(String(20), nullable=False)
  assistant_id: Mapped[UUID] = mapped_column(nullable=False)
  organization_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)

  # Marketplace metadata
  is_published: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
  pricing_type: Mapped[str] = mapped_column(String(20), default="free", server_default="'free'", nullable=False)

  # Cached metrics (updated manually)
  rating_avg: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=0, server_default="0", nullable=False)
  rating_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
  conversation_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  reviews: Mapped[list["MarketplaceReviewModel"]] = relationship("MarketplaceReviewModel", back_populates="assistant", cascade="all, delete-orphan")
  usage: Mapped[Optional["MarketplaceUsageModel"]] = relationship(
    "MarketplaceUsageModel", back_populates="assistant", cascade="all, delete-orphan", uselist=False
  )

  def __repr__(self) -> str:
    return (
      f"<MarketplaceAssistant(id={self.id}, assistant_type={self.assistant_type}, "
      f"assistant_id={self.assistant_id}, is_published={self.is_published}, "
      f"rating_avg={self.rating_avg})>"
    )


class MarketplaceReviewModel(CRUD):
  """Marketplace review model."""

  __tablename__ = "marketplace_reviews"
  __table_args__ = (
    UniqueConstraint("marketplace_assistant_id", "user_id", name="uq_marketplace_review_assistant_user"),
    CheckConstraint("rating >= 1 AND rating <= 5", name="ck_rating_range"),
  )

  marketplace_assistant_id: Mapped[UUID] = mapped_column(ForeignKey("marketplace_assistants.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

  rating: Mapped[int] = mapped_column(Integer, nullable=False)
  title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
  content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  assistant: Mapped["MarketplaceAssistantModel"] = relationship("MarketplaceAssistantModel", back_populates="reviews")

  def __repr__(self) -> str:
    return (
      f"<MarketplaceReview(id={self.id}, marketplace_assistant_id={self.marketplace_assistant_id}, "
      f"user_id={self.user_id}, rating={self.rating}, title={self.title})>"
    )


class MarketplaceUsageModel(CRUD):
  """Marketplace usage model."""

  __tablename__ = "marketplace_usage"
  __table_args__ = (
    UniqueConstraint("marketplace_assistant_id", name="uq_marketplace_usage_assistant"),
    CheckConstraint("usage_count >= 0", name="ck_usage_count_positive"),
  )

  marketplace_assistant_id: Mapped[UUID] = mapped_column(ForeignKey("marketplace_assistants.id", ondelete="CASCADE"), nullable=False)
  usage_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  # Relationships
  assistant: Mapped["MarketplaceAssistantModel"] = relationship("MarketplaceAssistantModel", back_populates="usage")

  def __repr__(self) -> str:
    return f"<MarketplaceUsage(id={self.id}, marketplace_assistant_id={self.marketplace_assistant_id}, usage_count={self.usage_count})>"
