from typing import Dict, Optional

from sqlalchemy import Boolean, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import CRUD


class SubscriptionModel(CRUD):
  """Subscription model for managing user subscriptions across different payment providers."""

  __tablename__ = "subscriptions"

  organization_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
  user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
  provider_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=False, index=True)
  plan_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("billing_plans.id", ondelete="RESTRICT"), nullable=False, index=True)
  subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
  settings: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
  is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False, index=True)

  # Relationships
  organization = relationship("OrganizationModel", lazy="select")
  user = relationship("UserModel", lazy="select")
  provider = relationship("PaymentProviderModel", back_populates="subscriptions", lazy="select")
  plan = relationship("BillingPlanModel", back_populates="subscriptions", lazy="select")

  def __repr__(self) -> str:
    return f"<Subscription {self.id}: org={self.organization_id}, plan={self.plan_id}, provider={self.provider_id}, active={self.is_active}>"

  @property
  def subscription_data(self) -> Dict:
    """Get subscription data from settings."""
    return self.settings or {}

  @property
  def provider_name(self) -> Optional[str]:
    """Get provider name if provider relationship is loaded."""
    if hasattr(self, "provider") and self.provider:
      return self.provider.name
    return None

  @property
  def plan_name(self) -> Optional[str]:
    """Get plan name if plan relationship is loaded."""
    if hasattr(self, "plan") and self.plan:
      return self.plan.name
    return None

  @property
  def plan_credits(self) -> Optional[int]:
    """Get plan credits if plan relationship is loaded."""
    if hasattr(self, "plan") and self.plan:
      return self.plan.credits
    return None

  def update_settings(self, new_settings: Dict) -> None:
    """Update subscription settings."""
    if self.settings:
      self.settings.update(new_settings)
    else:
      self.settings = new_settings
