from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from libs.response import LibResponse


class PaymentProviderInterface(ABC):
  """
  Streamlined abstract base class for payment providers (Stripe, Razorpay).

  Only includes methods that are actually used in the codebase to reduce complexity
  and maintenance overhead. All methods return LibResponse objects for consistent
  success/error handling.

  Currently used operations:
  - Plan creation and listing
  - Subscription creation
  - Customer management (create, find by email)
  - Invoice operations (create, fetch)
  - Webhook signature verification
  """

  # =============================================================================
  # PLAN MANAGEMENT
  # =============================================================================

  @abstractmethod
  def create_plan(
    self, name: str, amount: int, currency: str, period: str, interval: int = 1, description: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """
    Create a recurring payment plan.

    Args:
        name: Plan name
        amount: Amount in smallest currency unit (cents/paise)
        currency: Currency code (USD, INR, etc.)
        period: Billing period (daily, weekly, monthly, yearly)
        interval: Number of periods between charges
        description: Plan description (optional)

    Returns:
        Dict containing plan details with provider-specific plan ID
    """
    pass

  @abstractmethod
  def fetch_all_plans(self, count: int = 100, skip: int = 0, active_only: Optional[bool] = None) -> LibResponse[List[Dict[str, Any]]]:
    """
    Fetch all plans with pagination and filtering.

    Args:
        count: Number of plans to fetch
        skip: Number of plans to skip for pagination
        active_only: Filter by active status (optional)

    Returns:
        List of plan details
    """
    pass

  # =============================================================================
  # SUBSCRIPTION MANAGEMENT
  # =============================================================================

  @abstractmethod
  def create_subscription(
    self,
    plan_id: str,
    customer_id: str,
    start_at: Optional[int] = None,
    total_count: Optional[int] = None,
    trial_period: Optional[int] = None,
    customer_notify: bool = True,
    notes: Optional[Dict[str, Any]] = None,
    addons: Optional[List[Dict[str, Any]]] = None,
  ) -> LibResponse[Dict[str, Any]]:
    """
    Create a new subscription.

    Args:
        plan_id: Provider-specific plan ID
        customer_id: Provider-specific customer ID
        start_at: Subscription start timestamp (optional)
        total_count: Total billing cycles (optional, infinite if not specified)
        trial_period: Trial period in days (optional)
        customer_notify: Whether to notify customer
        notes: Additional metadata
        addons: List of add-on items (optional)

    Returns:
        Dict containing subscription details
    """
    pass

  # =============================================================================
  # CUSTOMER MANAGEMENT
  # =============================================================================

  @abstractmethod
  def create_customer(
    self, name: str, email: str, contact: Optional[str] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """
    Create a customer in the payment provider.

    Args:
        name: Customer name
        email: Customer email
        contact: Customer phone/contact (optional)
        notes: Additional metadata (optional)

    Returns:
        Dict containing customer details with provider-specific customer ID
    """
    pass

  @abstractmethod
  def fetch_customer_by_email(self, email: str) -> LibResponse[Optional[Dict[str, Any]]]:
    """
    Find customer by email address.

    Args:
        email: Customer email address

    Returns:
        Dict containing customer details or None if not found
    """
    pass

  # =============================================================================
  # INVOICE MANAGEMENT
  # =============================================================================

  @abstractmethod
  def fetch_invoice(self, invoice_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Fetch a specific invoice by ID.

    Args:
        invoice_id: Provider-specific invoice ID

    Returns:
        Dict containing invoice details
    """
    pass

  @abstractmethod
  def create_invoice(self, invoice_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Create an invoice for payment.

    Args:
        invoice_data: Invoice details including customer, line items, etc.

    Returns:
        Dict containing created invoice details
    """
    pass

  # =============================================================================
  # WEBHOOK VERIFICATION
  # =============================================================================

  @abstractmethod
  def verify_webhook_signature(self, payload: str, signature: str) -> LibResponse[bool]:
    """
    Verify webhook signature for security.

    Args:
        payload: Raw webhook payload string
        signature: Signature header from webhook request

    Returns:
        Boolean indicating if signature is valid
    """
    pass
