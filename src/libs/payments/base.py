from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from libs.response import LibResponse


class PaymentProviderInterface(ABC):
  """
  Abstract base class defining common interface for payment providers (Stripe, Razorpay).

  All methods return LibResponse objects for consistent success/error handling:
  - Success: LibResponse.success_response(data)
  - Error: LibResponse.error_response(errors)
  - Paginated: LibResponse.paginated_response(data, total, page, page_size)
  - With metadata: LibResponse.with_metadata(data, **meta)

  Return Types Guide:
  - Single items: LibResponse[Dict[str, Any]]
  - Lists/Collections: LibResponse[List[Dict[str, Any]]]
  - Boolean operations: LibResponse[bool]
  - Status messages: LibResponse[str]
  - Optional items: LibResponse[Optional[Dict[str, Any]]]

  Implementation Notes:
  - Use paginated_response() for lists with count/total info
  - Use error_response() for API failures or validation errors
  - Include provider-specific metadata using with_metadata()
  - Log operations using self.logger for consistent tracking
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
  def fetch_plan(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Fetch a specific plan by provider plan ID.

    Args:
        plan_id: Provider-specific plan ID

    Returns:
        Dict containing plan details
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
        Dict containing list of plans and pagination info
    """
    pass

  @abstractmethod
  def update_plan(self, plan_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Update plan details (if supported by provider).

    Args:
        plan_id: Provider-specific plan ID
        update_data: Fields to update

    Returns:
        Dict containing updated plan details
    """
    pass

  @abstractmethod
  def deactivate_plan(self, plan_id: str) -> LibResponse[bool]:
    """
    Deactivate/delete a plan.

    Args:
        plan_id: Provider-specific plan ID

    Returns:
        Dict containing operation result
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

  @abstractmethod
  def fetch_subscription(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Fetch a specific subscription by ID.

    Args:
        subscription_id: Provider-specific subscription ID

    Returns:
        Dict containing subscription details
    """
    pass

  @abstractmethod
  def fetch_all_subscriptions(
    self, count: int = 100, skip: int = 0, plan_id: Optional[str] = None, status: Optional[str] = None, customer_id: Optional[str] = None
  ) -> LibResponse[List[Dict[str, Any]]]:
    """
    Fetch all subscriptions with filtering options.

    Args:
        count: Number of subscriptions to fetch
        skip: Number to skip for pagination
        plan_id: Filter by plan ID (optional)
        status: Filter by status (optional)
        customer_id: Filter by customer ID (optional)

    Returns:
        Dict containing list of subscriptions and pagination info
    """
    pass

  @abstractmethod
  def update_subscription(self, subscription_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Update subscription details.

    Args:
        subscription_id: Provider-specific subscription ID
        update_data: Fields to update (plan_id, quantity, etc.)

    Returns:
        Dict containing updated subscription details
    """
    pass

  @abstractmethod
  def cancel_subscription(
    self, subscription_id: str, cancel_at_cycle_end: bool = True, cancellation_reason: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """
    Cancel a subscription.

    Args:
        subscription_id: Provider-specific subscription ID
        cancel_at_cycle_end: Whether to cancel at end of current cycle
        cancellation_reason: Reason for cancellation (optional)

    Returns:
        Dict containing updated subscription details
    """
    pass

  @abstractmethod
  def pause_subscription(self, subscription_id: str, pause_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """
    Pause a subscription (if supported by provider).

    Args:
        subscription_id: Provider-specific subscription ID
        pause_at: When to pause ('now' or 'cycle_end')

    Returns:
        Dict containing updated subscription details
    """
    pass

  @abstractmethod
  def resume_subscription(self, subscription_id: str, resume_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """
    Resume a paused subscription (if supported by provider).

    Args:
        subscription_id: Provider-specific subscription ID
        resume_at: When to resume ('now' or 'cycle_end')

    Returns:
        Dict containing updated subscription details
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
  def fetch_customer(self, customer_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Fetch customer details by ID.

    Args:
        customer_id: Provider-specific customer ID

    Returns:
        Dict containing customer details
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

  @abstractmethod
  def update_customer(self, customer_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Update customer information.

    Args:
        customer_id: Provider-specific customer ID
        update_data: Fields to update

    Returns:
        Dict containing updated customer details
    """
    pass

  # =============================================================================
  # ADD-ON MANAGEMENT
  # =============================================================================

  @abstractmethod
  def create_addon(self, subscription_id: str, addon_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Create an add-on charge for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID
        addon_data: Add-on details (item, amount, quantity, etc.)

    Returns:
        Dict containing add-on details
    """
    pass

  @abstractmethod
  def fetch_addons(self, subscription_id: str) -> LibResponse[List[Dict[str, Any]]]:
    """
    Fetch all add-ons for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID

    Returns:
        Dict containing list of add-ons
    """
    pass

  # =============================================================================
  # INVOICE MANAGEMENT
  # =============================================================================

  @abstractmethod
  def fetch_subscription_invoices(self, subscription_id: str, count: int = 100, skip: int = 0) -> LibResponse[List[Dict[str, Any]]]:
    """
    Fetch invoices for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID
        count: Number of invoices to fetch
        skip: Number to skip for pagination

    Returns:
        Dict containing list of invoices
    """
    pass

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
  # WEBHOOK & UTILITY METHODS
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

  @abstractmethod
  def convert_amount_to_smallest_unit(self, amount_in_currency: float) -> int:
    """
    Convert amount to smallest currency unit (cents/paise).

    Args:
        amount_in_currency: Amount in main currency unit

    Returns:
        Amount in smallest unit
    """
    pass

  @abstractmethod
  def convert_smallest_unit_to_currency(self, amount_in_smallest_unit: int) -> float:
    """
    Convert amount from smallest unit to main currency unit.

    Args:
        amount_in_smallest_unit: Amount in smallest unit

    Returns:
        Amount in main currency unit
    """
    pass

  # =============================================================================
  # PROVIDER-SPECIFIC INTEGRATION HELPERS
  # =============================================================================

  @abstractmethod
  def get_subscription_status_summary(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Get comprehensive status summary for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID

    Returns:
        Dict with subscription details, metrics, and status information
    """
    pass

  @abstractmethod
  def handle_webhook_event(self, event_type: str, payload: Dict[str, Any]) -> LibResponse[str]:
    """
    Process provider-specific webhook events.

    Args:
        event_type: Type of webhook event
        payload: Webhook payload data

    Returns:
        Dict containing processing result status and message
    """
    pass

  # =============================================================================
  # BILLING INTEGRATION HELPERS
  # =============================================================================

  @abstractmethod
  def create_subscription_from_billing_plan(
    self, billing_plan: Dict[str, Any], customer_id: str, trial_days: Optional[int] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """
    Helper to create subscription based on internal billing plan model.

    Args:
        billing_plan: Internal BillingPlanModel data
        customer_id: Provider-specific customer ID
        trial_days: Trial period in days (optional)
        notes: Additional metadata (optional)

    Returns:
        Dict containing both plan and subscription details from provider
    """
    pass

  @abstractmethod
  def sync_plan_with_provider(self, billing_plan: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """
    Sync internal billing plan with payment provider plan.

    Args:
        billing_plan: Internal BillingPlanModel data

    Returns:
        Dict containing provider plan details
    """
    pass

  # =============================================================================
  # ANALYTICS & REPORTING
  # =============================================================================

  @abstractmethod
  def get_subscription_metrics(
    self, subscription_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
  ) -> LibResponse[Dict[str, Any]]:
    """
    Get analytics metrics for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID
        start_date: Start date for metrics (optional)
        end_date: End date for metrics (optional)

    Returns:
        Dict containing metrics like revenue, payment success rate, etc.
    """
    pass

  @abstractmethod
  def get_plan_usage_stats(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Get usage statistics for a plan.

    Args:
        plan_id: Provider-specific plan ID

    Returns:
        Dict containing plan usage statistics
    """
    pass

  # =============================================================================
  # ERROR HANDLING & RETRY LOGIC
  # =============================================================================

  @abstractmethod
  def retry_failed_payment(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """
    Retry a failed subscription payment (if supported).

    Args:
        subscription_id: Provider-specific subscription ID

    Returns:
        Dict containing retry attempt result
    """
    pass

  @abstractmethod
  def get_payment_failures(self, subscription_id: str, limit: int = 10) -> LibResponse[List[Dict[str, Any]]]:
    """
    Get recent payment failures for a subscription.

    Args:
        subscription_id: Provider-specific subscription ID
        limit: Maximum number of failures to return

    Returns:
        List of payment failure details
    """
    pass

  # =============================================================================
  # PROVIDER INFO & CAPABILITIES
  # =============================================================================

  @abstractmethod
  def get_provider_name(self) -> str:
    """
    Get the payment provider name.

    Returns:
        String provider identifier (e.g., 'stripe', 'razorpay')
    """
    pass

  @abstractmethod
  def get_supported_currencies(self) -> List[str]:
    """
    Get list of currencies supported by this provider.

    Returns:
        List of currency codes
    """
    pass

  @abstractmethod
  def get_supported_billing_periods(self) -> List[str]:
    """
    Get list of billing periods supported by this provider.

    Returns:
        List of period strings (daily, weekly, monthly, yearly, etc.)
    """
    pass

  @abstractmethod
  def validate_plan_data(self, plan_data: Dict[str, Any]) -> LibResponse[bool]:
    """
    Validate plan data against provider constraints.

    Args:
        plan_data: Plan data to validate

    Returns:
        Boolean indicating if plan data is valid
    """
    pass

  @abstractmethod
  def validate_subscription_data(self, subscription_data: Dict[str, Any]) -> LibResponse[bool]:
    """
    Validate subscription data against provider constraints.

    Args:
        subscription_data: Subscription data to validate

    Returns:
        Boolean indicating if subscription data is valid
    """
    pass
