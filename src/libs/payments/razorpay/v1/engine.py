from datetime import datetime
from typing import Any, Dict, List, Optional

from razorpay.client import Client
from razorpay.errors import BadRequestError, GatewayError, ServerError, SignatureVerificationError

from common.logger import logger
from config.settings import settings
from libs.payments.base import PaymentProviderInterface
from libs.response import LibResponse


class RazorpayEngine(PaymentProviderInterface):
  """Engine for Razorpay payment processing implementing subscription and plan APIs."""

  def __init__(self):
    self.client = Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
    self.logger = logger

  # =============================================================================
  # PLAN MANAGEMENT
  # =============================================================================

  def create_plan(
    self, name: str, amount: int, currency: str, period: str, interval: int = 1, description: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a Razorpay plan for recurring payments."""
    try:
      plan_data: Dict[str, Any] = {"period": period, "interval": interval, "item": {"name": name, "amount": amount, "currency": currency}}

      if description:
        plan_data["item"]["description"] = description

      self.logger.info(f"Creating Razorpay plan: {name} - {amount} {currency}")

      # Call Razorpay client directly (it's already synchronous)
      plan = self.client.plan.create(plan_data)

      self.logger.info(f"Successfully created plan: {plan['id']}")
      return LibResponse.success_response(plan)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating plan: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "create_plan", "plan_name": name}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating plan: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_plan"}}])

  def fetch_plan(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific Razorpay plan by ID."""
    try:
      self.logger.info(f"Fetching Razorpay plan: {plan_id}")

      plan = self.client.plan.fetch(plan_id)

      return LibResponse.success_response(plan)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching plan {plan_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_plan", "plan_id": plan_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plan {plan_id}: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_plan", "plan_id": plan_id}}])

  def fetch_all_plans(self, count: int = 100, skip: int = 0, active_only: Optional[bool] = None) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all Razorpay plans with pagination."""
    try:
      options: Dict[str, Any] = {"count": count, "skip": skip}

      self.logger.info(f"Fetching Razorpay plans with options: {options}")

      plans_response = self.client.plan.all(options)

      plans = plans_response.get("items", [])
      total = plans_response.get("count", len(plans))

      # Filter by active status if requested (Razorpay doesn't have native filtering)
      if active_only is not None:
        # Note: Razorpay plans don't have an 'active' field, so we return all
        # This would need to be handled at the application level
        self.logger.warning("Razorpay plans don't support active/inactive filtering")

      return LibResponse.paginated_response(data=plans, total=total, page=(skip // count) + 1, page_size=count)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching plans: {str(e)}")
      return LibResponse.error_response([{"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plans: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])

  def update_plan(self, plan_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Update plan details - Razorpay doesn't support plan updates."""
    self.logger.warning(f"Razorpay doesn't support plan updates for plan: {plan_id}")
    return LibResponse.error_response([
      {
        "code": "operation_not_supported",
        "message": "Razorpay doesn't support updating plans after creation",
        "details": {"operation": "update_plan", "plan_id": plan_id, "suggestion": "Create a new plan instead"},
      }
    ])

  def deactivate_plan(self, plan_id: str) -> LibResponse[bool]:
    """Deactivate plan - Razorpay doesn't support plan deletion."""
    self.logger.warning(f"Razorpay doesn't support plan deactivation for plan: {plan_id}")
    return LibResponse.error_response([
      {
        "code": "operation_not_supported",
        "message": "Razorpay doesn't support deactivating or deleting plans",
        "details": {"operation": "deactivate_plan", "plan_id": plan_id, "suggestion": "Plans are immutable in Razorpay"},
      }
    ])

  # =============================================================================
  # SUBSCRIPTION MANAGEMENT
  # =============================================================================

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
    expire_by: Optional[int] = None,
    quantity: int = 1,
    notify_info: Optional[Dict[str, str]] = None,
  ) -> LibResponse[Dict[str, Any]]:
    """Create a new Razorpay subscription."""
    try:
      subscription_data: Dict[str, Any] = {
        "plan_id": plan_id,
        "customer_id": customer_id,
        "customer_notify": 1 if customer_notify else 0,
        "quantity": quantity,
      }

      if start_at:
        subscription_data["start_at"] = start_at
      if total_count:
        subscription_data["total_count"] = total_count
      if trial_period:
        subscription_data["trial_period"] = trial_period
      if expire_by:
        subscription_data["expire_by"] = expire_by
      if notes:
        subscription_data["notes"] = notes
      if addons:
        subscription_data["addons"] = addons
      if notify_info:
        subscription_data["notify_info"] = notify_info

      self.logger.info(f"Creating Razorpay subscription for customer {customer_id} with plan {plan_id}")

      subscription = self.client.subscription.create(subscription_data)

      self.logger.info(f"Successfully created subscription: {subscription['id']}")
      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating subscription: {str(e)}")
      return LibResponse.error_response([
        {
          "code": "razorpay_api_error",
          "message": str(e),
          "details": {"operation": "create_subscription", "plan_id": plan_id, "customer_id": customer_id},
        }
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating subscription: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_subscription"}}])

  def fetch_subscription(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific Razorpay subscription by ID."""
    try:
      self.logger.info(f"Fetching Razorpay subscription: {subscription_id}")

      subscription = self.client.subscription.fetch(subscription_id)

      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_subscription", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_subscription", "subscription_id": subscription_id}}
      ])

  def fetch_all_subscriptions(
    self, count: int = 100, skip: int = 0, plan_id: Optional[str] = None, status: Optional[str] = None, customer_id: Optional[str] = None
  ) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all Razorpay subscriptions with filtering options."""
    try:
      options: Dict[str, Any] = {"count": count, "skip": skip}

      if plan_id:
        options["plan_id"] = plan_id
      if status:
        options["status"] = status
      if customer_id:
        options["customer_id"] = customer_id

      self.logger.info(f"Fetching Razorpay subscriptions with options: {options}")

      subscriptions_response = self.client.subscription.all(options)

      subscriptions = subscriptions_response.get("items", [])
      total = subscriptions_response.get("count", len(subscriptions))

      return LibResponse.paginated_response(data=subscriptions, total=total, page=(skip // count) + 1, page_size=count)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching subscriptions: {str(e)}")
      return LibResponse.error_response([
        {
          "code": "razorpay_api_error",
          "message": str(e),
          "details": {"operation": "fetch_all_subscriptions", "filters": {"plan_id": plan_id, "status": status}},
        }
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching subscriptions: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_all_subscriptions"}}])

  def update_subscription(self, subscription_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Update Razorpay subscription details."""
    try:
      self.logger.info(f"Updating Razorpay subscription: {subscription_id} with data: {update_data}")

      subscription = self.client.subscription.edit(subscription_id, update_data)

      self.logger.info(f"Successfully updated subscription: {subscription_id}")
      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error updating subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "update_subscription", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error updating subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "update_subscription", "subscription_id": subscription_id}}
      ])

  def cancel_subscription(
    self, subscription_id: str, cancel_at_cycle_end: bool = True, cancellation_reason: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Cancel a Razorpay subscription."""
    try:
      cancel_data: Dict[str, Any] = {"cancel_at_cycle_end": 1 if cancel_at_cycle_end else 0}

      if cancellation_reason:
        cancel_data["notes"] = {"cancellation_reason": cancellation_reason}

      self.logger.info(f"Cancelling Razorpay subscription: {subscription_id}")

      subscription = self.client.subscription.cancel(subscription_id, cancel_data)

      self.logger.info(f"Successfully cancelled subscription: {subscription_id}")
      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error cancelling subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "cancel_subscription", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error cancelling subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "cancel_subscription", "subscription_id": subscription_id}}
      ])

  def pause_subscription(self, subscription_id: str, pause_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """Pause a Razorpay subscription."""
    try:
      pause_data: Dict[str, Any] = {"pause_at": pause_at or "now"}

      self.logger.info(f"Pausing Razorpay subscription: {subscription_id}")

      subscription = self.client.subscription.pause(subscription_id, pause_data)

      self.logger.info(f"Successfully paused subscription: {subscription_id}")
      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error pausing subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "pause_subscription", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error pausing subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "pause_subscription", "subscription_id": subscription_id}}
      ])

  def resume_subscription(self, subscription_id: str, resume_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """Resume a paused Razorpay subscription."""
    try:
      resume_data: Dict[str, Any] = {"resume_at": resume_at or "now"}

      self.logger.info(f"Resuming Razorpay subscription: {subscription_id}")

      subscription = self.client.subscription.resume(subscription_id, resume_data)

      self.logger.info(f"Successfully resumed subscription: {subscription_id}")
      return LibResponse.success_response(subscription)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error resuming subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "resume_subscription", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error resuming subscription {subscription_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "resume_subscription", "subscription_id": subscription_id}}
      ])

  # =============================================================================
  # UTILITY METHODS
  # =============================================================================

  def convert_amount_to_smallest_unit(self, amount_in_currency: float) -> int:
    """Convert amount to paise (smallest currency unit for INR/USD)."""
    return int(amount_in_currency * 100)

  def convert_smallest_unit_to_currency(self, amount_in_smallest_unit: int) -> float:
    """Convert amount from paise to main currency unit."""
    return amount_in_smallest_unit / 100

  def get_provider_name(self) -> str:
    """Get the payment provider name."""
    return "razorpay"

  def get_supported_currencies(self) -> List[str]:
    """Get list of currencies supported by Razorpay."""
    return ["INR", "USD", "EUR", "GBP", "AUD", "CAD", "SGD", "AED", "MYR"]

  def get_supported_billing_periods(self) -> List[str]:
    """Get list of billing periods supported by Razorpay."""
    return ["daily", "weekly", "monthly", "yearly"]

  # =============================================================================
  # REQUIRED ABSTRACT METHODS (NOT IMPLEMENTED FOR PLAN/SUBSCRIPTION FOCUS)
  # =============================================================================

  def create_customer(
    self, name: str, email: str, contact: Optional[str] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a customer in Razorpay."""
    try:
      customer_data: Dict[str, Any] = {"name": name, "email": email}

      if contact:
        customer_data["contact"] = contact
      if notes:
        customer_data["notes"] = notes

      self.logger.info(f"Creating Razorpay customer: {email}")

      customer = self.client.customer.create(data=customer_data)

      self.logger.info(f"Successfully created customer: {customer['id']}")
      return LibResponse.success_response(customer)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating customer: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "create_customer", "email": email}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating customer: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_customer"}}])

  def fetch_customer(self, customer_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a customer from Razorpay by ID."""
    try:
      self.logger.info(f"Fetching Razorpay customer: {customer_id}")

      customer = self.client.customer.fetch(customer_id)

      return LibResponse.success_response(customer)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching customer {customer_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_customer", "customer_id": customer_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching customer {customer_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_customer", "customer_id": customer_id}}
      ])

  def fetch_customer_by_email(self, email: str) -> LibResponse[Optional[Dict[str, Any]]]:
    """Find a customer by email address in Razorpay."""
    try:
      self.logger.info(f"Searching for Razorpay customer by email: {email}")

      customers = self.client.customer.all({"email": email})

      if customers and "items" in customers and len(customers["items"]) > 0:
        customer = customers["items"][0]
        self.logger.info(f"Found customer: {customer['id']}")
        return LibResponse.success_response(customer)
      else:
        self.logger.info(f"No customer found with email: {email}")
        return LibResponse.success_response(None)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error searching customer by email: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_customer_by_email", "email": email}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error searching customer by email: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_customer_by_email"}}])

  def update_customer(self, customer_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Customer management not implemented in this version"}])

  def create_addon(self, subscription_id: str, addon_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create an add-on for a Razorpay subscription."""
    try:
      self.logger.info(f"Creating add-on for subscription: {subscription_id}")

      addon = self.client.subscription.createAddon(subscription_id, addon_data)

      self.logger.info(f"Successfully created add-on: {addon.get('id', 'unknown')}")
      return LibResponse.success_response(addon)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating add-on: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "create_addon", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating add-on: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "create_addon", "subscription_id": subscription_id}}
      ])

  def fetch_addons(self, subscription_id: str) -> LibResponse[List[Dict[str, Any]]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Add-on management not implemented in this version"}])

  def fetch_subscription_invoices(self, subscription_id: str, count: int = 100, skip: int = 0) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch invoices for a Razorpay subscription."""
    try:
      self.logger.info(f"Fetching invoices for subscription: {subscription_id}")

      invoices_response = self.client.invoice.all({"subscription_id": subscription_id, "count": count, "skip": skip})

      invoices = invoices_response.get("items", [])
      total = invoices_response.get("count", len(invoices))

      return LibResponse.paginated_response(data=invoices, total=total, page=(skip // count) + 1, page_size=count)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching invoices: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_subscription_invoices", "subscription_id": subscription_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoices: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_subscription_invoices"}}])

  def fetch_invoice(self, invoice_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific Razorpay invoice by ID."""
    try:
      self.logger.info(f"Fetching Razorpay invoice: {invoice_id}")

      invoice = self.client.invoice.fetch(invoice_id)

      return LibResponse.success_response(invoice)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching invoice {invoice_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_invoice", "invoice_id": invoice_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoice {invoice_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_invoice", "invoice_id": invoice_id}}
      ])

  def create_invoice(self, invoice_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create a Razorpay invoice for payment."""
    try:
      self.logger.info("Creating Razorpay invoice")

      invoice = self.client.invoice.create(data=invoice_data)

      self.logger.info(f"Successfully created invoice: {invoice['id']}")
      return LibResponse.success_response(invoice)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating invoice: {str(e)}")
      return LibResponse.error_response([{"code": "razorpay_api_error", "message": str(e), "details": {"operation": "create_invoice"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error creating invoice: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_invoice"}}])

  def verify_webhook_signature(self, payload: str, signature: str) -> LibResponse[bool]:
    """Verify Razorpay webhook signature for security."""
    try:
      # Use the utility class for signature verification (raises exception if invalid)
      self.client.utility.verify_webhook_signature(payload, signature, settings.razorpay_webhook_secret)

      self.logger.info("Webhook signature verification successful")
      return LibResponse.success_response(True)

    except SignatureVerificationError:
      self.logger.warning("Webhook signature verification failed")
      return LibResponse.success_response(False)
    except Exception as e:
      self.logger.error(f"Error verifying webhook signature: {str(e)}")
      return LibResponse.error_response([
        {"code": "signature_verification_error", "message": str(e), "details": {"operation": "verify_webhook_signature"}}
      ])

  def get_subscription_status_summary(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Get comprehensive status summary for a Razorpay subscription."""
    try:
      self.logger.info(f"Getting status summary for subscription: {subscription_id}")

      # Fetch subscription details
      subscription = self.client.subscription.fetch(subscription_id)

      # Extract key metrics from subscription
      summary = {
        "subscription_id": subscription.get("id"),
        "status": subscription.get("status"),
        "plan_id": subscription.get("plan_id"),
        "customer_id": subscription.get("customer_id"),
        "current_start": subscription.get("current_start"),
        "current_end": subscription.get("current_end"),
        "next_charge_at": subscription.get("charge_at"),
        "total_count": subscription.get("total_count"),
        "paid_count": subscription.get("paid_count"),
        "remaining_count": subscription.get("remaining_count"),
        "trial_start": subscription.get("trial_start"),
        "trial_end": subscription.get("trial_end"),
        "has_scheduled_changes": subscription.get("has_scheduled_changes", False),
        "created_at": subscription.get("created_at"),
        "started_at": subscription.get("started_at"),
        "ended_at": subscription.get("ended_at"),
      }

      # Add computed fields
      summary["is_trial"] = bool(subscription.get("trial_end") and subscription.get("trial_start"))
      summary["is_active"] = subscription.get("status") in ["active", "authenticated"]
      summary["is_paused"] = subscription.get("status") == "halted"
      summary["completion_percentage"] = (
        (subscription.get("paid_count", 0) / subscription.get("total_count", 1)) * 100 if subscription.get("total_count") else 0
      )

      return LibResponse.success_response(summary)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error getting subscription status: {str(e)}")
      return LibResponse.error_response([
        {
          "code": "razorpay_api_error",
          "message": str(e),
          "details": {"operation": "get_subscription_status_summary", "subscription_id": subscription_id},
        }
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error getting subscription status: {str(e)}")
      return LibResponse.error_response([
        {
          "code": "unexpected_error",
          "message": str(e),
          "details": {"operation": "get_subscription_status_summary", "subscription_id": subscription_id},
        }
      ])

  def handle_webhook_event(self, event_type: str, payload: Dict[str, Any]) -> LibResponse[str]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Webhook handling not implemented in this version"}])

  def create_subscription_from_billing_plan(
    self, billing_plan: Dict[str, Any], customer_id: str, trial_days: Optional[int] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a Razorpay subscription from a billing plan."""
    try:
      # Get or create Razorpay plan ID for the billing plan
      razorpay_plan_id = billing_plan.get("plan_id")

      if not razorpay_plan_id:
        # Attempt to sync the billing plan with Razorpay first
        sync_response = self.sync_plan_with_provider(billing_plan)
        if not sync_response.is_successful():
          return sync_response

        razorpay_plan_id = sync_response.data.get("id")
        if not razorpay_plan_id:
          return LibResponse.error_response([{"code": "sync_failed", "message": "Failed to create Razorpay plan for billing plan"}])

      # Prepare subscription data
      subscription_data = {"plan_id": razorpay_plan_id, "customer_id": customer_id, "customer_notify": True}

      # Add trial period if specified
      if trial_days and trial_days > 0:
        subscription_data["trial_period"] = trial_days

      # Add notes if specified
      if notes:
        subscription_data["notes"] = notes

      # Create subscription
      return self.create_subscription(**subscription_data)

    except Exception as e:
      self.logger.error(f"Error creating subscription from billing plan: {str(e)}")
      return LibResponse.error_response([
        {"code": "subscription_creation_error", "message": str(e), "details": {"operation": "create_subscription_from_billing_plan"}}
      ])

  def sync_plan_with_provider(self, billing_plan: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Synchronize a billing plan with Razorpay by creating a corresponding plan."""
    try:
      # Extract billing plan details
      plan_name = billing_plan.get("name")
      amount_currency = billing_plan.get("amount")
      currency = billing_plan.get("currency", "INR")
      plan_credits = billing_plan.get("credits")

      if not all([plan_name, amount_currency, currency]):
        return LibResponse.error_response([
          {
            "code": "invalid_billing_plan",
            "message": "Billing plan missing required fields: name, amount, currency",
            "details": {"billing_plan": billing_plan},
          }
        ])

      # Convert amount to smallest unit (paise for INR)
      amount_paise = self.convert_amount_to_smallest_unit(float(amount_currency or 0))

      # Create description
      description = f"{plan_name} plan - {plan_credits} credits for {currency} {amount_currency}"

      # Create Razorpay plan
      response = self.create_plan(
        name=str(plan_name or ""), amount=amount_paise, currency=currency, period="monthly", interval=1, description=description
      )

      if response.is_successful():
        self.logger.info(f"Successfully synchronized billing plan '{plan_name}' with Razorpay")
        return response
      else:
        return response

    except Exception as e:
      self.logger.error(f"Error synchronizing billing plan with Razorpay: {str(e)}")
      return LibResponse.error_response([{"code": "sync_error", "message": str(e), "details": {"operation": "sync_plan_with_provider"}}])

  def get_subscription_metrics(
    self, subscription_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Metrics not implemented in this version"}])

  def get_plan_usage_stats(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Usage stats not implemented in this version"}])

  def retry_failed_payment(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Payment retry not implemented in this version"}])

  def get_payment_failures(self, subscription_id: str, limit: int = 10) -> LibResponse[List[Dict[str, Any]]]:
    """Not implemented - focusing on plans and subscriptions only."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Payment failure tracking not implemented in this version"}])

  def validate_plan_data(self, plan_data: Dict[str, Any]) -> LibResponse[bool]:
    """Validate plan data against Razorpay constraints."""
    try:
      errors = []

      # Check required fields
      required_fields = ["name", "amount", "currency", "period"]
      for field in required_fields:
        if field not in plan_data:
          errors.append(f"Missing required field: {field}")

      # Validate period
      valid_periods = self.get_supported_billing_periods()
      if plan_data.get("period") and plan_data["period"] not in valid_periods:
        errors.append(f"Invalid period: {plan_data['period']}. Must be one of: {valid_periods}")

      # Validate currency
      valid_currencies = self.get_supported_currencies()
      if plan_data.get("currency") and plan_data["currency"] not in valid_currencies:
        errors.append(f"Invalid currency: {plan_data['currency']}. Must be one of: {valid_currencies}")

      # Validate amount (must be positive integer in smallest unit)
      if plan_data.get("amount"):
        amount = plan_data["amount"]
        if not isinstance(amount, int) or amount <= 0:
          errors.append("Amount must be a positive integer in smallest currency unit (paise)")

      # Validate interval
      if plan_data.get("interval"):
        interval = plan_data["interval"]
        if not isinstance(interval, int) or interval <= 0:
          errors.append("Interval must be a positive integer")

      if errors:
        return LibResponse.error_response([
          {
            "code": "validation_error",
            "message": "Plan data validation failed",
            "details": {"validation_errors": errors, "operation": "validate_plan_data"},
          }
        ])

      return LibResponse.success_response(True)

    except Exception as e:
      self.logger.error(f"Error validating plan data: {str(e)}")
      return LibResponse.error_response([{"code": "validation_error", "message": str(e), "details": {"operation": "validate_plan_data"}}])

  def validate_subscription_data(self, subscription_data: Dict[str, Any]) -> LibResponse[bool]:
    """Validate subscription data against Razorpay constraints."""
    try:
      errors = []

      # Check required fields
      required_fields = ["plan_id", "customer_id"]
      for field in required_fields:
        if field not in subscription_data:
          errors.append(f"Missing required field: {field}")

      # Validate start_at timestamp (if provided)
      if subscription_data.get("start_at"):
        start_at = subscription_data["start_at"]
        if not isinstance(start_at, int) or start_at <= 0:
          errors.append("start_at must be a positive Unix timestamp")

      # Validate total_count (if provided)
      if subscription_data.get("total_count"):
        total_count = subscription_data["total_count"]
        if not isinstance(total_count, int) or total_count <= 0:
          errors.append("total_count must be a positive integer")

      # Validate trial_period (if provided)
      if subscription_data.get("trial_period"):
        trial_period = subscription_data["trial_period"]
        if not isinstance(trial_period, int) or trial_period < 0:
          errors.append("trial_period must be a non-negative integer (days)")

      # Validate customer_notify
      if "customer_notify" in subscription_data:
        customer_notify = subscription_data["customer_notify"]
        if not isinstance(customer_notify, (bool, int)) or customer_notify not in [0, 1, True, False]:
          errors.append("customer_notify must be boolean or 0/1")

      if errors:
        return LibResponse.error_response([
          {
            "code": "validation_error",
            "message": "Subscription data validation failed",
            "details": {"validation_errors": errors, "operation": "validate_subscription_data"},
          }
        ])

      return LibResponse.success_response(True)

    except Exception as e:
      self.logger.error(f"Error validating subscription data: {str(e)}")
      return LibResponse.error_response([{"code": "validation_error", "message": str(e), "details": {"operation": "validate_subscription_data"}}])


# Single instance of the engine
engine = RazorpayEngine()
logger.info("Razorpay engine initialized")
