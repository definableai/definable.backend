from typing import Any, Dict, List, Optional

import stripe

from common.logger import logger
from config.settings import settings
from libs.payments.base import PaymentProviderInterface
from libs.response import LibResponse


class StripeEngine(PaymentProviderInterface):
  """Streamlined Stripe payment engine with only essential functionality."""

  def __init__(self):
    """Initialize Stripe client with API key from settings."""
    self.logger = logger
    stripe.api_key = settings.stripe_secret_key
    self.webhook_secret = settings.stripe_webhook_secret

    # Stripe currency configuration
    self.supported_currencies = ["usd", "eur", "gbp", "cad", "aud", "jpy"]
    self.supported_periods = ["day", "week", "month", "year"]

    self.logger.info("Stripe payment provider initialized")

  # =============================================================================
  # PLAN MANAGEMENT
  # =============================================================================

  def create_plan(
    self, name: str, amount: int, currency: str, period: str, interval: int = 1, description: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a Stripe plan (Product + Price) for recurring payments."""
    try:
      # Create Product first
      product = stripe.Product.create(
        name=name,
        description=description or name,
        metadata={"plan_name": name, "created_via": "definable_backend"},
      )

      # Create Price for the product
      price = stripe.Price.create(
        product=product.id,
        unit_amount=amount,
        currency=currency.lower(),
        recurring={"interval": period, "interval_count": interval},  # type: ignore[typeddict-item]
        metadata={"plan_name": name, "created_via": "definable_backend"},
      )

      plan_data = {
        "id": price.id,
        "product_id": product.id,
        "name": name,
        "amount": amount,
        "currency": currency.lower(),
        "period": period,
        "interval": interval,
        "description": description,
        "active": True,
        "created": price.created,
        "provider": "stripe",
      }

      self.logger.info(f"Successfully created Stripe plan: {price.id}")
      return LibResponse.success_response(plan_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating plan: {e}")
      return LibResponse.error_response([
        {
          "code": "stripe_api_error",
          "message": str(e),
          "details": {"user_message": getattr(e, "user_message", None), "type": e.type if hasattr(e, "type") else None},
        }
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating plan: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": "Failed to create plan", "details": {"error": str(e)}}])

  def fetch_all_plans(self, count: int = 100, skip: int = 0, active_only: Optional[bool] = None) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all plans with pagination and filtering."""
    try:
      # Build query parameters for Stripe
      query_params = {"limit": min(count, 100)}  # Stripe has max limit of 100

      # Skip not implemented for simplicity - would need cursor-based pagination

      if active_only is not None:
        query_params["active"] = active_only

      # Fetch prices (plans) from Stripe
      if active_only is not None:
        prices_response = stripe.Price.list(
          limit=query_params["limit"],
          active=active_only,
        )
      else:
        prices_response = stripe.Price.list(
          limit=query_params["limit"],
        )
      plans = []

      for price in prices_response.data:
        # Get product ID from price object
        product_id = price.product if isinstance(price.product, str) else price.product.id
        product = stripe.Product.retrieve(product_id)

        # Safely access recurring data
        recurring_data = price.recurring
        period = "month"  # default
        interval_count = 1  # default

        if recurring_data:
          period = recurring_data.get("interval", "month")
          interval_count = recurring_data.get("interval_count", 1)

        plan_data = {
          "id": price.id,
          "product_id": product.id,
          "name": product.name,
          "amount": price.unit_amount,
          "currency": price.currency,
          "period": period,
          "interval": interval_count,
          "description": product.description,
          "active": price.active,
          "created": price.created,
          "provider": "stripe",
        }

        plans.append(plan_data)

      self.logger.info(f"Successfully fetched {len(plans)} Stripe plans")
      return LibResponse.paginated_response(plans, total=len(plans), page=1, page_size=len(plans))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching plans: {e}")
      return LibResponse.error_response([{"code": "stripe_api_error", "message": str(e), "details": {"type": getattr(e, "type", None)}}])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plans: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

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
  ) -> LibResponse[Dict[str, Any]]:
    """Create a new Stripe subscription."""
    try:
      self.logger.info(f"Creating Stripe subscription for customer {customer_id} with plan {plan_id}")

      # Build subscription parameters
      subscription_params: Dict[str, Any] = {
        "customer": customer_id,
        "items": [{"price": plan_id}],
      }

      if notes:
        subscription_params["metadata"] = notes  # type: ignore

      # Add optional fields if provided
      if trial_period is not None:
        subscription_params["trial_period_days"] = trial_period  # type: ignore

      # Note: Stripe doesn't support total_count directly like Razorpay
      # It would need to be handled via webhooks or scheduled cancellation

      subscription = stripe.Subscription.create(**subscription_params)  # type: ignore

      self.logger.info(f"Successfully created subscription: {subscription.id}")
      return LibResponse.success_response(dict(subscription))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating subscription: {e}")
      return LibResponse.error_response([
        {
          "code": "stripe_api_error",
          "message": str(e),
          "details": {"plan_id": plan_id, "customer_id": customer_id, "type": getattr(e, "type", None)},
        }
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating subscription: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  # =============================================================================
  # CUSTOMER MANAGEMENT
  # =============================================================================

  def create_customer(
    self, name: str, email: str, contact: Optional[str] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a new customer in Stripe."""
    try:
      self.logger.info(f"Creating Stripe customer: {email}")

      # Build customer parameters
      customer_params: Dict[str, Any] = {"name": name, "email": email}

      if notes:
        customer_params["metadata"] = notes  # type: ignore

      if contact:
        customer_params["phone"] = contact

      customer = stripe.Customer.create(**customer_params)  # type: ignore

      self.logger.info(f"Successfully created customer: {customer.id}")
      return LibResponse.success_response(dict(customer))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating customer: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"email": email, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating customer: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  def fetch_customer_by_email(self, email: str) -> LibResponse[Optional[Dict[str, Any]]]:
    """Find a customer by email address."""
    try:
      customers_response = stripe.Customer.list(email=email, limit=1)

      if not customers_response.data:
        self.logger.info(f"No customer found with email: {email}")
        return LibResponse.success_response(None)

      customer = customers_response.data[0]
      self.logger.info(f"Found customer: {customer.id} for email: {email}")
      return LibResponse.success_response(dict(customer))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching customer by email {email}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"email": email, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching customer by email {email}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  # =============================================================================
  # INVOICE MANAGEMENT
  # =============================================================================

  def fetch_invoice(self, invoice_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific invoice by ID."""
    try:
      invoice = stripe.Invoice.retrieve(invoice_id)
      self.logger.info(f"Successfully fetched invoice: {invoice.id}")
      return LibResponse.success_response(dict(invoice))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching invoice {invoice_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"invoice_id": invoice_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoice {invoice_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  def create_invoice(self, invoice_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create an invoice for payment."""
    try:
      self.logger.info("Creating Stripe invoice")

      # Pass invoice data directly to Stripe
      invoice = stripe.Invoice.create(**invoice_data)  # type: ignore

      self.logger.info(f"Successfully created invoice: {invoice.id}")
      return LibResponse.success_response(dict(invoice))

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating invoice: {e}")
      return LibResponse.error_response([{"code": "stripe_api_error", "message": str(e), "details": {"type": getattr(e, "type", None)}}])
    except Exception as e:
      self.logger.error(f"Unexpected error creating invoice: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  # =============================================================================
  # WEBHOOK VERIFICATION
  # =============================================================================

  def verify_webhook_signature(self, payload: str, signature: str) -> LibResponse[bool]:
    """Verify webhook signature for security."""
    try:
      # Verify the webhook signature using Stripe's utility
      stripe.Webhook.construct_event(payload, signature, self.webhook_secret)

      self.logger.info("Webhook signature verification: valid")
      return LibResponse.success_response(True)

    except stripe.SignatureVerificationError:
      self.logger.warning("Webhook signature verification: invalid")
      return LibResponse.success_response(False)
    except Exception as e:
      self.logger.error(f"Error verifying webhook signature: {str(e)}")
      return LibResponse.error_response([{"code": "signature_verification_error", "message": str(e)}])


# Global engine instance
engine = StripeEngine()
