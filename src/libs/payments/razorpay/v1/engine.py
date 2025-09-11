import base64
from typing import Any, Dict, List, Optional

import httpx
from razorpay.client import Client
from razorpay.errors import BadRequestError, GatewayError, ServerError

from common.logger import logger
from config.settings import settings
from libs.payments.base import PaymentProviderInterface
from libs.response import LibResponse


class RazorpayEngine(PaymentProviderInterface):
  """Streamlined Razorpay payment engine with only essential functionality."""

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

  def fetch_all_plans(self, count: int = 100, skip: int = 0, active_only: Optional[bool] = None) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all Razorpay plans with pagination and optional filtering."""
    try:
      plans_data = self.client.plan.all({"count": count, "skip": skip})
      plans = plans_data.get("items", [])

      self.logger.info(f"Successfully fetched {len(plans)} plans")
      return LibResponse.paginated_response(plans, total=plans_data.get("count", len(plans)), page=skip // count + 1, page_size=count)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching plans: {str(e)}")
      return LibResponse.error_response([{"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plans: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])

  # =============================================================================
  # SUBSCRIPTION MANAGEMENT
  # =============================================================================

  async def create_subscription(
    self,
    plan_id: str,
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
    """Create a new Razorpay subscription using httpx."""
    try:
      url = "https://api.razorpay.com/v1/subscriptions"

      # Create basic auth header
      auth_string = f"{settings.razorpay_key_id}:{settings.razorpay_key_secret}"
      auth_bytes = auth_string.encode("ascii")
      auth_header = base64.b64encode(auth_bytes).decode("ascii")

      headers = {"Content-Type": "application/json", "Authorization": f"Basic {auth_header}"}

      # Build payload
      payload: Dict[str, Any] = {
        "plan_id": plan_id,
        "customer_notify": 1 if customer_notify else 0,
        "quantity": quantity,
      }

      # Add optional fields if provided
      if start_at is not None:
        payload["start_at"] = start_at
      if total_count is not None:
        payload["total_count"] = total_count
      if trial_period is not None:
        payload["trial_period"] = trial_period
      if expire_by is not None:
        payload["expire_by"] = expire_by
      if addons is not None:
        payload["addons"] = addons
      if notes is not None:
        payload["notes"] = notes
      if notify_info is not None:
        payload["notify_info"] = notify_info

      self.logger.info(f"Creating Razorpay subscription with plan {plan_id}")

      async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=30.0)

        # Handle HTTP errors
        if response.status_code >= 400:
          error_detail = response.text
          try:
            error_json = response.json()
            error_detail = error_json.get("error", {}).get("description", error_detail)
          except Exception:
            pass

          self.logger.error(f"Razorpay API error creating subscription (HTTP {response.status_code}): {error_detail}")
          return LibResponse.error_response([
            {
              "code": "razorpay_api_error",
              "message": f"HTTP {response.status_code}: {error_detail}",
              "details": {"operation": "create_subscription", "plan_id": plan_id},
            }
          ])

        subscription = response.json()

      self.logger.info(f"Successfully created subscription: {subscription['id']}")
      return LibResponse.success_response(subscription)

    except Exception as e:
      self.logger.error(f"Unexpected error creating subscription: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_subscription"}}])

  # =============================================================================
  # CUSTOMER MANAGEMENT
  # =============================================================================

  def create_customer(
    self, name: str, email: str, contact: Optional[str] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a new customer in Razorpay."""
    try:
      customer_data: Dict[str, Any] = {"name": name, "email": email}

      if contact:
        customer_data["contact"] = contact
      if notes:
        customer_data["notes"] = notes

      self.logger.info(f"Creating Razorpay customer: {email}")

      customer = self.client.customer.create(customer_data)

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

  def fetch_customer_by_email(self, email: str) -> LibResponse[Optional[Dict[str, Any]]]:
    """Find a customer by email address."""
    try:
      # Use Razorpay's all() method to search by email
      customers_response = self.client.customer.all({"email": email})
      customers = customers_response.get("items", [])

      if not customers:
        self.logger.info(f"No customer found with email: {email}")
        return LibResponse.success_response(None)

      customer = customers[0]  # Take first match
      self.logger.info(f"Found customer: {customer['id']} for email: {email}")
      return LibResponse.success_response(customer)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching customer by email {email}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_customer_by_email", "email": email}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching customer by email {email}: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_customer_by_email"}}])

  # =============================================================================
  # INVOICE MANAGEMENT
  # =============================================================================

  def fetch_invoice(self, invoice_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific Razorpay invoice by ID."""
    try:
      invoice = self.client.invoice.fetch(invoice_id)
      self.logger.info(f"Successfully fetched invoice: {invoice['id']}")
      return LibResponse.success_response(invoice)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error fetching invoice {invoice_id}: {str(e)}")
      return LibResponse.error_response([
        {"code": "razorpay_api_error", "message": str(e), "details": {"operation": "fetch_invoice", "invoice_id": invoice_id}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoice {invoice_id}: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "fetch_invoice"}}])

  def create_invoice(self, invoice_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create an invoice for payment."""
    try:
      self.logger.info("Creating Razorpay invoice")

      invoice = self.client.invoice.create(invoice_data)

      self.logger.info(f"Successfully created invoice: {invoice['id']}")
      return LibResponse.success_response(invoice)

    except (BadRequestError, GatewayError, ServerError) as e:
      self.logger.error(f"Razorpay API error creating invoice: {str(e)}")
      return LibResponse.error_response([{"code": "razorpay_api_error", "message": str(e), "details": {"operation": "create_invoice"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error creating invoice: {str(e)}")
      return LibResponse.error_response([{"code": "unexpected_error", "message": str(e), "details": {"operation": "create_invoice"}}])

  # =============================================================================
  # WEBHOOK VERIFICATION
  # =============================================================================

  def verify_webhook_signature(self, payload: str, signature: str) -> LibResponse[bool]:
    """Verify webhook signature for security."""
    try:
      import hashlib
      import hmac

      # Extract signature from Razorpay header
      expected_signature = signature

      # Create our signature
      secret = settings.razorpay_webhook_secret
      computed_signature = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

      # Compare signatures
      is_valid = hmac.compare_digest(computed_signature, expected_signature)

      self.logger.info(f"Webhook signature verification: {'valid' if is_valid else 'invalid'}")
      return LibResponse.success_response(is_valid)

    except Exception as e:
      self.logger.error(f"Error verifying webhook signature: {str(e)}")
      return LibResponse.error_response([{"code": "signature_verification_error", "message": str(e)}])


# Global engine instance
engine = RazorpayEngine()
