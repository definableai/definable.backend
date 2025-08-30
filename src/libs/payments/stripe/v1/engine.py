from datetime import datetime
from typing import Any, Dict, List, Optional

import stripe

from common.logger import logger
from config.settings import settings
from libs.payments.base import PaymentProviderInterface
from libs.response import LibResponse


class StripeEngine(PaymentProviderInterface):
  """
  Stripe payment provider implementation following the PaymentProviderInterface.

  Handles all Stripe-specific operations including plans, subscriptions, customers,
  invoices, webhooks, and billing integrations.
  """

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
    """Create a recurring payment plan in Stripe."""
    try:
      # Create the product first
      product = stripe.Product.create(
        name=name, description=description or f"Subscription plan: {name}", metadata={"created_via": "definable_backend"}
      )

      # Create the price (plan) linked to the product
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

  def fetch_plan(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific plan by Stripe price ID."""
    try:
      price = stripe.Price.retrieve(plan_id)

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

      return LibResponse.success_response(plan_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching plan {plan_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"plan_id": plan_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plan {plan_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"plan_id": plan_id}}])

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
        if price.recurring:  # Only include recurring prices (plans)
          # Get product details
          product_id = price.product if isinstance(price.product, str) else price.product.id
          product = stripe.Product.retrieve(product_id)

          # Safely access recurring data
          recurring_data: Dict[str, Any] = price.recurring or {}

          plan_data = {
            "id": price.id,
            "product_id": product.id,
            "name": product.name,
            "amount": price.unit_amount,
            "currency": price.currency,
            "period": recurring_data.get("interval", "month"),
            "interval": recurring_data.get("interval_count", 1),
            "description": product.description,
            "active": price.active,
            "created": price.created,
            "provider": "stripe",
          }
          plans.append(plan_data)

      return LibResponse.paginated_response(data=plans, total=len(plans), page=(skip // count) + 1, page_size=count)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching plans: {e}")
      return LibResponse.error_response([{"code": "stripe_api_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching plans: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"operation": "fetch_all_plans"}}])

  def update_plan(self, plan_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Update plan details (Stripe prices are immutable, but we can update the product)."""
    try:
      price = stripe.Price.retrieve(plan_id)
      product_id = price.product if isinstance(price.product, str) else price.product.id

      # Update product details (name, description, etc.)
      product_updates = {}
      if "name" in update_data:
        product_updates["name"] = update_data["name"]
      if "description" in update_data:
        product_updates["description"] = update_data["description"]
      if "metadata" in update_data:
        product_updates["metadata"] = update_data["metadata"]

      if product_updates:
        stripe.Product.modify(product_id, **product_updates)

      # Get updated plan data
      return self.fetch_plan(plan_id)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error updating plan {plan_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"plan_id": plan_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error updating plan {plan_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"plan_id": plan_id}}])

  def deactivate_plan(self, plan_id: str) -> LibResponse[bool]:
    """Deactivate a plan by setting it to inactive."""
    try:
      stripe.Price.modify(plan_id, active=False)
      self.logger.info(f"Successfully deactivated Stripe plan: {plan_id}")
      return LibResponse.success_response(True)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error deactivating plan {plan_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"plan_id": plan_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error deactivating plan {plan_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"plan_id": plan_id}}])

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
      # Build subscription items
      items = [{"price": plan_id}]

      # Add addons if provided
      if addons:
        for addon in addons:
          if "price_id" in addon:
            items.append({"price": addon["price_id"], "quantity": addon.get("quantity", 1)})

      # Build subscription parameters
      sub_params: Dict[str, Any] = {
        "customer": customer_id,
        "items": items,
      }

      if start_at:
        sub_params["billing_cycle_anchor"] = start_at
      if trial_period:
        sub_params["trial_period_days"] = trial_period
      if notes:
        sub_params["metadata"] = notes

      # Create the subscription
      subscription = stripe.Subscription.create(**sub_params)

      # Build response data
      sub_data = {
        "id": subscription.id,
        "customer_id": subscription.customer,
        "plan_id": plan_id,
        "status": subscription.status,
        "current_period_start": getattr(subscription, "current_period_start", None),
        "current_period_end": getattr(subscription, "current_period_end", None),
        "trial_start": subscription.trial_start,
        "trial_end": subscription.trial_end,
        "created": subscription.created,
        "provider": "stripe",
      }

      self.logger.info(f"Successfully created Stripe subscription: {subscription.id}")
      return LibResponse.success_response(sub_data)

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
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"operation": "create_subscription"}}])

  def fetch_subscription(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific subscription by ID."""
    try:
      subscription = stripe.Subscription.retrieve(subscription_id)

      sub_data = {
        "id": subscription.id,
        "customer_id": subscription.customer,
        "status": subscription.status,
        "current_period_start": getattr(subscription, "current_period_start", None),
        "current_period_end": getattr(subscription, "current_period_end", None),
        "trial_start": subscription.trial_start,
        "trial_end": subscription.trial_end,
        "created": subscription.created,
        "provider": "stripe",
        "items": [{"price_id": item.price.id, "quantity": item.quantity} for item in subscription.items.data],
      }

      return LibResponse.success_response(sub_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching subscription {subscription_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching subscription {subscription_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def fetch_all_subscriptions(
    self, count: int = 100, skip: int = 0, plan_id: Optional[str] = None, status: Optional[str] = None, customer_id: Optional[str] = None
  ) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all subscriptions with filtering options."""
    try:
      # Build query parameters
      query_params: Dict[str, Any] = {"limit": min(count, 100)}

      if customer_id:
        query_params["customer"] = customer_id
      if status:
        query_params["status"] = status
      if plan_id:
        query_params["price"] = plan_id

      # Fetch subscriptions
      subs_response = stripe.Subscription.list(**query_params)
      subscriptions = []

      for sub in subs_response.data:
        sub_data = {
          "id": sub.id,
          "customer_id": sub.customer,
          "status": sub.status,
          "current_period_start": getattr(sub, "current_period_start", None),
          "current_period_end": getattr(sub, "current_period_end", None),
          "trial_start": sub.trial_start,
          "trial_end": sub.trial_end,
          "created": sub.created,
          "provider": "stripe",
          "items": [{"price_id": item.price.id, "quantity": item.quantity} for item in sub.items.data],
        }
        subscriptions.append(sub_data)

      return LibResponse.paginated_response(data=subscriptions, total=len(subscriptions), page=(skip // count) + 1, page_size=count)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching subscriptions: {e}")
      return LibResponse.error_response([{"code": "stripe_api_error", "message": str(e), "details": {"operation": "fetch_all_subscriptions"}}])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching subscriptions: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"operation": "fetch_all_subscriptions"}}])

  def update_subscription(self, subscription_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Update subscription details."""
    try:
      # Build update parameters
      update_params: Dict[str, Any] = {}

      if "items" in update_data:
        update_params["items"] = update_data["items"]
      if "metadata" in update_data:
        update_params["metadata"] = update_data["metadata"]
      if "trial_end" in update_data:
        update_params["trial_end"] = update_data["trial_end"]

      # Update the subscription
      subscription = stripe.Subscription.modify(subscription_id, **update_params)

      # Build response data
      sub_data = {
        "id": subscription.id,
        "customer_id": subscription.customer,
        "status": subscription.status,
        "current_period_start": getattr(subscription, "current_period_start", None),
        "current_period_end": getattr(subscription, "current_period_end", None),
        "trial_start": subscription.trial_start,
        "trial_end": subscription.trial_end,
        "created": subscription.created,
        "provider": "stripe",
      }

      self.logger.info(f"Successfully updated Stripe subscription: {subscription_id}")
      return LibResponse.success_response(sub_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error updating subscription {subscription_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error updating subscription {subscription_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def cancel_subscription(
    self, subscription_id: str, cancel_at_cycle_end: bool = True, cancellation_reason: Optional[str] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Cancel a Stripe subscription."""
    try:
      if cancel_at_cycle_end:
        # Schedule cancellation at period end
        subscription = stripe.Subscription.modify(
          subscription_id,
          cancel_at_period_end=True,
          metadata={"cancellation_reason": cancellation_reason} if cancellation_reason else {},
        )
      else:
        # Cancel immediately
        subscription = stripe.Subscription.cancel(subscription_id)

      sub_data = {
        "id": subscription.id,
        "customer_id": subscription.customer,
        "status": subscription.status,
        "current_period_start": getattr(subscription, "current_period_start", None),
        "current_period_end": getattr(subscription, "current_period_end", None),
        "canceled_at": getattr(subscription, "canceled_at", None),
        "cancel_at_period_end": getattr(subscription, "cancel_at_period_end", False),
        "provider": "stripe",
      }

      self.logger.info(f"Successfully cancelled Stripe subscription: {subscription_id}")
      return LibResponse.success_response(sub_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error cancelling subscription {subscription_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error cancelling subscription {subscription_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def pause_subscription(self, subscription_id: str, pause_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """Pause a subscription (Stripe uses pause_collection)."""
    try:
      stripe.Subscription.modify(
        subscription_id,
        pause_collection={"behavior": "mark_uncollectible"},  # type: ignore[typeddict-item]
      )

      self.logger.info(f"Successfully paused Stripe subscription: {subscription_id}")
      return self.fetch_subscription(subscription_id)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error pausing subscription {subscription_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error pausing subscription {subscription_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def resume_subscription(self, subscription_id: str, resume_at: Optional[str] = None) -> LibResponse[Dict[str, Any]]:
    """Resume a paused subscription."""
    try:
      stripe.Subscription.modify(subscription_id, pause_collection="")

      self.logger.info(f"Successfully resumed Stripe subscription: {subscription_id}")
      return self.fetch_subscription(subscription_id)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error resuming subscription {subscription_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error resuming subscription {subscription_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  # =============================================================================
  # CUSTOMER MANAGEMENT
  # =============================================================================

  def create_customer(
    self, name: str, email: str, contact: Optional[str] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create a customer in Stripe."""
    try:
      customer_data: Dict[str, Any] = {"name": name, "email": email}

      if contact:
        customer_data["phone"] = contact
      if notes:
        customer_data["metadata"] = notes

      customer = stripe.Customer.create(**customer_data)

      customer_result = {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "created": customer.created,
        "provider": "stripe",
        "metadata": dict(customer.metadata) if customer.metadata else {},
      }

      self.logger.info(f"Successfully created Stripe customer: {customer.id}")
      return LibResponse.success_response(customer_result)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating customer: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"email": email, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating customer: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"operation": "create_customer"}}])

  def fetch_customer(self, customer_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch customer details by ID."""
    try:
      customer = stripe.Customer.retrieve(customer_id)

      customer_data = {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "created": customer.created,
        "provider": "stripe",
        "metadata": dict(customer.metadata) if customer.metadata else {},
      }

      return LibResponse.success_response(customer_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching customer {customer_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"customer_id": customer_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching customer {customer_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"customer_id": customer_id}}])

  def fetch_customer_by_email(self, email: str) -> LibResponse[Optional[Dict[str, Any]]]:
    """Find customer by email address."""
    try:
      customers = stripe.Customer.list(email=email, limit=1)

      if customers.data:
        customer = customers.data[0]
        customer_data = {
          "id": customer.id,
          "name": customer.name,
          "email": customer.email,
          "phone": customer.phone,
          "created": customer.created,
          "provider": "stripe",
          "metadata": dict(customer.metadata) if customer.metadata else {},
        }
        return LibResponse.success_response(customer_data)
      else:
        return LibResponse.success_response(None)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error searching customer by email: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"email": email, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error searching customer by email: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"email": email}}])

  def update_customer(self, customer_id: str, update_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Update customer information."""
    try:
      customer = stripe.Customer.modify(customer_id, **update_data)

      customer_result = {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "created": customer.created,
        "provider": "stripe",
        "metadata": dict(customer.metadata) if customer.metadata else {},
      }

      self.logger.info(f"Successfully updated Stripe customer: {customer_id}")
      return LibResponse.success_response(customer_result)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error updating customer {customer_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"customer_id": customer_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error updating customer {customer_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"customer_id": customer_id}}])

  # =============================================================================
  # ADD-ON MANAGEMENT
  # =============================================================================

  def create_addon(self, subscription_id: str, addon_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create an add-on charge for a subscription (using invoice items)."""
    try:
      # Get the subscription to find the customer
      subscription = stripe.Subscription.retrieve(subscription_id)

      customer_id = subscription.customer if isinstance(subscription.customer, str) else subscription.customer.id
      invoice_item = stripe.InvoiceItem.create(
        customer=customer_id,
        amount=addon_data.get("amount", 0),
        currency=addon_data.get("currency", "usd"),
        description=addon_data.get("description", "Add-on charge"),
        subscription=subscription_id,
      )

      addon_result = {
        "id": invoice_item.id,
        "subscription_id": subscription_id,
        "customer_id": subscription.customer,
        "amount": invoice_item.amount,
        "currency": invoice_item.currency,
        "description": invoice_item.description,
        "created": invoice_item.date,
        "provider": "stripe",
      }

      self.logger.info(f"Successfully created Stripe add-on: {invoice_item.id}")
      return LibResponse.success_response(addon_result)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating add-on: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating add-on: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def fetch_addons(self, subscription_id: str) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch all add-ons for a subscription (invoice items)."""
    try:
      subscription = stripe.Subscription.retrieve(subscription_id)
      customer_id = subscription.customer if isinstance(subscription.customer, str) else subscription.customer.id
      invoice_items = stripe.InvoiceItem.list(customer=customer_id)

      addons = []
      for item in invoice_items.data:
        addon_data = {
          "id": item.id,
          "subscription_id": subscription_id,
          "customer_id": subscription.customer,
          "amount": item.amount,
          "currency": item.currency,
          "description": item.description,
          "created": item.date,
          "provider": "stripe",
        }
        addons.append(addon_data)

      return LibResponse.success_response(addons)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching add-ons: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching add-ons: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  # =============================================================================
  # INVOICE MANAGEMENT
  # =============================================================================

  def fetch_subscription_invoices(self, subscription_id: str, count: int = 100, skip: int = 0) -> LibResponse[List[Dict[str, Any]]]:
    """Fetch invoices for a subscription."""
    try:
      invoices_response = stripe.Invoice.list(subscription=subscription_id, limit=min(count, 100))
      invoices = []

      for invoice in invoices_response.data:
        invoice_data = {
          "id": invoice.id,
          "subscription_id": getattr(invoice, "subscription", None),
          "customer_id": invoice.customer,
          "amount_paid": invoice.amount_paid,
          "amount_due": invoice.amount_due,
          "currency": invoice.currency,
          "status": invoice.status,
          "created": invoice.created,
          "due_date": invoice.due_date,
          "provider": "stripe",
        }
        invoices.append(invoice_data)

      return LibResponse.paginated_response(data=invoices, total=len(invoices), page=(skip // count) + 1, page_size=count)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching invoices: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoices: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def fetch_invoice(self, invoice_id: str) -> LibResponse[Dict[str, Any]]:
    """Fetch a specific invoice by ID."""
    try:
      invoice = stripe.Invoice.retrieve(invoice_id)

      invoice_data = {
        "id": invoice.id,
        "subscription_id": getattr(invoice, "subscription", None),
        "customer_id": invoice.customer,
        "amount_paid": invoice.amount_paid,
        "amount_due": invoice.amount_due,
        "currency": invoice.currency,
        "status": invoice.status,
        "created": invoice.created,
        "due_date": invoice.due_date,
        "provider": "stripe",
      }

      return LibResponse.success_response(invoice_data)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error fetching invoice {invoice_id}: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"invoice_id": invoice_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error fetching invoice {invoice_id}: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"invoice_id": invoice_id}}])

  def create_invoice(self, invoice_data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Create an invoice for payment."""
    try:
      # Create invoice items first if provided
      line_items = invoice_data.get("line_items", [])
      customer_id = invoice_data["customer_id"]

      for item in line_items:
        stripe.InvoiceItem.create(
          customer=customer_id,
          amount=item["amount"],
          currency=item.get("currency", "usd"),
          description=item.get("description", ""),
        )

      # Create and finalize the invoice
      invoice = stripe.Invoice.create(customer=customer_id)
      if invoice.id:
        stripe.Invoice.finalize_invoice(invoice.id)

      invoice_result = {
        "id": invoice.id,
        "customer_id": customer_id,
        "amount_due": invoice.amount_due,
        "currency": invoice.currency,
        "status": invoice.status,
        "created": invoice.created,
        "provider": "stripe",
      }

      self.logger.info(f"Successfully created Stripe invoice: {invoice.id}")
      return LibResponse.success_response(invoice_result)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error creating invoice: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"operation": "create_invoice", "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error creating invoice: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"operation": "create_invoice"}}])

  # =============================================================================
  # WEBHOOK & UTILITY METHODS
  # =============================================================================

  def verify_webhook_signature(self, payload: str, signature: str) -> LibResponse[bool]:
    """Verify Stripe webhook signature for security."""
    try:
      stripe.Webhook.construct_event(payload, signature, self.webhook_secret)
      self.logger.info("Stripe webhook signature verification successful")
      return LibResponse.success_response(True)

    except ValueError:
      self.logger.warning("Invalid Stripe webhook payload")
      return LibResponse.success_response(False)
    except stripe.SignatureVerificationError:
      self.logger.warning("Invalid Stripe webhook signature")
      return LibResponse.success_response(False)
    except Exception as e:
      self.logger.error(f"Error verifying Stripe webhook signature: {e}")
      return LibResponse.error_response([{"code": "signature_verification_error", "message": str(e)}])

  def convert_amount_to_smallest_unit(self, amount_in_currency: float) -> int:
    """Convert amount to cents (smallest currency unit)."""
    return int(amount_in_currency * 100)

  def convert_smallest_unit_to_currency(self, amount_in_smallest_unit: int) -> float:
    """Convert amount from cents to main currency unit."""
    return amount_in_smallest_unit / 100

  def get_provider_name(self) -> str:
    """Get the payment provider name."""
    return "stripe"

  def get_supported_currencies(self) -> List[str]:
    """Get list of currencies supported by Stripe."""
    return self.supported_currencies

  def get_supported_billing_periods(self) -> List[str]:
    """Get list of billing periods supported by Stripe."""
    return self.supported_periods

  # =============================================================================
  # PROVIDER-SPECIFIC INTEGRATION HELPERS
  # =============================================================================

  def get_subscription_status_summary(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Get comprehensive status summary for a Stripe subscription."""
    try:
      subscription = stripe.Subscription.retrieve(subscription_id)
      customer_id = subscription.customer if isinstance(subscription.customer, str) else subscription.customer.id
      customer = stripe.Customer.retrieve(customer_id)

      summary = {
        "subscription_id": subscription.id,
        "customer_id": subscription.customer,
        "customer_email": customer.email,
        "status": subscription.status,
        "current_period_start": getattr(subscription, "current_period_start", None),
        "current_period_end": getattr(subscription, "current_period_end", None),
        "trial_start": subscription.trial_start,
        "trial_end": subscription.trial_end,
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "canceled_at": getattr(subscription, "canceled_at", None),
        "created": subscription.created,
        "provider": "stripe",
      }

      return LibResponse.success_response(summary)

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error getting subscription status: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error getting subscription status: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e), "details": {"subscription_id": subscription_id}}])

  def handle_webhook_event(self, event_type: str, payload: Dict[str, Any]) -> LibResponse[str]:
    """Process Stripe webhook events."""
    try:
      self.logger.info(f"Processing Stripe webhook event: {event_type}")

      # Handle different event types
      if event_type.startswith("invoice."):
        return LibResponse.success_response(f"Processed invoice event: {event_type}")
      elif event_type.startswith("subscription."):
        return LibResponse.success_response(f"Processed subscription event: {event_type}")
      elif event_type.startswith("customer."):
        return LibResponse.success_response(f"Processed customer event: {event_type}")
      else:
        self.logger.info(f"Unhandled Stripe webhook event type: {event_type}")
        return LibResponse.success_response(f"Event type {event_type} not handled")

    except Exception as e:
      self.logger.error(f"Error processing Stripe webhook event: {e}")
      return LibResponse.error_response([{"code": "webhook_processing_error", "message": str(e)}])

  # =============================================================================
  # BILLING INTEGRATION HELPERS
  # =============================================================================

  def create_subscription_from_billing_plan(
    self, billing_plan: Dict[str, Any], customer_id: str, trial_days: Optional[int] = None, notes: Optional[Dict[str, Any]] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Create subscription from billing plan."""
    try:
      plan_id = billing_plan.get("plan_id")
      if not plan_id:
        # Create plan first
        sync_response = self.sync_plan_with_provider(billing_plan)
        if not sync_response.is_successful():
          return sync_response
        plan_id = sync_response.data.get("id") if sync_response.data else None

      if not plan_id:
        return LibResponse.error_response([{"code": "missing_plan_id", "message": "Could not get or create plan ID"}])

      return self.create_subscription(
        plan_id=plan_id,
        customer_id=customer_id,
        trial_period=trial_days,
        notes=notes,
      )

    except Exception as e:
      self.logger.error(f"Error creating subscription from billing plan: {e}")
      return LibResponse.error_response([{"code": "subscription_creation_error", "message": str(e)}])

  def sync_plan_with_provider(self, billing_plan: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
    """Sync billing plan with Stripe."""
    try:
      name = billing_plan.get("name")
      amount = billing_plan.get("amount")
      currency = billing_plan.get("currency", "usd")
      plan_credits = billing_plan.get("credits")

      if not all([name, amount, currency]):
        return LibResponse.error_response([{"code": "invalid_billing_plan", "message": "Missing required fields: name, amount, currency"}])

      amount_cents = self.convert_amount_to_smallest_unit(float(amount or 0))
      description = f"{name} plan - {plan_credits} credits for {currency.upper()} {amount}"

      return self.create_plan(
        name=str(name),
        amount=amount_cents,
        currency=currency,
        period="month",
        interval=1,
        description=description,
      )

    except Exception as e:
      self.logger.error(f"Error syncing billing plan with Stripe: {e}")
      return LibResponse.error_response([{"code": "sync_error", "message": str(e)}])

  # =============================================================================
  # ANALYTICS & REPORTING
  # =============================================================================

  def get_subscription_metrics(
    self, subscription_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
  ) -> LibResponse[Dict[str, Any]]:
    """Get analytics metrics for a subscription."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Metrics not implemented in this version"}])

  def get_plan_usage_stats(self, plan_id: str) -> LibResponse[Dict[str, Any]]:
    """Get usage statistics for a plan."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Usage stats not implemented in this version"}])

  # =============================================================================
  # ERROR HANDLING & RETRY LOGIC
  # =============================================================================

  def retry_failed_payment(self, subscription_id: str) -> LibResponse[Dict[str, Any]]:
    """Retry a failed subscription payment."""
    try:
      # Get the latest invoice for the subscription
      invoices = stripe.Invoice.list(subscription=subscription_id, limit=1)
      if not invoices.data:
        return LibResponse.error_response([{"code": "no_invoices", "message": "No invoices found for subscription"}])

      invoice = invoices.data[0]
      if invoice.status == "open":
        # Attempt to pay the invoice
        paid_invoice = stripe.Invoice.pay(invoice.id) if invoice.id else invoice

        result = {
          "invoice_id": paid_invoice.id,
          "status": paid_invoice.status,
          "amount_paid": paid_invoice.amount_paid,
          "provider": "stripe",
        }

        return LibResponse.success_response(result)
      else:
        return LibResponse.success_response({"message": "No payment retry needed", "status": invoice.status})

    except stripe.StripeError as e:
      self.logger.error(f"Stripe API error retrying payment: {e}")
      return LibResponse.error_response([
        {"code": "stripe_api_error", "message": str(e), "details": {"subscription_id": subscription_id, "type": getattr(e, "type", None)}}
      ])
    except Exception as e:
      self.logger.error(f"Unexpected error retrying payment: {e}")
      return LibResponse.error_response([{"code": "internal_error", "message": str(e)}])

  def get_payment_failures(self, subscription_id: str, limit: int = 10) -> LibResponse[List[Dict[str, Any]]]:
    """Get recent payment failures for a subscription."""
    return LibResponse.error_response([{"code": "not_implemented", "message": "Payment failure tracking not implemented in this version"}])

  # =============================================================================
  # VALIDATION METHODS
  # =============================================================================

  def validate_plan_data(self, plan_data: Dict[str, Any]) -> LibResponse[bool]:
    """Validate plan data against Stripe constraints."""
    try:
      errors = []

      # Required fields
      required_fields = ["name", "amount", "currency", "period"]
      for field in required_fields:
        if field not in plan_data:
          errors.append(f"Missing required field: {field}")

      # Validate period
      if plan_data.get("period") and plan_data["period"] not in self.supported_periods:
        errors.append(f"Invalid period: {plan_data['period']}. Must be one of: {self.supported_periods}")

      # Validate currency
      if plan_data.get("currency") and plan_data["currency"].lower() not in self.supported_currencies:
        errors.append(f"Invalid currency: {plan_data['currency']}. Must be one of: {self.supported_currencies}")

      # Validate amount (must be positive integer in smallest unit)
      if plan_data.get("amount"):
        amount = plan_data["amount"]
        if not isinstance(amount, int) or amount <= 0:
          errors.append("Amount must be a positive integer in smallest currency unit (cents)")

      if errors:
        return LibResponse.error_response([
          {"code": "validation_error", "message": "Plan data validation failed", "details": {"validation_errors": errors}}
        ])

      return LibResponse.success_response(True)

    except Exception as e:
      self.logger.error(f"Error validating plan data: {e}")
      return LibResponse.error_response([{"code": "validation_error", "message": str(e)}])

  def validate_subscription_data(self, subscription_data: Dict[str, Any]) -> LibResponse[bool]:
    """Validate subscription data against Stripe constraints."""
    try:
      errors = []

      # Required fields
      required_fields = ["plan_id", "customer_id"]
      for field in required_fields:
        if field not in subscription_data:
          errors.append(f"Missing required field: {field}")

      if errors:
        return LibResponse.error_response([
          {"code": "validation_error", "message": "Subscription data validation failed", "details": {"validation_errors": errors}}
        ])

      return LibResponse.success_response(True)

    except Exception as e:
      self.logger.error(f"Error validating subscription data: {e}")
      return LibResponse.error_response([{"code": "validation_error", "message": str(e)}])


# Single instance of the engine
engine = StripeEngine()
logger.info("Stripe engine initialized")
