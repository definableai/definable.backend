from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from utils.charge import Charge

from .security import RBAC


class Usage:
  """
  Dependency for credit usage tracking.

  When background=True: Creates a charge (HOLD) only, letting the caller handle processing
  When background=False: Creates a charge (HOLD) and automatically finalizes it after function success
  """

  def __init__(
    self,
    charge_name: str,
    qty: int = 1,
    background: bool = False,  # Just one parameter!
    metadata: Optional[Dict[str, Any]] = None,
  ):
    """
    Initialize the usage dependency.

    Args:
        charge_name: Name of the charge to apply (must exist in ChargeModel)
        qty: Quantity to charge
        background: If True, only create HOLD (caller handles processing)
                   If False, automatically finalize charge after function completes
        metadata: Additional metadata to include with the transaction
    """
    self.charge_name = charge_name
    self.qty = qty
    self.background = background
    self.metadata = metadata or {}
    self.metadata["api_endpoint"] = charge_name

  async def __call__(
    self,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("billing", "read")),
  ) -> Dict[str, Any]:
    """
    Process usage tracking for an API endpoint.

    Creates a charge and handles finalization based on background setting.
    """
    user_id = UUID(user["id"])
    org_id = UUID(user["org_id"])

    # Create metadata with request information
    endpoint_metadata = {
      **self.metadata,
      "endpoint": request.url.path,
      "method": request.method,
      "user_agent": request.headers.get("user-agent", "unknown"),
    }

    try:
      # Create the charge (places a HOLD)
      charge = Charge(name=self.charge_name, user_id=user_id, org_id=org_id, session=session)

      try:
        await charge.create(qty=self.qty, metadata=endpoint_metadata)
      except HTTPException as e:
        if e.status_code == 402:
          raise HTTPException(status_code=402, detail=f"Insufficient credits for {self.charge_name}. Please add more credits to your account.")
        raise

      # Return value includes the charge and transaction info
      usage_info = {
        "charge": charge,
        "transaction_id": charge.transaction_id,
        "quantity": self.qty,
        # Include the user info from RBAC dependency
        **user,
      }

      # Only handle auto-finalization if background=False
      # If background=True, the caller will handle the charge
      if not self.background:
        # Add background task that runs immediately after the response
        # This ensures it runs after the endpoint handler completes successfully
        background_tasks.add_task(
          self._finalize_charge_after_response,
          charge=charge,
        )

      return usage_info

    except Exception:
      # Pass through any exceptions
      raise

  async def _finalize_charge_after_response(self, charge: Charge) -> None:
    """
    Finalize a charge after the response has been sent.

    This runs in a background task right after the endpoint handler completes
    successfully, ensuring we only convert HOLD to DEBIT if the function succeeds.
    """
    try:
      await charge.update(additional_metadata={"completed_by": "auto_finalize_after_response"})
      charge.logger.info(f"Auto-finalized charge {charge.transaction_id} after response - HOLD → DEBIT")
    except Exception as e:
      charge.logger.error(f"Error finalizing charge {charge.transaction_id} after response: {str(e)}")
      try:
        # If updating fails, try to release the hold
        await charge.delete(reason=f"Error finalizing: {str(e)}")
        charge.logger.info(f"Released charge {charge.transaction_id} after error")
      except Exception as release_error:
        charge.logger.error(f"Failed to release charge {charge.transaction_id}: {str(release_error)}")
