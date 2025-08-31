from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from common.error import ChargeNotFoundError, InvalidTransactionError
from common.logger import log
from models import ChargeModel, TransactionModel, TransactionStatus, TransactionType, WalletModel


class Charge:
  """Simplified charge utility for credit management."""

  def __init__(self, name: str, user_id: UUID, org_id: UUID, session: AsyncSession, service: str):
    self.id = uuid4()
    self.charge_id = str(self.id)  # String version for logging
    self.name = name
    self.user_id = user_id
    self.org_id = org_id
    self.session = session
    self.service = service
    self.transaction_id = None
    self.logger = log

    self.logger.debug(f"Charge object initialized [charge_id={self.charge_id}, name={name}, user_id={user_id}, service={service}]")

  @classmethod
  async def verify_balance(cls, name: str, org_id: UUID, qty: int = 1, session: Optional[AsyncSession] = None):
    """Check if organization has sufficient balance for charge."""
    # Get charge amount
    if session is None:
      raise ValueError("Database session is required")

    charge = await cls._get_charge_details(name, session)
    amount = charge.amount * qty

    # Get and lock wallet
    wallet = await cls._get_wallet(org_id, session, for_update=True)

    # Check balance
    if wallet.balance < amount:
      return False, amount, wallet.balance
    return True, amount, wallet.balance

  async def create(self, qty: int = 1, metadata: Optional[Dict[str, Any]] = None, description: Optional[str] = None):
    """Create charge hold."""
    charge = await self._get_charge_details(self.name, self.session)
    amount = charge.amount * qty

    self.logger.info(f"Creating charge {self.name} for {amount} credits")

    # Verify wallet has enough funds
    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)

    if wallet.balance < amount:
      self.logger.warning(f"Insufficient credits for charge {self.name}. Required: {amount}, Available: {wallet.balance}")
      raise HTTPException(status_code=402, detail="Insufficient credits")  # Payment Required

    # Use custom description if provided, otherwise use default
    hold_description = description or f"Hold for {self.name}"

    # Create transaction
    transaction = await self._create_transaction(TransactionType.HOLD, TransactionStatus.PENDING, amount, hold_description, metadata, qty)

    self.logger.info(f"Created transaction {transaction.id} for charge {self.name}")

    # Update wallet hold
    wallet.hold = (wallet.hold or 0) + amount

    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Updated wallet: balance={wallet.balance}, hold={wallet.hold}")
    self.transaction_id = transaction.id
    return self

  async def update(self, additional_metadata: Optional[Dict[str, Any]] = None, qty_increment: int = 0):
    """Complete charge by converting hold to debit or increment the quantity."""
    transaction = await self._get_transaction(for_update=True)

    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction update attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)
    self.logger.info(f"Current wallet state: balance={wallet.balance}, hold={wallet.hold or 0}")

    if qty_increment > 0:
      self.logger.info(f"Incrementing quantity for transaction {transaction.id} by {qty_increment}")

      charge = await self._get_charge_details(self.name, self.session)
      additional_amount = charge.amount * qty_increment

      wallet_hold = wallet.hold or 0
      available_balance = wallet.balance - wallet_hold

      if available_balance < additional_amount:
        self.logger.warning(
          f"Insufficient available balance for qty increment. Required: {additional_amount}, "
          f"Available: {available_balance} (Balance: {wallet.balance}, Hold: {wallet_hold})"
        )
        raise HTTPException(
          status_code=402, detail=f"Insufficient available credits for additional quantity. Need {additional_amount}, have {available_balance}"
        )

      current_metadata = transaction.transaction_metadata or {}
      current_qty = int(current_metadata.get("qty", 1))
      new_qty = current_qty + qty_increment
      current_metadata["qty"] = new_qty

      original_amount = transaction.credits
      transaction.credits = original_amount + additional_amount

      wallet.hold = (wallet.hold or 0) + additional_amount

      self.logger.info(f"Updated transaction {transaction.id} from qty={current_qty} to qty={new_qty}")
      self.logger.info(
        f"Credits increased from {original_amount} to {transaction.credits}. Updated wallet: balance={wallet.balance}, hold={wallet.hold}"
      )

      if additional_metadata:
        transaction.transaction_metadata = {**current_metadata, **additional_metadata}
      else:
        transaction.transaction_metadata = current_metadata

      await self.session.flush()
      await self.session.commit()
      return self

    self.logger.info(f"Converting HOLD to DEBIT for transaction {transaction.id}")
    transaction.type = TransactionType.DEBIT
    transaction.status = TransactionStatus.COMPLETED

    # Get existing metadata which already contains service information from create()
    current_metadata = transaction.transaction_metadata or {}
    service_name = self.service  # Use the service from initialization

    # Create more descriptive message based on service type
    if service_name == "chat":
      # For chat service, include model name without chat_id
      model = current_metadata.get("model", self.name)
      transaction.description = f"Credits used for {model} chat"
    else:
      # Generic description for other services
      transaction.description = f"Credits used for {self.name}" + (f" ({service_name})" if service_name else "")

    if additional_metadata:
      transaction.transaction_metadata = {**(transaction.transaction_metadata or {}), **additional_metadata}

    wallet.hold = (wallet.hold or 0) - transaction.credits
    if wallet.hold < 0:
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    wallet.balance -= transaction.credits
    wallet.credits_spent = (wallet.credits_spent or 0) + transaction.credits

    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Transaction completed - Updated wallet: balance={wallet.balance}, hold={wallet.hold}, credits_spent={wallet.credits_spent}")
    return self

  async def delete(self, reason: Optional[str] = None):
    """Cancel charge by releasing hold."""
    transaction = await self._get_transaction(for_update=True)

    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction deletion attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    self.logger.info(f"Converting HOLD to RELEASE for transaction {transaction.id}")
    transaction.type = TransactionType.RELEASE
    transaction.status = TransactionStatus.COMPLETED

    # Generic release description that works for all services
    transaction.description = f"Released hold for {self.name}"
    if reason:
      transaction.description += f": {reason}"

    if transaction.transaction_metadata:
      if reason:
        transaction.transaction_metadata["release_reason"] = reason
      transaction.transaction_metadata["original_status"] = "CANCELLED"

    # Update wallet hold
    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)
    self.logger.info(f"Current wallet state: balance={wallet.balance}, hold={wallet.hold or 0}")
    self.logger.info(f"Transaction credits: {transaction.credits}")
    wallet.hold = (wallet.hold or 0) - transaction.credits
    self.logger.info(f"Updated wallet hold: {wallet.hold}")
    if wallet.hold < 0:
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    # Commit changes to the database
    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Hold released - Updated wallet: balance={wallet.balance}, hold={wallet.hold}")
    return self

  async def calculate_and_update(self, content=None, metadata=None, status="completed", additional_metadata=None):
    """
    Calculate appropriate charge based on content type and measure,
    then update the transaction with complete billing information.
    """
    self.logger.info(f"Starting calculate_and_update for charge: {self.name}")

    try:
      # Get charge details to determine the appropriate measure
      charge_details = await self._get_charge_details(self.name, self.session)
      self.logger.info(f"Charge details obtained for charge: {self.name}")

      # Initialize metadata dictionary
      meta = additional_metadata or {}
      meta["processing_status"] = status

      # Calculate quantity based on charge's measure type
      measure_qty = 1  # Default
      if charge_details.measure == "page" and metadata and "pages" in metadata:
        measure_qty = metadata.get("pages", 1)
        meta["page_count"] = measure_qty
        self.logger.info(f"Page count calculated: {measure_qty}")
      elif charge_details.measure == "token":
        if metadata and "total_tokens" in metadata:
          measure_qty = metadata.get("total_tokens", 0)
          meta["token_count"] = measure_qty
          self.logger.info(f"Token count calculated: {measure_qty}")
          if "input_tokens" in metadata:
            meta["input_tokens"] = metadata.get("input_tokens", 0)
          if "output_tokens" in metadata:
            meta["output_tokens"] = metadata.get("output_tokens", 0)
        elif content:
          measure_qty = len(content.split()) if isinstance(content, str) else 0
          meta["token_count"] = measure_qty
          self.logger.info(f"Token count calculated from content: {measure_qty}")
      elif charge_details.measure == "sheet" and metadata and "sheet_count" in metadata:
        measure_qty = metadata.get("sheet_count", 1)
        meta["sheet_count"] = measure_qty
        self.logger.info(f"Sheet count calculated: {measure_qty}")

      # Include content length if provided
      if content:
        meta["content_length"] = len(content) if isinstance(content, str) else 0

      # Add billing metric details for transparency
      meta["measure_type"] = charge_details.measure
      meta["unit_type"] = charge_details.unit

      # Merge any additional metadata
      if metadata:
        meta.update({k: v for k, v in metadata.items() if k not in meta})

      # 1. First update with qty_increment if needed
      current_meta = (await self._get_transaction(for_update=True)).transaction_metadata or {}
      current_qty = int(current_meta.get("qty", 1))
      qty_increment = max(0, measure_qty - current_qty)

      # Handle the case where quantity increment would exceed available credits
      if qty_increment > 0:
        self.logger.info(f"Updating transaction with qty_increment: {qty_increment}")
        try:
          await self.update(additional_metadata=meta, qty_increment=qty_increment)
        except HTTPException as e:
          if "Insufficient available credits" in str(e):
            # Get wallet and just use all remaining balance
            wallet = await self._get_wallet(self.org_id, self.session, for_update=True)
            available_balance = wallet.balance - (wallet.hold or 0)

            if available_balance > 0:
              # Use all available balance regardless of quantity
              self.logger.warning(f"Insufficient credits for full quantity. Using all available balance of {available_balance} credits")

              # Calculate a custom transaction that uses exactly the available balance
              transaction = await self._get_transaction(for_update=True)
              original_amount = transaction.credits
              additional_amount = available_balance  # Use all available balance

              # Update metadata to indicate limited credits
              meta["credits_limited"] = True
              meta["requested_amount"] = charge_details.amount * qty_increment
              meta["applied_amount"] = additional_amount
              current_metadata = transaction.transaction_metadata or {}

              # Update transaction
              transaction.credits = original_amount + additional_amount
              transaction.transaction_metadata = {**current_metadata, **meta}

              # Update wallet
              wallet.hold = (wallet.hold or 0) + additional_amount

              await self.session.flush()
              await self.session.commit()

              self.logger.info(f"Credits increased from {original_amount} to {transaction.credits}. Used all available balance.")
            else:
              # If we can't increment at all, just add the metadata about the limitation
              self.logger.warning("No credits available for quantity increment")
              meta["credits_limited"] = True
              meta["requested_amount"] = charge_details.amount * qty_increment
              meta["applied_amount"] = 0
          else:
            # Re-raise if it's not a credits issue
            raise

      # 2. Then convert to DEBIT if status is completed (with empty qty_increment)
      if status == "completed":
        self.logger.info(f"Converting HOLD to DEBIT for charge: {self.name}")
        await self.update(additional_metadata=meta)

      self.logger.info(f"Completed calculate_and_update for charge: {self.name}")
      return self
    except Exception as e:
      self.logger.error(f"Failed to calculate and update charge: {str(e)}")

      # For credit-related errors, still try to complete the transaction
      if isinstance(e, HTTPException) and "Insufficient" in str(e):
        try:
          meta = additional_metadata or {}
          meta["processing_status"] = status
          meta["credits_limited"] = True
          meta["error"] = str(e)

          # Convert the hold to a debit with whatever amount we originally held
          self.logger.info(f"Completing transaction with limited credits due to: {str(e)}")
          await self.update(additional_metadata=meta)
          return self
        except Exception as complete_error:
          self.logger.error(f"Failed to complete charge with limited credits: {str(complete_error)}")
          raise complete_error
      raise

  # Helper methods
  @staticmethod
  async def _get_charge_details(name: str, session: AsyncSession):
    query = select(ChargeModel).where(ChargeModel.name == name, ChargeModel.is_active)
    result = await session.execute(query)
    charge = result.scalar_one_or_none()
    if not charge:
      raise ChargeNotFoundError(f"Charge not found: {name}")
    return charge

  @staticmethod
  async def _get_wallet(org_id: UUID, session: AsyncSession, for_update: bool = False):
    if for_update:
      lock_query = text("SELECT * FROM wallets WHERE organization_id = :organization_id FOR UPDATE")
      await session.execute(lock_query, {"organization_id": str(org_id)})

    query = select(WalletModel).where(WalletModel.organization_id == org_id)
    result = await session.execute(query)
    wallet = result.scalar_one_or_none()

    if not wallet:
      wallet = WalletModel(id=uuid4(), organization_id=org_id, balance=0, hold=0, credits_spent=0)
      session.add(wallet)
      await session.commit()
      await session.refresh(wallet)

    return wallet

  async def _get_transaction(self, for_update: bool = False):
    if for_update:
      lock_query = text("SELECT * FROM transactions WHERE id = :tx_id FOR UPDATE")
      await self.session.execute(lock_query, {"tx_id": str(self.transaction_id)})

    query = select(TransactionModel).where(TransactionModel.id == self.transaction_id)
    result = await self.session.execute(query)
    return result.scalar_one_or_none()

  @classmethod
  def from_transaction_id(cls, transaction_id: str, session):
    """Create a Charge instance from an existing transaction ID for sync sessions."""
    from sqlalchemy import select
    from models import TransactionModel

    # Get the transaction
    query = select(TransactionModel).where(TransactionModel.id == UUID(transaction_id))
    result = session.execute(query)
    transaction = result.scalar_one_or_none()

    if not transaction:
      raise ValueError(f"Transaction {transaction_id} not found")

    # Extract metadata to recreate charge
    metadata = transaction.transaction_metadata or {}
    charge_name = metadata.get("charge_name", "unknown")
    service = metadata.get("service", "default")

    # Create charge instance
    charge = cls(name=charge_name, user_id=transaction.user_id, org_id=transaction.organization_id, session=session, service=service)

    # Set the transaction_id to link to existing transaction
    charge.transaction_id = transaction.id

    return charge

  async def calculate_and_update_sync(self, content=None, metadata=None, status="completed", additional_metadata=None):
    """
    Sync version of calculate_and_update for use with synchronous sessions.
    """
    self.logger.info(f"Starting sync calculate_and_update for charge: {self.name}")

    try:
      # Get charge details to determine the appropriate measure
      charge_details = await self._get_charge_details_sync(self.name, self.session)
      self.logger.info(f"Charge details obtained for charge: {self.name}")

      # Initialize metadata dictionary
      meta = additional_metadata or {}
      meta["processing_status"] = status

      # For sync version, we'll keep it simple and just update with metadata
      # without complex quantity calculations
      if metadata:
        meta.update({k: v for k, v in metadata.items() if k not in meta})

      # Include content length if provided
      if content:
        meta["content_length"] = len(content) if isinstance(content, str) else 0

      # Add billing metric details for transparency
      meta["measure_type"] = charge_details.measure
      meta["unit_type"] = charge_details.unit

      # Update the transaction
      await self.update_sync(additional_metadata=meta)

      self.logger.info(f"Completed sync calculate_and_update for charge: {self.name}")
      return self
    except Exception as e:
      self.logger.error(f"Failed to calculate and update charge: {str(e)}")
      raise

  async def update_sync(self, additional_metadata=None):
    """Sync version of update to complete charge by converting hold to debit."""
    transaction = await self._get_transaction_sync(for_update=True)

    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction update attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    wallet = await self._get_wallet_sync(self.org_id, self.session, for_update=True)
    self.logger.info(f"Current wallet state: balance={wallet.balance}, hold={wallet.hold or 0}")

    self.logger.info(f"Converting HOLD to DEBIT for transaction {transaction.id}")
    transaction.type = TransactionType.DEBIT
    transaction.status = TransactionStatus.COMPLETED

    # Get existing metadata which already contains service information from create()
    current_metadata = transaction.transaction_metadata or {}
    service_name = self.service  # Use the service from initialization

    # Create more descriptive message based on service type
    if service_name == "chat":
      # For chat service, include model name without chat_id
      model = current_metadata.get("model", self.name)
      transaction.description = f"Credits used for {model} chat"
    else:
      # Generic description for other services
      transaction.description = f"Credits used for {self.name}" + (f" ({service_name})" if service_name else "")

    if additional_metadata:
      transaction.transaction_metadata = {**(transaction.transaction_metadata or {}), **additional_metadata}

    wallet.hold = (wallet.hold or 0) - transaction.credits
    if wallet.hold < 0:
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    wallet.balance -= transaction.credits
    wallet.credits_spent = (wallet.credits_spent or 0) + transaction.credits

    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Transaction completed - Updated wallet: balance={wallet.balance}, hold={wallet.hold}, credits_spent={wallet.credits_spent}")
    return self

  async def delete_sync(self, reason=None):
    """Sync version of delete to cancel charge by releasing hold."""
    transaction = await self._get_transaction_sync(for_update=True)

    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction deletion attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    self.logger.info(f"Converting HOLD to RELEASE for transaction {transaction.id}")
    transaction.type = TransactionType.RELEASE
    transaction.status = TransactionStatus.COMPLETED

    # Generic release description that works for all services
    transaction.description = f"Released hold for {self.name}"
    if reason:
      transaction.description += f": {reason}"

    if transaction.transaction_metadata:
      if reason:
        transaction.transaction_metadata["release_reason"] = reason
      transaction.transaction_metadata["original_status"] = "CANCELLED"

    # Update wallet hold
    wallet = await self._get_wallet_sync(self.org_id, self.session, for_update=True)
    self.logger.info(f"Current wallet state: balance={wallet.balance}, hold={wallet.hold or 0}")
    self.logger.info(f"Transaction credits: {transaction.credits}")
    wallet.hold = (wallet.hold or 0) - transaction.credits
    self.logger.info(f"Updated wallet hold: {wallet.hold}")
    if wallet.hold < 0:
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    # Commit changes to the database
    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Hold released - Updated wallet: balance={wallet.balance}, hold={wallet.hold}")
    return self

  # Sync helper methods
  async def _get_charge_details_sync(self, name: str, session):
    query = select(ChargeModel).where(ChargeModel.name == name, ChargeModel.is_active)
    result = await session.execute(query)
    charge = result.scalar_one_or_none()
    if not charge:
      raise ChargeNotFoundError(f"Charge not found: {name}")
    return charge

  async def _get_wallet_sync(self, org_id: UUID, session, for_update: bool = False):
    from sqlalchemy.sql import text

    if for_update:
      lock_query = text("SELECT * FROM wallets WHERE organization_id = :organization_id FOR UPDATE")
      await session.execute(lock_query, {"organization_id": str(org_id)})

    query = select(WalletModel).where(WalletModel.organization_id == org_id)
    result = await session.execute(query)
    wallet = result.scalar_one_or_none()

    if not wallet:
      wallet = WalletModel(id=uuid4(), organization_id=org_id, balance=0, hold=0, credits_spent=0)
      session.add(wallet)
      await session.commit()
      await session.refresh(wallet)

    return wallet

  async def _get_transaction_sync(self, for_update: bool = False):
    from sqlalchemy.sql import text

    if for_update:
      lock_query = text("SELECT * FROM transactions WHERE id = :tx_id FOR UPDATE")
      await self.session.execute(lock_query, {"tx_id": str(self.transaction_id)})

    query = select(TransactionModel).where(TransactionModel.id == self.transaction_id)
    result = await self.session.execute(query)
    return result.scalar_one_or_none()

  async def _create_transaction(self, tx_type, status, amount, description=None, metadata=None, qty=1):
    # First, get the charge details to include in metadata
    charge_details = await self._get_charge_details(self.name, self.session)

    # Prepare base metadata with charge details
    charge_metadata = {
      "charge_name": charge_details.name,
      "charge_amount": charge_details.amount,
      "charge_unit": charge_details.unit,
      "charge_measure": charge_details.measure,
      "service": self.service,
      "action": charge_details.action,
      "charge_description": charge_details.description,
      # Add quantity information - use qty from metadata or default to 1
      "qty": qty,
    }

    # Merge with any additional metadata
    combined_metadata = {**(metadata or {}), **charge_metadata}

    # Create the transaction
    transaction = TransactionModel(
      id=uuid4(),
      user_id=self.user_id,
      organization_id=self.org_id,
      type=tx_type,
      status=status,
      credits=amount,
      description=description or f"{tx_type} for {self.name}",
      transaction_metadata=combined_metadata,
    )

    self.session.add(transaction)
    await self.session.flush()
    await self.session.refresh(transaction)
    return transaction
