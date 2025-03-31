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

  def __init__(self, name: str, user_id: UUID, org_id: UUID, session: AsyncSession):
    self.id = uuid4()
    self.charge_id = str(self.id)  # String version for logging
    self.name = name
    self.user_id = user_id
    self.org_id = org_id
    self.session = session
    self.transaction_id = None
    self.logger = log

    self.logger.debug(f"Charge object initialized [charge_id={self.charge_id}, name={name}, user_id={user_id}]")

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

  async def create(self, qty: int = 1, metadata: Optional[Dict[str, Any]] = None):
    """Create charge hold."""
    charge = await self._get_charge_details(self.name, self.session)
    amount = charge.amount * qty

    self.logger.info(f"Creating charge {self.name} for {amount} credits")

    # Verify wallet has enough funds
    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)

    # Only log balance details at debug level for development
    self.logger.debug(f"Wallet balance: {wallet.balance}, required: {amount}")

    if wallet.balance < amount:
      self.logger.warning(f"Insufficient credits for charge {self.name}. Required: {amount}, Available: {wallet.balance}")
      raise HTTPException(status_code=402, detail="Insufficient credits")  # Payment Required

    # Create transaction
    self.logger.debug("Creating transaction")
    transaction = await self._create_transaction(TransactionType.HOLD, TransactionStatus.PENDING, amount, f"Hold for {self.name}", metadata, qty)

    # Log transaction ID at info level for production tracking
    self.logger.info(f"Created transaction {transaction.id} for charge {self.name}")

    # Update wallet
    wallet.hold = (wallet.hold or 0) + amount  # Ensure hold is not None

    # Explicitly flush and commit changes to ensure wallet update is persisted
    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Updated wallet: balance={wallet.balance}, hold={wallet.hold}")
    self.transaction_id = transaction.id
    return self

  async def update(self, additional_metadata: Optional[Dict[str, Any]] = None, qty_increment: int = 0):
    """
    Complete charge by converting hold to debit or increment the quantity.

    If qty_increment is provided, the transaction will be updated with the new quantity
    and additional credits will be reserved, provided the wallet has sufficient balance.
    """
    # Directly attempt to get the transaction - will raise appropriate errors
    transaction = await self._get_transaction(for_update=True)

    # Continue with validation and updates
    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction update attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    # Get the wallet for updating
    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)
    self.logger.info(f"Current wallet state: balance={wallet.balance}, hold={wallet.hold or 0}")

    # If qty_increment is provided, update the transaction quantity and credits
    if qty_increment > 0:
      self.logger.info(f"Incrementing quantity for transaction {transaction.id} by {qty_increment}")

      # Get charge details to calculate additional amount
      charge = await self._get_charge_details(self.name, self.session)
      additional_amount = charge.amount * qty_increment

      # Calculate available balance (total balance minus current hold)
      wallet_hold = wallet.hold or 0  # Handle None explicitly
      available_balance = wallet.balance - wallet_hold

      if available_balance < additional_amount:
        self.logger.warning(
          f"Insufficient available balance for qty increment. Required: {additional_amount}, "
          f"Available: {available_balance} (Balance: {wallet.balance}, Hold: {wallet_hold})"
        )
        raise HTTPException(
          status_code=402, detail=f"Insufficient available credits for additional quantity. Need {additional_amount}, have {available_balance}"
        )

      # Update transaction metadata
      current_metadata = transaction.transaction_metadata or {}
      current_qty = int(current_metadata.get("qty", 1))
      new_qty = current_qty + qty_increment
      current_metadata["qty"] = new_qty

      # Update transaction amount
      original_amount = transaction.credits
      transaction.credits = original_amount + additional_amount

      # Update wallet hold - ensure not None and force update
      wallet.hold = (wallet.hold or 0) + additional_amount

      self.logger.info(f"Updated transaction {transaction.id} from qty={current_qty} to qty={new_qty}")
      self.logger.info(
        f"Credits increased from {original_amount} to {transaction.credits}. Updated wallet: balance={wallet.balance}, hold={wallet.hold}"
      )

      # Update transaction metadata with provided additional metadata
      if additional_metadata:
        transaction.transaction_metadata = {**current_metadata, **additional_metadata}
      else:
        transaction.transaction_metadata = current_metadata

      # Ensure changes are committed to the database
      await self.session.flush()
      await self.session.commit()
      return self

    # If no qty_increment, proceed with converting HOLD to DEBIT
    self.logger.info(f"Converting HOLD to DEBIT for transaction {transaction.id}")
    transaction.type = TransactionType.DEBIT
    transaction.status = TransactionStatus.COMPLETED

    if additional_metadata:
      transaction.transaction_metadata = {**(transaction.transaction_metadata or {}), **additional_metadata}

    # Update wallet - reduce hold and balance, increment credits_spent
    # Handle potential None values in wallet properties
    wallet.hold = (wallet.hold or 0) - transaction.credits
    if wallet.hold < 0:  # Defensive check
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    wallet.balance -= transaction.credits
    wallet.credits_spent = (wallet.credits_spent or 0) + transaction.credits

    # Ensure changes are committed
    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Transaction completed - Updated wallet: balance={wallet.balance}, hold={wallet.hold}, credits_spent={wallet.credits_spent}")
    return self

  async def delete(self, reason: Optional[str] = None):
    """Cancel charge by releasing hold."""
    # Directly attempt to get the transaction - will raise appropriate errors
    transaction = await self._get_transaction(for_update=True)

    # Continue with validation and updates
    if not transaction or transaction.type != TransactionType.HOLD or transaction.status != TransactionStatus.PENDING:
      self.logger.warning(f"Invalid transaction deletion attempt for transaction_id: {self.transaction_id}")
      raise InvalidTransactionError("Invalid or already processed transaction")

    # Update transaction - change to RELEASE instead of creating a new transaction
    self.logger.info(f"Converting HOLD to RELEASE for transaction {transaction.id}")
    transaction.type = TransactionType.RELEASE
    transaction.status = TransactionStatus.COMPLETED

    if transaction.transaction_metadata:
      if reason:
        transaction.transaction_metadata["release_reason"] = reason
      transaction.transaction_metadata["original_status"] = "CANCELLED"

    # Update wallet - handle None values and ensure proper update
    wallet = await self._get_wallet(self.org_id, self.session, for_update=True)
    wallet.hold = (wallet.hold or 0) - transaction.credits
    if wallet.hold < 0:  # Defensive check
      self.logger.warning(f"Negative hold detected: {wallet.hold}, resetting to 0")
      wallet.hold = 0

    # Ensure changes are committed
    await self.session.flush()
    await self.session.commit()

    self.logger.info(f"Hold released - Updated wallet: balance={wallet.balance}, hold={wallet.hold}")
    return self

  # Helper methods
  @staticmethod
  async def _get_charge_details(name: str, session: AsyncSession):
    query = select(ChargeModel).where(ChargeModel.name == name)
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
      wallet = WalletModel(organization_id=org_id, balance=0, hold=0, credits_spent=0)
      session.add(wallet)
      await session.flush()
      await session.refresh(wallet)

    return wallet

  async def _get_transaction(self, for_update: bool = False):
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
      "service": charge_details.service,
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
