import contextlib
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.error import ChargeNotFoundError, InsufficientCreditsError, InvalidTransactionError
from common.logger import logger
from models import ChargeModel, TransactionModel, WalletModel


# Import enums directly if they're simple enums
class TransactionType:
  CREDIT = "CREDIT"
  DEBIT = "DEBIT"
  HOLD = "HOLD"
  RELEASE = "RELEASE"


class TransactionStatus:
  PENDING = "PENDING"
  COMPLETED = "COMPLETED"
  FAILED = "FAILED"
  CANCELLED = "CANCELLED"


class ChargeError(Exception):
  """Base exception for charge-related errors."""

  pass


class Charge:
  """
  Utility class for managing billing charges throughout the application.
  """

  def __init__(self, name: str, user_id: UUID, org_id: UUID, session: AsyncSession):
    """
    Initialize a charge object.

    Args:
        name: Name of the charge from charges table
        user_id: User ID to charge
        org_id: Organization ID
        session: Database session (required)

    Raises:
        ValueError: If session is None
    """
    if session is None:
      raise ValueError("Database session is required")

    # Store the parameters for later validation during create()
    self.id = uuid4()
    self.name = name
    self.user_id = user_id
    self.org_id = org_id
    self.session = session
    self.transaction_id: Optional[UUID] = None
    self.metadata: Dict[str, Any] = {}
    self.balance_checked = False  # Flag to track if balance has been verified

  @classmethod
  async def create_with_balance_check(cls, name: str, user_id: UUID, org_id: UUID, session: AsyncSession, qty: int = 1):
    """Factory method to create Charge after validating balance."""
    logger.info(f"Creating charge with balance check: {name}, {user_id}, {org_id}, {qty}")

    # Get charge from database directly
    query = select(ChargeModel).where(ChargeModel.name == name)
    result = await session.execute(query)
    charge_details = result.scalar_one_or_none()

    if not charge_details:
      raise ChargeNotFoundError(f"Charge not found: {name}")

    # Calculate total amount
    amount = charge_details.amount * qty

    # Check wallet balance
    wallet_query = select(WalletModel).where(WalletModel.user_id == user_id)
    wallet_result = await session.execute(wallet_query)
    wallet = wallet_result.scalar_one_or_none()

    if not wallet:
      # Create wallet if not exists
      wallet = WalletModel(user_id=user_id, balance=0, hold=0, credits_spent=0)
      session.add(wallet)
      await session.flush()
      await session.refresh(wallet)

    if wallet.balance < amount:
      raise InsufficientCreditsError("Insufficient credits")

    # Create charge instance if balance is sufficient
    charge = cls(name, user_id, org_id, session)
    charge.balance_checked = True
    return charge

  def _convert_uuids_to_str(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert UUID objects to strings in a dict."""
    if not data:
      return {}

    result = {}
    for key, value in data.items():
      if isinstance(value, UUID):
        result[key] = str(value)
      else:
        result[key] = value
    return result

  @contextlib.asynccontextmanager
  async def _get_session(self, session: Optional[AsyncSession] = None):
    """Context manager for database session handling."""
    use_session = session or self.session
    try:
      yield use_session
    except Exception as e:
      await use_session.rollback()
      raise ChargeError(f"Database operation failed: {str(e)}") from e

  async def create(self, qty: int = 1, metadata: Optional[Dict[str, Any]] = None, session: Optional[AsyncSession] = None) -> "Charge":
    """
    Create a new charge by placing a hold on the user's credits.

    Args:
        qty: Quantity of units to charge
        metadata: Additional context information
        session: Database session (optional)

    Returns:
        self for method chaining

    Raises:
        ChargeNotFoundError: If the charge name doesn't exist
        InsufficientCreditsError: If user has insufficient credits
        ChargeError: For other charge-related errors
    """
    async with self._get_session(session) as use_session:
      try:
        # Get charge details
        charge = await use_session.get(ChargeModel, self.name)
        if not charge:
          raise ChargeNotFoundError(f"Charge not found: {self.name}")

        # Calculate total amount
        amount = charge.amount * qty

        # Check wallet balance (only if not already checked)
        if not self.balance_checked:
          wallet = await use_session.get(WalletModel, self.user_id)
          if wallet.balance < amount:
            raise InsufficientCreditsError(f"Required: {amount}, Available: {wallet.balance}")

        # Create charge metadata
        charge_metadata = {
          "charge_id": str(charge.id),
          "service": charge.service,
          "action": charge.action,
          "qty": qty,
          "unit_amount": charge.amount,
          "total_amount": amount,
          "charge_name": self.name,
        }

        # Merge with additional metadata
        metadata_str = self._convert_uuids_to_str(metadata)
        combined_metadata = {**metadata_str, **charge_metadata}

        # Create transaction
        transaction = await self.create_transaction(
          self.user_id, self.org_id, TransactionType.HOLD, TransactionStatus.PENDING, amount, f"Hold for {self.name}", combined_metadata, session
        )

        # Update wallet
        wallet.hold += amount
        await use_session.flush()

        # Update instance
        self.transaction_id = UUID(str(transaction.id))
        self.metadata = combined_metadata

        logger.info(f"Created charge: {self.name}, transaction_id: {self.transaction_id}")
        return self

      except (ChargeNotFoundError, InsufficientCreditsError):
        raise
      except Exception as e:
        logger.error(f"Failed to create charge: {str(e)}")
        raise ChargeError(f"Failed to create charge: {str(e)}") from e

  async def update(self, additional_metadata: Optional[Dict[str, Any]] = None, session: Optional[AsyncSession] = None) -> "Charge":
    """
    Complete the charge by converting the hold to a debit.

    Args:
        additional_metadata: Additional metadata to add
        session: Database session (optional)

    Returns:
        self for method chaining

    Raises:
        ValueError: If charge hasn't been created
        InvalidTransactionError: If hold transaction is invalid
        ChargeError: For other charge-related errors
    """
    if not self.transaction_id:
      raise ValueError("Cannot update a charge that hasn't been created")

    async with self._get_session(session) as use_session:
      try:
        # Get the hold transaction
        transaction = await use_session.get(TransactionModel, self.transaction_id)
        if not transaction or transaction.type != TransactionType.HOLD:
          raise InvalidTransactionError("Invalid hold transaction")

        # Update the existing transaction metadata
        if additional_metadata:
          if transaction.transaction_metadata is None:
            transaction.transaction_metadata = {}
          transaction.transaction_metadata.update(additional_metadata)

        # Update the transaction type and status
        transaction.type = TransactionType.DEBIT
        transaction.status = TransactionStatus.COMPLETED

        # Update the description
        transaction.description = f"Charge for {self.name}"

        # Save changes
        await use_session.commit()

        # Update wallet
        wallet = await use_session.get(WalletModel, self.user_id)
        wallet.hold -= transaction.credits
        await use_session.flush()

        logger.info(f"Updated charge: {self.name}, transaction_id: {self.transaction_id}")
        return self

      except InvalidTransactionError:
        raise
      except Exception as e:
        logger.error(f"Failed to update charge: {str(e)}")
        raise ChargeError(f"Failed to update charge: {str(e)}") from e

  async def delete(self, reason: Optional[str] = None, session: Optional[AsyncSession] = None) -> "Charge":
    """
    Delete the charge by releasing the hold.

    Args:
        reason: Optional reason for deletion
        session: Database session (optional)

    Returns:
        self for method chaining

    Raises:
        ValueError: If charge hasn't been created
        InvalidTransactionError: If hold transaction is invalid
        ChargeError: For other charge-related errors
    """
    if not self.transaction_id:
      raise ValueError("Cannot delete a charge that hasn't been created")

    async with self._get_session(session) as use_session:
      try:
        # Get the hold transaction
        transaction = await use_session.get(TransactionModel, self.transaction_id)
        if not transaction or transaction.type != TransactionType.HOLD:
          raise InvalidTransactionError("Invalid hold transaction")

        # Create metadata with reason
        metadata = {**(transaction.transaction_metadata or {})}
        if reason:
          metadata["deletion_reason"] = reason

        # Create RELEASE transaction
        release_transaction = await self.create_transaction(
          self.user_id,
          self.org_id,
          TransactionType.RELEASE,
          TransactionStatus.COMPLETED,
          transaction.credits,
          f"Release hold for {self.name}",
          metadata,
          session,
        )
        use_session.add(release_transaction)
        await use_session.flush()
        await use_session.refresh(release_transaction)

        # Update hold transaction status
        transaction.status = TransactionStatus.FAILED
        if reason and transaction.transaction_metadata:
          transaction.transaction_metadata["deletion_reason"] = reason
        await use_session.commit()

        # Update wallet
        wallet = await use_session.get(WalletModel, self.user_id)
        wallet.hold -= transaction.credits
        await use_session.flush()

        logger.info(f"Deleted charge: {self.name}, transaction_id: {self.transaction_id}")
        return self

      except InvalidTransactionError:
        raise
      except Exception as e:
        logger.error(f"Failed to delete charge: {str(e)}")
        raise ChargeError(f"Failed to delete charge: {str(e)}") from e

  async def create_transaction(
    self,
    user_id: UUID,
    org_id: UUID,
    transaction_type: str,
    status: str,
    credit_amount: int,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    session: Optional[AsyncSession] = None,
  ) -> TransactionModel:
    """Create a transaction record."""
    use_session = session or self.session

    transaction = TransactionModel(
      user_id=user_id,
      org_id=org_id,
      type=transaction_type,
      status=status,
      credits=credit_amount,
      description=description,
      transaction_metadata=metadata,
    )

    use_session.add(transaction)
    await use_session.flush()
    await use_session.refresh(transaction)
    return transaction

  async def update_wallet_for_hold(self, user_id: UUID, amount: int, session: AsyncSession) -> WalletModel:
    """Update wallet for hold operation."""
    wallet_query = select(WalletModel).where(WalletModel.user_id == user_id)
    wallet_result = await session.execute(wallet_query)
    wallet = wallet_result.scalar_one_or_none()

    if not wallet:
      wallet = WalletModel(user_id=user_id, balance=0, hold=0, credits_spent=0)
      session.add(wallet)

    wallet.hold += amount
    await session.flush()
    return wallet
