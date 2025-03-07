from uuid import UUID

from fastapi import Depends

from src.dependencies.security import JWTBearer


async def get_current_user_id(user: dict = Depends(JWTBearer())) -> UUID:
  """Get current user ID from JWT token."""
  return UUID(user["id"])
