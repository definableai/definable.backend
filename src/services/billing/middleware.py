from fastapi import Depends
from src.dependencies.security import JWTBearer
from uuid import UUID


async def get_current_user_id(user: dict = Depends(JWTBearer())) -> UUID:
    """Get current user ID from JWT token."""
    return UUID(user["id"])
