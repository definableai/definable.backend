import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from config.settings import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
  """Verify password."""
  return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def get_password_hash(password: str) -> str:
  """Get password hash."""
  return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
  """Create access token."""
  to_encode = data.copy()
  if expires_delta:
    expire = datetime.now(timezone.utc) + expires_delta
  else:
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
  to_encode.update({"exp": expire})
  encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
  return encoded_jwt


def verify_jwt_token(token: str) -> Optional[dict]:
  """Verify and decode JWT token."""
  try:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    return payload
  except jwt.ExpiredSignatureError:
    raise ValueError("Token has expired")
  except jwt.InvalidTokenError:
    raise ValueError("Invalid token")
