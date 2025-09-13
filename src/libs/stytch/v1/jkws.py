from datetime import datetime, timezone
from typing import Any, Dict, Optional

import jwt
from jwt import PyJWKClient

from config.settings import settings


class StytchLocalVerifier:
  def __init__(self, project_id: str, jwks_url: Optional[str] = None, environment: Optional[str] = "test"):
    """
    Initialize the local verifier with your Stytch project ID.

    Args:
        project_id: Your Stytch project ID
        jwks_url: Optional custom JWKS URL (defaults to Stytch's public JWKS endpoint)
    """
    self.project_id = project_id

    # Default JWKS URL for Stytch
    if jwks_url is None:
      # For live projects
      if environment == "test":
        self.jwks_url = f"https://test.stytch.com/v1/sessions/jwks/{project_id}"
        print("testing environment")
      else:
        self.jwks_url = f"https://api.stytch.com/v1/sessions/jwks/{project_id}"
    else:
      self.jwks_url = jwks_url

    # Initialize the JWKS client with caching
    self.jwks_client = PyJWKClient(
      self.jwks_url,
      cache_keys=True,  # Cache the keys locally
      max_cached_keys=100,  # Maximum number of keys to cache
      cache_jwk_set=True,  # Cache the entire JWK set
      lifespan=600,  # Cache for 10 minutes (adjust as needed)
    )

  def verify_session_token(
    self, session_token: str, max_token_age_seconds: Optional[int] = None, required_claims: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """
    Verify a Stytch session JWT locally using JWKS.

    Args:
        session_token: The JWT session token to verify
        max_token_age_seconds: Optional maximum age for the token (in seconds)
        required_claims: Optional dictionary of claims that must be present and match

    Returns:
        Dict containing the decoded token claims if valid

    Raises:
        jwt.InvalidTokenError: If the token is invalid
        jwt.ExpiredSignatureError: If the token has expired
        jwt.InvalidSignatureError: If the signature is invalid
    """
    try:
      # Get the signing key from JWKS endpoint
      signing_key = self.jwks_client.get_signing_key_from_jwt(session_token)

      # Decode and verify the token
      decoded_token = jwt.decode(
        session_token,
        signing_key.key,
        algorithms=["RS256"],  # Stytch uses RS256
        audience=self.project_id,  # Verify the audience claim
        options={
          "verify_signature": True,
          "verify_exp": True,
          "verify_iat": True,
          "verify_aud": True,
          "require": ["exp", "iat", "aud", "sub", "iss"],
        },
      )
      # Additional validation checks
      self._validate_additional_claims(decoded_token, max_token_age_seconds, required_claims)

      return decoded_token

    except jwt.ExpiredSignatureError:
      raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidSignatureError:
      raise jwt.InvalidSignatureError("Invalid token signature")
    except jwt.InvalidTokenError as e:
      raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")
    except Exception as e:
      raise jwt.InvalidTokenError(f"Token verification failed: {str(e)}")

  def _validate_additional_claims(
    self, decoded_token: Dict[str, Any], max_token_age_seconds: Optional[int], required_claims: Optional[Dict[str, Any]]
  ):
    """
    Perform additional validation on token claims.
    """
    current_time = datetime.now(timezone.utc).timestamp()

    # Check token age if specified
    if max_token_age_seconds is not None:
      token_age = current_time - decoded_token.get("iat", 0)
      if token_age > max_token_age_seconds:
        raise jwt.InvalidTokenError(f"Token age ({token_age}s) exceeds maximum ({max_token_age_seconds}s)")

    # Validate required claims if specified
    if required_claims:
      for claim_name, expected_value in required_claims.items():
        if claim_name not in decoded_token:
          raise jwt.InvalidTokenError(f"Required claim '{claim_name}' not found in token")

        if decoded_token[claim_name] != expected_value:
          raise jwt.InvalidTokenError(f"Claim '{claim_name}' value mismatch. Expected: {expected_value}, Got: {decoded_token[claim_name]}")

  def get_user_id_from_token(self, session_token: str) -> str:
    """
    Extract the user ID from a verified session token.

    Args:
        session_token: The JWT session token

    Returns:
        The user ID from the token's subject claim
    """
    decoded = self.verify_session_token(session_token)
    return decoded.get("sub", "").replace("user-", "")  # Stytch prefixes with "user-"

  def is_token_valid(self, session_token: str) -> bool:
    """
    Check if a token is valid without raising exceptions.

    Args:
        session_token: The JWT session token

    Returns:
        True if the token is valid, False otherwise
    """
    try:
      self.verify_session_token(session_token)
      return True
    except Exception:
      return False


# single instance
stytch_local_verifier = StytchLocalVerifier(project_id=settings.stytch_project_id, environment=settings.stytch_environment)


__all__ = ["stytch_local_verifier"]
