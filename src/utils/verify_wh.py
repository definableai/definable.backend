import base64
import hashlib
import hmac

from config.settings import settings


def verify_svix_signature(svix_id, svix_timestamp, body, remote_signature):
  # Create the signed content string
  signed_content = f"{svix_id}.{svix_timestamp}.{body}"

  # Extract and decode the secret
  secret_part = settings.stytch_webhook_secret.split("_")[1]
  secret_bytes = base64.b64decode(secret_part)

  # Create HMAC signature
  signature = hmac.new(key=secret_bytes, msg=signed_content.encode("utf-8"), digestmod=hashlib.sha256)
  encoded_signature = base64.b64encode(signature.digest()).decode("utf-8")
  # Return base64 encoded signature
  return encoded_signature == remote_signature.split(",")[1]
