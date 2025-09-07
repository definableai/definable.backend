"""Module for settings."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
  """Settings class."""

  agent_base_url: str
  app_name: str
  app_base_url: str
  job_update_url: str
  internal_token: str
  jwt_secret: str
  master_api_key: str
  openai_api_key: str
  environment: str = "dev"  # can be dev | beta | prod
  database_url: str
  resend_api_key: str
  frontend_url: str
  composio_callback_url: str
  firebase_creds: str
  firebase_rtdb: str
  stripe_secret_key: str
  stripe_publishable_key: str
  stripe_webhook_secret: str
  stripe_success_url: str
  stripe_cancel_url: str
  jwt_expire_minutes: int
  kb_settings_version: int
  s3_bucket: str
  s3_access_key: str
  s3_secret_key: str
  s3_endpoint: str
  firecrawl_api_key: str
  celery_broker_url: str
  celery_result_backend: str
  anthropic_api_key: str
  public_s3_bucket: str
  python_sandbox_testing_url: str
  deepseek_api_key: str
  prompt_buffer_size: int
  stytch_project_id: str
  stytch_secret: str
  stytch_environment: str
  stytch_webhook_secret: str
  razorpay_key_id: str
  razorpay_webhook_secret: str
  razorpay_key_secret: str
  composio_api_key: str
  composio_base_url: str

  # API Key Configuration
  api_key_length: int = 32
  api_key_default_expiry_days: int = 365
  auth_token_expiry_hours: int = 24
  enable_api_key_rate_limiting: bool = True
  api_key_rate_limit_per_minute: int = 100

  class Config:
    """Config class."""

    env_file = ".env"
    env_file_encoding = "utf-8"


# Singleton pattern to ensure only one instance of Settings is used
settings = Settings()  # type: ignore
