"""Module for settings."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
  """Settings class."""

  app_name: str
  jwt_secret: str
  master_api_key: str
  openai_api_key: str
  environment: str = "dev"  # can be dev | beta | prod
  database_url: str
  resend_api_key: str
  frontend_url: str
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

  class Config:
    """Config class."""

    env_file = ".env"
    env_file_encoding = "utf-8"


# Singleton pattern to ensure only one instance of Settings is used
settings = Settings()  # type: ignore
