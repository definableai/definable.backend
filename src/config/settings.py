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
  environment: str = "development"
  database_url: str
  resend_api_key: str
  frontend_url: str
  stripe_secret_key: str
  stripe_publishable_key: str
  stripe_webhook_secret: str
  jwt_expire_minutes: int

  class Config:
    """Config class."""

    env_file = ".env"
    env_file_encoding = "utf-8"


# Singleton pattern to ensure only one instance of Settings is used
settings = Settings()  # type: ignore
