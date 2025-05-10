from loguru import logger

from config.settings import settings

env = settings.environment

if env == "prod":
  logger.add("logs/app.log", rotation="10 MB", level="INFO")
else:
  logger.add("logs/app.log", rotation="10 MB", level="DEBUG")

log = logger
