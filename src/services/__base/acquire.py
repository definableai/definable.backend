"""Module for acquiring services."""

import utils
from common.cache import Cache, deps_cache
from common.logger import log
from common.websocket import WebSocketManager
from config.settings import Settings, settings
from database import async_session


class Acquire:
  """Acquire class."""

  def __init__(self):
    self.db_session = async_session
    self.schemas = self._register_schemas()
    self.services = self._register_services()
    self.settings: Settings = settings
    self.utils = utils
    self.logger = log
    self.cache = Cache()
    self.deps_cache = deps_cache
    self.ws_manager = WebSocketManager()

  def _register_services(self):
    """Register services."""
    return utils.ModuleLoader.load_modules_from_services(self, file_name="service.py", class_suffix="Service")

  def _register_schemas(self):
    """Register schemas."""
    return utils.ModuleLoader.load_modules_from_services(self, file_name="schema.py", class_suffix="Schema")
