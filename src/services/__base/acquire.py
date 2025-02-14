"""Module for acquiring services."""

from typing import List

import utils
from common.cache import Cache, deps_cache
from common.logger import log
from common.websocket import WebSocketManager
from config.settings import Settings, settings
from database import Base, async_session


class Acquire:
  """Acquire class."""

  def __init__(self):
    self.db_session = async_session
    self.models: List[Base] = self._register_models()
    self.schemas = self._register_schemas()
    self.settings: Settings = settings
    self.utils = utils
    self.logger = log
    self.cache = Cache()
    self.deps_cache = deps_cache
    self.ws_manager = WebSocketManager()

  def _register_models(self):
    """Register models."""
    return utils.ModuleLoader.load_modules_from_services(self, file_name="model.py", class_suffix="Model")

  def _register_schemas(self):
    """Register schemas."""
    return utils.ModuleLoader.load_modules_from_services(self, file_name="schema.py", class_suffix="Schema")
