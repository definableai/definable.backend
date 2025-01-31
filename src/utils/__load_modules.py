"""Utility module for dynamically loading other modules."""

import importlib
import os
from typing import Any


class ModuleLoader:
  """Class for loading modules dynamically."""

  @staticmethod
  def load_modules_from_services(instance: Any, file_name: str, class_suffix: str) -> dict[str, Any]:
    """
    Load modules from services.
    """
    services_dir = os.path.join(os.path.dirname(__file__), "..", "services")
    if not os.path.exists(services_dir):
      raise FileNotFoundError(f"Services directory not found: {services_dir}")
    loaded_modules = {}
    for service_name in os.listdir(services_dir):
      if service_name.startswith("__"):
        continue
      service_path = os.path.join(services_dir, service_name)
      if os.path.isdir(service_path):
        module_path = os.path.join(service_path, file_name)
        if os.path.isfile(module_path) and module_path.endswith(".py"):
          module_name = f"services.{service_name}.{file_name[:-3]}"  # Remove '.py' from file_name
          module = importlib.import_module(module_name)
          for attribute_name in dir(module):
            if attribute_name.endswith(class_suffix) and isinstance(getattr(module, attribute_name), type):
              loaded_modules[attribute_name] = getattr(module, attribute_name)
    return loaded_modules
