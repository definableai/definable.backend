"""Utility module for dynamically loading other modules."""

import importlib
import os
from typing import Any, List


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

  @staticmethod
  def load_models() -> List[Any]:
    """
    Load all model classes from the models directory.

    Returns:
        List[Any]: A list of model classes that extend Base
    """
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    if not os.path.exists(models_dir):
      raise FileNotFoundError(f"Models directory not found: {models_dir}")

    all_models = []

    for model_file in os.listdir(models_dir):
      if model_file.startswith("__") or not model_file.endswith(".py"):
        continue

      module_name = f"models.{model_file[:-3]}"  # Remove '.py' from file_name
      try:
        module = importlib.import_module(module_name)

        # Find all classes in the module that are likely models
        for attribute_name in dir(module):
          attribute = getattr(module, attribute_name)
          if isinstance(attribute, type) and attribute_name.endswith("Model") and hasattr(attribute, "__tablename__"):
            all_models.append(attribute)
      except ImportError as e:
        print(f"Error importing {module_name}: {e}")

    return all_models
