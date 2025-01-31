import importlib
import inspect
import os

from fastapi import APIRouter, FastAPI

from .acquire import Acquire


# TODO: clean the manager class
class Manager:
  """
  Manager class for registering services and middlewares.

  Args:
    app (FastAPI): The FastAPI application instance.
    prefix (str): The base URL prefix for all services.

  Attributes:
    app (FastAPI): The FastAPI application instance.
    prefix (str): The base URL prefix for all services.
    services_dir (str): The directory containing service modules.
    mws_dir (str): The directory containing middleware modules.
  """

  def __init__(self, app: FastAPI, prefix: str = "/api"):
    self.app = app
    self.prefix = prefix
    self.acquire = Acquire()
    self.services_dir = os.path.join(os.path.dirname(__file__), "..")
    if not os.path.exists(self.services_dir):
      raise FileNotFoundError(f"Services directory not found: {self.services_dir}")
    self.mws_dir = os.path.join(os.path.dirname(__file__), "..", "..", "middlewares")
    if not os.path.exists(self.mws_dir):
      raise FileNotFoundError(f"Middlewares directory not found: {self.mws_dir}")

  def register_services(self) -> None:
    """
    Register services with the FastAPI application.
    """
    for service_name in os.listdir(self.services_dir):
      if service_name.startswith("__"):
        continue
      service_path = os.path.join(self.services_dir, service_name)
      if os.path.isdir(service_path):
        service_module_path = f"services.{service_name}.service"
        try:
          service_module = importlib.import_module(service_module_path)
          service_class = next((cls for name, cls in inspect.getmembers(service_module, inspect.isclass) if name.endswith("Service")), None)
          if service_class:
            # Check if '__init__' accepts 'acquire' parameter
            init_params = inspect.signature(service_class.__init__).parameters
            if "acquire" in init_params:
              service_instance = service_class(acquire=self.acquire)
            else:
              service_instance = service_class()
            router = APIRouter(prefix=f"{self.prefix}/{service_name}")

            # Register core methods (GET, POST, PUT, DELETE)
            for method in ["get", "post", "put", "delete"]:
              if hasattr(service_instance, method):
                endpoint = getattr(service_instance, method)
                router.add_api_route(path="", endpoint=endpoint, methods=[method.upper()])

            # Register exposed methods specified in http_exposed
            if hasattr(service_instance, "http_exposed"):
              for route in service_instance.http_exposed:
                http_method, sub_path = route.split("=")
                endpoint_name = f"{http_method}_{sub_path}"
                # print(f"endpoint_name: {endpoint_name}")
                if hasattr(service_instance, endpoint_name):
                  endpoint = getattr(service_instance, endpoint_name)
                  router.add_api_route(path=f"/{sub_path}", endpoint=endpoint, methods=[http_method.upper()])

          self.app.include_router(router)
        except ModuleNotFoundError:
          pass

  def register_middlewares(self) -> None:
    """
    Register middlewares with the FastAPI application.
    """
    # ignore files starting with __
    for mw_name in os.listdir(self.mws_dir):
      if mw_name.startswith("__"):
        continue
      mw_path = os.path.join(self.mws_dir, mw_name)
      if os.path.isfile(mw_path) and mw_name.endswith(".py"):
        mw_module_name = mw_name[:-3]
        mw_module_path = f"middlewares.{mw_module_name}"
        mw_module = importlib.import_module(mw_module_path)
        mw_class = getattr(mw_module, "Middleware", None)
        if mw_class:
          self.app.add_middleware(mw_class)
        else:
          print(f"Middleware class not found: {mw_name}")
      else:
        print(f"Not a valid middleware: {mw_path}")
