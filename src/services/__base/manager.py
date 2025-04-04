import importlib
import inspect
import os
from typing import Any, Dict, Type

from fastapi import APIRouter, FastAPI, WebSocket

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
    self.ws_routes: Dict[str, Type] = {}

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
                # register ws routes
                self.register_ws_routes(router, service_instance, service_name)
                http_method, sub_path = route.split("=")
                endpoint_name = f"{http_method}_{sub_path.replace('/', '_')}"
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
          init_params = inspect.signature(mw_class.__init__).parameters
          if "acquire" in init_params:
            self.app.add_middleware(mw_class, acquire=self.acquire)
          else:
            self.app.add_middleware(mw_class)
        else:
          print(f"Middleware class not found: {mw_name}")
      else:
        print(f"Not a valid middleware: {mw_path}")

  def register_ws_routes(self, router: APIRouter, service_instance: Any, service_name: str) -> None:
    """Register WebSocket routes for a service."""
    for route in service_instance.http_exposed:
      if not route.startswith("ws="):
        continue

      _, sub_path = route.split("=")
      endpoint_name = f"ws_{sub_path}"

      if hasattr(service_instance, endpoint_name):
        endpoint = getattr(service_instance, endpoint_name)

        # Create WebSocket endpoint wrapper
        async def ws_endpoint(websocket: WebSocket) -> None:
          try:
            await endpoint(websocket)
          except Exception as e:
            # Log error
            print(f"WebSocket error in {service_name}: {str(e)}")
            if websocket.client_state.CONNECTED:
              await websocket.close(code=4000, reason=str(e))

        # Register WebSocket route
        router.add_api_websocket_route(path=f"/{sub_path}", endpoint=endpoint)

        # Store WebSocket route for reference
        self.ws_routes[f"{service_name}.{sub_path}"] = endpoint
