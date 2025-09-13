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
    Recursively scans all directories under services_dir and registers any service.py files found.
    """
    self._register_services_recursive(self.services_dir, [])

  def _register_services_recursive(self, current_dir: str, path_segments: list[str]) -> None:
    """
    Recursively register services from directories.

    Args:
        current_dir (str): Current directory being scanned
        path_segments (list[str]): List of path segments built up through recursion
    """
    for item in os.listdir(current_dir):
      if item.startswith("__"):
        continue

      item_path = os.path.join(current_dir, item)

      if os.path.isdir(item_path):
        # Add this directory to the path segments and recurse
        self._register_services_recursive(item_path, path_segments + [item])
      elif item == "service.py":
        # Found a service.py file, register it with the accumulated path
        self._register_service_from_path(path_segments)

  def _register_service_from_path(self, path_segments: list[str]) -> None:
    """
    Register a service from a path sequence.

    Args:
        path_segments (list[str]): List of path segments (directories) leading to the service
    """
    if not path_segments:
      return

    # Create module path like "services.v1.kb.service"
    module_path = "services." + ".".join(path_segments) + ".service"

    # Create API path like "/api/v1/kb"
    api_path = f"{self.prefix}/" + "/".join(path_segments)

    try:
      service_module = importlib.import_module(module_path)
      service_class = next((cls for name, cls in inspect.getmembers(service_module, inspect.isclass) if name.endswith("Service")), None)

      if service_class:
        # Check if '__init__' accepts 'acquire' parameter
        init_params = inspect.signature(service_class.__init__).parameters
        if "acquire" in init_params:
          service_instance = service_class(acquire=self.acquire)
        else:
          service_instance = service_class()

        # Create tag name from the service name (last path segment, capitalized)
        tag_name = path_segments[-1].replace("_", " ").title()

        # Extract description from class docstring if it exists
        tag_description = None
        if service_class.__doc__:
          # Get the first line of the docstring and clean it up
          docstring_lines = service_class.__doc__.strip().split("\n")
          if docstring_lines:
            tag_description = docstring_lines[0].strip().rstrip(".")

        # Create router with tags for Swagger documentation
        router = APIRouter(prefix=api_path, tags=[tag_name])

        # Add tag description to OpenAPI metadata if docstring exists
        if tag_description:
          if not hasattr(self.app, "openapi_tags") or self.app.openapi_tags is None:
            self.app.openapi_tags = []
          # Only add if tag doesn't already exist
          if not any(tag.get("name") == tag_name for tag in self.app.openapi_tags):
            self.app.openapi_tags.append({"name": tag_name, "description": tag_description})

        # Register core methods (GET, POST, PUT, DELETE)
        for method in ["get", "post", "put", "delete"]:
          if hasattr(service_instance, method):
            endpoint = getattr(service_instance, method)
            router.add_api_route(path="", endpoint=endpoint, methods=[method.upper()])

        # Register exposed methods specified in http_exposed
        if hasattr(service_instance, "http_exposed"):
          for route in service_instance.http_exposed:
            # register ws routes
            self.register_ws_routes(router, service_instance, path_segments[-1])
            http_method, sub_path = route.split("=")
            endpoint_name = f"{http_method}_{sub_path}"
            if hasattr(service_instance, endpoint_name):
              endpoint = getattr(service_instance, endpoint_name)
              router.add_api_route(path=f"/{sub_path}", endpoint=endpoint, methods=[http_method.upper()])

        if router:
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
