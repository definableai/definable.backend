import asyncio
import json
from base64 import b64decode
from typing import Any, Dict, List

from firebase_admin import App, credentials, db, get_app, initialize_app  # type: ignore

from config.settings import settings
from libs.response import LibResponse


class FirebaseManager:
  """Singleton Firebase manager for efficient database operations."""

  _instance: "FirebaseManager | None" = None
  _app: App | None = None
  _rtdb_ref = None

  def __new__(cls) -> "FirebaseManager":
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __init__(self):
    # Prevent re-initialization
    if hasattr(self, "_initialized"):
      return
    self._initialized = True

  def _get_app(self) -> App:
    """Get or create Firebase app."""
    if self._app is not None:
      return self._app

    try:
      # Check if default app already exists
      try:
        self._app = get_app()
        return self._app
      except ValueError:
        # No default app exists, create one
        pass

      # Decode credentials
      creds_64 = settings.firebase_creds
      creds_data = json.loads(b64decode(creds_64))

      # Initialize Firebase app
      cred = credentials.Certificate(creds_data)
      self._app = initialize_app(cred, {"databaseURL": settings.firebase_rtdb})
      return self._app

    except Exception as e:
      raise Exception(f"Failed to create Firebase app: {e}")

  def get_rtdb(self):
    """Get Firebase Realtime Database reference."""
    if self._rtdb_ref is None:
      self._get_app()  # Ensure app is initialized
      self._rtdb_ref = db.reference()
    return self._rtdb_ref


# Global singleton instance
_firebase_manager = FirebaseManager()


# Legacy functions for backwards compatibility
def get_firebase_app() -> App:
  """Get or create a Firebase app with the provided credentials and database URL."""
  return _firebase_manager._get_app()


def get_rtdb():
  """Get Firebase Realtime Database reference, initializing app if needed."""
  return _firebase_manager.get_rtdb()


# For backwards compatibility, provide rtdb as a function call
rtdb = get_rtdb


# === Async LibResponse-based Firebase Operations ===


async def write_data(path: str, data: Any) -> LibResponse[Dict[str, Any]]:
  """Write data to Firebase Realtime Database with LibResponse (async)."""
  try:

    def _write():
      rtdb_ref = _firebase_manager.get_rtdb()
      ref = rtdb_ref.child(path) if path else rtdb_ref
      ref.set(data)
      return {"path": path, "operation": "write", "data_written": True}

    result = await asyncio.to_thread(_write)
    return LibResponse.success_response(result)

  except Exception as e:
    return LibResponse.error_response([{"code": "FIREBASE_WRITE_ERROR", "message": f"Failed to write data to path '{path}': {str(e)}", "path": path}])


async def read_data(path: str) -> LibResponse[Any]:
  """Read data from Firebase Realtime Database with LibResponse (async)."""
  try:

    def _read():
      rtdb_ref = _firebase_manager.get_rtdb()
      ref = rtdb_ref.child(path) if path else rtdb_ref
      return ref.get()

    data = await asyncio.to_thread(_read)

    return LibResponse.success_response(
      data, meta={"path": path, "operation": "read", "data_type": type(data).__name__ if data is not None else "null"}
    )

  except Exception as e:
    return LibResponse.error_response([{"code": "FIREBASE_READ_ERROR", "message": f"Failed to read data from path '{path}': {str(e)}", "path": path}])


async def update_data(path: str, data: Dict[str, Any]) -> LibResponse[Dict[str, Any]]:
  """Update data in Firebase Realtime Database with LibResponse (async)."""
  try:

    def _update():
      rtdb_ref = _firebase_manager.get_rtdb()
      ref = rtdb_ref.child(path) if path else rtdb_ref
      ref.update(data)
      return {"path": path, "operation": "update", "updated_keys": list(data.keys()), "updated_count": len(data)}

    result = await asyncio.to_thread(_update)
    return LibResponse.success_response(result)

  except Exception as e:
    return LibResponse.error_response([
      {"code": "FIREBASE_UPDATE_ERROR", "message": f"Failed to update data at path '{path}': {str(e)}", "path": path}
    ])


async def delete_data(path: str) -> LibResponse[Dict[str, Any]]:
  """Delete data from Firebase Realtime Database with LibResponse (async)."""
  try:

    def _delete():
      rtdb_ref = _firebase_manager.get_rtdb()
      ref = rtdb_ref.child(path) if path else rtdb_ref
      ref.delete()
      return {"path": path, "operation": "delete", "deleted": True}

    result = await asyncio.to_thread(_delete)
    return LibResponse.success_response(result)

  except Exception as e:
    return LibResponse.error_response([
      {"code": "FIREBASE_DELETE_ERROR", "message": f"Failed to delete data at path '{path}': {str(e)}", "path": path}
    ])


async def delete_keys(keys: List[str], base_path: str = "") -> LibResponse[Dict[str, Any]]:
  """Delete multiple keys from Firebase Realtime Database with LibResponse (async)."""
  try:

    def _bulk_delete():
      rtdb_ref = _firebase_manager.get_rtdb()
      succeeded = []
      failed = []

      for key in keys:
        try:
          full_path = f"{base_path}/{key}" if base_path else key
          ref = rtdb_ref.child(full_path)
          ref.delete()
          succeeded.append(key)
        except Exception as e:
          failed.append({"key": key, "error": str(e)})

      return succeeded, failed

    succeeded, failed = await asyncio.to_thread(_bulk_delete)

    if len(failed) == 0:
      return LibResponse.success_response({"operation": "bulk_delete", "deleted_keys": succeeded, "deleted_count": len(succeeded)})
    elif len(succeeded) > 0:
      return LibResponse.partial_success(
        data={"operation": "bulk_delete", "deleted_keys": succeeded, "deleted_count": len(succeeded), "failed_count": len(failed)},
        succeeded=succeeded,
        failed=failed,
      )
    else:
      return LibResponse.error_response([{"code": "FIREBASE_BULK_DELETE_ERROR", "message": "Failed to delete any keys", "failed_items": failed}])

  except Exception as e:
    return LibResponse.error_response([{"code": "FIREBASE_BULK_DELETE_ERROR", "message": f"Failed to perform bulk delete: {str(e)}", "keys": keys}])


async def exists(path: str) -> LibResponse[bool]:
  """Check if a path exists in Firebase Realtime Database with LibResponse (async)."""
  try:

    def _exists():
      rtdb_ref = _firebase_manager.get_rtdb()
      ref = rtdb_ref.child(path)
      data = ref.get()
      return data is not None

    exists_result = await asyncio.to_thread(_exists)

    return LibResponse.success_response(exists_result, meta={"path": path, "operation": "exists_check"})

  except Exception as e:
    return LibResponse.error_response([
      {"code": "FIREBASE_EXISTS_ERROR", "message": f"Failed to check if path '{path}' exists: {str(e)}", "path": path}
    ])


# === Batch Operations ===


async def batch_write(operations: List[Dict[str, Any]]) -> LibResponse[Dict[str, Any]]:
  """Perform multiple write operations efficiently with single instance."""
  try:

    def _batch_write():
      rtdb_ref = _firebase_manager.get_rtdb()
      succeeded = []
      failed = []

      for op in operations:
        try:
          path = op.get("path", "")
          data = op.get("data")
          operation_type = op.get("type", "write")  # write, update, delete

          ref = rtdb_ref.child(path) if path else rtdb_ref

          if operation_type == "write":
            ref.set(data)
          elif operation_type == "update":
            ref.update(data)
          elif operation_type == "delete":
            ref.delete()

          succeeded.append({"path": path, "type": operation_type})

        except Exception as e:
          failed.append({"path": op.get("path", ""), "type": op.get("type", "write"), "error": str(e)})

      return succeeded, failed

    succeeded, failed = await asyncio.to_thread(_batch_write)

    if len(failed) == 0:
      return LibResponse.success_response({"operation": "batch_write", "succeeded_operations": succeeded, "succeeded_count": len(succeeded)})
    elif len(succeeded) > 0:
      return LibResponse.partial_success(
        data={"operation": "batch_write", "succeeded_operations": succeeded, "succeeded_count": len(succeeded), "failed_count": len(failed)},
        succeeded=succeeded,
        failed=failed,
      )
    else:
      return LibResponse.error_response([{"code": "FIREBASE_BATCH_WRITE_ERROR", "message": "All batch operations failed", "failed_items": failed}])

  except Exception as e:
    return LibResponse.error_response([{"code": "FIREBASE_BATCH_WRITE_ERROR", "message": f"Failed to perform batch operations: {str(e)}"}])
