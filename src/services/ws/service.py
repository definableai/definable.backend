from fastapi import Depends, WebSocket

from dependencies.security import RBAC
from services.__base.acquire import Acquire


class WebSocketService:
  http_exposed = ["ws=connect"]

  def __init__(self, acquire: Acquire):
    self.ws_manager = acquire.ws_manager

  async def ws_connect(self, websocket: WebSocket, payload: dict = Depends(RBAC("*", "read"))):
    try:
      await self.ws_manager.connect(websocket, payload["org_id"], payload["id"], payload["permissions"])
    except Exception as e:
      if not websocket.client_state.CONNECTED:
        await websocket.close(code=4000, reason=str(e))
