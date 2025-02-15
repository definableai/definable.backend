import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set
from uuid import UUID

from fastapi import Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import log
from database import get_db


class WebSocketMessageType(Enum):
  CONNECT = "connect"
  SUBSCRIBE = "subscribe"
  UNSUBSCRIBE = "unsubscribe"
  ERROR = "error"
  MESSAGE = "message"


@dataclass
class WebSocketConnection:
  """Represents a WebSocket connection with its context."""

  websocket: WebSocket
  user_id: UUID
  org_id: UUID
  channel_id: str
  permissions: list


# TODO : more robust approach is required to handle socket clients.
class WebSocketManager:
  """WebSocket connection manager with RBAC integration."""

  def __init__(self):
    self._connections: Dict[str, WebSocketConnection] = {}  # channel_id -> connection
    self._org_connections: Dict[str, Set[str]] = {}  # org_id -> set of channel_ids
    self._logger = log.bind(service="websocket")

  async def connect(self, websocket: WebSocket, org_id: UUID, user_id: UUID, permissions: list, session: AsyncSession = Depends(get_db)) -> None:
    """Handle new WebSocket connection."""
    try:
      channel_id = f"{user_id}_{org_id}"
      connection = WebSocketConnection(websocket=websocket, user_id=user_id, org_id=org_id, channel_id=channel_id, permissions=permissions)
      # Accept connection
      await websocket.accept()

      # Store connection
      self._connections[connection.channel_id] = connection

      if connection.org_id not in self._org_connections:
        self._org_connections[str(connection.org_id)] = set()
      self._org_connections[str(connection.org_id)].add(connection.channel_id)

      # Send connection confirmation with available events
      await self._send_connection_info(connection)

      # Handle incoming messages
      try:
        while True:
          message = await websocket.receive_json()
          print(message)
          await self._handle_message(connection, message)

      except WebSocketDisconnect:
        await self.disconnect(connection.channel_id)

    except Exception as e:
      self._logger.error(f"WebSocket error: {str(e)}")
      await websocket.close(code=4000, reason=str(e))

  async def disconnect(self, channel_id: str) -> None:
    """Handle client disconnection."""
    if channel_id in self._connections:
      connection = self._connections[channel_id]

      # Clean up organization connections
      if connection.org_id in self._org_connections:
        self._org_connections[str(connection.org_id)].remove(channel_id)
        if not self._org_connections[str(connection.org_id)]:
          del self._org_connections[str(connection.org_id)]

      # Remove connection
      del self._connections[channel_id]

      self._logger.info("Client disconnected", channel_id=channel_id)

  # TODO: convert this to pydantic model especially for data
  async def broadcast(self, org_id: UUID, data: dict, resource: str, required_action: str = "read", exclude_channel: Optional[str] = None) -> None:
    """Broadcast message to all eligible clients in an organization."""
    if str(org_id) not in self._org_connections.keys():
      return
    org_conns = list(self._org_connections[str(org_id)])  # type : ignore
    for channel_id in org_conns:
      if channel_id == exclude_channel:
        continue
      connection = self._connections[channel_id]

      # Check if client has required permission
      if f"{resource}_{required_action}" in connection.permissions:
        try:
          await connection.websocket.send_json({
            "type": WebSocketMessageType.MESSAGE.value,
            "event": f"{resource}_{required_action}",
            "data": json.dumps(data),
          })
        except Exception as e:
          self._logger.error(f"Failed to send to {channel_id}: {str(e)}")
          await self.disconnect(channel_id)

  ### Private methods ###

  async def _send_connection_info(self, connection: WebSocketConnection):
    """Send initial connection information."""
    await connection.websocket.send_json({
      "type": WebSocketMessageType.CONNECT.value,
      "channel_id": connection.channel_id,
      "events": connection.permissions,
    })

  async def _handle_message(self, connection: WebSocketConnection, message: dict) -> None:
    """Handle incoming WebSocket messages."""
    try:
      message_type = message.get("type")

      if message_type == WebSocketMessageType.ERROR.value:
        self._logger.error("Client error", channel_id=connection.channel_id, error=message.get("error"))

    except Exception as e:
      self._logger.error(f"Message handling error: {str(e)}")
      await connection.websocket.send_json({"type": WebSocketMessageType.ERROR.value, "error": str(e)})
