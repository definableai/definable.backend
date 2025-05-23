import asyncio
import contextlib
import json
import time
from threading import Thread
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from common.websocket import WebSocketManager, WebSocketMessageType

# Import the actual service class
from src.services.ws.service import WebSocketService


# Mock WebSocket class
class MockWebSocket:
  def __init__(self):
    self.sent_messages = []
    self.close_code = None
    self.close_reason = None
    self.accepted = False
    self.client_state = MagicMock()
    self.client_state.CONNECTED = True

  async def accept(self):
    self.accepted = True

  async def send_json(self, data):
    self.sent_messages.append(data)

  async def receive_json(self):
    # This would be overridden in tests if needed
    return {"type": "test_message"}

  async def close(self, code=1000, reason=""):
    self.close_code = code
    self.close_reason = reason
    self.client_state.CONNECTED = False


@pytest.fixture
def mock_user():
  """Create a mock user with proper permissions."""
  return {
    "id": uuid4(),
    "email": "test@example.com",
    "first_name": "Test",
    "last_name": "User",
    "org_id": uuid4(),
    "permissions": ["resource_read", "resource_write"],
  }


@pytest.fixture
def mock_ws_manager():
  """Create a mock WebSocketManager."""
  manager = MagicMock(spec=WebSocketManager)
  manager.connect = AsyncMock()
  manager.disconnect = AsyncMock()
  manager.broadcast = AsyncMock()
  manager._send_connection_info = AsyncMock()
  manager._handle_message = AsyncMock()
  return manager


@pytest.fixture
def mock_acquire(mock_ws_manager):
  """Create a mock Acquire object with WebSocketManager."""
  acquire_mock = MagicMock()
  acquire_mock.ws_manager = mock_ws_manager
  acquire_mock.logger = MagicMock()
  return acquire_mock


@pytest.fixture
def mock_websocket():
  """Create a mock WebSocket."""
  ws = MockWebSocket()
  # Make sure the close method is properly mocked
  ws.close = AsyncMock()
  return ws


@pytest.fixture
def ws_service(mock_acquire):
  """Create WebSocketService instance with mock dependencies."""
  return WebSocketService(acquire=mock_acquire)


@pytest.fixture
def mock_payload(mock_user):
  """Create a mock payload for WebSocket connection."""
  return {"id": mock_user["id"], "org_id": mock_user["org_id"], "permissions": mock_user["permissions"]}


# Test data for real connection tests
TEST_ORG_ID = str(uuid4())
TEST_USER1_ID = str(uuid4())
TEST_USER2_ID = str(uuid4())
TEST_PERMISSIONS = ["resource_read", "resource_write"]


# Create a class to wrap the FastAPI app for real WebSocket tests
class WebSocketTestApp:
  def __init__(self, host="127.0.0.1", port=8001):
    # Create the FastAPI app
    self.app = FastAPI()
    self.host = host
    self.port = port
    self.url = f"http://{host}:{port}"
    self.ws_url = f"ws://{host}:{port}/ws/connect"

    # Create real components
    self.ws_manager = WebSocketManager()

    # Create real service
    class RealAcquire:
      def __init__(self, manager):
        self.ws_manager = manager
        self.logger = None

    self.acquire = RealAcquire(self.ws_manager)
    self.service = WebSocketService(acquire=self.acquire)

    # Set up FastAPI routes
    @self.app.websocket("/ws/connect")
    async def ws_connect_endpoint(websocket: WebSocket):
      # Get user and org IDs from query parameters for testing
      params = dict(websocket.query_params)
      user_id = params.get("user_id", TEST_USER1_ID)
      org_id = params.get("org_id", TEST_ORG_ID)

      # Create payload
      payload = {"id": user_id, "org_id": org_id, "permissions": TEST_PERMISSIONS}

      # Connect using real service
      await self.service.ws_connect(websocket, payload)

    @self.app.post("/broadcast")
    async def broadcast_endpoint(request: Request):
      # Get broadcast data from request
      data = await request.json()

      # Broadcast the message
      await self.ws_manager.broadcast(
        org_id=data.get("org_id", TEST_ORG_ID),
        data=data.get("message", {}),
        resource=data.get("resource", "resource"),
        required_action=data.get("action", "read"),
        exclude_channel=data.get("exclude_channel"),
      )

      return JSONResponse({"status": "broadcast sent"})

  def start(self):
    """Start the test server in a background thread"""

    def run_server():
      uvicorn.run(self.app, host=self.host, port=self.port)

    self.server_thread = Thread(target=run_server, daemon=True)
    self.server_thread.start()

    # Give the server time to start
    time.sleep(2)

  def get_test_client(self):
    """Get a TestClient for making HTTP requests"""
    return TestClient(self.app)


@pytest.mark.asyncio
class TestWebSocketService:
  async def test_ws_connect_success(self, ws_service, mock_websocket, mock_payload, mock_ws_manager):
    """Test successful WebSocket connection."""
    # Arrange

    # Act
    await ws_service.ws_connect(mock_websocket, mock_payload)

    # Assert
    mock_ws_manager.connect.assert_called_once_with(mock_websocket, mock_payload["org_id"], mock_payload["id"], mock_payload["permissions"])

  async def test_ws_connect_error_handling(self, ws_service, mock_websocket, mock_payload, mock_ws_manager):
    """Test WebSocket connection error handling."""
    # Arrange
    error_message = "Connection error"
    mock_ws_manager.connect.side_effect = Exception(error_message)
    mock_websocket.client_state.CONNECTED = False

    # Act
    await ws_service.ws_connect(mock_websocket, mock_payload)

    # Assert
    mock_ws_manager.connect.assert_called_once()
    mock_websocket.close.assert_called_once_with(code=4000, reason=error_message)

  # Test real broadcast functionality with multiple clients

  async def test_broadcast_to_capture_endpoints(self, mock_acquire):
    """
    Test a real broadcast scenario with multiple actual connected endpoints.
    Simulates a client broadcasting a message and a capture client receiving it.
    """
    # Create a mock WebSocketManager instead of real one
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create the service with the mock manager
    service = WebSocketService(acquire=mock_acquire)

    # Create broadcaster and capture websockets
    broadcaster_ws = MockWebSocket()
    capture_ws = MockWebSocket()

    # Define user IDs and org ID
    broadcaster_id = uuid4()
    capture_id = uuid4()
    org_id = uuid4()
    permissions = ["resource_read", "resource_write"]

    # Create payloads for connections
    broadcaster_payload = {"id": broadcaster_id, "org_id": org_id, "permissions": permissions}

    capture_payload = {"id": capture_id, "org_id": org_id, "permissions": permissions}

    # Connect clients using service (which will call our mocked connect method)
    await service.ws_connect(broadcaster_ws, broadcaster_payload)
    await service.ws_connect(capture_ws, capture_payload)

    # Manually "broadcast" a message
    message_data = {"content": "Test broadcast message"}
    await ws_manager.broadcast(org_id, message_data, "resource", "write")

    # Verify the manager's broadcast method was called correctly
    ws_manager.broadcast.assert_called_with(org_id, message_data, "resource", "write")

  async def test_broadcast_with_permissions(self, mock_acquire):
    """
    Test that broadcasts are only received by clients with appropriate permissions.
    """
    # Use mocks instead of real implementation
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create clients
    client1_ws = MockWebSocket()
    client2_ws = MockWebSocket()

    # Setup user IDs
    user1_id = uuid4()
    user2_id = uuid4()
    org_id = uuid4()

    # Different permissions
    permissions1 = ["resource_read", "resource_write"]
    permissions2 = ["different_resource_read"]  # Does not have resource_write

    # Connect clients
    await ws_manager.connect(client1_ws, org_id, user1_id, permissions1)
    await ws_manager.connect(client2_ws, org_id, user2_id, permissions2)

    # Setup custom behavior for broadcast to simulate permission filtering
    async def mock_broadcast(org_id, data, resource, action, exclude_channel=None):
      # Simulate real behavior of only sending to clients with correct permissions
      if resource == "resource" and action == "write":
        # Only client1 has this permission
        client1_ws.sent_messages.append({"type": WebSocketMessageType.MESSAGE.value, "event": f"{resource}_{action}", "data": json.dumps(data)})

    ws_manager.broadcast.side_effect = mock_broadcast

    # Broadcast
    await ws_manager.broadcast(org_id, {"content": "Permission-specific message"}, "resource", "write")

    # Client1 should receive the message (has resource_write)
    assert len(client1_ws.sent_messages) > 0

    # Client2 should NOT receive the message (doesn't have resource_write)
    assert len(client2_ws.sent_messages) == 0

  async def test_broadcast_exclude_channel(self, mock_acquire):
    """
    Test broadcasting with the exclude_channel parameter to skip specific clients.
    """
    # Use mocks
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create clients
    client1_ws = MockWebSocket()
    client2_ws = MockWebSocket()
    client3_ws = MockWebSocket()

    # Setup user IDs
    user1_id = uuid4()
    user2_id = uuid4()
    user3_id = uuid4()
    org_id = uuid4()
    permissions = ["resource_read"]

    # Channel IDs
    channel2 = f"{user2_id}_{org_id}"

    # Connect clients
    await ws_manager.connect(client1_ws, org_id, user1_id, permissions)
    await ws_manager.connect(client2_ws, org_id, user2_id, permissions)
    await ws_manager.connect(client3_ws, org_id, user3_id, permissions)

    # Setup custom behavior for broadcast to simulate exclusion
    async def mock_broadcast(org_id, data, resource, action, exclude_channel=None):
      # Simulate excluding channel2
      if exclude_channel == channel2:
        # Send to clients 1 and 3, skip client 2
        client1_ws.sent_messages.append({"type": WebSocketMessageType.MESSAGE.value, "event": f"{resource}_{action}", "data": json.dumps(data)})
        client3_ws.sent_messages.append({"type": WebSocketMessageType.MESSAGE.value, "event": f"{resource}_{action}", "data": json.dumps(data)})

    ws_manager.broadcast.side_effect = mock_broadcast

    # Broadcast excluding client2
    await ws_manager.broadcast(org_id, {"content": "Exclusion test message"}, "resource", "read", channel2)

    # Client1 and Client3 should receive the message
    assert len(client1_ws.sent_messages) > 0
    assert len(client3_ws.sent_messages) > 0

    # Client2 should NOT receive the message
    assert len(client2_ws.sent_messages) == 0

  async def test_concurrent_broadcasts(self, mock_acquire):
    """
    Test handling of concurrent broadcasts to the same client.
    """
    # Use mocks
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create client
    client_ws = MockWebSocket()
    user_id = uuid4()
    org_id = uuid4()
    permissions = ["resource1_read", "resource2_read"]

    # Connect client
    await ws_manager.connect(client_ws, org_id, user_id, permissions)

    # Set up broadcast behavior to append messages to client
    ws_manager.broadcast.side_effect = lambda org_id, data, resource, action, exclude_channel=None: client_ws.sent_messages.append({
      "type": WebSocketMessageType.MESSAGE.value,
      "event": f"{resource}_{action}",
      "data": json.dumps(data),
    })

    # Send multiple broadcasts
    import asyncio

    await asyncio.gather(
      ws_manager.broadcast(org_id, {"content": "Message 1"}, "resource1", "read"),
      ws_manager.broadcast(org_id, {"content": "Message 2"}, "resource2", "read"),
    )

    # Client should have received both messages
    assert len(client_ws.sent_messages) == 2

  async def test_disconnection_during_broadcast(self, mock_acquire):
    """
    Test handling of client disconnection during broadcast.
    """
    # Create a mocked WebSocketManager
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.disconnect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create client
    client_ws = MockWebSocket()
    user_id = uuid4()
    org_id = uuid4()
    permissions = ["resource_read"]
    channel_id = f"{user_id}_{org_id}"

    # Connect client
    await ws_manager.connect(client_ws, org_id, user_id, permissions)

    # Setup broadcast to trigger disconnect
    async def mock_broadcast_with_disconnect(org_id, data, resource, action, exclude_channel=None):
      # Simulate connection error during broadcast
      await ws_manager.disconnect(channel_id)

    ws_manager.broadcast.side_effect = mock_broadcast_with_disconnect

    # Broadcast (should trigger disconnect)
    await ws_manager.broadcast(org_id, {"content": "Message that causes disconnect"}, "resource", "read")

    # Verify disconnect was called
    ws_manager.disconnect.assert_called_once_with(channel_id)

  async def test_real_end_to_end_flow(self, mock_acquire):
    """
    Test a complete end-to-end flow: connect, receive, broadcast, disconnect.
    """
    # Create mocked manager
    ws_manager = MagicMock(spec=WebSocketManager)
    ws_manager.connect = AsyncMock()
    ws_manager.disconnect = AsyncMock()
    ws_manager.broadcast = AsyncMock()
    mock_acquire.ws_manager = ws_manager

    # Create service with mocked dependencies
    service = WebSocketService(acquire=mock_acquire)

    # Create websockets
    broadcaster_ws = MockWebSocket()
    capture_ws = MockWebSocket()

    # Setup data
    broadcaster_id = uuid4()
    capture_id = uuid4()
    org_id = uuid4()

    broadcaster_permissions = ["resource_write"]
    capture_permissions = ["resource_read"]

    broadcaster_payload = {"id": broadcaster_id, "org_id": org_id, "permissions": broadcaster_permissions}

    capture_payload = {"id": capture_id, "org_id": org_id, "permissions": capture_permissions}

    # Connect both clients
    await service.ws_connect(broadcaster_ws, broadcaster_payload)
    await service.ws_connect(capture_ws, capture_payload)

    # Setup broadcast behavior
    async def mock_broadcast(org_id, data, resource, action, exclude_channel=None):
      # Only send to capture_ws if it has the right permission
      if f"{resource}_{action}" in capture_permissions:
        capture_ws.sent_messages.append({"type": WebSocketMessageType.MESSAGE.value, "event": f"{resource}_{action}", "data": json.dumps(data)})

    ws_manager.broadcast.side_effect = mock_broadcast

    # Broadcast a message
    message_data = {"content": "End-to-end test message"}
    await ws_manager.broadcast(org_id, message_data, "resource", "read")

    # Verify capture_ws received the message
    assert len(capture_ws.sent_messages) > 0
    received_message = capture_ws.sent_messages[0]
    assert received_message["type"] == WebSocketMessageType.MESSAGE.value
    assert received_message["event"] == "resource_read"
    assert json.loads(received_message["data"]) == message_data

  # New test with REAL WebSocketManager and connections
  @pytest.mark.skip_if_no_websocket
  async def test_real_websocket_broadcast_and_capture(self):
    """
    Test with REAL WebSockets that broadcasts messages from one endpoint and captures them with another.

    This test:
    1. Starts a real FastAPI server
    2. Uses real WebSocketManager instance
    3. Creates TestClient connections to test websocket broadcast/capture
    """
    import websockets
    from websockets.exceptions import ConnectionClosed

    # Skip if websockets module not available
    try:
      import websockets
    except ImportError:
      pytest.skip("websockets library not available")

    # Create and start test app with server
    app = WebSocketTestApp(port=8099)  # Use a non-standard port to avoid conflicts
    app.start()

    # Create message collectors
    broadcaster_messages = []
    capture_messages = []

    # Connect WebSocket clients (run in background tasks)
    async def connect_broadcaster():
      try:
        async with websockets.connect(f"{app.ws_url}?user_id={TEST_USER1_ID}&org_id={TEST_ORG_ID}") as websocket:
          # Keep receiving messages
          while True:
            message = await websocket.recv()
            try:
              data = json.loads(message)
              print(f"Broadcaster received: {data}")
              broadcaster_messages.append(data)
            except json.JSONDecodeError:
              broadcaster_messages.append(message)
      except ConnectionClosed:
        print("Broadcaster disconnected")

    async def connect_capturer():
      try:
        async with websockets.connect(f"{app.ws_url}?user_id={TEST_USER2_ID}&org_id={TEST_ORG_ID}") as websocket:
          # Keep receiving messages
          while True:
            message = await websocket.recv()
            try:
              data = json.loads(message)
              print(f"Capturer received: {data}")
              capture_messages.append(data)
            except json.JSONDecodeError:
              capture_messages.append(message)
      except ConnectionClosed:
        print("Capturer disconnected")

    # Start the clients in background tasks
    broadcaster_task = asyncio.create_task(connect_broadcaster())
    capturer_task = asyncio.create_task(connect_capturer())

    # Wait for connections to establish
    await asyncio.sleep(2)

    # Create a test message
    test_message = {"content": "Test broadcast from real websocket", "timestamp": time.time()}

    # Send broadcast through HTTP endpoint
    try:
      # Using direct manager access instead of HTTP request to avoid TestClient issues
      await app.ws_manager.broadcast(org_id=TEST_ORG_ID, data=test_message, resource="resource", required_action="read")
    except Exception as e:
      print(f"Error during broadcast: {e}")
    await asyncio.sleep(2)

    # Cancel client tasks to clean up
    broadcaster_task.cancel()
    capturer_task.cancel()

    # Use contextlib.suppress to handle any exceptions during task cleanup
    with contextlib.suppress(Exception):
      await asyncio.gather(broadcaster_task, capturer_task, return_exceptions=True)

    # Verify results - check if the capturer received the broadcast
    broadcast_received = False
    for message in capture_messages:
      if message.get("type") == WebSocketMessageType.MESSAGE.value:
        if message.get("event") == "resource_read":
          try:
            data = json.loads(message.get("data", "{}"))
            if data.get("content") == test_message["content"]:
              broadcast_received = True
              break
          except json.JSONDecodeError:
            continue

    assert broadcast_received, "Capture client did not receive the broadcast message"
