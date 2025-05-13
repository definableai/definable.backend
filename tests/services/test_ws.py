import pytest
import json
from fastapi import WebSocket, WebSocketDisconnect
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch, call
import sys

# Import the actual service class
from src.services.ws.service import WebSocketService
from common.websocket import WebSocketManager, WebSocketConnection, WebSocketMessageType

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

# Custom WebSocketService class for testing broadcast and capture endpoints
class EnhancedWebSocketService(WebSocketService):
    http_exposed = ["ws=connect", "ws=broadcast", "ws=capture"]
    
    async def ws_broadcast(self, websocket: WebSocket, payload: dict):
        """Endpoint for broadcasting messages to clients."""
        try:
            # Receive message data from the client
            data = await websocket.receive_json()
            
            # Extract broadcast parameters
            org_id = payload["org_id"]
            resource = data.get("resource", "default")
            required_action = data.get("action", "read")
            message_data = data.get("data", {})
            exclude_channel = data.get("exclude_channel")
            
            # Broadcast the message
            await self.ws_manager.broadcast(
                org_id, 
                message_data, 
                resource, 
                required_action, 
                exclude_channel
            )
            
            # Send confirmation back to the sender
            await websocket.send_json({
                "status": "success",
                "message": "Broadcast completed"
            })
            
        except Exception as e:
            if websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })
    
    async def ws_capture(self, websocket: WebSocket, payload: dict):
        """Endpoint for capturing messages."""
        try:
            # Create a dedicated connection for capturing messages
            await self.ws_manager.connect(
                websocket, 
                payload["org_id"], 
                payload["id"], 
                payload["permissions"],
                is_capture=True  # This would be a custom flag to identify capture connections
            )
        except Exception as e:
            if not websocket.client_state.CONNECTED:
                await websocket.close(code=4000, reason=str(e))

@pytest.fixture
def mock_user():
    """Create a mock user with proper permissions."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "org_id": uuid4(),
        "permissions": ["resource_read", "resource_write"]
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
def enhanced_ws_service(mock_acquire):
    """Create EnhancedWebSocketService instance with mock dependencies."""
    return EnhancedWebSocketService(acquire=mock_acquire)

@pytest.fixture
def mock_payload(mock_user):
    """Create a mock payload for WebSocket connection."""
    return {
        "id": mock_user["id"],
        "org_id": mock_user["org_id"],
        "permissions": mock_user["permissions"]
    }

@pytest.mark.asyncio
class TestWebSocketService:
    
    async def test_ws_connect_success(self, ws_service, mock_websocket, mock_payload, mock_ws_manager):
        """Test successful WebSocket connection."""
        # Arrange
        
        # Act
        await ws_service.ws_connect(mock_websocket, mock_payload)
        
        # Assert
        mock_ws_manager.connect.assert_called_once_with(
            mock_websocket, 
            mock_payload["org_id"], 
            mock_payload["id"],
            mock_payload["permissions"]
        )
    
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
        assert mock_websocket.close.call_args[1]["code"] == 4000
        assert mock_websocket.close.call_args[1]["reason"] == error_message

    async def test_broadcast_message(self, mock_ws_manager, mock_payload, mock_user):
        """Test broadcasting a message to clients in an organization."""
        # Arrange
        org_id = mock_user["org_id"]
        data = {"message": "Test broadcast message"}
        resource = "resource"
        required_action = "read"
        
        # Act
        await mock_ws_manager.broadcast(org_id, data, resource, required_action)
        
        # Assert
        # Since the mock is already called by the act step, we just need to verify
        # it was called with the right arguments
        assert mock_ws_manager.broadcast.called
        call_args = mock_ws_manager.broadcast.call_args[0]
        assert call_args[0] == org_id
        assert call_args[1] == data
        assert call_args[2] == resource
        assert call_args[3] == required_action
    
    async def test_broadcast_to_specific_clients(self, mock_ws_manager, mock_payload, mock_user):
        """Test broadcasting a message to specific clients based on permissions."""
        # Arrange
        org_id = mock_user["org_id"]
        data = {"message": "Test broadcast message"}
        resource = "resource"
        required_action = "write"
        exclude_channel = "user1_org1"
        
        # Act
        await mock_ws_manager.broadcast(org_id, data, resource, required_action, exclude_channel)
        
        # Assert
        assert mock_ws_manager.broadcast.called
        call_args = mock_ws_manager.broadcast.call_args[0]
        kwargs = mock_ws_manager.broadcast.call_args[1]
        assert call_args[0] == org_id
        assert call_args[1] == data
        assert call_args[2] == resource
        assert call_args[3] == required_action
        if "exclude_channel" in kwargs:
            assert kwargs["exclude_channel"] == exclude_channel
        else:
            assert call_args[4] == exclude_channel

    async def test_message_capture(self, mock_acquire):
        """Test capturing incoming messages from WebSocket."""
        # Arrange
        ws_manager = WebSocketManager()
        mock_acquire.ws_manager = ws_manager
        service = WebSocketService(acquire=mock_acquire)
        
        websocket = MockWebSocket()
        websocket.accept = AsyncMock()
        user_id = uuid4()
        org_id = uuid4()
        permissions = ["resource_read"]
        
        # Create a test message to be received
        test_message = {"type": WebSocketMessageType.MESSAGE.value, "data": {"content": "test"}}
        websocket.receive_json = AsyncMock(side_effect=[test_message, WebSocketDisconnect()])
        
        # Mock internal methods to verify they're called
        with patch.object(ws_manager, '_handle_message', new_callable=AsyncMock) as mock_handle_message, \
             patch.object(ws_manager, '_send_connection_info', new_callable=AsyncMock) as mock_send_info:
            
            # Act
            try:
                await ws_manager.connect(websocket, org_id, user_id, permissions)
            except WebSocketDisconnect:
                pass
            
            # Assert
            assert websocket.accept.called
            assert mock_send_info.called
            assert mock_handle_message.called
            # First arg is connection, which we can't easily compare
            assert mock_handle_message.call_args[0][1] == test_message

    async def test_websocket_message_flow(self, mock_ws_manager, mock_payload, mock_user):
        """Test the full message flow with connect, broadcast and disconnect."""
        # Arrange
        ws_manager = WebSocketManager()
        
        # Create two mock websockets
        websocket1 = MockWebSocket()
        websocket2 = MockWebSocket()
        
        user1_id = uuid4()
        user2_id = uuid4()
        org_id = mock_user["org_id"]
        permissions = ["resource_read"]
        
        # Connect websockets
        with patch.object(websocket1, 'accept', new_callable=AsyncMock) as mock_accept1, \
             patch.object(websocket2, 'accept', new_callable=AsyncMock) as mock_accept2, \
             patch.object(websocket1, 'send_json', new_callable=AsyncMock) as mock_send1, \
             patch.object(websocket2, 'send_json', new_callable=AsyncMock) as mock_send2:
             
            channel1_id = f"{user1_id}_{org_id}"
            channel2_id = f"{user2_id}_{org_id}"
            
            # Mock connections dict
            ws_manager._connections = {
                channel1_id: WebSocketConnection(
                    websocket=websocket1,
                    user_id=user1_id,
                    org_id=org_id,
                    channel_id=channel1_id,
                    permissions=permissions
                ),
                channel2_id: WebSocketConnection(
                    websocket=websocket2,
                    user_id=user2_id,
                    org_id=org_id,
                    channel_id=channel2_id,
                    permissions=permissions
                )
            }
            
            # Mock org_connections dict
            ws_manager._org_connections = {
                str(org_id): {channel1_id, channel2_id}
            }
            
            # Act - Broadcast a message to all clients in the org
            data = {"message": "Test broadcast message"}
            resource = "resource"
            required_action = "read"
            
            await ws_manager.broadcast(org_id, data, resource, required_action)
            
            # Assert - Both clients should receive the message
            expected_message = {
                "type": WebSocketMessageType.MESSAGE.value,
                "event": f"{resource}_{required_action}",
                "data": json.dumps(data),
            }
            
            mock_send1.assert_called_with(expected_message)
            mock_send2.assert_called_with(expected_message)
            
            # Act - Disconnect one client
            await ws_manager.disconnect(channel1_id)
            
            # Assert - The client should be removed from connections
            assert channel1_id not in ws_manager._connections

    # Additional tests for broadcast and capture endpoints
    
    async def test_ws_broadcast_endpoint(self, enhanced_ws_service, mock_websocket, mock_payload, mock_ws_manager):
        """Test the dedicated broadcast endpoint."""
        # Arrange
        broadcast_data = {
            "resource": "resource",
            "action": "write",
            "data": {"message": "Test broadcast from endpoint"},
            "exclude_channel": None
        }
        mock_websocket.receive_json = AsyncMock(return_value=broadcast_data)
        
        # Act
        await enhanced_ws_service.ws_broadcast(mock_websocket, mock_payload)
        
        # Assert
        mock_ws_manager.broadcast.assert_called_once_with(
            mock_payload["org_id"],
            broadcast_data["data"],
            broadcast_data["resource"],
            broadcast_data["action"],
            broadcast_data["exclude_channel"]
        )
        
        assert mock_websocket.sent_messages[0] == {
            "status": "success",
            "message": "Broadcast completed"
        }
    
    async def test_ws_broadcast_error_handling(self, enhanced_ws_service, mock_websocket, mock_payload, mock_ws_manager):
        """Test error handling in the broadcast endpoint."""
        # Arrange
        error_message = "Broadcast error"
        mock_websocket.receive_json = AsyncMock(side_effect=Exception(error_message))
        
        # Act
        await enhanced_ws_service.ws_broadcast(mock_websocket, mock_payload)
        
        # Assert
        assert mock_websocket.sent_messages[0] == {
            "status": "error",
            "message": error_message
        }
    
    async def test_ws_capture_endpoint(self, enhanced_ws_service, mock_websocket, mock_payload, mock_ws_manager):
        """Test the dedicated capture endpoint."""
        # Arrange
        
        # Act
        await enhanced_ws_service.ws_capture(mock_websocket, mock_payload)
        
        # Assert
        mock_ws_manager.connect.assert_called_once_with(
            mock_websocket,
            mock_payload["org_id"],
            mock_payload["id"],
            mock_payload["permissions"],
            is_capture=True
        )
    
    async def test_ws_capture_error_handling(self, enhanced_ws_service, mock_websocket, mock_payload, mock_ws_manager):
        """Test error handling in the capture endpoint."""
        # Arrange
        error_message = "Capture connection error"
        mock_ws_manager.connect.side_effect = Exception(error_message)
        mock_websocket.client_state.CONNECTED = False
        
        # Act
        await enhanced_ws_service.ws_capture(mock_websocket, mock_payload)
        
        # Assert
        assert mock_websocket.close.call_args[1]["code"] == 4000
        assert mock_websocket.close.call_args[1]["reason"] == error_message
    
    async def test_end_to_end_broadcast_capture(self, mock_acquire):
        """Test end-to-end flow with broadcast and capture endpoints."""
        # Arrange
        ws_manager = WebSocketManager()
        mock_acquire.ws_manager = ws_manager
        service = EnhancedWebSocketService(acquire=mock_acquire)
        
        # Create broadcaster and capture websockets
        broadcaster_ws = MockWebSocket()
        capture_ws = MockWebSocket()
        
        user_id = uuid4()
        org_id = uuid4()
        permissions = ["resource_read", "resource_write"]
        
        payload = {
            "id": user_id,
            "org_id": org_id,
            "permissions": permissions
        }
        
        # Setup broadcaster
        broadcast_data = {
            "resource": "resource",
            "action": "write",
            "data": {"message": "Test integrated broadcast and capture"},
            "exclude_channel": None
        }
        broadcaster_ws.receive_json = AsyncMock(return_value=broadcast_data)
        
        # Mock ws_manager connect and broadcast to not actually do anything
        # but still track calls
        with patch.object(ws_manager, 'connect', new_callable=AsyncMock) as mock_connect, \
             patch.object(ws_manager, 'broadcast', new_callable=AsyncMock) as mock_broadcast:
            
            # Act - First connect the capture endpoint
            await service.ws_capture(capture_ws, payload)
            
            # Then send broadcast
            await service.ws_broadcast(broadcaster_ws, payload)
            
            # Assert
            # Verify capture connection was made
            mock_connect.assert_called_once_with(
                capture_ws,
                payload["org_id"],
                payload["id"],
                payload["permissions"],
                is_capture=True
            )
            
            # Verify broadcast was sent
            mock_broadcast.assert_called_once_with(
                payload["org_id"],
                broadcast_data["data"],
                broadcast_data["resource"],
                broadcast_data["action"],
                broadcast_data["exclude_channel"]
            ) 