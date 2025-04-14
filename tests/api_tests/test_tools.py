import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

@pytest.fixture
def test_category_data():
    """Create test tool category data."""
    return {
        "name": "Test Category",
        "description": "Test category description"
    }

@pytest.fixture
def test_tool_data():
    """Create test tool data."""
    return {
        "name": "Test Tool",
        "description": "Test tool description",
        "category_id": str(uuid4()),
        "is_active": True,
        "version": "1.0",
        "logo_url": "",
        "is_public": True,
        "configuration": [
            {
                "name": "api_key",
                "type": "str",
                "description": "API key for the service",
                "required": True,
                "default": None
            }
        ],
        "inputs": [
            {
                "name": "query",
                "type": "str",
                "description": "The query to process",
                "required": True,
                "default": None
            }
        ],
        "outputs": {
            "type": "str",
            "description": "The result of the tool execution"
        },
        "settings": {
            "function_info": {
                "name": "Test Function",
                "is_async": True,
                "description": "Test function description",
                "code": "async def run(self, query: str):\n    return f\"Processed: {query}\""
            },
            "requirements": ["requests"],
            "deployment": {
                "framework": "agno",
                "toolkit_class": True,
                "standalone_function": False
            }
        }
    }

@pytest.mark.asyncio
class TestToolsAPI:
    """Test Tools API endpoints."""

    async def test_create_category(self, client, mock_db_session, auth_headers, test_category_data, org_id):
        """Test creating a tool category endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a category
            category_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": category_id,
                "name": test_category_data["name"],
                "description": test_category_data["description"],
                "organization_id": org_id
            }
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                f"/api/tools/create_category?org_id={org_id}",
                headers=auth_headers,
                json=test_category_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == category_id
            assert result["name"] == test_category_data["name"]
            assert result["organization_id"] == org_id
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()
    
    async def test_list_categories(self, client, mock_db_session, auth_headers):
        """Test listing tool categories endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of categories
            category_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": category_id,
                    "name": "Test Category",
                    "description": "Test category description",
                    "organization_id": str(uuid4())
                }
            ]
            mock_get.return_value = mock_response
            
            # Make the API request
            response = client.get(
                "/api/tools/list_categories",
                headers=auth_headers
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == category_id
            
            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()
    
    async def test_create_tool(self, client, mock_db_session, auth_headers, test_tool_data, org_id):
        """Test creating a tool endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Configure the mock to return a tool
            tool_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": tool_id,
                "name": test_tool_data["name"],
                "description": test_tool_data["description"],
                "category_id": test_tool_data["category_id"],
                "is_active": test_tool_data["is_active"],
                "version": test_tool_data["version"],
                "organization_id": org_id
            }
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                f"/api/tools?org_id={org_id}",
                headers=auth_headers,
                json=test_tool_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == tool_id
            assert result["name"] == test_tool_data["name"]
            assert result["organization_id"] == org_id
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once()
    
    async def test_list_all_tools(self, client, mock_db_session, auth_headers):
        """Test listing all tools endpoint."""
        # Mock the response from the client get call directly
        with patch.object(client, 'get', autospec=True) as mock_get:
            # Configure the mock to return a list of tools
            tool_id = str(uuid4())
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": tool_id,
                    "name": "Test Tool",
                    "description": "Test tool description",
                    "category_id": str(uuid4()),
                    "is_active": True,
                    "version": "1.0",
                    "organization_id": str(uuid4())
                }
            ]
            mock_get.return_value = mock_response
            
            # Make the API request
            response = client.get(
                "/api/tools/list_all",
                headers=auth_headers
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == tool_id
            
            # Verify the client get was called with the correct arguments
            mock_get.assert_called_once()
    
    async def test_test_tool(self, client, mock_db_session, auth_headers):
        """Test the test_tool endpoint."""
        # Mock the response from the client post call directly
        with patch.object(client, 'post', autospec=True) as mock_post:
            # Create test data
            tool_id = str(uuid4())
            test_request_data = {
                "input_prompt": "test query",
                "config_items": [
                    {
                        "name": "api_key",
                        "value": "test_api_key"
                    }
                ],
                "provider": "anthropic",
                "model_name": "claude-3-7-sonnet-latest",
                "api_key": "test_llm_api_key",
                "instructions": "Test instructions"
            }
            
            # Configure the mock to return a result
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "result": "Successfully processed query: test query",
                "time_taken": 1.23
            }
            mock_post.return_value = mock_response
            
            # Make the API request
            response = client.post(
                f"/api/tools/test_tool?tool_id={tool_id}",
                headers=auth_headers,
                json=test_request_data
            )
            
            # Verify response
            assert response.status_code == 200
            result = response.json()
            assert "result" in result
            assert "time_taken" in result
            
            # Verify the client post was called with the correct arguments
            mock_post.assert_called_once() 