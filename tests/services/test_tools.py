import pytest
import json
import pytest_asyncio
from uuid import UUID, uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select

from src.services.tools.service import ToolService
from src.services.tools.schema import (
    ToolCategoryCreate,
    ToolCategoryResponse,
    ToolCategoryUpdate,
    ToolCreate,
    ToolResponse,
    ToolUpdate,
    ToolTestRequest,
    ToolParameter,
    ToolOutput,
    ToolFunctionInfo,
    ToolSettings,
    PaginatedToolResponse
)
from src.services.__base.acquire import Acquire
from src.models import ToolCategoryModel, ToolModel


# TestAcquire - mock of the Acquire class for service initialization
class TestAcquire(Acquire):
    def __init__(self):
        self.settings = type('Settings', (), {
            'python_sandbox_testing_url': 'http://localhost:8000/test'
        })()
        self.logger = MagicMock()


@pytest.fixture
def test_user():
    """Create a test user dictionary."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
        "is_admin": True
    }


@pytest.fixture
def test_organization():
    """Create a test organization ID."""
    return uuid4()


@pytest.fixture
def tools_service():
    """Create a ToolService instance."""
    service = ToolService(acquire=TestAcquire())

    # Create a mock for the generator
    generator_mock = MagicMock()
    generator_mock.generate_toolkit_from_json = AsyncMock(return_value="test generated code")
    service.generator = generator_mock

    return service


class AsyncResultMock:
    """Mock for SQLAlchemy result that properly handles async patterns."""
    def __init__(self, value=None):
        self.value = value

    async def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self


class AsyncScalarsMock:
    """Mock for scalars result."""
    def __init__(self, values=None):
        self.values = values or []

    def all(self):
        return self.values


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()

    # Make add synchronous for simplicity
    session.add = MagicMock()
    # Make delete async since it's being awaited
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    # For scalar queries on count
    session.scalar = AsyncMock(return_value=1)

    return session


# Helper function for mock models
def create_model_mock(**kwargs):
    """Create a mock model with common attributes."""
    instance = MagicMock()

    # Set default attributes
    instance.id = kwargs.get('id', uuid4())
    instance.created_at = kwargs.get('created_at', datetime.now(timezone.utc))
    instance.updated_at = kwargs.get('updated_at', datetime.now(timezone.utc))

    # Set any additional attributes
    for key, value in kwargs.items():
        setattr(instance, key, value)

    return instance


@pytest.fixture
def sample_tool_create_data():
    """Create a sample tool creation data."""
    return ToolCreate(
        name="Test Tool",
        description="Test Description",
        category_id=uuid4(),
        version="1.0.0",
        is_public=False,
        is_verified=False,
        is_active=True,
        logo_url="https://example.com/logo.png",
        inputs=[
            ToolParameter(
                name="input1",
                type="string",
                description="Input 1",
                required=True
            )
        ],
        outputs=ToolOutput(
            type="string",
            description="Output"
        ),
        configuration=[
            ToolParameter(
                name="config1",
                type="string",
                description="Config 1",
                required=True
            )
        ],
        settings=ToolSettings(
            function_info=ToolFunctionInfo(
                name="test_function",
                is_async=True,
                description="Test Function",
                code="async def test_function(): return 'test'"
            ),
            requirements=["requests"]
        )
    )


@pytest.fixture
def sample_category_create_data():
    """Create a sample category creation data."""
    return ToolCategoryCreate(
        name="Test Category",
        description="Test Category Description"
    )


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool_id = uuid4()
    return create_model_mock(
        id=tool_id,
        name="Test Tool",
        description="Test Description",
        organization_id=uuid4(),
        user_id=uuid4(),
        category_id=uuid4(),
        logo_url="https://example.com/logo.png",
        is_active=True,
        version="1.0.0",
        is_public=False,
        is_verified=False,
        inputs=[{"name": "input1", "type": "string", "description": "Input 1", "required": True}],
        outputs={"type": "string", "description": "Output"},
        configuration=[{"name": "config1", "type": "string", "description": "Config 1", "required": True}],
        settings={
            "function_info": {
                "name": "test_function",
                "is_async": True,
                "description": "Test Function",
                "code": "async def test_function(): return 'test'"
            },
            "requirements": ["requests"]
        },
        generated_code="test generated code"
    )


@pytest.fixture
def mock_category():
    """Create a mock category."""
    category_id = uuid4()
    return create_model_mock(
        id=category_id,
        name="Test Category",
        description="Test Category Description"
    )


@pytest.mark.asyncio
class TestToolService:
    """Tests for the ToolService."""

    async def test_post_create_tool_success(self, tools_service, mock_db_session, test_user,
                                            test_organization, sample_tool_create_data, mock_category):
        """Test creating a new tool successfully."""
        # Mock the queries directly in the right order
        mock_execute = AsyncMock()

        # First check (if tool exists) - return None
        first_result = MagicMock()
        first_result.scalar_one_or_none.return_value = None

        # Second check (if category exists) - return category
        second_result = MagicMock()
        second_result.scalar_one_or_none.return_value = mock_category

        # Set up side effect sequence
        mock_execute.side_effect = [first_result, second_result]
        mock_db_session.execute = mock_execute

        # Mock the model creation
        with patch('src.models.ToolModel') as mock_model_class:
            db_tool = create_model_mock(id=uuid4())
            mock_model_class.return_value = db_tool

            response = await tools_service.post(
                org_id=test_organization,
                tool_data=sample_tool_create_data,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, JSONResponse)
        assert response.status_code == 201
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        assert tools_service.generator.generate_toolkit_from_json.called

    async def test_post_create_tool_existing(self, tools_service, mock_db_session, test_user, test_organization, sample_tool_create_data, mock_tool):
        """Test creating a tool that already exists."""
        # Setup mock session to return an existing tool
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_tool

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.post(
                org_id=test_organization,
                tool_data=sample_tool_create_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 400
        assert "already exists" in excinfo.value.detail

    async def test_post_create_tool_no_category(self, tools_service, mock_db_session, test_user, test_organization, sample_tool_create_data):
        """Test creating a tool with a non-existent category."""
        # Mock the queries directly in the right order
        mock_execute = AsyncMock()

        # First check (if tool exists) - return None
        first_result = MagicMock()
        first_result.scalar_one_or_none.return_value = None

        # Second check (if category exists) - return None to fail
        second_result = MagicMock()
        second_result.scalar_one_or_none.return_value = None

        # Set up side effect sequence
        mock_execute.side_effect = [first_result, second_result]
        mock_db_session.execute = mock_execute

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.post(
                org_id=test_organization,
                tool_data=sample_tool_create_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 400
        assert "Category not found" in excinfo.value.detail

    async def test_put_update_tool_success(self, tools_service, mock_db_session, test_user, test_organization, mock_tool):
        """Test updating a tool successfully."""
        # Setup mock session to return a tool
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_tool

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create update data
        update_data = ToolUpdate(
            name="Updated Tool",
            description="Updated Description"
        )

        response = await tools_service.put(
            org_id=test_organization,
            tool_id=mock_tool.id,
            tool_data=update_data,
            session=mock_db_session,
            user=test_user
        )

        # Assertions
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        assert mock_db_session.commit.called
        assert mock_tool.name == "Updated Tool"
        assert mock_tool.description == "Updated Description"

    async def test_put_update_tool_not_found(self, tools_service, mock_db_session, test_user, test_organization):
        """Test updating a non-existent tool."""
        # Setup mock session to not find the tool
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create update data
        update_data = ToolUpdate(
            name="Updated Tool",
            description="Updated Description"
        )

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.put(
                org_id=test_organization,
                tool_id=uuid4(),
                tool_data=update_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 404
        assert "Tool not found" in excinfo.value.detail

    async def test_post_test_tool_success(self, tools_service, mock_db_session, test_user, mock_tool):
        """Test the tool test functionality."""
        # Setup mock session to return a tool
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_tool

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create test request data
        test_request = ToolTestRequest(
            input_prompt="Test prompt",
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test_api_key",
            config_items=[{"name": "config1", "value": "test_value"}],
            instructions="Test instructions"
        )

        # Mock aiohttp.ClientSession
        with patch('aiohttp.ClientSession') as mock_session:
            # Create mockable context managers
            session_context = MagicMock()
            post_response = MagicMock()

            # Setup the mock chain
            mock_session.return_value.__aenter__.return_value = session_context
            session_context.post.return_value.__aenter__.return_value = post_response
            post_response.json = AsyncMock(return_value={"result": "success"})

            # Run the test
            response = await tools_service.post_test_tool(
                tool_id=mock_tool.id,
                tool_test_request=test_request,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        assert session_context.post.called

    async def test_post_test_tool_not_found(self, tools_service, mock_db_session, test_user):
        """Test testing a non-existent tool."""
        # Setup mock session to not find the tool
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create test request data
        test_request = ToolTestRequest(
            input_prompt="Test prompt",
            provider="openai",
            model_name="gpt-4o-mini",
            api_key="test_api_key",
            config_items=[{"name": "config1", "value": "test_value"}],
            instructions="Test instructions"
        )

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.post_test_tool(
                tool_id=uuid4(),
                tool_test_request=test_request,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 404
        assert "Tool not found" in excinfo.value.detail

    async def test_get_list_tools(self, tools_service, mock_db_session, test_user, test_organization, mock_tool):
        """Test listing tools."""
        # Mock the count query
        mock_db_session.scalar = AsyncMock(return_value=1)

        # Mock the main query result
        all_mock = MagicMock()
        all_mock.all.return_value = [mock_tool]

        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value = all_mock

        mock_db_session.execute = AsyncMock(return_value=scalars_mock)

        # Mock the validation
        with patch('src.services.tools.schema.ToolResponse.model_validate', return_value=ToolResponse(
            id=mock_tool.id,
            name=mock_tool.name,
            description=mock_tool.description,
            organization_id=mock_tool.organization_id,
            user_id=mock_tool.user_id,
            category_id=mock_tool.category_id,
            logo_url=mock_tool.logo_url,
            is_active=mock_tool.is_active,
            version=mock_tool.version,
            is_public=mock_tool.is_public,
            is_verified=mock_tool.is_verified,
            inputs=mock_tool.inputs,
            outputs=mock_tool.outputs,
            configuration=mock_tool.configuration,
            settings=mock_tool.settings,
            created_at=mock_tool.created_at,
            updated_at=mock_tool.updated_at
        )):
            response = await tools_service.get_list(
                org_id=test_organization,
                offset=0,
                limit=10,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, PaginatedToolResponse)
        assert len(response.tools) == 1
        assert response.total == 1
        assert response.has_more is False

    async def test_post_create_category_success(self, tools_service, mock_db_session, test_user, test_organization, sample_category_create_data):
        """Test creating a new category successfully."""
        # Mock category existence check - return None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Mock the model
        with patch('src.models.ToolCategoryModel') as mock_model_class:
            db_category = create_model_mock(
                id=uuid4(),
                name=sample_category_create_data.name,
                description=sample_category_create_data.description
            )
            mock_model_class.return_value = db_category

            # Mock the validation
            with patch('src.services.tools.schema.ToolCategoryResponse.model_validate', return_value=ToolCategoryResponse(
                id=uuid4(),
                name=sample_category_create_data.name,
                description=sample_category_create_data.description
            )):
                response = await tools_service.post_create_category(
                    org_id=test_organization,
                    category_data=sample_category_create_data,
                    session=mock_db_session,
                    user=test_user
                )

        # Assertions
        assert isinstance(response, ToolCategoryResponse)
        assert mock_db_session.add.called
        assert mock_db_session.commit.called

    async def test_put_update_category_success(self, tools_service, mock_db_session, test_user, test_organization, mock_category):
        """Test updating a category successfully."""
        # Setup mock session to return a category
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_category

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create update data
        update_data = ToolCategoryUpdate(
            name="Updated Category",
            description="Updated Description"
        )

        # Mock the validation
        with patch('src.services.tools.schema.ToolCategoryResponse.model_validate', return_value=ToolCategoryResponse(
            id=mock_category.id,
            name="Updated Category",
            description="Updated Description"
        )):
            response = await tools_service.put_update_category(
                org_id=test_organization,
                category_id=mock_category.id,
                category_data=update_data,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, ToolCategoryResponse)
        assert mock_db_session.commit.called
        assert mock_category.name == "Updated Category"
        assert mock_category.description == "Updated Description"

    async def test_put_update_category_not_found(self, tools_service, mock_db_session, test_user, test_organization):
        """Test updating a non-existent category."""
        # Setup mock session to not find the category
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create update data
        update_data = ToolCategoryUpdate(
            name="Updated Category",
            description="Updated Description"
        )

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.put_update_category(
                org_id=test_organization,
                category_id=uuid4(),
                category_data=update_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 404
        assert "Category not found" in excinfo.value.detail

    async def test_delete_category_success(self, tools_service, mock_db_session, test_user, mock_category):
        """Test deleting a category successfully."""
        # Setup mock for category existence check
        category_result = MagicMock()
        category_result.scalar_one_or_none.return_value = mock_category

        # Setup mock for tools existence check - no tools
        tools_result = MagicMock()
        tools_result.scalar_one_or_none.return_value = None

        # Configure mock_db_session to return different results
        mock_execute = AsyncMock()
        mock_execute.side_effect = [category_result, tools_result]
        mock_db_session.execute = mock_execute

        response = await tools_service.delete_delete_category(
            category_id=mock_category.id,
            session=mock_db_session,
            user=test_user
        )

        # Assertions
        assert isinstance(response, dict)
        assert response["message"] == "Category deleted successfully"
        assert mock_db_session.delete.called
        assert mock_db_session.commit.called

    async def test_delete_category_not_found(self, tools_service, mock_db_session, test_user):
        """Test deleting a non-existent category."""
        # Setup mock session to not find the category
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.delete_delete_category(
                category_id=uuid4(),
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 404
        assert "Category not found" in excinfo.value.detail

    async def test_get_list_categories(self, tools_service, mock_db_session, test_user, mock_category):
        """Test listing categories."""
        # Setup mock session for categories
        all_mock = MagicMock()
        all_mock.all.return_value = [mock_category]

        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value = all_mock

        mock_db_session.execute = AsyncMock(return_value=scalars_mock)

        # Mock the validation
        with patch('src.services.tools.schema.ToolCategoryResponse.model_validate', return_value=ToolCategoryResponse(
            id=mock_category.id,
            name=mock_category.name,
            description=mock_category.description
        )):
            response = await tools_service.get_list_categories(
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, list)
        assert len(response) == 1
        assert isinstance(response[0], ToolCategoryResponse)

    async def test_create_tool_with_invalid_generator(
        self,
        tools_service,
        mock_db_session,
        test_user,
        test_organization,
        sample_tool_create_data,
        mock_category
    ):
        """Test creating a tool when the generator fails."""
        # Mock the queries directly in the right order
        mock_execute = AsyncMock()

        # First check (if tool exists) - return None
        first_result = MagicMock()
        first_result.scalar_one_or_none.return_value = None

        # Second check (if category exists) - return category
        second_result = MagicMock()
        second_result.scalar_one_or_none.return_value = mock_category

        # Set up side effect sequence
        mock_execute.side_effect = [first_result, second_result]
        mock_db_session.execute = mock_execute

        # Make the generator raise an exception
        tools_service.generator.generate_toolkit_from_json = AsyncMock(side_effect=Exception("Generator error"))

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.post(
                org_id=test_organization,
                tool_data=sample_tool_create_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 400
        assert "Error generating tool" in excinfo.value.detail


# ============================================================================
# INTEGRATION TESTS - RUN WITH: INTEGRATION_TEST=1 pytest tests/services/test_tools.py
# ============================================================================

import os

# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.

    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")

# Only import these modules for integration tests
if is_integration_test():
    from sqlalchemy import select, text
    from models import ToolModel, ToolCategoryModel
    import pytest_asyncio


@pytest.fixture
def test_integration_user():
    """Create a test user for integration tests."""
    user_id = uuid4()
    org_id = uuid4()
    return {
        "id": user_id,
        "email": f"test-integration-{user_id}@example.com",
        "first_name": "Test",
        "last_name": "Integration",
        "organization_id": org_id,
        "is_admin": True
    }


@pytest.fixture
def test_integration_org():
    """Create a test organization ID for integration tests."""
    return uuid4()


@pytest_asyncio.fixture
async def setup_test_db_integration(db_session):
    """Setup the test database for tools integration tests."""
    # Skip if not running integration tests
    if not is_integration_test():
        pytest.skip("Integration tests are skipped. Set INTEGRATION_TEST=1 to run them.")

    async for session in db_session:
        try:
            # Create necessary database tables if they don't exist
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS tool_categories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))

            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS tools (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    organization_id UUID NOT NULL,
                    user_id UUID NOT NULL,
                    category_id UUID NOT NULL REFERENCES tool_categories(id),
                    logo_url VARCHAR(255),
                    is_active BOOLEAN DEFAULT TRUE,
                    version VARCHAR(50) NOT NULL,
                    is_public BOOLEAN DEFAULT FALSE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    inputs JSONB,
                    outputs JSONB,
                    configuration JSONB,
                    settings JSONB,
                    generated_code TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, version)
                )
            """))

            # Commit the table creation
            await session.commit()

            # Clean up any existing test data first
            await session.execute(text("DELETE FROM tools WHERE name LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM tool_categories WHERE name LIKE 'Test Integration%'"))
            await session.commit()

            # Create a test category for use in tests
            test_category_id = uuid4()
            await session.execute(
                text("""
                    INSERT INTO tool_categories (id, name, description)
                    VALUES (:id, 'Test Integration Category', 'Category for integration tests')
                """),
                {"id": str(test_category_id)}
            )
            await session.commit()

            yield test_category_id

            # Clean up after tests
            await session.execute(text("DELETE FROM tools WHERE name LIKE 'Test Integration%'"))
            await session.execute(text("DELETE FROM tool_categories WHERE name LIKE 'Test Integration%'"))
            await session.commit()

        except Exception as e:
            print(f"Error in setup: {e}")
            await session.rollback()
            raise
        finally:
            # Only process the first yielded session
            break


@pytest.mark.asyncio
class TestToolServiceIntegration:
    """Integration tests for Tool service with real database."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_create_category_integration(self, tools_service, db_session, test_integration_user):
        """Test creating a category with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                # Create a category
                category_data = ToolCategoryCreate(
                    name="Test Integration Category Creation",
                    description="Test integration category description"
                )

                # Execute
                response = await tools_service.post_create_category(
                    org_id=test_integration_user["organization_id"],
                    category_data=category_data,
                    session=session,
                    user=test_integration_user
                )

                # Assert
                assert response is not None
                assert response.name == category_data.name
                assert response.description == category_data.description
                assert response.id is not None

                # Verify in database
                query = select(ToolCategoryModel).where(ToolCategoryModel.name == category_data.name)
                result = await session.execute(query)
                db_category = result.scalar_one_or_none()
                assert db_category is not None
                assert db_category.name == category_data.name

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_create_tool_integration(self, tools_service, db_session, test_integration_user,
                                         setup_test_db_integration):
        """Test creating a tool with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                category_id = setup_test_db_integration

                # Create tool data
                tool_data = ToolCreate(
                    name="Test Integration Tool",
                    description="Test integration tool description",
                    category_id=category_id,
                    version="1.0.0",
                    is_public=False,
                    is_verified=False,
                    is_active=True,
                    logo_url="https://example.com/logo.png",
                    inputs=[
                        ToolParameter(
                            name="input1",
                            type="string",
                            description="Input 1",
                            required=True
                        )
                    ],
                    outputs=ToolOutput(
                        type="string",
                        description="Output"
                    ),
                    configuration=[
                        ToolParameter(
                            name="config1",
                            type="string",
                            description="Config 1",
                            required=True
                        )
                    ],
                    settings=ToolSettings(
                        function_info=ToolFunctionInfo(
                            name="test_function",
                            is_async=True,
                            description="Test Function",
                            code="async def test_function(): return 'test'"
                        ),
                        requirements=["requests"]
                    )
                )

                # Execute
                response = await tools_service.post(
                    org_id=test_integration_user["organization_id"],
                    tool_data=tool_data,
                    session=session,
                    user=test_integration_user
                )

                # Assert
                assert isinstance(response, JSONResponse)
                assert response.status_code == 201

                # Verify in database
                tool_id = UUID(json.loads(response.body)["id"])
                query = select(ToolModel).where(ToolModel.id == tool_id)
                result = await session.execute(query)
                db_tool = result.scalar_one_or_none()
                assert db_tool is not None
                assert db_tool.name == tool_data.name
                assert db_tool.version == tool_data.version
                assert db_tool.generated_code is not None

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_update_tool_integration(self, tools_service, db_session, test_integration_user,
                                         setup_test_db_integration):
        """Test updating a tool with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                category_id = setup_test_db_integration

                # First create a tool
                tool_data = ToolCreate(
                    name="Test Integration Tool Update",
                    description="Original description",
                    category_id=category_id,
                    version="1.0.0",
                    is_public=False,
                    is_verified=False,
                    is_active=True,
                    logo_url="https://example.com/logo.png",
                    inputs=[
                        ToolParameter(
                            name="input1",
                            type="string",
                            description="Input 1",
                            required=True
                        )
                    ],
                    outputs=ToolOutput(
                        type="string",
                        description="Output"
                    ),
                    configuration=[
                        ToolParameter(
                            name="config1",
                            type="string",
                            description="Config 1",
                            required=True
                        )
                    ],
                    settings=ToolSettings(
                        function_info=ToolFunctionInfo(
                            name="test_function",
                            is_async=True,
                            description="Test Function",
                            code="async def test_function(): return 'test'"
                        ),
                        requirements=["requests"]
                    )
                )

                create_response = await tools_service.post(
                    org_id=test_integration_user["organization_id"],
                    tool_data=tool_data,
                    session=session,
                    user=test_integration_user
                )

                tool_id = UUID(json.loads(create_response.body)["id"])

                # Now update the tool
                update_data = ToolUpdate(
                    name="Test Integration Tool Updated",
                    description="Updated description",
                    is_active=False
                )

                # Execute update
                update_response = await tools_service.put(
                    org_id=test_integration_user["organization_id"],
                    tool_id=tool_id,
                    tool_data=update_data,
                    session=session,
                    user=test_integration_user
                )

                # Assert
                assert isinstance(update_response, JSONResponse)
                assert update_response.status_code == 200

                # Verify in database
                query = select(ToolModel).where(ToolModel.id == tool_id)
                result = await session.execute(query)
                db_tool = result.scalar_one_or_none()
                assert db_tool is not None
                assert db_tool.name == update_data.name
                assert db_tool.description == update_data.description
                assert db_tool.is_active == update_data.is_active

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break

    async def test_list_tools_integration(self, tools_service, db_session, test_integration_user,
                                        setup_test_db_integration):
        """Test listing tools with integration database."""
        # Get the actual session from the generator
        async for session in db_session:
            try:
                category_id = setup_test_db_integration

                # Create multiple tools
                tool_names = [
                    "Test Integration Tool List 1",
                    "Test Integration Tool List 2",
                    "Test Integration Tool List 3"
                ]

                for name in tool_names:
                    tool_data = ToolCreate(
                        name=name,
                        description=f"Description for {name}",
                        category_id=category_id,
                        version="1.0.0",
                        is_public=False,
                        is_verified=False,
                        is_active=True,
                        logo_url="https://example.com/logo.png",
                        inputs=[
                            ToolParameter(
                                name="input1",
                                type="string",
                                description="Input 1",
                                required=True
                            )
                        ],
                        outputs=ToolOutput(
                            type="string",
                            description="Output"
                        ),
                        configuration=[
                            ToolParameter(
                                name="config1",
                                type="string",
                                description="Config 1",
                                required=True
                            )
                        ],
                        settings=ToolSettings(
                            function_info=ToolFunctionInfo(
                                name="test_function",
                                is_async=True,
                                description="Test Function",
                                code="async def test_function(): return 'test'"
                            ),
                            requirements=["requests"]
                        )
                    )

                    await tools_service.post(
                        org_id=test_integration_user["organization_id"],
                        tool_data=tool_data,
                        session=session,
                        user=test_integration_user
                    )

                # Execute list query
                response = await tools_service.get_list(
                    org_id=test_integration_user["organization_id"],
                    offset=0,
                    limit=10,
                    session=session,
                    user=test_integration_user
                )

                # Assert
                assert isinstance(response, PaginatedToolResponse)
                assert response.total >= len(tool_names)

                # Verify all created tools are in the response
                tool_names_in_response = [tool.name for tool in response.tools]
                for name in tool_names:
                    assert name in tool_names_in_response

            except Exception:
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session
                break


@pytest.mark.asyncio
class TestToolServiceEdgeCases:
    """Test edge cases for the Tool service."""

    async def test_update_nonexistent_category(self, tools_service, mock_db_session, test_user, test_organization):
        """Test updating a category that doesn't exist."""
        # Create update data
        update_data = ToolCategoryUpdate(
            name="Updated Category",
            description="Updated Description"
        )

        # Set up mock to return None (category not found)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Test and verify exception
        with pytest.raises(HTTPException) as excinfo:
            await tools_service.put_update_category(
                org_id=test_organization,
                category_id=uuid4(),
                category_data=update_data,
                session=mock_db_session,
                user=test_user
            )

        assert excinfo.value.status_code == 404
        assert "Category not found" in excinfo.value.detail

    async def test_list_tools_with_category_filter(self, tools_service, mock_db_session, test_user, test_organization, mock_tool):
        """Test listing tools with category filter."""
        # Mock the count query
        mock_db_session.scalar = AsyncMock(return_value=1)

        # Mock the main query result
        all_mock = MagicMock()
        all_mock.all.return_value = [mock_tool]

        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value = all_mock

        mock_db_session.execute = AsyncMock(return_value=scalars_mock)

        # Mock the validation
        with patch('src.services.tools.schema.ToolResponse.model_validate', return_value=ToolResponse(
            id=mock_tool.id,
            name=mock_tool.name,
            description=mock_tool.description,
            organization_id=mock_tool.organization_id,
            user_id=mock_tool.user_id,
            category_id=mock_tool.category_id,
            logo_url=mock_tool.logo_url,
            is_active=mock_tool.is_active,
            version=mock_tool.version,
            is_public=mock_tool.is_public,
            is_verified=mock_tool.is_verified,
            inputs=mock_tool.inputs,
            outputs=mock_tool.outputs,
            configuration=mock_tool.configuration,
            settings=mock_tool.settings,
            created_at=mock_tool.created_at,
            updated_at=mock_tool.updated_at
        )):
            # Execute with category filter
            category_id = uuid4()
            response = await tools_service.get_list(
                org_id=test_organization,
                offset=0,
                limit=10,
                category_id=category_id,
                session=mock_db_session,
                user=test_user
            )

        # Assertions
        assert isinstance(response, PaginatedToolResponse)
        assert len(response.tools) == 1
        assert response.total == 1
        assert response.has_more is False
        # Verify the mock was called with category_id filter
        assert mock_db_session.execute.called


@pytest.mark.asyncio
class TestToolServicePerformance:
    """Performance tests for the Tool service."""

    # Skip if not in integration test mode
    pytestmark = pytest.mark.skipif(
        not is_integration_test(),
        reason="Integration tests are skipped. Set INTEGRATION_TEST=1 to run them."
    )

    async def test_bulk_category_creation(self, tools_service, db_session, test_integration_user):
        """Test creating multiple categories in bulk."""
        # Create 5 categories (smaller number for faster testing)
        categories_to_create = 5
        tasks = []

        # Get the actual session from the generator
        async for session in db_session:
            try:
                # First clean any existing test data
                await session.execute(text("DELETE FROM tool_categories WHERE name LIKE 'Bulk Test%'"))
                await session.commit()

                for i in range(categories_to_create):
                    category_data = ToolCategoryCreate(
                        name=f"Bulk Test Category {i}",
                        description=f"Bulk test description {i}"
                    )
                    tasks.append(
                        tools_service.post_create_category(
                            org_id=test_integration_user["organization_id"],
                            category_data=category_data,
                            session=session,
                            user=test_integration_user
                        )
                    )

                # Execute concurrently
                import asyncio
                import time

                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                # Analyze results
                successful_creations = [r for r in results if not isinstance(r, Exception)]
                [r for r in results if isinstance(r, Exception)]

                # Assert
                assert len(successful_creations) > 0
                print(f"Bulk creation: {len(successful_creations)} categories created in {end_time - start_time:.2f} seconds")

                # Verify in the database
                query = select(ToolCategoryModel).where(ToolCategoryModel.name.like("Bulk Test%"))
                result = await session.execute(query)
                db_categories = list(result.scalars().all())
                assert len(db_categories) == len(successful_creations)

                # Clean up
                await session.execute(text("DELETE FROM tool_categories WHERE name LIKE 'Bulk Test%'"))
                await session.commit()
            except Exception as e:
                print(f"Error in bulk category creation test: {e}")
                await session.rollback()
                raise
            finally:
                # Only process the first yielded session and ensure it's closed properly
                break
