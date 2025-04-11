import os
import sys
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load test environment variables
env_test_path = '.env.test'
load_dotenv(dotenv_path=env_test_path)

# Create module-wide mocks for database
mock_settings = MagicMock()
mock_settings.database_url = os.environ.get("DATABASE_URL")
mock_settings.jwt_secret = os.environ.get("JWT_SECRET")
mock_settings.jwt_expire_minutes = int(os.environ.get("JWT_EXPIRE_MINUTES", "30"))
mock_settings.app_name = os.environ.get("APP_NAME")
mock_settings.master_api_key = os.environ.get("MASTER_API_KEY")
mock_settings.environment = os.environ.get("ENVIRONMENT")
mock_settings.frontend_url = os.environ.get("FRONTEND_URL")

# Mock the database modules
mock_get_db = AsyncMock()
mock_async_session = AsyncMock()
mock_Base = MagicMock()
mock_CRUD = MagicMock()

# Add mocks for both import paths (database and src.database)
sys.modules['database'] = MagicMock()
sys.modules['database.postgres'] = MagicMock()
sys.modules['database.models'] = MagicMock()
sys.modules['src.database'] = MagicMock()
sys.modules['src.database.postgres'] = MagicMock()
sys.modules['src.database.models'] = MagicMock()

# Set up the necessary attributes and functions
sys.modules['database'].get_db = mock_get_db
sys.modules['src.database'].get_db = mock_get_db
sys.modules['database'].Base = mock_Base
sys.modules['src.database'].Base = mock_Base
sys.modules['database'].CRUD = mock_CRUD
sys.modules['src.database'].CRUD = mock_CRUD
sys.modules['database'].async_session = mock_async_session
sys.modules['src.database'].async_session = mock_async_session

# Patch config.settings
with patch.dict('sys.modules', {
    'config': MagicMock(),
    'config.settings': MagicMock(),
}):
    # Patch the settings module
    sys.modules['config.settings'].settings = mock_settings

@pytest.fixture
def app():
    """Create a FastAPI application instance."""
    return FastAPI()

@pytest.fixture
def client(app):
    """Create a TestClient instance."""
    return TestClient(app)

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    return session

@pytest.fixture
def auth_headers():
    """Return headers with a valid test token."""
    return {"Authorization": "Bearer test_token_for_testing_only"} 