import os
import pytest
import asyncio
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Load test environment variables
env_test_path = os.path.join(os.path.dirname(__file__), ".env.test")
load_dotenv(dotenv_path=env_test_path)

# Import project modules after environment setup
from src.database import Base

# Create test database engine
TEST_DATABASE_URL = "postgresql+asyncpg://testuser:testpassword@localhost:5432/zyeta_test"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for pytest-asyncio."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_test_db():
    """Create test database schema."""
    # Drop test database if it exists, then create it
    async with test_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))

    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Clean up after tests complete
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(setup_test_db):
    """Create a test database session."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
def app():
    """Create a FastAPI application instance."""
    from src.app import app as application
    return application


@pytest.fixture
def client(app):
    """Create a TestClient instance."""
    return TestClient(app)


@pytest.fixture
def request_context():
    """Create a request context."""
    return {}


@pytest.fixture
def auth_headers():
    """Return headers with a valid test token."""
    return {"Authorization": "Bearer test_token_for_testing_only"}


@pytest.fixture
def generate_id():
    """Generate a random id for test resources."""
    import uuid
    return lambda: str(uuid.uuid4())[:8]
