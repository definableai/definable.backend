from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

from config.settings import settings

# Async engine and session (for FastAPI endpoints)
async_engine = create_async_engine(
  settings.database_url,
  echo=False,
  poolclass=AsyncAdaptedQueuePool,
  pool_size=1000,
  max_overflow=2000,
  pool_timeout=30,
  pool_pre_ping=True,
)

# Fixed sessionmaker configuration
async_session = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine and session (for Celery tasks)
sync_database_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
sync_engine = create_engine(
  sync_database_url,
  echo=False,
  poolclass=QueuePool,
  pool_size=100,
  max_overflow=200,
  pool_timeout=30,
  pool_pre_ping=True,
)

sync_session = sessionmaker(sync_engine, class_=Session, expire_on_commit=False)


async def get_db():
  async with async_session() as session:
    yield session
