from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from config.settings import settings

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


async def get_db():
  async with async_session() as session:
    yield session
