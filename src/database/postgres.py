from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings

async_engine = create_async_engine(settings.database_url, echo=False)

# Fixed sessionmaker configuration
async_session = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
  async with async_session() as session:
    yield session
