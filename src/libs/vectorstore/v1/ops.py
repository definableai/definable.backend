from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text

from database import async_engine, async_session

"""
create docker container using :
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain\
    -p 6024:5432 -d pgvector/pgvector:pg16

Note: It doesn't required to create extension in the database.
"""


async def ensure_vector_extension() -> None:
  """Ensure the vector extension is installed in the database."""
  async with async_session() as session:
    try:
      # Check if vector extension exists
      result = await session.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
      if not result.fetchone():
        # Install vector extension
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()
        print("Vector extension installed successfully")
      else:
        print("Vector extension already exists")
    except Exception as e:
      await session.rollback()
      raise Exception(f"Failed to ensure vector extension: {e}")


async def create_vectorstore(org_name: str, collection_name: str) -> UUID:
  try:
    # Ensure vector extension is installed first
    await ensure_vector_extension()

    # Create vectorstore without trying to create extension
    vectorstore = PGVector(
      embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
      connection=async_engine,
      collection_name=f"{org_name}_{collection_name}",
      use_jsonb=True,
      create_extension=False,
      async_mode=True,
    )

    await vectorstore.acreate_collection()
    async with async_session() as session:
      collection = await vectorstore.aget_collection(session)

    return collection.uuid
  except Exception as e:
    raise Exception(f"libs.pg_vector.create.create_vectorstore: {e}")
