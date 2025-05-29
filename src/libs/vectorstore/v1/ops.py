from uuid import UUID

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from database import async_engine, async_session


async def create_vectorstore(org_name: str, collection_name: str) -> UUID:
  try:
    vectorstore = PGVector(
      embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
      connection=async_engine,
      collection_name=f"{org_name}_{collection_name}",
      use_jsonb=True,
      create_extension=True,
    )
    await vectorstore.acreate_collection()
    async with async_session() as session:
      collection = await vectorstore.aget_collection(session)

    return collection.uuid
  except Exception as e:
    raise Exception(f"libs.pg_vector.create.create_vectorstore: {e}")
