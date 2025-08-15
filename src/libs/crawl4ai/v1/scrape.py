import asyncio
import json
import os
from typing import AsyncIterator
from uuid import uuid4

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig  # type: ignore
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import create_async_engine

os.environ["OPENAI_API_KEY"] = (
  "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"
)

config = CrawlerRunConfig(
  css_selector="div.theme-doc-markdown.markdown",
  word_count_threshold=10,
  excluded_tags=["header", "footer", "a", "img", "script", "style", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"],
  exclude_external_links=True,
  exclude_social_media_links=True,
  exclude_domains=["ads.com", "spammytrackers.net"],
  exclude_external_images=True,
  cache_mode=CacheMode.BYPASS,
)


class Crawl4AILoader(BaseLoader):
  def __init__(
    self,
    browser_config: BrowserConfig = None,
  ):
    """Initialize the Crawl4AI document loader.

    Args:
        browser_config: Optional BrowserConfig for the crawler
    """
    self.browser_config = browser_config
    self._crawler = AsyncWebCrawler(config=browser_config)

  async def alazy_load(self, **kwargs) -> AsyncIterator[Document]:
    url = kwargs.get("url")
    """Asynchronously load documents using crawl4ai."""
    await self._crawler.start()
    results = await self._crawler.arun(url, config=config)
    metadata = {"url": results.url, "id": str(uuid4())}
    content = results.markdown
    yield Document(page_content=content, metadata=metadata)
    await self._crawler.close()


loader = Crawl4AILoader()


async def process_single_url(url: str, vectorstore, text_splitter) -> None:
  """Process a single URL - crawl, chunk, and store in vector DB"""
  print(f"Processing URL: {url}")

  # Crawl

  async for doc in loader.alazy_load(url=url):
    # Chunk
    chunks = text_splitter.split_documents([doc])
    print(f"Created {len(chunks)} chunks")

    # Store in vector DB
    await vectorstore.aadd_documents(chunks)
    print(f"Stored chunks for {url}")


# Asynchronous usage
async def main():
  # create async engine
  engine = create_async_engine(
    "postgresql+asyncpg://postgres:AnandeshSharma9996944943@db.afjxxgitligynkbbpaqx.supabase.co:5432/postgres"
    # "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
  )

  # Initialize vectorstore asynchronously
  vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="langchain",
    connection=engine,
    use_jsonb=True,
    create_extension=False,
  )

  # Rest of your code
  # docs = [await Crawl4AILoader("https://www.google.com").aload()]

  urls = json.load(open("links.json"))
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
  for url in urls:
    try:
      await process_single_url(url, vectorstore, text_splitter)
    except Exception as e:
      print(f"Error processing URL {url}: {str(e)}")
      continue


if __name__ == "__main__":
  asyncio.run(main())
