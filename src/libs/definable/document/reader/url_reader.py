import asyncio
from time import sleep
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from libs.definable.document.base import Document
from libs.definable.document.reader.base import Reader


class URLReader(Reader):
  """Reader for general URL content"""

  def __init__(self, proxy: Optional[str] = None, **kwargs):
    super().__init__(**kwargs)
    self.proxy = proxy

  def read(self, url: str) -> List[Document]:
    if not url:
      raise ValueError("No url provided")

    # Retry the request up to 3 times with exponential backoff
    for attempt in range(3):
      try:
        response = httpx.get(url, proxy=self.proxy) if self.proxy else httpx.get(url)
        break
      except httpx.RequestError:
        if attempt == 2:  # Last attempt
          raise
        wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
        sleep(wait_time)

    try:
      response.raise_for_status()
    except httpx.HTTPStatusError:
      raise

    document = self._create_document(url, response.text)
    if self.chunk:
      return self.chunk_document(document)
    return [document]

  async def async_read(self, url: str) -> List[Document]:
    """Async version of read method"""
    if not url:
      raise ValueError("No url provided")

    client_args = {"proxy": self.proxy} if self.proxy else {}
    async with httpx.AsyncClient(**client_args) as client:  # type: ignore
      for attempt in range(3):
        try:
          response = await client.get(url)
          break
        except httpx.RequestError:
          if attempt == 2:  # Last attempt
            raise
          wait_time = 2**attempt
          await asyncio.sleep(wait_time)

      try:
        response.raise_for_status()
      except httpx.HTTPStatusError:
        raise

      document = self._create_document(url, response.text)
      if self.chunk:
        return await self.chunk_documents_async([document])
      return [document]

  def _create_document(self, url: str, content: str) -> Document:
    """Helper method to create a document from URL content"""
    parsed_url = urlparse(url)
    doc_name = parsed_url.path.strip("/").replace("/", "_").replace(" ", "_")
    if not doc_name:
      doc_name = parsed_url.netloc

    return Document(
      name=doc_name,
      id=doc_name,
      meta_data={"url": url},
      content=content,
    )
