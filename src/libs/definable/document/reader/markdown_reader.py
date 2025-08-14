import asyncio
import uuid
from pathlib import Path
from typing import IO, Any, List, Optional, Union

from libs.definable.document.base import Document
from libs.definable.document.chunking.markdown import MarkdownChunking
from libs.definable.document.chunking.strategy import ChunkingStrategy
from libs.definable.document.reader.base import Reader


class MarkdownReader(Reader):
  """Reader for Markdown files"""

  def __init__(
    self,
    chunk: bool = True,
    chunk_size: int = 5000,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    **kwargs,  # Add this to accept additional parameters
  ) -> None:
    # Use MarkdownChunking as default if no chunking_strategy is provided
    if chunking_strategy is None:
      chunking_strategy = MarkdownChunking()

    super().__init__(chunk=chunk, chunk_size=chunk_size, chunking_strategy=chunking_strategy, **kwargs)

  def read(self, file: Union[Path, IO[Any]]) -> List[Document]:
    try:
      if isinstance(file, Path):
        if not file.exists():
          raise FileNotFoundError(f"Could not find file: {file}")
        file_name = file.stem
        file_contents = file.read_text("utf-8")
      else:
        file_name = file.name.split(".")[0]
        file.seek(0)
        file_contents = file.read().decode("utf-8")

      documents = [Document(name=file_name, id=str({uuid.uuid4()}), content=file_contents)]
      if self.chunk:
        chunked_documents = []
        for document in documents:
          chunked_documents.extend(self.chunk_document(document))
        return chunked_documents
      return documents
    except Exception:
      return []

  async def async_read(self, file: Union[Path, IO[Any]]) -> List[Document]:
    try:
      if isinstance(file, Path):
        if not file.exists():
          raise FileNotFoundError(f"Could not find file: {file}")

        file_name = file.stem

        try:
          import aiofiles

          async with aiofiles.open(file, "r", encoding="utf-8") as f:
            file_contents = await f.read()
        except ImportError:
          file_contents = file.read_text("utf-8")
      else:
        file_name = file.name.split(".")[0]
        file.seek(0)
        file_contents = file.read().decode("utf-8")

      document = Document(
        name=file_name,
        id=str(uuid.uuid4()),  # Fixed an issue with the id creation
        content=file_contents,
      )

      if self.chunk:
        return await self._async_chunk_document(document)
      return [document]
    except Exception:
      return []

  async def _async_chunk_document(self, document: Document) -> List[Document]:
    if not self.chunk or not document:
      return [document]

    async def process_chunk(chunk_doc: Document) -> Document:
      return chunk_doc

    chunked_documents = self.chunk_document(document)

    if not chunked_documents:
      return [document]

    tasks = [process_chunk(chunk_doc) for chunk_doc in chunked_documents]
    return await asyncio.gather(*tasks)
