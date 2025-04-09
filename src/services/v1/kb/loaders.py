from pathlib import Path
from typing import AsyncIterator, List, Optional
from uuid import UUID, uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
  CSVLoader,
  PyPDFLoader,
  TextLoader,
  UnstructuredEmailLoader,
  UnstructuredEPubLoader,
  UnstructuredExcelLoader,
  UnstructuredHTMLLoader,
  UnstructuredMarkdownLoader,
  UnstructuredPowerPointLoader,
  UnstructuredWordDocumentLoader,
  UnstructuredXMLLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from database import async_engine
from libs.s3.v1 import s3_client
from models import KBDocumentModel

from .schema import AllowedFileExtension


class FileLoader:
  """File loader factory for different file types."""

  def __init__(self, kb_id: UUID, document: KBDocumentModel):
    self.kb_id = kb_id
    self.document = document
    self.file_path = ""

  async def load(self) -> AsyncIterator[Document]:
    """Load document based on file type."""
    try:
      # Download file from S3 to temp location
      file_content = await s3_client.download_file(self.document.s3_key)
      temp_path = f"/tmp/{self.document.id}"

      with open(temp_path, "wb") as f:
        f.write(file_content.read())

      self.file_path = temp_path

      # Get appropriate loader
      loader = self._get_loader()

      # Load documents
      if loader:
        docs = loader.load()
        for doc in docs:
          # Enhance metadata
          doc.metadata.update({
            "id": str(uuid4()),
            "kb_id": str(self.kb_id),
            "doc_id": str(self.document.id),
            "title": self.document.title,
            "file_type": self.document.file_type,
            "original_filename": self.document.original_filename,
            "tokens": len(doc.page_content),
          })
          yield doc

    finally:
      # Cleanup temp file
      if self.file_path and Path(self.file_path).exists():
        Path(self.file_path).unlink()

  def _get_loader(self) -> Optional[BaseLoader]:
    """Get appropriate loader based on file type."""
    loaders = {
      AllowedFileExtension.PDF.value: lambda: PyPDFLoader(self.file_path),
      AllowedFileExtension.DOCX.value: lambda: UnstructuredWordDocumentLoader(self.file_path),
      AllowedFileExtension.HTML.value: lambda: UnstructuredHTMLLoader(self.file_path),
      AllowedFileExtension.HTM.value: lambda: UnstructuredHTMLLoader(self.file_path),
      AllowedFileExtension.MARKDOWN.value: lambda: UnstructuredMarkdownLoader(self.file_path),
      AllowedFileExtension.MDX.value: lambda: UnstructuredMarkdownLoader(self.file_path),
      AllowedFileExtension.XML.value: lambda: UnstructuredXMLLoader(self.file_path),
      AllowedFileExtension.EPUB.value: lambda: UnstructuredEPubLoader(self.file_path),
      AllowedFileExtension.CSV.value: lambda: CSVLoader(self.file_path),
      AllowedFileExtension.EML.value: lambda: UnstructuredEmailLoader(self.file_path),
      AllowedFileExtension.MSG.value: lambda: UnstructuredEmailLoader(self.file_path),
      AllowedFileExtension.PPTX.value: lambda: UnstructuredPowerPointLoader(self.file_path),
      AllowedFileExtension.PPT.value: lambda: UnstructuredPowerPointLoader(self.file_path),
      AllowedFileExtension.XLSX.value: lambda: UnstructuredExcelLoader(self.file_path),
      AllowedFileExtension.XLS.value: lambda: UnstructuredExcelLoader(self.file_path),
      AllowedFileExtension.TXT.value: lambda: TextLoader(self.file_path),
    }

    return loaders.get(self.document.file_type, lambda: TextLoader(self.file_path))()


class DocumentProcessor:
  """Process documents for vector storage."""

  def __init__(self, embedding_model: str = "text-embedding-3-large", chunk_size: int = 500, chunk_overlap: int = 50):
    self.embedding_model = embedding_model
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

  async def process_document(self, kb_id: UUID, documents: List[KBDocumentModel], collection_name: str) -> None:
    """Process a single document - load, chunk, and store in vector DB."""
    try:
      # Initialize vector store
      vectorstore = PGVector(
        embeddings=OpenAIEmbeddings(model=self.embedding_model),
        collection_name=collection_name,
        connection=async_engine,
        use_jsonb=True,
        create_extension=False,
      )

      lc_documents: List[Document] = []
      for document in documents:
        if document.content:
          lc_documents.append(Document(page_content=document.content, metadata={**document.source_metadata, "doc_id": str(document.id)}))

      # Chunk documents
      chunks = self.text_splitter.split_documents(lc_documents)

      # add index to metadata
      for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

      # Store in vector DB
      await vectorstore.aadd_documents(chunks)

    except Exception as e:
      raise Exception(f"Error processing document {document.id}: {str(e)}")
