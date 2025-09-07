import tempfile
from pathlib import Path
from typing import AsyncIterator, List
from uuid import UUID, uuid4

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
  DocumentConverter,
  ExcelFormatOption,
  PdfFormatOption,
  WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from langchain_core.documents import Document as LangChainDocument
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import async_engine
from libs.s3.v1 import s3_client
from models import KBDocumentModel


class DoclingFileLoader:
  """File loader using Docling for document processing."""

  def __init__(self, kb_id: UUID, document: KBDocumentModel):
    self.kb_id = kb_id
    self.document = document
    self.file_path = ""

    # Initialize the document converter with format options
    self.converter = DocumentConverter(
      allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.XLSX,
        InputFormat.PPTX,
        InputFormat.HTML,
        InputFormat.MD,
        InputFormat.ASCIIDOC,
        InputFormat.CSV,
        InputFormat.XML_JATS,
        InputFormat.XML_USPTO,
      ],
      format_options={
        InputFormat.PDF: PdfFormatOption(
          pipeline_cls=StandardPdfPipeline,
        ),
        InputFormat.DOCX: WordFormatOption(
          pipeline_cls=SimplePipeline,
        ),
        InputFormat.XLSX: ExcelFormatOption(
          pipeline_cls=SimplePipeline,
        ),
      },
    )

    self.chunker = HybridChunker()

  def _get_input_format(self) -> InputFormat:
    """Determine the input format based on file extension."""
    file_ext = self.document.source_metadata.get("file_type", "").lower()

    # Map file extensions to Docling input formats
    format_mapping = {
      "pdf": InputFormat.PDF,
      "docx": InputFormat.DOCX,
      "xlsx": InputFormat.XLSX,
      "pptx": InputFormat.PPTX,
      "html": InputFormat.HTML,
      "htm": InputFormat.HTML,
      "md": InputFormat.MD,
      "asciidoc": InputFormat.ASCIIDOC,
      "adoc": InputFormat.ASCIIDOC,
      "csv": InputFormat.CSV,
      "xml": InputFormat.XML_JATS,
      "nxml": InputFormat.XML_JATS,
      "uspto": InputFormat.XML_USPTO,
      "jpg": InputFormat.IMAGE,
      "jpeg": InputFormat.IMAGE,
      "png": InputFormat.IMAGE,
      "tiff": InputFormat.IMAGE,
      "bmp": InputFormat.IMAGE,
    }

    return format_mapping.get(file_ext, InputFormat.MD)

  async def load(self) -> AsyncIterator[LangChainDocument]:
    """Load document using Docling or text fallback."""
    print(f"Loading document: {self.document.source_metadata.get('original_filename')}")

    try:
      # Download file from S3 to temp location asynchronously
      file_content = await s3_client.download_file(self.document.source_metadata["s3_key"])
      temp_path = str(Path(tempfile.gettempdir()) / str(self.document.id))

      # Write file asynchronously using asyncio
      import aiofiles  # type: ignore

      async with aiofiles.open(temp_path, "wb") as f:
        await f.write(file_content.read())

      self.file_path = temp_path
      file_ext = self.document.source_metadata.get("file_type", "").lower()

      # Try Docling for supported formats (including images)
      if file_ext in [
        "pdf",
        "docx",
        "xlsx",
        "pptx",
        "html",
        "htm",
        "md",
        "asciidoc",
        "adoc",
        "csv",
        "xml",
        "nxml",
        "uspto",
        "jpg",
        "jpeg",
        "png",
        "tiff",
        "bmp",
      ]:
        try:
          print(f"Using Docling for {file_ext}")
          async for doc in self._load_with_docling():
            yield doc
          return
        except Exception as e:
          print(f"Docling failed for {file_ext}: {str(e)}")

      # Fallback to text processing for txt, json, and failed Docling attempts
      print(f"Using text processing for {file_ext}")
      async for doc in self._load_with_text():
        yield doc

    finally:
      if self.file_path and Path(self.file_path).exists():
        Path(self.file_path).unlink()

  async def _load_with_docling(self) -> AsyncIterator[LangChainDocument]:
    """Load document using Docling."""
    conversion_result = self.converter.convert(self.file_path)
    doc = conversion_result.document
    chunks = self.chunker.chunk(doc)

    for idx, chunk in enumerate(chunks, start=1):
      metadata = {
        "id": str(uuid4()),
        "kb_id": str(self.kb_id),
        "doc_id": str(self.document.id),
        "title": self.document.title,
        "file_type": self.document.source_metadata["file_type"],
        "original_filename": self.document.source_metadata["original_filename"],
        "tokens": len(chunk.text),
        "chunk_id": idx,
      }

      yield LangChainDocument(page_content=chunk.text, metadata=metadata)

  async def _load_with_text(self) -> AsyncIterator[LangChainDocument]:
    """Load document using simple text processing."""
    with open(self.file_path, "r", encoding="utf-8") as f:
      content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=50,
      length_function=len,
    )

    chunks = text_splitter.split_text(content)

    for idx, chunk_text in enumerate(chunks, start=1):
      text_metadata = {
        "id": str(uuid4()),
        "kb_id": str(self.kb_id),
        "doc_id": str(self.document.id),
        "title": self.document.title,
        "file_type": self.document.source_metadata["file_type"],
        "original_filename": self.document.source_metadata["original_filename"],
        "tokens": len(chunk_text.split()),
        "chunk_id": idx,
      }

      yield LangChainDocument(page_content=chunk_text, metadata=text_metadata)


class DocumentProcessor:
  """Process documents for vector storage."""

  def __init__(self, embedding_model: str = "text-embedding-3-large", chunk_size: int = 500, chunk_overlap: int = 50):
    self.embedding_model = embedding_model
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

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

      lc_documents: List[LangChainDocument] = []
      for document in documents:
        # Check if document has s3_key (file document) or existing content (URL document)
        if document.source_metadata.get("s3_key"):
          # File document - use DoclingFileLoader to download from S3 and process
          loader = DoclingFileLoader(kb_id=kb_id, document=document)
          async for doc in loader.load():
            lc_documents.append(doc)
        elif document.content:
          # URL document - use existing content directly
          chunks = await self._chunk_text_content(document, kb_id)
          lc_documents.extend(chunks)
        else:
          raise Exception(f"Document {document.id} has no s3_key and no content available")

      # Store in vector DB
      await vectorstore.aadd_documents(lc_documents)

    except Exception as e:
      raise Exception(f"Error processing document {document.id}: {str(e)}")

  async def _chunk_text_content(self, document: KBDocumentModel, kb_id: UUID) -> List[LangChainDocument]:
    """Chunk text content for URL documents that don't have files."""
    # Create text splitter with the same settings as file processing
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=self.chunk_size,
      chunk_overlap=self.chunk_overlap,
      length_function=len,
    )

    # Split the content into chunks
    text_chunks = text_splitter.split_text(document.content)

    # Convert to LangChain documents with metadata
    lc_documents = []
    for idx, chunk_text in enumerate(text_chunks, start=1):
      metadata = {
        "id": str(uuid4()),
        "kb_id": str(kb_id),
        "doc_id": str(document.id),
        "title": document.title,
        "chunk_id": idx,
        "tokens": len(chunk_text.split()),
        "source_type": "url",
      }

      # Add URL-specific metadata if available
      if document.source_metadata.get("url"):
        metadata["source_url"] = document.source_metadata["url"]

      lc_doc = LangChainDocument(page_content=chunk_text, metadata=metadata)
      lc_documents.append(lc_doc)

    return lc_documents
