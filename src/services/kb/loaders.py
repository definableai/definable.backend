from pathlib import Path
from typing import Any, AsyncIterator, List
from uuid import UUID, uuid4
import tempfile

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

from .schema import AllowedFileExtension


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
    file_ext = self.document.file_type.lower()

    # Map file extensions to Docling input formats
    format_mapping = {
      AllowedFileExtension.PDF: InputFormat.PDF,
      # Office formats
      AllowedFileExtension.DOCX: InputFormat.DOCX,
      AllowedFileExtension.XLSX: InputFormat.XLSX,
      AllowedFileExtension.PPTX: InputFormat.PPTX,
      # Web formats
      AllowedFileExtension.HTML: InputFormat.HTML,
      AllowedFileExtension.HTM: InputFormat.HTML,
      # Image formats
      AllowedFileExtension.JPG: InputFormat.IMAGE,
      AllowedFileExtension.JPEG: InputFormat.IMAGE,
      AllowedFileExtension.PNG: InputFormat.IMAGE,
      AllowedFileExtension.TIFF: InputFormat.IMAGE,
      AllowedFileExtension.BMP: InputFormat.IMAGE,
      # Text formats
      AllowedFileExtension.MD: InputFormat.MD,
      AllowedFileExtension.ASCIIDOC: InputFormat.ASCIIDOC,
      AllowedFileExtension.ADOC: InputFormat.ASCIIDOC,
      # Data formats
      AllowedFileExtension.CSV: InputFormat.CSV,
      # XML formats
      AllowedFileExtension.XML: InputFormat.XML_JATS,  # Default to JATS for generic XML
      AllowedFileExtension.NXML: InputFormat.XML_JATS,  # NXML is typically JATS
      AllowedFileExtension.USPTO: InputFormat.XML_USPTO,  # USPTO patent XML
    }

    return format_mapping.get(file_ext, InputFormat.MD)

  async def load(self) -> AsyncIterator[LangChainDocument]:
    """Load document using Docling."""
    try:
      # Download file from S3 to temp location asynchronously
      file_content = await s3_client.download_file(self.document.source_metadata["s3_key"])
      temp_path = str(Path(tempfile.gettempdir()) / str(self.document.id))

      # Write file asynchronously using asyncio
      import aiofiles  # type: ignore

      async with aiofiles.open(temp_path, "wb") as f:
        await f.write(file_content.read())

      self.file_path = temp_path

      try:
        # Convert the document
        conversion_result = self.converter.convert(self.file_path)
        doc = conversion_result.document

        # Chunk the document using HybridChunker
        chunks = self.chunker.chunk(doc)

        # Convert chunks to LangChain documents
        for idx, chunk in enumerate(chunks, start=1):
          # Extract metadata from the chunk
          metadata: dict[str, Any] = {
            "id": str(uuid4()),
            "kb_id": str(self.kb_id),
            "doc_id": str(self.document.id),
            "title": self.document.title,
            "file_type": self.document.source_metadata["file_type"],
            "original_filename": self.document.source_metadata["original_filename"],
            "tokens": len(chunk.text),
            "chunk_id": idx,
          }

          # Add page number and bounding box if available in the chunk's attributes
          if hasattr(chunk, "page_number"):
            metadata["page_number"] = chunk.page_number
          if hasattr(chunk, "bounding_box"):
            metadata["bounding_box"] = chunk.bounding_box

          # Create LangChain document
          lc_doc = LangChainDocument(page_content=chunk.text, metadata=metadata)
          yield lc_doc
      except Exception as e:
        print(f"Error processing {self.document.source_metadata['file_type']} file: {str(e)}")
        return

    finally:
      # Cleanup temp file
      if self.file_path and Path(self.file_path).exists():
        Path(self.file_path).unlink()


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
