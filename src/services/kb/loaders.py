from pathlib import Path
from typing import Any, AsyncIterator, List
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
from services.kb.source_handlers.file import FileReaderFactory


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
      # Office formats
      "docx": InputFormat.DOCX,
      "xlsx": InputFormat.XLSX,
      "pptx": InputFormat.PPTX,
      # Web formats
      "html": InputFormat.HTML,
      "htm": InputFormat.HTML,
      # Image formats
      "jpg": InputFormat.IMAGE,
      "jpeg": InputFormat.IMAGE,
      "png": InputFormat.IMAGE,
      "tiff": InputFormat.IMAGE,
      "bmp": InputFormat.IMAGE,
      # Text formats
      "md": InputFormat.MD,
      "markdown": InputFormat.MD,
      "asciidoc": InputFormat.ASCIIDOC,
      "adoc": InputFormat.ASCIIDOC,
      # Data formats
      "csv": InputFormat.CSV,
      # XML formats
      "xml": InputFormat.XML_JATS,  # Default to JATS for generic XML
      "nxml": InputFormat.XML_JATS,  # NXML is typically JATS
      "uspto": InputFormat.XML_USPTO,  # USPTO patent XML
    }

    return format_mapping.get(file_ext, InputFormat.MD)

  def _is_docling_supported(self) -> bool:
    """Check if the file type is supported by Docling."""
    file_ext = self.document.source_metadata.get("file_type", "").lower()
    print(f"File extension: {file_ext}")

    # Remove markdown from Docling support since it fails in practice
    docling_supported = {
      "pdf",
      "docx",
      "xlsx",
      "pptx",
      "html",
      "htm",
      # "md",        # Remove this
      # "markdown",  # Remove this
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
    }

    # Special handling for text-based formats that should use custom readers
    text_formats = {"md", "markdown", "txt", "text", "json"}
    if file_ext in text_formats:
      print(f"Using custom reader for text format: {file_ext}")
      return False

    return file_ext in docling_supported

  async def load(self) -> AsyncIterator[LangChainDocument]:
    """Load document using Docling or fallback to custom readers."""
    try:
      # Download file from S3 to temp location
      file_content = await s3_client.download_file(self.document.source_metadata["s3_key"])
      temp_path = f"/tmp/{self.document.id}"
      print(f"Downloading file from S3 to {temp_path}")

      with open(temp_path, "wb") as f:
        f.write(file_content.read())
      print(f"File downloaded to {temp_path}")
      self.file_path = temp_path

      document_yielded = False

      if self._is_docling_supported():
        try:
          # Use Docling for supported formats
          print("Using Docling for supported formats")
          async for doc in self._load_with_docling():
            yield doc
            document_yielded = True
        except Exception as e:
          print(f"Docling failed, falling back to custom reader: {str(e)}")
          # Fallback to custom reader
          async for doc in self._load_with_custom_reader():
            yield doc
            document_yielded = True
      else:
        # Use custom readers for unsupported formats
        print("Using custom readers for unsupported formats")
        async for doc in self._load_with_custom_reader():
          yield doc
          document_yielded = True

      if not document_yielded:
        print("No documents were generated!")

    finally:
      # Cleanup temp file
      if self.file_path and Path(self.file_path).exists():
        Path(self.file_path).unlink()

  async def _load_with_docling(self) -> AsyncIterator[LangChainDocument]:
    """Load document using Docling."""
    try:
      conversion_result = self.converter.convert(self.file_path)
      doc = conversion_result.document

      # Check if conversion actually succeeded
      if not doc or not hasattr(doc, "text") or not doc.text.strip():
        print("Docling conversion produced empty result, falling back to custom reader")
        raise ValueError("Docling conversion failed - empty document")

      # Chunk the document using HybridChunker
      chunks = list(self.chunker.chunk(doc))

      if not chunks:
        print("No chunks generated, falling back to custom reader")
        raise ValueError("Docling chunking failed - no chunks generated")

      print(f"Generated {len(chunks)} chunks")

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

        print(f"Metadata: {metadata}")

        # Add page number and bounding box if available in the chunk's attributes
        if hasattr(chunk, "page_number"):
          metadata["page_number"] = chunk.page_number
        if hasattr(chunk, "bounding_box"):
          metadata["bounding_box"] = chunk.bounding_box

        # Create LangChain document
        lc_doc = LangChainDocument(page_content=chunk.text, metadata=metadata)
        yield lc_doc
    except Exception as e:
      print(f"Error processing document with Docling: {str(e)}")
      raise

  async def _load_with_custom_reader(self) -> AsyncIterator[LangChainDocument]:
    """Load document using custom readers for unsupported formats."""
    file_type = self.document.source_metadata.get("file_type", "")

    # Get appropriate reader
    reader = FileReaderFactory.get_reader(file_type)

    # Read the document
    documents = await reader.async_read(Path(self.file_path))

    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=50,
      length_function=len,
    )

    # Process each document
    for doc_idx, doc in enumerate(documents, start=1):
      # Split content into chunks
      chunks = text_splitter.split_text(doc.content)

      # Convert chunks to LangChain documents
      for chunk_idx, chunk_text in enumerate(chunks, start=1):
        metadata: dict[str, Any] = {
          "id": str(uuid4()),
          "kb_id": str(self.kb_id),
          "doc_id": str(self.document.id),
          "title": self.document.title,
          "file_type": self.document.source_metadata["file_type"],
          "original_filename": self.document.source_metadata["original_filename"],
          "tokens": len(chunk_text),
          "chunk_id": f"{doc_idx}_{chunk_idx}",
        }

        # Add metadata from the original document if available
        if hasattr(doc, "meta_data") and doc.meta_data:
          metadata.update(doc.meta_data)

        # Create LangChain document
        lc_doc = LangChainDocument(page_content=chunk_text, metadata=metadata)
        yield lc_doc


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

      print(f"Processing {len(documents)} documents")
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
      if lc_documents:  # Only add if we have documents
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
