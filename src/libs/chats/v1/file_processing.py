"""File processing utilities for chat using Agno readers and OCR."""

import tempfile
from pathlib import Path
import httpx
import os

from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.docx_reader import DocxReader

from common.logger import log as logger


class FileProcessor:
  """Process files for chat using Agno knowledge bases and OCR."""

  def __init__(self):
    self.temp_dir = Path(tempfile.gettempdir())

  async def extract_content(self, file_url: str, filename: str, content_type: str) -> str:
    """Extract text content from file using appropriate Agno reader or OCR."""
    temp_path = None
    try:
      # Download file
      temp_path = await self._download_file(file_url, filename)

      # Determine file type and extract content
      file_ext = filename.split(".")[-1].lower() if "." in filename else ""

      if content_type.startswith("image/") or file_ext in ["jpg", "jpeg", "png", "tiff", "bmp"]:
        return await self._extract_image_content(temp_path)
      elif file_ext == "pdf" or content_type == "application/pdf":
        return await self._extract_pdf_content(temp_path)
      elif file_ext == "csv" or content_type == "text/csv":
        return await self._extract_csv_content(temp_path)
      elif file_ext in ["docx", "doc"] or "wordprocessingml" in content_type:
        return await self._extract_docx_content(temp_path)
      elif file_ext in ["txt", "md"] or content_type.startswith("text/"):
        return await self._extract_text_content(temp_path)
      else:
        # Fallback to text extraction
        return await self._extract_text_content(temp_path)

    except Exception as e:
      logger.error(f"Error processing file {filename}: {str(e)}")
      return f"[Error processing file {filename}: {str(e)}]"
    finally:
      # Cleanup
      if temp_path and os.path.exists(temp_path):
        try:
          os.unlink(temp_path)
        except Exception as cleanup_error:
          logger.warning(f"Failed to cleanup temp file {temp_path}: {str(cleanup_error)}")

  async def _download_file(self, url: str, filename: str) -> str:
    """Download file from URL to temp location."""
    temp_path = self.temp_dir / f"chat_file_{hash(filename)}_{filename}"

    async with httpx.AsyncClient() as client:
      response = await client.get(url)
      response.raise_for_status()

      with open(temp_path, "wb") as f:
        f.write(response.content)

    return str(temp_path)

  async def _extract_pdf_content(self, file_path: str) -> str:
    """Extract content from PDF using Agno PDFReader."""
    try:
      reader = PDFReader()
      documents = reader.read(file_path)

      # Extract text from all documents
      content_parts = []
      for doc in documents:
        if hasattr(doc, "content"):
          content_parts.append(doc.content)
        elif hasattr(doc, "page_content"):
          content_parts.append(doc.page_content)
        else:
          content_parts.append(str(doc))

      return "\n".join(content_parts) if content_parts else ""
    except Exception as e:
      logger.error(f"Error extracting PDF content: {str(e)}")
      return f"[Error reading PDF: {str(e)}]"

  async def _extract_csv_content(self, file_path: str) -> str:
    """Extract content from CSV using Agno CSVReader."""
    try:
      reader = CSVReader()
      documents = reader.read(file_path)

      # Extract data from CSV
      content_parts = []
      for doc in documents:
        if hasattr(doc, "content"):
          content_parts.append(doc.content)
        elif hasattr(doc, "page_content"):
          content_parts.append(doc.page_content)
        else:
          content_parts.append(str(doc))

      return "\n".join(content_parts) if content_parts else ""
    except Exception as e:
      logger.error(f"Error extracting CSV content: {str(e)}")
      return f"[Error reading CSV: {str(e)}]"

  async def _extract_docx_content(self, file_path: str) -> str:
    """Extract content from DOCX using Agno DocxReader."""
    try:
      reader = DocxReader()
      documents = reader.read(file_path)

      # Extract text from document
      content_parts = []
      for doc in documents:
        if hasattr(doc, "content"):
          content_parts.append(doc.content)
        elif hasattr(doc, "page_content"):
          content_parts.append(doc.page_content)
        else:
          content_parts.append(str(doc))

      return "\n".join(content_parts) if content_parts else ""
    except Exception as e:
      logger.error(f"Error extracting DOCX content: {str(e)}")
      return f"[Error reading DOCX: {str(e)}]"

  async def _extract_text_content(self, file_path: str) -> str:
    """Extract content from text files."""
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    except Exception as e:
      logger.error(f"Error extracting text content: {str(e)}")
      return f"[Error reading text file: {str(e)}]"

  async def _extract_image_content(self, file_path: str) -> str:
    """Extract text from images using OCR."""
    try:
      # Try using Tesseract OCR
      import pytesseract
      from PIL import Image

      # Open and process image
      image = Image.open(file_path)

      # Extract text using OCR
      text = pytesseract.image_to_string(image, lang="eng")

      return text.strip() if text else "[No text found in image]"

    except ImportError:
      logger.warning("pytesseract not available, trying easyocr")
      try:
        import easyocr

        reader = easyocr.Reader(["en"])
        results = reader.readtext(file_path, detail=0)

        return "\n".join(results) if results else "[No text found in image]"

      except ImportError:
        logger.error("No OCR libraries available (pytesseract, easyocr)")
        return "[OCR not available - cannot read image content]"
    except Exception as e:
      logger.error(f"Error extracting image content: {str(e)}")
      return f"[Error reading image: {str(e)}]"


# Global instance
file_processor = FileProcessor()


async def extract_file_content(file_url: str, filename: str, content_type: str) -> str:
  """Extract text content from file - main entry point."""
  return await file_processor.extract_content(file_url, filename, content_type)
