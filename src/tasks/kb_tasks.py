"""Celery bridge with simple synchronous task functions."""

import platform
import sys
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select

from common.logger import logger
from common.q import task
from database import sync_session
from libs.firebase.v1 import get_rtdb
from models import JobModel, JobStatus

RTDB = get_rtdb()


def _safe_status_value(status_field):
  """Safely extract status value, handling both enum and int types."""
  if not status_field:
    return None
  if hasattr(status_field, "value"):
    return status_field.value
  return status_field  # Assume it's already an int/string


def get_file_details_for_websocket(doc_model) -> Dict[str, Any]:
  """Extract file details from document model for WebSocket updates (excluding content).

  Args:
      doc_model: KBDocumentModel instance

  Returns:
      Dict containing file metadata for WebSocket transmission
  """
  if not doc_model:
    return {}

  try:
    # Determine overall status - prioritize extraction status, fallback to indexing status
    status = _safe_status_value(doc_model.extraction_status) or _safe_status_value(doc_model.indexing_status)
    if status == 2:  # COMPLETED = 2
      status = "completed"
    elif status == 1:  # PROCESSING = 1
      status = "processing"
    elif status == 3:  # FAILED = 3
      status = "failed"
    elif status == 0:  # PENDING = 0
      status = "pending"
    else:
      status = "pending"  # Default fallback

    # Determine file_type from source metadata or source type
    file_type = ""
    if doc_model.source_metadata:
      # For URL-based documents, always use "webpage"
      if doc_model.source_metadata.get("operation") in ["scrape", "crawl", "map"] or doc_model.source_metadata.get("url"):
        file_type = "webpage"
      else:
        # First try to get file_type from metadata for non-URL documents
        file_type = doc_model.source_metadata.get("file_type", "")

        # If not found, determine based on content type or s3_key
        if not file_type:
          if doc_model.source_metadata.get("content_type"):
            content_type = doc_model.source_metadata.get("content_type", "")
            # Extract file type from content type (e.g., "application/pdf" -> "pdf")
            if "/" in content_type:
              file_type = content_type.split("/")[-1].lower()
          elif doc_model.source_metadata.get("s3_key"):
            # Extract from s3_key file extension
            s3_key = doc_model.source_metadata.get("s3_key", "")
            if "." in s3_key:
              file_type = s3_key.split(".")[-1].lower()

    # Use "markdown" as fallback if no file_type could be determined
    if not file_type:
      file_type = "markdown"

    # Extract URL for URL-based documents
    url = None
    if doc_model.source_metadata and file_type == "webpage":
      url = doc_model.source_metadata.get("url")

    return {
      "id": str(doc_model.id),
      "title": doc_model.title or "",
      "description": doc_model.description or "",
      "file_type": file_type,
      "size": len(doc_model.content) if doc_model.content else 0,
      "status": status,
      "created_at": doc_model.created_at.isoformat() if doc_model.created_at else None,
      "updated_at": doc_model.updated_at.isoformat() if doc_model.updated_at else None,
      "download_url": None,
      "url": url,  # Include URL for webpage documents
    }
  except Exception as e:
    logger.error(f"Failed to extract file details for WebSocket: {str(e)}")
    return {
      "id": str(doc_model.id) if doc_model and hasattr(doc_model, "id") else "unknown",
      "title": "",
      "description": "",
      "file_type": "",
      "size": 0,
      "status": "pending",
      "created_at": None,
      "updated_at": None,
      "download_url": None,
      "url": None,
    }


def send_job_update(
  org_id: UUID,
  job_id: UUID,
  status: str,
  message: str,
  progress: float = 0.0,
  kb_id: Optional[UUID] = None,
  file_id: Optional[UUID] = None,
  timeout: int = 10,
  url_details: Optional[Dict[str, Any]] = None,
  file_details: Optional[Dict[str, Any]] = None,
) -> bool:
  """Send job status update via Firebase to the kb_write path.

  Args:
      org_id: Organization UUID
      job_id: Job UUID
      status: Job status string ("pending" | "uploading" | "extracting" | "indexing" | "completed" | "failed")
      message: Status message
      progress: Progress percentage (0-100)
      kb_id: Optional Knowledge Base UUID
      file_id: Optional File/Document UUID
      timeout: Request timeout in seconds
      url_details: Optional URL processing details (discovered_urls, processed_urls, etc.)
      file_details: Optional file/document details (metadata, status, etc. - excluding content)

  Returns:
      bool: True if update was sent successfully, False otherwise
  """
  try:
    from datetime import datetime

    # Map string status to numeric status for backend
    status_mapping = {
      "pending": 0,
      "uploading": 1,
      "extracting": 1,
      "indexing": 1,
      "processing": 1,
      "uploaded": 2,  # Intermediate completion - upload finished
      "processed": 2,  # Intermediate completion - processing finished
      "step_completed": 2,  # Generic intermediate completion
      "completed": 2,
      "failed": 3,
    }
    numeric_status = status_mapping.get(status, 1)  # Default to PROCESSING if unknown

    # Prepare simplified payload for Firebase
    payload = {
      "org_id": str(org_id),
      "job_id": str(job_id),
      "status": numeric_status,  # Use correct numeric status
      "message": message,
      "data": {
        "jobId": str(job_id),
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "message": message,
        "progress": progress,
        "kbId": str(kb_id) if kb_id else None,
        "fileId": str(file_id) if file_id else None,
        **(url_details or {}),  # Include URL processing details if provided
        **({"file_details": file_details} if file_details else {}),  # Include file details if provided
      },
    }

    logger.info(f"Sending job update for {job_id}: status={status}, progress={progress}%")
    logger.debug(f"Update payload: {payload}")

    # Write to Firebase at org_id/kb_write/ path using sync operations
    firebase_path = f"{org_id}/kb_write/"

    # Use synchronous Firebase operations to avoid event loop issues
    try:
      ref = RTDB.child(firebase_path)
      ref.set(payload)

      logger.info(f"Job update written successfully to Firebase for {job_id}: {firebase_path}")
      return True
    except Exception as firebase_error:
      logger.error(f"Job update failed for {job_id}: Firebase error - {str(firebase_error)}")
      return False

  except Exception as e:
    logger.error(f"Unexpected error sending job update to Firebase for {job_id}: {str(e)}")
    return False


# NOTE: create_url_folder_sync function removed as folder creation is now handled by the API


def create_child_document_sync(parent_doc, page_url: str, content: str, folder_id, session, index: int):
  """Create child document for a crawled page (sync version)."""
  try:
    # Import here to avoid circular imports
    import copy
    from datetime import datetime

    # Generate child document title from URL
    from urllib.parse import urlparse

    from models import DocumentStatus, KBDocumentModel

    parsed_url = urlparse(page_url)
    path = parsed_url.path.strip("/") or "index"
    title = f"{path.replace('/', '_')}" if path != "index" else f"page_{index + 1}"

    # Create child metadata
    child_metadata = copy.deepcopy(parent_doc.source_metadata)
    child_metadata.update({
      "url": page_url,
      "is_parent": False,
      "parent_id": str(parent_doc.id),
      "crawl_index": index,
      "size": len(content.encode("utf-8")),
      "size_characters": len(content),
      "size_words": len(content.split()),
      "size_lines": len(content.splitlines()),
      "file_type": "webpage",  # Always set file_type as webpage for URL documents
    })

    # Create child document
    child_doc = KBDocumentModel(
      title=title,
      description=f"Crawled content from {page_url}",
      kb_id=parent_doc.kb_id,
      folder_id=folder_id,
      content=content,
      source_type_id=parent_doc.source_type_id,
      source_metadata=child_metadata,
      extraction_status=DocumentStatus.COMPLETED,
      indexing_status=DocumentStatus.PENDING,
      extraction_completed_at=datetime.now(),
    )

    session.add(child_doc)
    session.commit()
    session.refresh(child_doc)

    logger.info(f"Created child document: {child_doc.title} (ID: {child_doc.id})")
    return child_doc

  except Exception as e:
    logger.error(f"Failed to create child document for {page_url}: {str(e)}")
    return None


def update_job_progress(
  job_id: UUID,
  progress: float,
  message: str,
  session,
  metadata: Optional[dict] = None,
  celery_task_id: Optional[str] = None,
  org_id: Optional[UUID] = None,
  send_update: bool = True,
):
  """Update job progress in database and optionally send HTTP update for WebSocket broadcasting."""
  try:
    job = session.get(JobModel, job_id)
    if job:
      # Get previous progress and status to check if we should send update
      previous_progress = job.context.get("progress", 0.0) if job.context else 0.0
      previous_status = job.context.get("last_sent_status") if job.context else None

      # Update job with progress information
      job.message = message

      # Store progress and metadata in job context
      if job.context is None:
        job.context = {}

      job.context.update({"progress": progress, "last_update": str(time.time()), "metadata": metadata or {}})

      # Store Celery task ID for Flower monitoring
      if celery_task_id:
        job.context["celery_task_id"] = celery_task_id

      session.commit()
      logger.info(f"Updated job progress: {job_id} - {progress}% - {message}")

      # Only send HTTP update if explicitly requested and there's meaningful change
      if send_update and org_id:
        # Determine status string based on job status and metadata
        status_str = "pending"
        if job.status == JobStatus.PROCESSING:
          task_type = metadata.get("task_type", "") if metadata else ""
          if "upload" in task_type or "file" in task_type:
            status_str = "uploading"
          elif "extract" in task_type or "processing" in task_type or "document_processing" in task_type:
            status_str = "processing"
          elif "index" in task_type:
            status_str = "indexing"
          else:
            status_str = "processing"  # Default for processing
        elif job.status == JobStatus.COMPLETED:
          # Differentiate between intermediate completions and final completion
          task_type = metadata.get("task_type", "") if metadata else ""
          if "document_indexing" in task_type:
            status_str = "completed"  # Final job - indexing is the last step
          else:
            # Intermediate jobs (upload, processing) - use task-specific completion status
            if "file_upload" in task_type:
              status_str = "uploaded"
            elif "document_processing" in task_type:
              status_str = "processed"
            else:
              status_str = "step_completed"  # Generic intermediate completion
        elif job.status == JobStatus.FAILED:
          status_str = "failed"

        # Determine if we should send update based on status change or significant progress change
        status_changed = previous_status != status_str
        is_completion = progress >= 100.0
        is_start = progress == 0.0 and previous_progress == 0.0
        is_first_update = previous_status is None

        should_send_update = (
          status_changed  # Status actually changed
          or is_completion  # Always send completion
          or is_first_update  # Always send first update
          or (is_start and status_str in ["uploading", "processing", "indexing"])  # Starting a new phase
        )

        if should_send_update:
          # Store the status we're about to send to avoid duplicates
          job.context["last_sent_status"] = status_str
          session.commit()

          # Extract kb_id and file_id from metadata
          kb_id = None
          file_id = None

          if metadata:
            # Try to get kb_id from metadata
            if metadata.get("kb_id"):
              try:
                kb_id = UUID(metadata.get("kb_id"))
              except (ValueError, TypeError):
                kb_id = None

            # Try to get file_id from doc_id
            if metadata.get("doc_id"):
              try:
                file_id = UUID(metadata.get("doc_id"))
              except (ValueError, TypeError):
                file_id = None

          # Send progress update
          logger.info(f"Sending job update: {job_id} status={status_str} (was {previous_status}), progress={progress}%")
          send_job_update(
            org_id=org_id,
            job_id=job_id,
            status=status_str,
            message=message,
            progress=progress,
            kb_id=kb_id,
            file_id=file_id,
          )
        else:
          logger.debug(f"Skipping job update for {job_id}: status={status_str}, progress={progress}% (no significant change)")

  except Exception as e:
    logger.error(
      f"Failed to update job progress for job {job_id}: {str(e)}",
      extra={
        "job_id": str(job_id) if job_id else None,
        "org_id": str(org_id) if org_id else None,
        "error": str(e),
      },
    )


@task(bind=True)
def process_document_task(
  self,
  job_id: str,
  source_type_name: str,
  source_type_config: dict,
  doc_id: str,
  org_id: str,
  user_id: str,
  charge_id: Optional[str] = None,
  complete_charge: bool = True,
) -> dict:
  """Process document using simple synchronous operations."""
  logger.info(f"Starting document processing task for document ID: {doc_id}, job ID: {job_id}")

  with sync_session() as session:
    try:
      # Get job and check if parent job (upload) is completed
      job = session.get(JobModel, UUID(job_id))
      if job and job.parent_job_id:
        parent_job = session.get(JobModel, job.parent_job_id)
        if parent_job and parent_job.status != JobStatus.COMPLETED:
          logger.warning(f"Parent job {parent_job.id} not completed (status: {parent_job.status}), cannot proceed with processing")
          if job:
            job.status = JobStatus.FAILED
            job.message = "Upload not completed, cannot process document"
            session.commit()
          return {"success": False, "error": "Upload not completed"}

      # Update job status to processing
      if job:
        job.status = JobStatus.PROCESSING
        job.message = "Starting document processing"
        session.commit()

      # Update job progress with Celery task ID (send update since this is the start)
      update_job_progress(
        job_id=UUID(job_id),
        progress=0.0,
        message="Starting document processing",
        session=session,
        metadata={
          "doc_id": doc_id,
          "source_type": source_type_name,
          "task_type": "document_processing",
          "kb_id": job.context.get("kb_id") if job and job.context else None,
        },
        celery_task_id=self.request.id,
        org_id=UUID(org_id),
        send_update=True,
      )

      # Import models locally to avoid circular imports
      from models import DocumentStatus, KBDocumentModel, SourceTypeModel

      # Get document
      doc = session.get(KBDocumentModel, UUID(doc_id))
      if not doc:
        if job:
          job.status = JobStatus.FAILED
          job.message = f"Document {doc_id} not found"
          session.commit()
        return {"success": False, "error": f"Document {doc_id} not found"}

      # Update document status
      doc.extraction_status = DocumentStatus.PROCESSING
      session.commit()

      # Update job progress (send update for significant progress change)
      update_job_progress(
        job_id=UUID(job_id),
        progress=25.0,
        message="Processing document content",
        session=session,
        metadata={
          "doc_id": doc_id,
          "source_type": source_type_name,
          "task_type": "document_processing",
          "kb_id": job.context.get("kb_id") if job and job.context else None,
        },
        org_id=UUID(org_id),
        send_update=True,
      )

      # Get source type
      source_type = session.query(SourceTypeModel).filter_by(name=source_type_name).first()
      if not source_type:
        doc.extraction_status = DocumentStatus.FAILED
        doc.error_message = f"Source type {source_type_name} not found"
        session.commit()
        if job:
          job.status = JobStatus.FAILED
          job.message = f"Source type {source_type_name} not found"
          session.commit()
        return {"success": False, "error": f"Source type {source_type_name} not found"}

      # Process document based on source type
      if source_type_name == "file":
        # Extract content using existing DoclingFileLoader
        try:
          import asyncio

          from services.kb.loaders import DoclingFileLoader

          logger.info(f"Extracting content from file document: {doc.id}")

          # Create and run async content extraction in sync context
          async def extract_content():
            loader = DoclingFileLoader(kb_id=UUID("00000000-0000-0000-0000-000000000000"), document=doc)
            chunks = []
            async for chunk_doc in loader.load():
              chunks.append(chunk_doc.page_content)
            return "\n\n".join(chunks)

          loop = asyncio.new_event_loop()
          asyncio.set_event_loop(loop)
          try:
            content = loop.run_until_complete(extract_content())
            if not content:
              content = "No content could be extracted from this document"
            logger.info(f"Content extracted successfully, length: {len(content)} characters")
          finally:
            loop.close()

        except Exception as e:
          logger.error(f"Failed to extract content using DoclingFileLoader: {str(e)}")
          # Update document status to failed
          doc.extraction_status = DocumentStatus.FAILED
          doc.error_message = f"Content extraction failed: {str(e)}"
          session.commit()

          # Update job status to failed
          if job:
            job.status = JobStatus.FAILED
            job.message = f"Document processing error: Content extraction failed - {str(e)}"
            session.commit()

            # Update job progress to trigger failure status update
            update_job_progress(
              job_id=UUID(job_id),
              progress=0.0,
              message=f"Document processing error: Content extraction failed - {str(e)}",
              session=session,
              metadata={
                "doc_id": doc_id,
                "task_type": "document_processing",
                "kb_id": job.context.get("kb_id") if job and job.context else None,
                "error_type": "content_extraction_failed",
              },
              org_id=UUID(org_id),
              send_update=True,
            )

          # Return failure result to terminate the workflow
          return {"success": False, "error": f"Content extraction failed: {str(e)}"}
      elif source_type_name == "url":
        # Extract content using dedicated URL processing task
        try:
          # Get URL from document metadata
          url = doc.source_metadata.get("url")
          operation = doc.source_metadata.get("operation", "scrape")
          settings = doc.source_metadata.get("settings", {})

          logger.info(f"Extracting content from URL: {url} using operation: {operation}")

          # Use the dedicated URL processing task function directly
          import asyncio

          # Run async URL extraction in sync context
          loop = asyncio.new_event_loop()
          asyncio.set_event_loop(loop)
          try:
            # Call the URL processing function directly
            if operation == "scrape":
              result = loop.run_until_complete(_process_url_scrape(url, settings, org_id, job_id, doc_id, str(doc.kb_id)))
            elif operation == "crawl":
              result = loop.run_until_complete(_process_url_crawl(url, settings, org_id, job_id, doc_id, str(doc.kb_id), user_id))
            elif operation == "map":
              result = loop.run_until_complete(_process_url_map(url, settings, org_id, job_id, doc_id, str(doc.kb_id)))
            else:
              raise ValueError(f"Unsupported URL operation: {operation}")

            content = result.get("content", "")
            if not content or not content.strip():
              content = f"No content could be extracted from URL: {url}"
            logger.info(f"URL content extraction completed, length: {len(content)} characters")
          finally:
            loop.close()

        except Exception as e:
          logger.error(f"Failed to extract content from URL: {str(e)}")
          error_details = str(e)

          # Check if it's a specific API error that users should know about
          if any(code in error_details.lower() for code in ["502", "503", "504", "timeout"]):
            content = f"URL processing temporarily failed due to API issues. The service may be experiencing high load. Error: {error_details[:200]}"
          elif "both" in error_details.lower() and "failed" in error_details.lower():
            content = f"URL processing failed - both primary and fallback crawlers encountered issues. Error: {error_details[:200]}"
          else:
            content = f"URL content extraction failed: {error_details[:200]}"

          # If doc already has some content, append the error info instead of replacing
          if doc.content and len(doc.content.strip()) > 0:
            content = f"{doc.content}\n\n--- Processing Error ---\n{content}"
      else:
        content = doc.content or "Content extracted"

      # Update document with processed content
      from datetime import datetime

      doc.content = content
      doc.extraction_status = DocumentStatus.COMPLETED
      doc.extraction_completed_at = datetime.now()

      # Ensure URL documents have correct file_type in metadata
      if source_type_name == "url":
        if not doc.source_metadata:
          doc.source_metadata = {}
        doc.source_metadata["file_type"] = "webpage"

      session.commit()

      # Update job progress (send update for significant progress change)
      update_job_progress(
        job_id=UUID(job_id),
        progress=75.0,
        message="Document content extracted successfully",
        session=session,
        metadata={
          "doc_id": doc_id,
          "content_length": len(content),
          "task_type": "document_processing",
          "kb_id": job.context.get("kb_id") if job and job.context else None,
        },
        org_id=UUID(org_id),
        send_update=True,
      )

      # Handle charging if provided
      if charge_id and complete_charge:
        try:
          from utils.charge import Charge

          charge = Charge.from_transaction_id(charge_id, session=session)
          if charge:
            charge.calculate_and_update_sync()
        except Exception as e:
          logger.warning(f"Failed to complete charge {charge_id}: {e}")

      # Update job status to completed
      if job:
        job.status = JobStatus.COMPLETED
        job.message = "Document processing completed successfully"
        session.commit()

      # Update job progress (completion will trigger status update automatically)
      update_job_progress(
        job_id=UUID(job_id),
        progress=100.0,
        message="Document processing completed successfully",
        session=session,
        metadata={
          "doc_id": doc_id,
          "content_length": len(content),
          "task_type": "document_processing",
          "kb_id": job.context.get("kb_id") if job and job.context else None,
        },
        org_id=UUID(org_id),
        send_update=True,
      )

      # Create and trigger indexing job
      if job:
        kb_id = job.context.get("kb_id")
        if kb_id:
          if source_type_name == "url":
            # For URL documents, look for existing child indexing job (created in API)
            logger.info(f"Looking for existing child indexing job with parent_job_id: {job_id}")

            # Query for child jobs
            parent_job_uuid = UUID(job_id)
            child_jobs_query = select(JobModel).where(JobModel.parent_job_id == parent_job_uuid)
            child_jobs_result = session.execute(child_jobs_query)
            child_jobs = child_jobs_result.scalars().all()

            logger.info(f"Found {len(child_jobs)} child jobs")

            indexing_job_found = False
            for child_job in child_jobs:
              logger.info(f"Child job {child_job.id} context: {child_job.context}")
              if child_job.context and "doc_ids" in child_job.context:
                # This is an indexing job
                logger.info(f"Triggering existing child indexing job: {child_job.id}")
                try:
                  task_id = submit_index_documents_task(
                    job_id=child_job.id,
                    doc_ids=[UUID(doc_id) for doc_id in child_job.context["doc_ids"]],
                    org_id=UUID(org_id),
                    user_id=UUID(child_job.context.get("user_id", user_id)),
                    kb_id=UUID(child_job.context.get("kb_id", kb_id)),
                    initial_charge_id=child_job.context.get("initial_charge_id", charge_id),
                  )
                  logger.info(f"Successfully submitted index task {task_id} for existing child job {child_job.id}")
                  indexing_job_found = True
                  break
                except Exception as e:
                  logger.error(
                    f"Failed to submit index task for child job {child_job.id}: {str(e)}",
                    extra={
                      "child_job_id": str(child_job.id),
                      "job_id": str(job_id),
                      "org_id": str(org_id),
                      "doc_id": str(doc_id),
                      "error": str(e),
                    },
                  )

            if not indexing_job_found:
              logger.warning("No existing indexing job found for URL document, creating new one as fallback")
              # Fallback: create new indexing job if none found
              index_job = JobModel(
                name="Document Indexing",
                description="Index document after processing",
                status=JobStatus.PENDING,
                created_by=job.created_by,
                parent_job_id=job.id,
                context={
                  "doc_ids": [str(doc_id)],
                  "kb_id": kb_id,
                  "user_id": str(user_id),
                  "initial_charge_id": charge_id,
                },
              )
              session.add(index_job)
              session.commit()
              session.refresh(index_job)

              logger.info(f"Created fallback indexing job: {index_job.id}")
              submit_index_documents_task(
                job_id=index_job.id,
                doc_ids=[UUID(doc_id)],
                org_id=UUID(org_id),
                user_id=UUID(user_id),
                kb_id=UUID(kb_id),
                initial_charge_id=charge_id,
              )

          else:
            # For file documents, create new indexing job (original behavior)
            index_job = JobModel(
              name="Document Indexing",
              description="Index document after processing",
              status=JobStatus.PENDING,
              created_by=job.created_by,
              parent_job_id=job.id,
              context={
                "doc_ids": [str(doc_id)],
                "kb_id": kb_id,
                "user_id": str(user_id),
                "initial_charge_id": charge_id,
              },
            )
            session.add(index_job)
            session.commit()
            session.refresh(index_job)

            logger.info(f"Creating and triggering index job: {index_job.id}")
            submit_index_documents_task(
              job_id=index_job.id,
              doc_ids=[UUID(doc_id)],
              org_id=UUID(org_id),
              user_id=UUID(user_id),
              kb_id=UUID(kb_id),
              initial_charge_id=charge_id,
            )

      return {
        "success": True,
        "doc_id": str(doc_id),
        "content_length": len(content),
      }

    except Exception as e:
      logger.error(
        f"Document processing failed for document {doc_id}: {str(e)}",
        extra={
          "doc_id": str(doc_id),
          "job_id": str(job_id),
          "org_id": str(org_id),
          "error": str(e),
        },
      )

      # Update document status on error
      try:
        doc = session.get(KBDocumentModel, UUID(doc_id))
        if doc:
          doc.extraction_status = DocumentStatus.FAILED
          doc.error_message = str(e)
          session.commit()
      except Exception:
        pass

      # Update job status on error
      try:
        job = session.get(JobModel, UUID(job_id))
        if job:
          job.status = JobStatus.FAILED
          job.message = f"Document processing error: {str(e)}"
          session.commit()

          # Update job progress to trigger failure status update
          update_job_progress(
            job_id=UUID(job_id),
            progress=0.0,
            message=f"Document processing error: {str(e)}",
            session=session,
            metadata={
              "doc_id": doc_id,
              "task_type": "document_processing",
              "kb_id": job.context.get("kb_id") if job and job.context else None,
            },
            org_id=UUID(org_id),
            send_update=True,
          )
      except Exception:
        pass

      return {"success": False, "error": str(e)}


@task(bind=True)
def index_documents_task(
  self,
  job_id: str,
  doc_ids: List[str],
  org_id: str,
  user_id: str,
  kb_id: str,
  initial_charge_id: Optional[str] = None,
) -> dict:
  """Index documents with embedding generation and billing."""
  logger.info(f"Starting indexing task for documents: {doc_ids}")
  logger.info(f"Task parameters - org_id: {org_id}, kb_id: {kb_id}, initial_charge_id: {initial_charge_id}, user_id: {user_id}, job_id: {job_id}")

  with sync_session() as session:
    try:
      # Get job and check if parent job (processing) is completed
      job = session.get(JobModel, UUID(job_id))
      if job and job.parent_job_id:
        parent_job = session.get(JobModel, job.parent_job_id)
        if parent_job and parent_job.status != JobStatus.COMPLETED:
          logger.warning(f"Parent job {parent_job.id} not completed (status: {parent_job.status}), cannot proceed with indexing")
          if job:
            job.status = JobStatus.FAILED
            job.message = "Document processing not completed, cannot index"
            session.commit()
          return {"success": False, "error": "Document processing not completed"}

      # Update job status to processing
      if job:
        job.status = JobStatus.PROCESSING
        job.message = "Starting document indexing"
        session.commit()
        logger.info(f"Job {job_id} status updated to PROCESSING")

        # Update job progress with Celery task ID and send initial status
        update_job_progress(
          job_id=UUID(job_id),
          progress=0.0,
          message="Starting document indexing",
          session=session,
          metadata={"kb_id": kb_id, "doc_count": len(doc_ids), "task_type": "document_indexing", "doc_id": doc_ids[0] if doc_ids else None},
          celery_task_id=self.request.id,
          org_id=UUID(org_id),
          send_update=True,
        )

      # Import models locally
      from datetime import datetime

      from models import DocumentStatus, KBDocumentModel, KnowledgeBaseModel
      from utils.charge import Charge

      # Get documents and verify they exist
      doc_models = []
      for doc_id_str in doc_ids:
        doc = session.get(KBDocumentModel, UUID(doc_id_str))
        if doc:
          # Verify the document belongs to the correct KB and organization
          kb = session.get(KnowledgeBaseModel, doc.kb_id)
          if kb and str(kb.organization_id) == org_id:
            doc_models.append(doc)
          else:
            logger.error(f"Document {doc_id_str} does not belong to organization {org_id}")
        else:
          logger.error(f"Document {doc_id_str} not found")

      if not doc_models:
        logger.error(f"No valid documents found for IDs: {doc_ids}")
        if job:
          job.status = JobStatus.FAILED
          job.message = "No valid documents found"
          session.commit()
        return {"success": False, "error": "No valid documents found"}

      logger.info(f"Documents found: {[doc.id for doc in doc_models]}")

      # Get knowledge base
      kb_model = session.get(KnowledgeBaseModel, UUID(kb_id))
      if not kb_model:
        logger.error(f"Knowledge base not found for kb_id: {kb_id}, org_id: {org_id}")
        if job:
          job.status = JobStatus.FAILED
          job.message = f"Knowledge base {kb_id} not found"
          session.commit()
        return {"success": False, "error": f"Knowledge base {kb_id} not found"}

      logger.info(f"Knowledge base found: {kb_model.name}")

      # Get existing charge if provided for updating
      embedding_charge = None
      if initial_charge_id:
        try:
          logger.info(f"Using existing charge for indexing operation: {initial_charge_id}")
          embedding_charge = Charge.from_transaction_id(initial_charge_id, session=session)
          if embedding_charge:
            logger.info(f"Found existing charge: {embedding_charge.transaction_id}")
          else:
            logger.warning(f"Could not find charge with ID: {initial_charge_id}")
        except Exception as e:
          logger.error(
            f"Failed to get existing charge {initial_charge_id}: {str(e)}",
            extra={
              "charge_id": initial_charge_id,
              "kb_id": str(kb_id),
              "job_id": str(job_id),
              "error": str(e),
            },
          )
          embedding_charge = None

      # Process documents one by one
      logger.info("Starting document processing for indexing")
      indexed_count = 0
      total_docs = len(doc_models)

      for doc_index, doc_model in enumerate(doc_models, 1):
        logger.info(f"Processing document {doc_index}/{total_docs}: {doc_model.id}")
        logger.debug(f"Document details - title: {doc_model.title}, content_length: {len(doc_model.content) if doc_model.content else 0}")

        doc_model.indexing_status = DocumentStatus.PROCESSING
        session.add(doc_model)
        session.commit()
        logger.info(f"Document {doc_model.id} status updated to PROCESSING")

        # Update job progress (only send updates for significant progress changes)
        progress = (doc_index - 1) / total_docs * 80.0  # Reserve 20% for final steps
        update_job_progress(
          job_id=UUID(job_id),
          progress=progress,
          message=f"Processing document {doc_index}/{total_docs}: {doc_model.title}",
          session=session,
          metadata={"current_doc": str(doc_model.id), "doc_progress": f"{doc_index}/{total_docs}", "task_type": "document_indexing"},
          org_id=UUID(org_id),
          send_update=False,  # Don't send update for each document, only for significant changes
        )

        try:
          # Estimate embedding cost based on document content
          if doc_model.content:
            token_count = len(doc_model.content.split())
            estimated_chunks = max(1, token_count // kb_model.settings["max_chunk_size"])
            logger.info(f"Content analysis - tokens: {token_count}, estimated_chunks: {estimated_chunks}")

            # Update charge with document-specific information if we have a charge
            if embedding_charge and embedding_charge.transaction_id:
              try:
                logger.debug("Charge update skipped in sync context - will be handled asynchronously")
                logger.info(f"Document {doc_model.id} analysis: {token_count} tokens, {estimated_chunks} estimated chunks")
              except Exception as e:
                logger.error(f"Failed to log charge info: {str(e)}")
          else:
            logger.warning(f"Document {doc_model.id} has no content for embedding")

          # Initialize document processor for actual embeddings
          logger.info("Processing document embeddings")

          # Generate actual embeddings using existing loaders and vectorstore
          if doc_model.content:
            try:
              from langchain_core.documents import Document
              from langchain_openai import OpenAIEmbeddings
              from langchain_postgres import PGVector

              from database.postgres import sync_engine

              # Initialize embeddings and vectorstore
              logger.info(f"Initializing embeddings with model: {kb_model.embedding_model}")
              embeddings = OpenAIEmbeddings(model=kb_model.embedding_model)

              collection_name = f"{kb_model.organization_id}_{kb_model.name}"
              logger.info(f"Using collection name: {collection_name}")

              # Use existing sync engine for PGVector
              logger.info("Initializing PGVector with sync database engine")
              vectorstore = PGVector(
                collection_name=collection_name,
                connection=sync_engine,
                embeddings=embeddings,
                use_jsonb=True,
                create_extension=False,
              )

              # Check if document content is already available (optimization)
              if doc_model.content and len(doc_model.content.strip()) > 0:
                logger.info(f"Using existing processed content for document {doc_model.id} (length: {len(doc_model.content)})")

                # Use text splitter to chunk the existing content
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                text_splitter = RecursiveCharacterTextSplitter(
                  chunk_size=kb_model.settings.get("max_chunk_size", 1000),
                  chunk_overlap=kb_model.settings.get("chunk_overlap", 200),
                  length_function=len,
                  separators=["\n\n", "\n", " ", ""],
                )

                # Split content into chunks
                text_chunks = text_splitter.split_text(doc_model.content)
                logger.info(f"Split content into {len(text_chunks)} chunks using text splitter")

                # Create Document objects with metadata
                chunks = []
                for chunk_idx, chunk_text in enumerate(text_chunks):
                  # Determine source type from document metadata
                  source_type = "file"  # Default
                  if doc_model.source_metadata and (
                    doc_model.source_metadata.get("operation") in ["scrape", "crawl", "map"] or doc_model.source_metadata.get("url")
                  ):
                    source_type = "url"

                  chunk_metadata = {
                    "doc_id": str(doc_model.id),
                    "kb_id": str(kb_id),
                    "chunk_id": chunk_idx,
                    "title": doc_model.title,
                    "source_type": source_type,
                    "file_type": "webpage" if source_type == "url" else doc_model.source_metadata.get("file_type", "file"),
                    **doc_model.source_metadata,
                  }

                  doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                  chunks.append(doc)

                chunk_count = len(chunks)
              else:
                # Fallback to DoclingFileLoader if content is not available
                logger.info("Content not available, using DoclingFileLoader for document processing")
                import asyncio

                from services.kb.loaders import DoclingFileLoader

                loader = DoclingFileLoader(kb_id=UUID(kb_id), document=doc_model)

                # Collect all chunks - handle async generator in sync context
                chunks = []

                # Create async function to collect chunks
                async def collect_chunks():
                  chunk_list = []
                  count = 0
                  async for chunk_doc in loader.load():  # DoclingFileLoader returns AsyncIterator
                    # Determine source type from document metadata
                    source_type = "file"  # Default
                    if doc_model.source_metadata and (
                      doc_model.source_metadata.get("operation") in ["scrape", "crawl", "map"] or doc_model.source_metadata.get("url")
                    ):
                      source_type = "url"

                    # Create proper metadata for each chunk
                    chunk_metadata = {
                      "doc_id": str(doc_model.id),
                      "kb_id": str(kb_id),
                      "chunk_id": count,
                      "title": doc_model.title,
                      "source_type": source_type,
                      "file_type": "webpage" if source_type == "url" else doc_model.source_metadata.get("file_type", "file"),
                      **doc_model.source_metadata,
                    }

                    # Create langchain Document with metadata
                    doc = Document(page_content=chunk_doc.page_content, metadata=chunk_metadata)
                    chunk_list.append(doc)
                    count += 1
                  return chunk_list, count

                # Run async function in sync context
                chunks, chunk_count = asyncio.run(collect_chunks())

              logger.info(f"Generated {chunk_count} chunks for document {doc_model.id}")

              # Store chunks in vectorstore if we have any
              if chunks:
                logger.info(f"Storing {len(chunks)} chunks in vectorstore")
                chunk_ids = vectorstore.add_documents(chunks)  # Sync method
                if len(chunk_ids) > 3:
                  logger.info(f"Successfully stored {len(chunk_ids)} chunks with IDs: {chunk_ids[:3]}...")
                else:
                  logger.info(f"Successfully stored chunks with IDs: {chunk_ids}")

                actual_chunks = len(chunks)
                indexed_count += 1
              else:
                logger.warning(f"No chunks generated for document {doc_model.id}")
                actual_chunks = 0

              # Update status to completed
              doc_model.indexing_status = DocumentStatus.COMPLETED
              doc_model.indexing_completed_at = datetime.now()
              logger.info(f"Document {doc_model.id} indexing status updated to COMPLETED")

              # Update metadata with chunk information
              metadata = doc_model.source_metadata or {}
              metadata["actual_chunks"] = actual_chunks
              doc_model.source_metadata = metadata
              logger.debug(f"Document {doc_model.id} metadata updated with chunk information")

              session.add(doc_model)
              session.commit()
              logger.info(f"Document {doc_model.id} changes committed to database")

              # Update charge with actual chunk information (no async/await in sync context)
              if embedding_charge and embedding_charge.transaction_id:
                try:
                  logger.debug("Updating charge with actual chunk count - skipping in sync context")
                  logger.info(f"Embedding charge will be updated asynchronously for {actual_chunks} chunks")
                except Exception as e:
                  logger.error(
                    f"Failed to log charge update: {str(e)}",
                    extra={
                      "doc_id": str(doc_model.id) if doc_model else None,
                      "kb_id": str(kb_id),
                      "job_id": str(job_id),
                      "actual_chunks": actual_chunks if "actual_chunks" in locals() else None,
                      "error": str(e),
                    },
                  )

            except Exception as embed_error:
              logger.error(
                f"Failed to create embeddings for document {doc_model.id}: {str(embed_error)}",
                extra={
                  "doc_id": str(doc_model.id),
                  "kb_id": str(kb_id),
                  "job_id": str(job_id),
                  "error": str(embed_error),
                },
              )
              # Mark as failed but continue with other documents
              doc_model.indexing_status = DocumentStatus.FAILED
              doc_model.error_message = f"Embedding generation failed: {str(embed_error)}"
              session.add(doc_model)
              session.commit()
              logger.info(f"Document {doc_model.id} marked as FAILED due to embedding error")

        except Exception as e:
          logger.error(
            f"Error processing document {doc_model.id} for indexing: {str(e)}",
            extra={
              "doc_id": str(doc_model.id),
              "kb_id": str(kb_id),
              "job_id": str(job_id),
              "error": str(e),
            },
          )
          # Update status to failed
          doc_model.indexing_status = DocumentStatus.FAILED
          doc_model.error_message = str(e)
          session.add(doc_model)
          session.commit()
          logger.info(f"Document {doc_model.id} status updated to FAILED")

          # Update charge with failure information (handled asynchronously elsewhere)
          if embedding_charge and embedding_charge.transaction_id:
            try:
              logger.info(f"Charge error handling for document {doc_model.id} will be processed asynchronously")
              logger.info(f"Error details logged for charge: {embedding_charge.transaction_id}")
            except Exception as charge_error:
              logger.error(
                f"Failed to log charge error info: {str(charge_error)}",
                extra={
                  "kb_id": str(kb_id),
                  "job_id": str(job_id),
                  "doc_id": str(doc_model.id) if doc_model else None,
                  "error": str(charge_error),
                },
              )

      # Complete charge if provided (handled asynchronously elsewhere)
      if embedding_charge and embedding_charge.transaction_id:
        try:
          logger.info("Indexing charge completion will be handled asynchronously")
          logger.info(f"Indexing completed: {indexed_count}/{total_docs} documents, charge: {embedding_charge.transaction_id}")
        except Exception as e:
          logger.warning(f"Failed to log indexing completion: {e}")
      elif initial_charge_id:
        try:
          logger.info(f"Initial charge {initial_charge_id} completion will be handled by charge service")
        except Exception as e:
          logger.warning(f"Failed to log initial charge completion {initial_charge_id}: {e}")

      # Update job status to completed
      if job:
        job.status = JobStatus.COMPLETED
        job.message = f"Document indexing completed. Indexed {indexed_count}/{total_docs} documents"
        session.commit()

      # Update final job progress (completion will trigger status update automatically)
      update_job_progress(
        job_id=UUID(job_id),
        progress=100.0,
        message=f"Document indexing completed. Indexed {indexed_count}/{total_docs} documents",
        session=session,
        metadata={
          "kb_id": kb_id,
          "indexed_count": indexed_count,
          "total_docs": total_docs,
          "task_type": "document_indexing",
          "doc_id": doc_ids[0] if doc_ids else None,
        },  # noqa: E501
        org_id=UUID(org_id),
        send_update=True,
      )

      logger.info(f"Indexing task completed for all documents: {doc_ids}")

      return {
        "success": True,
        "kb_id": str(kb_id),
        "indexed_count": indexed_count,
        "total_docs": total_docs,
      }

    except Exception as e:
      logger.error(
        f"Document indexing failed for KB {kb_id}: {str(e)}",
        extra={
          "kb_id": str(kb_id),
          "job_id": str(job_id),
          "org_id": str(org_id),
          "error": str(e),
        },
      )

      # Update job status on error
      try:
        job = session.get(JobModel, UUID(job_id))
        if job:
          job.status = JobStatus.FAILED
          job.message = f"Document indexing error: {str(e)}"
          session.commit()

          # Update job progress to trigger failure status update
          update_job_progress(
            job_id=UUID(job_id),
            progress=0.0,
            message=f"Document indexing error: {str(e)}",
            session=session,
            metadata={"kb_id": kb_id, "task_type": "document_indexing", "doc_id": doc_ids[0] if doc_ids and len(doc_ids) > 0 else None},
            org_id=UUID(org_id),
            send_update=True,
          )
      except Exception:
        pass

      return {"success": False, "error": str(e)}


@task(bind=True)
def upload_file_task(
  self,
  job_id: str,
  doc_id: str,
  file_content: bytes,
  s3_key: str,
  content_type: str,
  org_id: str,
  user_id: str,
  charge_id: str,
) -> dict:
  """Upload file using simple synchronous operations."""
  logger.info(f"Starting file upload task for document ID: {doc_id}, job ID: {job_id}")

  with sync_session() as session:
    try:
      # Update job status to processing
      job = session.get(JobModel, UUID(job_id))
      if job:
        job.status = JobStatus.PROCESSING
        job.message = "Starting file upload"
        session.commit()

        # Send HTTP update for upload start
        # Get kb_id from document
        from models import KBDocumentModel

        doc = session.get(KBDocumentModel, UUID(doc_id))
        kb_id_value = doc.kb_id if doc else None

        # Update job progress with Celery task ID and send initial status
        update_job_progress(
          job_id=UUID(job_id),
          progress=0.0,
          message="Starting file upload to S3",
          session=session,
          metadata={
            "doc_id": doc_id,
            "s3_key": s3_key,
            "task_type": "file_upload",
            "kb_id": str(kb_id_value) if kb_id_value else None,
          },
          celery_task_id=self.request.id,
          org_id=UUID(org_id),
          send_update=True,
        )

      # Upload file to S3 using async client in sync context
      try:
        import asyncio

        from libs.s3.v1 import s3_client

        # Create and run async upload in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          s3_url = loop.run_until_complete(s3_client.upload_file(file=file_content, key=s3_key, content_type=content_type))
          logger.info(f"File uploaded to S3: {s3_url}")
        finally:
          loop.close()

        # Update job progress (send update for significant progress change)
        update_job_progress(
          job_id=UUID(job_id),
          progress=75.0,
          message="File uploaded to S3 successfully",
          session=session,
          metadata={
            "doc_id": doc_id,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "task_type": "file_upload",
            "kb_id": str(kb_id_value) if kb_id_value else None,
          },
          org_id=UUID(org_id),
          send_update=True,
        )

      except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        if job:
          job.status = JobStatus.FAILED
          job.message = f"S3 upload failed: {str(e)}"
          session.commit()

          # Update job progress to trigger failure status update
          update_job_progress(
            job_id=UUID(job_id),
            progress=0.0,
            message=f"S3 upload failed: {str(e)}",
            session=session,
            metadata={
              "doc_id": doc_id,
              "s3_key": s3_key,
              "task_type": "file_upload",
              "kb_id": str(kb_id_value) if kb_id_value else None,
            },
            org_id=UUID(org_id),
            send_update=True,
          )
        return {"success": False, "error": f"S3 upload failed: {str(e)}"}

      # Update document with S3 info

      doc = session.get(KBDocumentModel, UUID(doc_id))
      if doc:
        # Update source_metadata with S3 information
        metadata = doc.source_metadata or {}
        metadata.update({"s3_key": s3_key, "content_type": content_type, "upload_status": "completed"})
        doc.source_metadata = metadata
        session.commit()

      # Complete charge (handled asynchronously in service layer)
      try:
        # Note: Charge completion is handled asynchronously in the service layer
        logger.info(f"Upload completed, charge {charge_id} will be processed by background service")
      except Exception as e:
        logger.warning(f"Failed to log charge info {charge_id}: {e}")

      # Update job status to completed
      if job:
        job.status = JobStatus.COMPLETED
        job.message = "File upload completed successfully"
        session.commit()

      # Update job progress (completion will trigger status update automatically)
      update_job_progress(
        job_id=UUID(job_id),
        progress=100.0,
        message="File upload completed successfully",
        session=session,
        metadata={"doc_id": doc_id, "s3_key": s3_key, "task_type": "file_upload", "kb_id": str(kb_id_value) if kb_id_value else None},
        org_id=UUID(org_id),
        send_update=True,
      )

      # Trigger child process job
      logger.info(f"Looking for child jobs with parent_job_id: {job_id}")

      # Query for child jobs
      parent_job_uuid = UUID(job_id)
      logger.info(f"Parent job UUID: {parent_job_uuid}")
      child_jobs_query = select(JobModel).where(JobModel.parent_job_id == parent_job_uuid)
      child_jobs_result = session.execute(child_jobs_query)
      child_jobs = child_jobs_result.scalars().all()

      logger.info(f"Found {len(child_jobs)} child jobs")

      # If no child jobs found, let's also check all jobs to debug
      if len(child_jobs) == 0:
        logger.warning("No child jobs found, checking all jobs for debugging...")
        all_jobs_query = select(JobModel)
        all_jobs_result = session.execute(all_jobs_query)
        all_jobs = all_jobs_result.scalars().all()
        logger.info(f"Total jobs in database: {len(all_jobs)}")
        for job in all_jobs:
          if job.parent_job_id:
            logger.info(f"Job {job.id} has parent_job_id: {job.parent_job_id}")

      for child_job in child_jobs:
        logger.info(f"Child job {child_job.id} context: {child_job.context}")
        if child_job.context and "source_type_name" in child_job.context:
          # This is a process job
          logger.info(f"Triggering child process job: {child_job.id}")
          try:
            task_id = submit_process_document_task(
              job_id=child_job.id,
              source_type_name=child_job.context["source_type_name"],
              source_type_config=child_job.context["source_type_config"],
              doc_id=UUID(child_job.context["doc_id"]),
              org_id=UUID(org_id),
              user_id=UUID(user_id),
              charge_id=child_job.context.get("charge_id"),
              complete_charge=child_job.context.get("complete_charge", True),
            )
            logger.info(f"Successfully submitted process task {task_id} for child job {child_job.id}")
          except Exception as e:
            logger.error(
              f"Failed to submit process task for child job {child_job.id}: {str(e)}",
              extra={
                "child_job_id": str(child_job.id),
                "job_id": str(job_id),
                "org_id": str(org_id),
                "doc_id": str(doc_id),
                "error": str(e),
              },
            )
        else:
          logger.warning(f"Child job {child_job.id} does not have source_type_name in context")

      return {
        "success": True,
        "doc_id": str(doc_id),
        "s3_key": s3_key,
      }

    except Exception as e:
      logger.error(
        f"File upload failed for document {doc_id}: {str(e)}",
        extra={
          "doc_id": str(doc_id),
          "job_id": str(job_id),
          "org_id": str(org_id),
          "error": str(e),
        },
      )

      # Update job status on error
      try:
        job = session.get(JobModel, UUID(job_id))
        if job:
          job.status = JobStatus.FAILED
          job.message = f"File upload error: {str(e)}"
          session.commit()

          # Get kb_id from document for metadata
          doc = session.get(KBDocumentModel, UUID(doc_id))
          kb_id_value = doc.kb_id if doc else None
          # Update job progress to trigger failure status update
          update_job_progress(
            job_id=UUID(job_id),
            progress=0.0,
            message=f"File upload error: {str(e)}",
            session=session,
            metadata={"doc_id": doc_id, "task_type": "file_upload", "kb_id": str(kb_id_value) if kb_id_value else None},
            org_id=UUID(org_id),
            send_update=True,
          )
      except Exception:
        pass

      return {"success": False, "error": str(e)}


# Helper functions to submit tasks (unchanged)
def submit_process_document_task(
  job_id: UUID,
  source_type_name: str,
  source_type_config: dict,
  doc_id: UUID,
  org_id: UUID,
  user_id: UUID,
  charge_id: Optional[str] = None,
  complete_charge: bool = True,
) -> str:
  """Submit document processing task."""
  task_result = process_document_task.delay(
    job_id=str(job_id),
    source_type_name=source_type_name,
    source_type_config=source_type_config,
    doc_id=str(doc_id),
    org_id=str(org_id),
    user_id=str(user_id),
    charge_id=charge_id,
    complete_charge=complete_charge,
  )
  return task_result.id


def submit_index_documents_task(
  job_id: UUID,
  doc_ids: List[UUID],
  org_id: UUID,
  user_id: UUID,
  kb_id: UUID,
  initial_charge_id: Optional[str] = None,
) -> str:
  """Submit document indexing task."""
  task_result = index_documents_task.delay(
    job_id=str(job_id),
    doc_ids=[str(doc_id) for doc_id in doc_ids],
    org_id=str(org_id),
    user_id=str(user_id),
    kb_id=str(kb_id),
    initial_charge_id=initial_charge_id,
  )
  return task_result.id


def submit_upload_file_task(
  job_id: UUID,
  doc_id: UUID,
  file_content: bytes,
  s3_key: str,
  content_type: str,
  org_id: UUID,
  user_id: UUID,
  charge_id: str,
) -> str:
  """Submit file upload task."""
  task_result = upload_file_task.delay(
    job_id=str(job_id),
    doc_id=str(doc_id),
    file_content=file_content,
    s3_key=s3_key,
    content_type=content_type,
    org_id=str(org_id),
    user_id=str(user_id),
    charge_id=charge_id,
  )
  return task_result.id


@task(bind=True)
def process_url_task(
  self,
  url: str,
  operation: str,
  settings: Dict[str, Any],  # noqa: F811
  org_id: str,
  job_id: str,
  doc_id: str,
  kb_id: str,
  user_id: str,
) -> Dict[str, Any]:
  """
  Process URL with comprehensive WebSocket progress updates.

  Handles scrape, crawl, and map operations with real-time progress notifications.
  """
  logger.info(f"Starting URL processing task: {operation} for {url}")

  try:
    # Send initial progress update
    send_job_update(
      org_id=UUID(org_id),
      job_id=UUID(job_id),
      status="processing",
      message=f"Starting {operation} operation for {url}",
      progress=5.0,
      kb_id=UUID(kb_id),
      file_id=UUID(doc_id),
    )

    import asyncio

    # Run async operations in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      if operation == "scrape":
        result = loop.run_until_complete(_process_url_scrape(url, settings, org_id, job_id, doc_id, kb_id))
      elif operation == "crawl":
        result = loop.run_until_complete(_process_url_crawl(url, settings, org_id, job_id, doc_id, kb_id, user_id))
      elif operation == "map":
        result = loop.run_until_complete(_process_url_map(url, settings, org_id, job_id, doc_id, kb_id))
      else:
        raise ValueError(f"Unsupported URL operation: {operation}")
      return result
    finally:
      loop.close()

  except Exception as e:
    logger.error(f"URL processing failed: {str(e)}")
    send_job_update(
      org_id=UUID(org_id),
      job_id=UUID(job_id),
      status="failed",
      message=f"URL processing failed: {str(e)}",
      progress=0.0,
      kb_id=UUID(kb_id),
      file_id=UUID(doc_id),
    )
    raise


async def _process_url_scrape(
  url: str,
  settings: Dict[str, Any],  # noqa: F811
  org_id: str,
  job_id: str,
  doc_id: str,
  kb_id: str,
) -> Dict[str, Any]:
  """Process single URL scraping with dynamic progress updates."""

  total_urls = 1  # Only 1 URL for scrape operation

  # Get document details for WebSocket updates
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    if not doc:
      raise ValueError(f"Document {doc_id} not found")

    # Use folder_id from document (set by API) - don't create new folder
    folder_id = doc.folder_id
    logger.info(f"Using folder {folder_id} for scrape operation (set by API)")

    doc_details = get_file_details_for_websocket(doc)

  # Send initial URL discovery update (0-20% range)
  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Discovered {total_urls} URL to scrape: {url}",
    progress=10.0,
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "scrape",
      "discovered_urls": total_urls,
      "processed_urls": 0,
      "current_url": url,
      "progress_stage": "discovered",
      "folder_id": str(folder_id),
    },
    file_details=doc_details,
  )

  # Send scraping progress update (20-70% range)
  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Scraping content from {url}",
    progress=30.0,
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "scrape",
      "discovered_urls": total_urls,
      "processed_urls": 0,
      "current_url": url,
      "status": "scraping",
      "progress_stage": "processing",
      "folder_id": str(folder_id),
    },
    file_details=doc_details,
  )

  from libs.firecrawl.v2 import firecrawl

  # Implement retry logic for scrape operation
  max_retries = 2
  retry_count = 0
  content = None

  while retry_count <= max_retries:
    try:
      content = await firecrawl.scrape_url(url=url, settings=settings)
      break  # Success, exit retry loop
    except Exception as api_error:
      error_msg = str(api_error).lower()
      is_retryable_error = any(code in error_msg for code in ["502", "503", "504", "timeout", "connection", "network"])

      if is_retryable_error and retry_count < max_retries:
        retry_count += 1
        wait_time = min(2**retry_count, 10)  # Exponential backoff, max 10 seconds
        logger.warning(f"Retryable scrape error (attempt {retry_count}/{max_retries + 1}): {api_error}. Retrying in {wait_time} seconds...")

        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Scrape error detected, retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries + 1})",
          progress=30.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "scrape",
            "discovered_urls": total_urls,
            "attempt": retry_count + 1,
            "error": str(api_error)[:100],
            "progress_stage": "retrying",
            "folder_id": str(folder_id),
          },
          file_details=doc_details,
        )

        # Wait before retry
        import asyncio

        await asyncio.sleep(wait_time)
        continue
      if is_retryable_error and retry_count >= max_retries:
        # Max retries exceeded, try fallback crawler
        logger.error("Max retries exceeded for scrape, attempting fallback to crawl4ai")
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Firecrawl scrape failed after {max_retries + 1} attempts, trying crawl4ai fallback",
          progress=30.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "scrape",
            "discovered_urls": total_urls,
            "crawler_type": "api_fallback",
            "original_error": str(api_error)[:100],
            "progress_stage": "fallback",
            "folder_id": str(folder_id),
          },
          file_details=doc_details,
        )

        try:
          from libs.crawl4ai.v2 import crawl4ai

          # Use crawl4ai for single URL scraping
          crawl_result = await crawl4ai.crawl(url=url, params={**settings, "limit": 1})
          if crawl_result.get("success") and crawl_result.get("data"):
            first_page = crawl_result["data"][0]
            content = first_page.get("markdown", "") or first_page.get("content", "")
            logger.info("Successfully fell back to crawl4ai for scraping after Firecrawl API failures")
          else:
            content = ""
          break  # Exit retry loop
        except Exception as fallback_error:
          logger.error(f"Fallback scraper also failed: {fallback_error}")
          # Use minimal content as final fallback
          content = f"Content extraction failed after multiple attempts. Original error: {str(api_error)[:100]}"
          break
      else:
        # Non-retryable error, raise immediately
        raise api_error

  if not content:
    content = f"No content could be extracted from URL: {url}"

  # Create document entry for the scraped URL (already exists as parent doc)
  with sync_session() as session:
    from models import KBDocumentModel

    # Update the parent document with scraped content
    doc = session.get(KBDocumentModel, UUID(doc_id))
    if doc:
      # Update metadata with scraping details
      if not doc.source_metadata:
        doc.source_metadata = {}
      doc.source_metadata.update({
        "scraped_content_size": len(content),
        "scraped_characters": len(content),
        "scraped_words": len(content.split()) if content else 0,
        "scraping_completed": True,
        "file_type": "webpage",  # Always set file_type as webpage for URL documents
      })
      session.commit()
      logger.info(f"Updated document {doc_id} with scraped content metadata")

      # Update document details after scraping
      doc_details = get_file_details_for_websocket(doc)

  # Send completion update with detailed URL processing info (70-100% range)
  progress = 90.0  # Single URL completion at 90%

  # Prepare all files for final summary (just the parent document for scrape)
  all_files = [doc_details]

  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Successfully scraped {len(content)} characters from 1 URL",
    progress=progress,
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "scrape",
      "discovered_urls": total_urls,
      "processed_urls": 1,
      "scraped_content_size": len(content),
      "completed_urls": [{"url": url, "content_size": len(content), "doc_id": doc_id}],
      "status": "completed",
      "progress_stage": "completed",
      "folder_id": str(folder_id),
      "created_files": all_files,
    },
    file_details=doc_details,
  )

  logger.info(f"Successfully scraped {len(content)} characters from {url}")
  return {"success": True, "content": content, "operation": "scrape", "urls_processed": 1}


async def _process_url_crawl(
  url: str,
  settings: Dict[str, Any],  # noqa: F811
  org_id: str,
  job_id: str,
  doc_id: str,
  kb_id: str,
  user_id: str,
) -> Dict[str, Any]:
  """Process URL crawling with comprehensive progress updates."""
  logger.info(f"Starting comprehensive crawl of {url}")

  # Get crawler type from settings
  crawler_type = settings.get("crawler", "premium")

  # Get document details for WebSocket updates
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    doc_details = get_file_details_for_websocket(doc)

  # Get the folder_id from document (set by API)
  folder_id = None
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    folder_id = doc.folder_id if doc else None
    logger.info(f"Using folder {folder_id} for crawl operation (set by API)")

  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Initializing {crawler_type} crawler for {url}",
    progress=10.0,
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "crawl",
      "source_url": url,
      "crawler_type": crawler_type,
      "progress_stage": "initializing",
      "folder_id": str(folder_id),
    },
    file_details=doc_details,
  )

  # Windows Store Python subprocess limitation detection and fallback
  is_windows_subprocess_limited = platform.system() == "Windows" and "WindowsApps" in sys.executable

  # Perform crawl operation with appropriate crawler and retry logic
  results = None
  crawler_used = crawler_type
  max_retries = 2
  retry_count = 0

  while retry_count <= max_retries:
    try:
      if crawler_type == "base":
        from libs.crawl4ai.v2 import crawl4ai

        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Starting crawl4ai crawl of {url}" + (f" (attempt {retry_count + 1})" if retry_count > 0 else ""),
          progress=15.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "crawl",
            "crawler_type": "crawl4ai",
            "attempt": retry_count + 1,
            "progress_stage": "crawling",
            "folder_id": str(folder_id),
          },
        )
        results = await crawl4ai.crawl(url=url, params=settings)
        break  # Success, exit retry loop
      else:  # premium (firecrawl)
        if is_windows_subprocess_limited:
          logger.warning("Windows Store Python detected - using crawl4ai instead of firecrawl")
          from libs.crawl4ai.v2 import crawl4ai

          send_job_update(
            org_id=UUID(org_id),
            job_id=UUID(job_id),
            status="processing",
            message=f"Using crawl4ai fallback for {url}" + (f" (attempt {retry_count + 1})" if retry_count > 0 else ""),
            progress=15.0,
            kb_id=UUID(kb_id),
            file_id=UUID(doc_id),
            url_details={
              "operation": "crawl",
              "crawler_type": "crawl4ai_fallback",
              "attempt": retry_count + 1,
              "progress_stage": "crawling",
              "folder_id": str(folder_id),
            },
          )
          results = await crawl4ai.crawl(url=url, params=settings)
          crawler_used = "base"
          break  # Success, exit retry loop
        else:
          from libs.firecrawl.v2 import firecrawl

          send_job_update(
            org_id=UUID(org_id),
            job_id=UUID(job_id),
            status="processing",
            message=f"Starting Firecrawl crawl of {url}" + (f" (attempt {retry_count + 1})" if retry_count > 0 else ""),
            progress=15.0,
            kb_id=UUID(kb_id),
            file_id=UUID(doc_id),
            url_details={
              "operation": "crawl",
              "crawler_type": "firecrawl",
              "attempt": retry_count + 1,
              "progress_stage": "crawling",
              "folder_id": str(folder_id),
            },
          )
          results = await firecrawl.crawl(url=url, params=settings)
          break  # Success, exit retry loop

    except (NotImplementedError, OSError) as subprocess_error:
      if "subprocess" in str(subprocess_error).lower() or isinstance(subprocess_error, NotImplementedError):
        logger.warning(f"Subprocess error detected: {subprocess_error}")
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message="Fallback to crawl4ai due to system limitations",
          progress=15.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={"operation": "crawl", "crawler_type": "system_fallback", "progress_stage": "fallback", "folder_id": str(folder_id)},
        )
        try:
          from libs.crawl4ai.v2 import crawl4ai

          results = await crawl4ai.crawl(url=url, params=settings)
          crawler_used = "base"
          break  # Success, exit retry loop
        except Exception as fallback_error:
          raise ValueError(f"Both primary crawler and fallback failed. Primary: {subprocess_error}, Fallback: {fallback_error}")
      else:
        raise subprocess_error
    except Exception as api_error:
      error_msg = str(api_error).lower()
      is_retryable_error = any(code in error_msg for code in ["502", "503", "504", "timeout", "connection", "network"])

      if is_retryable_error and retry_count < max_retries:
        retry_count += 1
        wait_time = min(2**retry_count, 10)  # Exponential backoff, max 10 seconds
        logger.warning(f"Retryable error detected (attempt {retry_count}/{max_retries + 1}): {api_error}. Retrying in {wait_time} seconds...")

        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"API error detected, retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries + 1})",
          progress=15.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "crawl",
            "crawler_type": crawler_used,
            "attempt": retry_count + 1,
            "error": str(api_error)[:100],
            "progress_stage": "retrying",
            "folder_id": str(folder_id),
          },
        )

        # Wait before retry
        import asyncio

        await asyncio.sleep(wait_time)
        continue
      if is_retryable_error and retry_count >= max_retries:
        # Max retries exceeded, try fallback crawler
        logger.error(f"Max retries exceeded for {crawler_type}, attempting fallback to crawl4ai")
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Firecrawl failed after {max_retries + 1} attempts, trying crawl4ai fallback",
          progress=15.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "crawl",
            "crawler_type": "api_fallback",
            "original_error": str(api_error)[:100],
            "progress_stage": "fallback",
            "folder_id": str(folder_id),
          },
        )

        try:
          from libs.crawl4ai.v2 import crawl4ai

          results = await crawl4ai.crawl(url=url, params=settings)
          crawler_used = "base"
          logger.info("Successfully fell back to crawl4ai after Firecrawl API failures")
          break  # Success with fallback
        except Exception as fallback_error:
          logger.error(f"Fallback crawler also failed: {fallback_error}")
          error_msg = (
            f"Both Firecrawl API (after {max_retries + 1} attempts) and fallback crawl4ai failed. Firecrawl: {api_error}, Fallback: {fallback_error}"
          )
          raise ValueError(error_msg)
      else:
        # Non-retryable error, raise immediately
        raise api_error

  if not results or not results.get("success"):
    raise ValueError(f"Crawl operation failed: {results.get('error', 'Unknown error') if results else 'No results returned'}")

  scraped_pages = results.get("data", [])
  if not scraped_pages:
    raise ValueError("No pages were crawled")

  logger.info(f"Successfully crawled {len(scraped_pages)} pages using {crawler_used} crawler")

  # Send detailed URLs discovered notification
  discovered_urls = [page_data.get("metadata", {}).get("url", f"{url}/page_{idx}") for idx, page_data in enumerate(scraped_pages)]
  total_urls = len(scraped_pages)

  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Discovered {total_urls} URLs to process from crawl operation",
    progress=25.0,  # URLs discovered at 25%
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "crawl",
      "discovered_urls": total_urls,
      "processed_urls": 0,
      "crawler_used": crawler_used,
      "discovered_url_list": discovered_urls[:10],  # First 10 URLs for preview
      "status": "discovered",
      "progress_stage": "urls_discovered",
      "folder_id": str(folder_id),
    },
    file_details=doc_details,
  )

  # Process each crawled page using the folder_id from API
  with sync_session() as session:
    doc = session.get(KBDocumentModel, UUID(doc_id))
    if not doc:
      raise ValueError(f"Document {doc_id} not found")

    # Ensure parent document has correct file_type in metadata
    if not doc.source_metadata:
      doc.source_metadata = {}
    doc.source_metadata["file_type"] = "webpage"
    session.add(doc)
    session.commit()
    logger.info(f"Updated parent document {doc_id} with file_type: webpage")

    main_content = ""
    child_documents: List[dict] = []
    processed_urls = []

    for idx, page_data in enumerate(scraped_pages):
      page_url = page_data.get("metadata", {}).get("url", f"{url}/page_{idx}")
      page_content = page_data.get("markdown", "")

      if not page_content.strip():
        logger.warning(f"Skipping empty content from {page_url}")
        # Still track as processed even if empty
        processed_urls.append({"url": page_url, "status": "skipped", "reason": "empty_content"})

        # Send progress update for skipped URL
        progress = 25.0 + ((len(processed_urls) / total_urls) * 60.0)
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Skipped URL {len(processed_urls)}/{total_urls}: {page_url} (empty content)",
          progress=progress,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "crawl",
            "discovered_urls": total_urls,
            "processed_urls": len(processed_urls),
            "current_url": page_url,
            "status": "skipping_empty_url",
            "progress_stage": "processing_urls",
            "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1),
            "folder_id": str(folder_id),
          },
        )
        continue

      # Check if this is the main page
      if page_url == url or page_url.rstrip("/") == url.rstrip("/"):
        main_content = page_content
        logger.info(f"Found main page content: {len(page_content)} characters")
        processed_urls.append({"url": page_url, "status": "main_page", "content_size": len(page_content), "doc_id": doc_id})

        # Send progress update for main page
        progress = 25.0 + ((len(processed_urls) / total_urls) * 60.0)
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Processed main page {len(processed_urls)}/{total_urls}: {page_url}",
          progress=progress,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "crawl",
            "discovered_urls": total_urls,
            "processed_urls": len(processed_urls),
            "current_url": page_url,
            "content_size": len(page_content),
            "status": "processing_main_page",
            "progress_stage": "processing_urls",
            "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1),
            "folder_id": str(folder_id),
          },
        )
      else:
        # Create child document for this crawled page
        child_doc = create_child_document_sync(
          parent_doc=doc, page_url=page_url, content=page_content, folder_id=folder_id, session=session, index=len(child_documents)
        )

        if child_doc:
          child_documents.append(child_doc)
          processed_urls.append({"url": page_url, "status": "document_created", "content_size": len(page_content), "doc_id": str(child_doc.id)})
          logger.info(f"Created child document {child_doc.id} for {page_url}")

          # Send detailed progress update for each document created
          # Dynamic progress: 25% (discovery) + 60% * (processed_urls / total_urls) = 25-85% range
          progress = 25.0 + ((len(processed_urls) / total_urls) * 60.0)

          # Get child document details for WebSocket
          child_doc_details = get_file_details_for_websocket(child_doc)

          send_job_update(
            org_id=UUID(org_id),
            job_id=UUID(job_id),
            status="processing",
            message=f"Created document {len(child_documents)}/{total_urls}: {child_doc.title} from {page_url}",
            progress=progress,
            kb_id=UUID(kb_id),
            file_id=child_doc.id,
            url_details={
              "operation": "crawl",
              "discovered_urls": total_urls,
              "processed_urls": len(processed_urls),
              "created_documents": len(child_documents),
              "current_url": page_url,
              "current_doc_id": str(child_doc.id),
              "content_size": len(page_content),
              "status": "creating_documents",
              "progress_stage": "processing_urls",
              "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1),
              "folder_id": str(folder_id),
            },
            file_details=child_doc_details,
          )
        else:
          processed_urls.append({"url": page_url, "status": "failed", "reason": "document_creation_failed"})

          # Send progress update for failed document creation
          progress = 25.0 + ((len(processed_urls) / total_urls) * 60.0)
    send_job_update(
      org_id=UUID(org_id),
      job_id=UUID(job_id),
      status="processing",
      message=f"Failed to create document {len(processed_urls)}/{total_urls}: {page_url}",
      progress=progress,
      kb_id=UUID(kb_id),
      file_id=UUID(doc_id),
      url_details={
        "operation": "crawl",
        "discovered_urls": total_urls,
        "processed_urls": len(processed_urls),
        "current_url": page_url,
        "status": "document_creation_failed",
        "progress_stage": "processing_urls",
        "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1),
        "folder_id": str(folder_id),
      },
    )

    # Collect all files (parent + children) for final summary
    all_files = [doc_details]
    for child_doc in child_documents:
      child_doc_details = get_file_details_for_websocket(child_doc)
      all_files.append(child_doc_details)

    # Send completion notification with detailed URL processing summary (85-95% range)
    send_job_update(
      org_id=UUID(org_id),
      job_id=UUID(job_id),
      status="processing",
      message=f"Crawl completed: Created {len(child_documents)} documents from {total_urls} discovered URLs",
      progress=90.0,  # Completion at 90%
      kb_id=UUID(kb_id),
      file_id=UUID(doc_id),
      url_details={
        "operation": "crawl",
        "discovered_urls": total_urls,
        "processed_urls": len(processed_urls),
        "created_documents": len(child_documents),
        "main_content_size": len(main_content) if main_content else 0,
        "completed_urls": processed_urls,
        "crawler_used": crawler_used,
        "status": "completed",
        "progress_stage": "completed",
        "completion_percentage": 100.0,
        "folder_id": str(folder_id),
        "created_files": all_files,
      },
      file_details=doc_details,
    )

    logger.info(f"Crawl completed: main content ({len(main_content)} chars), {len(child_documents)} child documents")

    return {
      "success": True,
      "content": main_content or f"Crawled {len(scraped_pages)} pages from {url}",
      "operation": "crawl",
      "child_documents": len(child_documents),
      "total_pages": len(scraped_pages),
      "urls_processed": len(processed_urls),
      "processed_urls": processed_urls,
    }


async def _process_url_map(
  url: str,
  settings: Dict[str, Any],  # noqa: F811
  org_id: str,
  job_id: str,
  doc_id: str,
  kb_id: str,
) -> Dict[str, Any]:
  """Process URL mapping with individual document creation for each discovered URL."""

  # Get document details for WebSocket updates
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    doc_details = get_file_details_for_websocket(doc)

  # Get the folder_id from document (set by API)
  folder_id = None
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    folder_id = doc.folder_id if doc else None
    logger.info(f"Using folder {folder_id} for map operation (set by API)")

  # Send initial mapping progress update
  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Starting URL mapping discovery from {url}",
    progress=10.0,
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={"operation": "map", "source_url": url, "status": "discovering", "folder_id": str(folder_id)},
    file_details=doc_details,
  )

  from libs.firecrawl.v2 import firecrawl

  # Discover URLs using mapping with retry logic
  max_retries = 2
  retry_count = 0
  results = None

  while retry_count <= max_retries:
    try:
      results = firecrawl.map_url(url=url, settings=settings)
      break  # Success, exit retry loop
    except Exception as api_error:
      error_msg = str(api_error).lower()
      is_retryable_error = any(code in error_msg for code in ["502", "503", "504", "timeout", "connection", "network"])

      if is_retryable_error and retry_count < max_retries:
        retry_count += 1
        wait_time = min(2**retry_count, 10)  # Exponential backoff, max 10 seconds
        logger.warning(f"Retryable map error (attempt {retry_count}/{max_retries + 1}): {api_error}. Retrying in {wait_time} seconds...")

        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Map error detected, retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries + 1})",
          progress=10.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "map",
            "source_url": url,
            "attempt": retry_count + 1,
            "error": str(api_error)[:100],
            "progress_stage": "retrying",
            "folder_id": str(folder_id),
          },
        )

        # Wait before retry
        import asyncio

        await asyncio.sleep(wait_time)
        continue
      if is_retryable_error and retry_count >= max_retries:
        # Max retries exceeded, return basic URL list as fallback
        logger.error("Max retries exceeded for map operation, using basic fallback")
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Map operation failed after {max_retries + 1} attempts, using basic URL fallback",
          progress=15.0,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "map",
            "source_url": url,
            "crawler_type": "basic_fallback",
            "original_error": str(api_error)[:100],
            "progress_stage": "fallback",
            "folder_id": str(folder_id),
          },
        )

        # Use basic URL as fallback (just process the source URL itself)
        results = [{"url": url, "title": "Source URL"}]
        logger.info("Using basic URL fallback for mapping after API failures")
        break
      else:
        # Non-retryable error, raise immediately
        raise api_error

  if not results:
    results = [{"url": url, "title": "Source URL"}]  # Fallback to at least process the source URL

  discovered_urls = [r.get("url", "") for r in results if r.get("url")]

  # Send URL discovery notification
  total_urls = len(discovered_urls)
  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Discovered {total_urls} URLs from mapping operation",
    progress=20.0,  # Discovery completed at 20%
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "map",
      "discovered_urls": total_urls,
      "processed_urls": 0,
      "source_url": url,
      "discovered_url_list": discovered_urls[:10],  # First 10 URLs for preview
      "status": "discovered",
      "progress_stage": "urls_discovered",
      "folder_id": str(folder_id),
    },
    file_details=doc_details,
  )

  # Process each discovered URL by scraping its content and creating individual documents
  # Using the folder_id from API (already set in document)
  with sync_session() as session:
    from models import KBDocumentModel

    doc = session.get(KBDocumentModel, UUID(doc_id))
    if not doc:
      raise ValueError(f"Document {doc_id} not found")

    # Ensure parent document has correct file_type in metadata
    if not doc.source_metadata:
      doc.source_metadata = {}
    doc.source_metadata["file_type"] = "webpage"
    session.add(doc)
    session.commit()
    logger.info(f"Updated parent document {doc_id} with file_type: webpage")

    main_content = ""
    child_documents: list = []
    processed_urls = []
    combined_content_parts = []

    for idx, url_result in enumerate(results):
      mapped_url = url_result.get("url", "")
      url_title = url_result.get("title", "")

      if not mapped_url:
        logger.warning(f"Skipping result {idx} - no URL found")
        continue

      try:
        # Scrape content from each discovered URL with retry logic
        logger.info(f"Scraping content from discovered URL: {mapped_url}")

        # Use firecrawl to scrape the individual URL with retry logic
        scraped_content = None
        scrape_retries = 1  # Fewer retries for individual URLs in batch
        scrape_retry_count = 0

        while scrape_retry_count <= scrape_retries:
          try:
            scraped_content = await firecrawl.scrape_url(url=mapped_url, settings={})
            break  # Success, exit retry loop
          except Exception as scrape_error:
            error_msg = str(scrape_error).lower()
            is_retryable_error = any(code in error_msg for code in ["502", "503", "504", "timeout", "connection"])

            if is_retryable_error and scrape_retry_count < scrape_retries:
              scrape_retry_count += 1
              wait_time = min(2**scrape_retry_count, 5)  # Shorter wait for batch processing
              logger.warning(f"Retryable error scraping {mapped_url} (attempt {scrape_retry_count + 1}): {scrape_error}")

              # Wait before retry
              import asyncio

              await asyncio.sleep(wait_time)
              continue
            if is_retryable_error:
              # Try fallback crawler for this specific URL
              logger.info(f"Trying fallback crawler for {mapped_url}")
              try:
                from libs.crawl4ai.v2 import crawl4ai

                crawl_result = await crawl4ai.crawl(url=mapped_url, params={"limit": 1})
                if crawl_result.get("success") and crawl_result.get("data"):
                  first_page = crawl_result["data"][0]
                  scraped_content = first_page.get("markdown", "") or first_page.get("content", "")
                  logger.info(f"Successfully used fallback crawler for {mapped_url}")
                else:
                  scraped_content = ""
              except Exception:
                scraped_content = ""
              break
            else:
              # Non-retryable error or max retries reached
              logger.error(f"Failed to scrape {mapped_url}: {scrape_error}")
              scraped_content = ""
              break

        if scraped_content is None:
          scraped_content = ""

        if not scraped_content or not scraped_content.strip():
          logger.warning(f"No content scraped from {mapped_url}")
          processed_urls.append({"url": mapped_url, "status": "empty", "reason": "no_content"})

          # Send progress update for empty URL
          progress = 20.0 + ((len(processed_urls) / total_urls) * 65.0) if total_urls > 0 else 20.0
          send_job_update(
            org_id=UUID(org_id),
            job_id=UUID(job_id),
            status="processing",
            message=f"Skipped URL {len(processed_urls)}/{total_urls}: {mapped_url} (no content)",
            progress=progress,
            kb_id=UUID(kb_id),
            file_id=UUID(doc_id),
            url_details={
              "operation": "map",
              "discovered_urls": total_urls,
              "processed_urls": len(processed_urls),
              "current_url": mapped_url,
              "status": "skipping_empty_url",
              "progress_stage": "processing_urls",
              "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1) if total_urls > 0 else 0,
              "folder_id": str(folder_id),
            },
          )
          continue

        # Check if this is the main/source URL
        if mapped_url == url or mapped_url.rstrip("/") == url.rstrip("/"):
          main_content = scraped_content
          processed_urls.append({"url": mapped_url, "status": "main_page", "content_size": len(scraped_content), "doc_id": doc_id})
          logger.info(f"Found main page content: {len(scraped_content)} characters")

          # Send progress update for main page
          progress = 20.0 + ((len(processed_urls) / total_urls) * 65.0) if total_urls > 0 else 20.0
          send_job_update(
            org_id=UUID(org_id),
            job_id=UUID(job_id),
            status="processing",
            message=f"Processed main page {len(processed_urls)}/{total_urls}: {mapped_url}",
            progress=progress,
            kb_id=UUID(kb_id),
            file_id=UUID(doc_id),
            url_details={
              "operation": "map",
              "discovered_urls": total_urls,
              "processed_urls": len(processed_urls),
              "current_url": mapped_url,
              "content_size": len(scraped_content),
              "status": "processing_main_page",
              "progress_stage": "processing_urls",
              "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1) if total_urls > 0 else 0,
              "folder_id": str(folder_id),
            },
          )
        else:
          # Create child document for each mapped URL
          child_doc = create_child_document_sync(
            parent_doc=doc, page_url=mapped_url, content=scraped_content, folder_id=folder_id, session=session, index=len(child_documents)
          )

          if child_doc:
            # Update child document title with URL title if available
            if url_title:
              child_doc.title = url_title[:100]  # Limit title length
              session.add(child_doc)
              session.commit()

            child_documents.append(child_doc)
            processed_urls.append({
              "url": mapped_url,
              "status": "document_created",
              "content_size": len(scraped_content),
              "doc_id": str(child_doc.id),
              "title": url_title,
            })
            logger.info(f"Created child document {child_doc.id} for mapped URL: {mapped_url}")

            # Send progress update for each document created
            # Dynamic progress: 20% (discovery) + 65% * (processed_urls / total_urls) = 20-85% range
            progress = 20.0 + ((len(processed_urls) / total_urls) * 65.0) if total_urls > 0 else 20.0

            # Get child document details for WebSocket
            child_doc_details = get_file_details_for_websocket(child_doc)

            send_job_update(
              org_id=UUID(org_id),
              job_id=UUID(job_id),
              status="processing",
              message=f"Created document {len(child_documents)}/{total_urls}: {child_doc.title} from {mapped_url}",
              progress=progress,
              kb_id=UUID(kb_id),
              file_id=child_doc.id,
              url_details={
                "operation": "map",
                "discovered_urls": total_urls,
                "processed_urls": len(processed_urls),
                "created_documents": len(child_documents),
                "current_url": mapped_url,
                "current_doc_id": str(child_doc.id),
                "content_size": len(scraped_content),
                "status": "creating_documents",
                "progress_stage": "processing_urls",
                "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1) if total_urls > 0 else 0,
                "folder_id": str(folder_id),
              },
              file_details=child_doc_details,
            )
          else:
            processed_urls.append({"url": mapped_url, "status": "failed", "reason": "document_creation_failed"})

            # Send progress update for failed document creation
            progress = 20.0 + ((len(processed_urls) / total_urls) * 65.0) if total_urls > 0 else 20.0
            send_job_update(
              org_id=UUID(org_id),
              job_id=UUID(job_id),
              status="processing",
              message=f"Failed to create document {len(processed_urls)}/{total_urls}: {mapped_url}",
              progress=progress,
              kb_id=UUID(kb_id),
              file_id=UUID(doc_id),
              url_details={
                "operation": "map",
                "discovered_urls": total_urls,
                "processed_urls": len(processed_urls),
                "current_url": mapped_url,
                "status": "document_creation_failed",
                "progress_stage": "processing_urls",
                "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1) if total_urls > 0 else 0,
                "folder_id": str(folder_id),
              },
            )

        # Also add to combined content for the parent document
        combined_content_parts.append(f"## {url_title or mapped_url}\n{scraped_content}")

      except Exception as e:
        logger.error(f"Failed to process mapped URL {mapped_url}: {str(e)}")
        processed_urls.append({"url": mapped_url, "status": "failed", "reason": str(e)})

        # Send progress update for exception during processing
        progress = 20.0 + ((len(processed_urls) / total_urls) * 65.0) if total_urls > 0 else 20.0
        send_job_update(
          org_id=UUID(org_id),
          job_id=UUID(job_id),
          status="processing",
          message=f"Error processing URL {len(processed_urls)}/{total_urls}: {mapped_url}",
          progress=progress,
          kb_id=UUID(kb_id),
          file_id=UUID(doc_id),
          url_details={
            "operation": "map",
            "discovered_urls": total_urls,
            "processed_urls": len(processed_urls),
            "current_url": mapped_url,
            "status": "processing_error",
            "error": str(e),
            "progress_stage": "processing_urls",
            "completion_percentage": round((len(processed_urls) / total_urls) * 100, 1) if total_urls > 0 else 0,
            "folder_id": str(folder_id),
          },
        )

  # Create combined content for the parent document if no main content was found
  if not main_content and combined_content_parts:
    main_content = "\n\n".join(combined_content_parts)

  # Collect all files (parent + children) for final summary
  all_files = [doc_details]
  for child_doc in child_documents:
    child_doc_details = get_file_details_for_websocket(child_doc)
    all_files.append(child_doc_details)

  # Send completion notification (85-95% range)
  send_job_update(
    org_id=UUID(org_id),
    job_id=UUID(job_id),
    status="processing",
    message=f"Map operation completed: Created {len(child_documents)} documents from {total_urls} discovered URLs",
    progress=90.0,  # Completion at 90%
    kb_id=UUID(kb_id),
    file_id=UUID(doc_id),
    url_details={
      "operation": "map",
      "discovered_urls": total_urls,
      "processed_urls": len(processed_urls),
      "created_documents": len(child_documents),
      "main_content_size": len(main_content) if main_content else 0,
      "completed_urls": processed_urls,
      "status": "completed",
      "progress_stage": "completed",
      "completion_percentage": 100.0,
      "folder_id": str(folder_id),
      "created_files": all_files,
    },
    file_details=doc_details,
  )

  logger.info(f"Successfully mapped and processed {total_urls} URLs, created {len(child_documents)} child documents")
  return {
    "success": True,
    "content": main_content or f"Mapped {total_urls} URLs from {url}",
    "operation": "map",
    "urls_mapped": total_urls,
    "child_documents": len(child_documents),
    "processed_urls": processed_urls,
  }
