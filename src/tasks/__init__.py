"""Simple tasks package for background job processing."""

# Simple Celery bridge tasks
from .kb_tasks import (
  process_document_task,
  index_documents_task,
  upload_file_task,
  submit_process_document_task,
  submit_index_documents_task,
  submit_upload_file_task,
)

__all__ = [
  "process_document_task",
  "index_documents_task",
  "upload_file_task",
  "submit_process_document_task",
  "submit_index_documents_task",
  "submit_upload_file_task",
]
