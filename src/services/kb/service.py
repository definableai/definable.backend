from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from database import async_engine, async_session, get_db
from dependencies.security import RBAC
from dependencies.usage import Usage
from libs.s3.v1 import s3_client
from libs.vectorstore.v1 import create_vectorstore
from models import DocumentStatus, KBDocumentModel, KBFolder, KnowledgeBaseModel, SourceTypeModel
from models.job_model import JobModel, JobStatus
from services.__base.acquire import Acquire
from tasks.kb_tasks import submit_index_documents_task, submit_process_document_task, submit_upload_file_task

from .schema import (
  CrawlerType,
  DocumentChunk,
  DocumentChunkCreate,
  DocumentChunkDelete,
  DocumentChunkUpdate,
  FileDocumentData,
  KBDocumentChunksResponse,
  KBDocumentResponse,
  KBFolderCreate,
  KnowledgeBaseCreate,
  KnowledgeBaseDetailResponse,
  KnowledgeBaseResponse,
  KnowledgeBaseUpdate,
  URLDocumentData,
  validate_file_document_data,
)


class FolderItem(TypedDict):
  id: str
  name: str
  folder_info: dict
  created_at: Optional[str]
  updated_at: str
  item_count: int


class FileItem(TypedDict):
  id: str
  title: str
  description: str
  file_type: str
  size: int
  status: str
  created_at: Optional[str]
  updated_at: str
  download_url: Optional[str]


class CurrentFolder(TypedDict):
  id: str
  name: str
  parent_id: Optional[str]


class BreadcrumbItem(TypedDict):
  id: str
  name: str


class KnowledgeResponse(TypedDict, total=False):
  folders: List[FolderItem]
  files: List[FileItem]
  current_folder: CurrentFolder
  breadcrumbs: List[BreadcrumbItem]


class KnowledgeBaseService:
  """Knowledge base service."""

  http_exposed = [
    "post=create",
    "put=update",
    "delete=remove",
    "get=get",
    "get=list",
    "delete=remove_document",
    "post=testing",
    "get=get_document_chunks",
    "put=update_document_chunk",
    "delete=delete_document_chunks",
    "post=search_chunks",
    "post=add_url_document",
    "post=add_file_document",
    "get=get_document_content",
    "post=index_documents",
    "post=add_document_chunk",
    "get=list_knowledge",
    "post=create_folder",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.settings = acquire.settings
    self.ws_manager = acquire.ws_manager
    self.utils = acquire.utils
    self.logger = acquire.logger

  def _get_crawler_config(self, crawler_type: CrawlerType) -> tuple[str, str]:
    """
    Get crawler implementation details from enum type.

    Args:
        crawler_type: The abstracted crawler type (base or premium)

    Returns:
        tuple: (crawler_name, version) where version is only relevant for firecrawl
    """
    if crawler_type == CrawlerType.BASE:
      return ("crawl4ai", "")  # crawl4ai doesn't use versioning
    elif crawler_type == CrawlerType.PREMIUM:
      return ("firecrawl", "v2")  # Always use v2 for firecrawl
    else:
      raise ValueError(f"Unsupported crawler type: {crawler_type}")

  def _get_status_string(self, status: DocumentStatus) -> str:
    """
    Convert DocumentStatus enum to string.

    Args:
        status: DocumentStatus enum value

    Returns:
        String representation of the status
    """
    status_mapping = {
      DocumentStatus.PENDING: "pending",
      DocumentStatus.PROCESSING: "extracting",  # For document processing
      DocumentStatus.COMPLETED: "completed",
      DocumentStatus.FAILED: "failed",
    }
    return status_mapping.get(status, "pending")

  def _get_job_status_string(self, status: JobStatus) -> str:
    """
    Convert JobStatus enum to string.

    Args:
        status: JobStatus enum value

    Returns:
        String representation of the job status
    """
    status_mapping = {
      JobStatus.PENDING: "pending",
      JobStatus.PROCESSING: "processing",
      JobStatus.COMPLETED: "completed",
      JobStatus.FAILED: "failed",
      JobStatus.CANCELLED: "cancelled",
    }
    return status_mapping.get(status, "pending")

  def _get_document_overall_status(self, extraction_status: DocumentStatus, indexing_status: DocumentStatus) -> str:
    """
    Determine overall document status based on extraction and indexing statuses.

    Args:
        extraction_status: Document extraction status
        indexing_status: Document indexing status

    Returns:
        Overall status string: pending, extracting, indexing, completed, failed
    """
    # If either is failed, overall is failed
    if extraction_status == DocumentStatus.FAILED or indexing_status == DocumentStatus.FAILED:
      return "failed"

    # If both are completed, overall is completed
    if extraction_status == DocumentStatus.COMPLETED and indexing_status == DocumentStatus.COMPLETED:
      return "completed"

    # If extraction is processing, overall is extracting
    if extraction_status == DocumentStatus.PROCESSING:
      return "extracting"

    # If indexing is processing, overall is indexing
    if indexing_status == DocumentStatus.PROCESSING:
      return "indexing"

    # If extraction is completed but indexing is pending, we're ready for indexing
    if extraction_status == DocumentStatus.COMPLETED and indexing_status == DocumentStatus.PENDING:
      return "indexing"

    # Default to pending
    return "pending"

  # TODO: add types for embedding models
  async def post_create(
    self,
    org_id: UUID,
    kb_data: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> KnowledgeBaseResponse:
    """Create a new knowledge base."""
    print(user)
    # Create vectorstore
    collection_id = await create_vectorstore(org_id, kb_data.name)
    # TODO : create a unique index for org_id and KB(name)
    # Create knowledge base
    db_kb = KnowledgeBaseModel(
      organization_id=org_id,
      user_id=user["id"],
      collection_id=collection_id,
      name=kb_data.name,
      embedding_model=kb_data.settings.embedding_model,
      settings=kb_data.settings.model_dump(),
    )
    session.add(db_kb)
    await session.commit()
    await session.refresh(db_kb)
    return KnowledgeBaseResponse.model_validate(db_kb)

  async def put_update(
    self,
    org_id: UUID,
    kb_id: UUID,
    kb_data: KnowledgeBaseUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> KnowledgeBaseResponse:
    """Update a knowledge base."""
    # Get knowledge base
    db_kb = await self._get_kb(kb_id, org_id, session)
    if not db_kb:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Update fields
    update_data = kb_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(db_kb, field, value)

    await session.commit()
    await session.refresh(db_kb)
    return KnowledgeBaseResponse.model_validate(db_kb)

  async def delete_remove(
    self,
    org_id: UUID,
    kb_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "delete")),
  ) -> dict:
    """Delete a knowledge base."""
    # Get knowledge base
    db_kb = await self._get_kb(kb_id, org_id, session)
    if not db_kb:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    await session.delete(db_kb)
    await session.commit()
    return {"message": "Knowledge base deleted successfully"}

  async def get_get(
    self,
    org_id: UUID,
    kb_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ) -> KnowledgeBaseDetailResponse:
    """Get a knowledge base with its documents."""
    # Get knowledge base with documents
    query = (
      select(KnowledgeBaseModel, KBDocumentModel)
      .outerjoin(KBDocumentModel)
      .where(KnowledgeBaseModel.id == kb_id, KnowledgeBaseModel.organization_id == org_id)
    )
    result = await session.execute(query)
    rows = result.unique().all()

    if not rows:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Process results
    kb = rows[0].KnowledgeBaseModel
    documents = []
    for row in rows:
      if row.KBDocumentModel is not None:
        doc_dict = row.KBDocumentModel.__dict__
        if doc_dict.get("s3_key"):
          try:
            # Generate pre-signed URL with 1 hour expiration
            download_url = await s3_client.get_presigned_url(
              doc_dict["s3_key"],
              expires_in=3600,
              operation="get_object",
            )
            doc_dict["download_url"] = download_url
          except Exception as e:
            print(str(e))
            doc_dict["download_url"] = None

        # Convert status enums to strings before adding to response
        doc_dict["status"] = self._get_document_overall_status(row.KBDocumentModel.extraction_status, row.KBDocumentModel.indexing_status)
        documents.append(KBDocumentResponse.model_validate(doc_dict))

    return KnowledgeBaseDetailResponse(**kb.__dict__, documents=documents)

  async def get_list(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ) -> List[KnowledgeBaseResponse]:
    """Get all knowledge bases for an organization."""
    query = select(KnowledgeBaseModel).where(KnowledgeBaseModel.organization_id == org_id)
    result = await session.execute(query)
    data = list(result.scalars().all())
    return [KnowledgeBaseResponse.model_validate(item) for item in data]

  async def post_add_file_document(
    self,
    org_id: UUID,
    kb_id: UUID,
    document_data: Annotated[FileDocumentData, Depends(validate_file_document_data)],
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
    usage: dict = Depends(Usage("pdf_extraction", qty=1, background=True, metadata={"operation": "file_document_upload"})),
  ) -> KBDocumentResponse:
    """Add file document to knowledge base."""
    try:
      # Get charge from usage dependency
      charge = usage["charge"]
      self.logger.info(f"Created initial charge for file upload: {charge.transaction_id}")

      self.logger.info(f"document_data: {document_data}")

      # Get source type model
      source_type_model = await session.get(SourceTypeModel, 1)
      if not source_type_model:
        # Release charge if source type not found
        await charge.delete(reason="Source type not found")
        raise HTTPException(status_code=404, detail="Source type not found")

      # Extract file metadata
      file_metadata = document_data.get_metadata()

      # Get file size without fully reading content to avoid blocking
      file_size = document_data.file.size or 0
      file_size_mb = file_size / (1024 * 1024)
      charge_qty = max(1, int(file_size_mb))
      file_metadata["charge_qty"] = charge_qty

      # Update charge with initial file information
      await charge.calculate_and_update(
        metadata={
          "file_name": document_data.file.filename,
          "file_size_mb": file_size_mb,
          "file_type": document_data.file.content_type,
          "pages": file_metadata.get("pages", 1),
          "sheet_count": file_metadata.get("sheet_count", 1),
          "charge_qty": charge_qty,
        },
        status="processing",
      )

      # Process folder path if provided
      folder_id = document_data.folder_id
      print(document_data)

      # Check if a file with the same name already exists in this folder
      existing_file_query = select(KBDocumentModel).where(
        KBDocumentModel.kb_id == kb_id, KBDocumentModel.folder_id == folder_id, KBDocumentModel.title == document_data.title
      )
      existing_file = await session.execute(existing_file_query)
      if existing_file.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"A file with name '{document_data.title}' already exists in this folder")

      # Prepare S3 configuration
      config_schema = source_type_model.config_schema
      storage = config_schema.get("storage")
      if not storage:
        await charge.delete(reason="Storage configuration not found")
        raise HTTPException(status_code=400, detail="Storage not found in source type config")

      s3_key = f"{storage['bucket']}/{storage['path']}/{org_id}-{kb_id}/{document_data.file.filename}"

      # Store S3 key immediately without blocking upload
      file_metadata["s3_key"] = s3_key
      file_metadata["upload_status"] = "pending"
      file_metadata["billing_transaction_id"] = str(charge.transaction_id)

      # Create document record immediately (before file upload)
      db_doc = KBDocumentModel(
        title=document_data.title,
        description=document_data.description,
        kb_id=kb_id,
        folder_id=folder_id,
        source_type_id=1,
        source_metadata=file_metadata,
        extraction_status=DocumentStatus.PENDING,
        indexing_status=DocumentStatus.PENDING,
      )

      session.add(db_doc)
      await session.commit()
      await session.refresh(db_doc)

      # Create upload and process jobs only
      async with async_session() as task_session:
        # Create file upload job
        upload_job = JobModel(
          name="File Upload",
          description=f"Upload file {document_data.file.filename}",
          status=JobStatus.PENDING,
          created_by=UUID(user["id"]),
          context={
            "doc_id": str(db_doc.id),
            "s3_key": s3_key,
            "content_type": document_data.file.content_type or "application/octet-stream",
            "charge_id": str(charge.transaction_id),
          },
        )
        task_session.add(upload_job)

        # Flush to get the upload job ID
        await task_session.flush()
        await task_session.refresh(upload_job)

        # Create document processing job as child of upload job
        process_job = JobModel(
          name="Document Processing",
          description=f"Process document {document_data.title}",
          status=JobStatus.PENDING,
          created_by=UUID(user["id"]),
          parent_job_id=upload_job.id,
          context={
            "source_type_name": source_type_model.name,
            "source_type_config": source_type_model.config_schema,
            "doc_id": str(db_doc.id),
            "kb_id": str(kb_id),
            "charge_id": str(charge.transaction_id),
            "complete_charge": False,
          },
        )
        task_session.add(process_job)
        await task_session.commit()
        await task_session.refresh(process_job)

        self.logger.info(f"Created upload job: {upload_job.id}")
        self.logger.info(f"Created process job: {process_job.id} with parent_job_id: {process_job.parent_job_id}")

      # Submit only the file upload task to Celery
      file_content = await document_data.file.read()
      submit_upload_file_task(
        job_id=upload_job.id,
        doc_id=db_doc.id,
        file_content=file_content,
        s3_key=s3_key,
        content_type=document_data.file.content_type or "application/octet-stream",
        org_id=org_id,
        user_id=UUID(user["id"]),
        charge_id=str(charge.transaction_id),
      )

      # Return response with overall status
      response_dict = db_doc.__dict__.copy()
      response_dict["status"] = "pending"  # Initial status for newly created document
      return KBDocumentResponse.model_validate(response_dict)

    except Exception as e:
      # If we have a charge and encounter any error, release the hold
      if "charge" in locals():
        try:
          await charge.delete(reason=f"Error in file document upload: {str(e)}")
          self.logger.info(f"Released charge for transaction {charge.transaction_id} due to error")
        except Exception as release_error:
          self.logger.error(f"Failed to release charge: {str(release_error)}")

      # Re-raise the original exception
      raise

  async def post_add_url_document(
    self,
    org_id: UUID,
    kb_id: UUID,
    document_data: URLDocumentData,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
    usage: dict = Depends(Usage("url_extraction", qty=1, background=True, metadata={"operation": "url_document_extraction"})),
  ) -> KBDocumentResponse:
    """Add URL document to knowledge base."""
    self.logger.info(f"Starting URL document addition for org_id={org_id}, kb_id={kb_id}")
    self.logger.info(f"URL document data: url={document_data.url}, operation={document_data.operation}, title={document_data.title}")

    try:
      # Get charge from usage dependency
      charge = usage["charge"]
      self.logger.info(f"Created initial charge for URL extraction: {charge.transaction_id}")
      self.logger.debug(f"Charge details: name=url_extraction, transaction_id={charge.transaction_id}")

      self.logger.debug(f"Full document_data: {document_data}")

      # get source type model
      self.logger.debug("Fetching source type model for URL (ID=2)")
      source_type_model = await session.get(SourceTypeModel, 2)
      if not source_type_model:
        self.logger.error("Source type model not found for URL processing")
        # Release charge if source type not found
        await charge.delete(reason="Source type not found")
        self.logger.info(f"Released charge {charge.transaction_id} due to missing source type")
        raise HTTPException(status_code=404, detail="Source type not found")

      self.logger.info(f"Source type model found: {source_type_model.name}")

      # Convert folder_id from string to UUID if provided
      folder_id = None
      if document_data.folder_id:
        self.logger.debug(f"Converting folder_id from string: {document_data.folder_id}")
        try:
          folder_id = UUID(document_data.folder_id)
          self.logger.info(f"Folder ID converted successfully: {folder_id}")
        except ValueError as ve:
          self.logger.error(f"Invalid folder ID format: {document_data.folder_id}, error: {str(ve)}")
          # Release charge if folder ID is invalid
          await charge.delete(reason="Invalid folder ID format")
          self.logger.info(f"Released charge {charge.transaction_id} due to invalid folder ID")
          raise HTTPException(status_code=400, detail="Invalid folder ID format")
      else:
        self.logger.info("No folder_id provided, document will be placed at root level")

      # Parse URLs and config
      self.logger.debug("Generating metadata from document data")
      metadata = document_data.get_metadata()
      metadata["is_parent"] = True
      metadata["parent_id"] = None

      # Add actual crawler implementation details
      crawler_name, firecrawl_version = self._get_crawler_config(document_data.crawler)
      metadata["crawler_impl"] = crawler_name
      if firecrawl_version:
        metadata["firecrawl_version"] = firecrawl_version

      self.logger.debug(f"Generated metadata: {metadata}")

      # Calculate charge quantity based on operation type and settings
      self.logger.info("Calculating charge quantity based on operation and settings")
      url_metadata = document_data.get_metadata()
      charge_qty = 1  # Base charge for URL extraction

      # Adjust charge based on operation type
      if document_data.operation == "crawl" and document_data.settings:
        # Import the CrawlerOptions type for type checking
        from .schema import CrawlerOptions

        if isinstance(document_data.settings, CrawlerOptions):
          original_charge_qty = charge_qty
          charge_qty = max(1, document_data.settings.limit // 10)  # 1 charge per 10 pages
          self.logger.info(
            f"Crawl operation - adjusted charge quantity from {original_charge_qty} to {charge_qty} based on limit={document_data.settings.limit}"
          )
        else:
          self.logger.warning(f"Crawl operation but settings is not CrawlerOptions type: {type(document_data.settings)}")
      elif document_data.operation == "map":
        self.logger.info("Map operation - using fixed charge quantity of 1")
        charge_qty = 1  # Fixed charge for mapping
      else:  # scrape
        self.logger.info("Scrape operation - using fixed charge quantity of 1")
        charge_qty = 1  # Fixed charge for single page scrape

      url_metadata["charge_qty"] = charge_qty
      self.logger.info(f"Final charge quantity calculated: {charge_qty}")

      # Update charge with URL-specific information
      self.logger.info("Updating charge with URL-specific metadata")
      await charge.calculate_and_update(
        metadata={
          "url": document_data.url,
          "operation": document_data.operation,
          "settings": url_metadata.get("settings", {}),
          "charge_qty": charge_qty,
          "folder_id": str(folder_id) if folder_id else None,
        },
        status="processing",  # It's still being processed at this point
      )
      self.logger.info(f"Charge updated successfully for transaction {charge.transaction_id}")

      # Store the transaction ID in metadata for tracking
      metadata["billing_transaction_id"] = str(charge.transaction_id)
      self.logger.debug(f"Added billing transaction ID to metadata: {charge.transaction_id}")

      # Always create a new folder for organizing URL processing documents
      self.logger.info("Creating folder for URL processing documents")
      from datetime import datetime

      # Use title as folder name, add timestamp only if there's a conflict
      folder_name = document_data.title or f"URL Content ({document_data.operation.title()})"

      # Check if folder with this name already exists
      existing_folder_query = select(KBFolder).where(
        KBFolder.kb_id == kb_id,
        KBFolder.name == folder_name,
        KBFolder.parent_id == folder_id,  # folder_id is the parent from request data
      )
      existing_result = await session.execute(existing_folder_query)
      existing_folder = existing_result.scalar_one_or_none()

      # If folder exists, add timestamp to make it unique
      if existing_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{folder_name}_{timestamp}"
        self.logger.info(f"Folder name conflict detected, using unique name: {folder_name}")

      # Create new folder for URL processing
      new_folder = KBFolder(
        name=folder_name,
        kb_id=kb_id,
        parent_id=folder_id,  # Use the parent folder from request data (can be None for root)
        folder_info={
          "created_by": f"{document_data.operation}_operation",
          "source_url": document_data.url,
          "created_for": "url_processing",
          "created_at": datetime.now().isoformat(),
          "parent_folder_id": str(folder_id) if folder_id else None,
          "original_title": document_data.title,
        },
      )

      session.add(new_folder)
      await session.commit()
      await session.refresh(new_folder)
      url_folder_id = new_folder.id
      self.logger.info(f"Created new folder for URL processing: {new_folder.name} (ID: {new_folder.id}) under parent: {folder_id}")

      # Create parent document to track crawl job
      self.logger.info("Creating parent document for URL processing")
      parent_doc = KBDocumentModel(
        title=document_data.title,
        description=document_data.description,
        kb_id=kb_id,
        folder_id=url_folder_id,  # Use the created/existing URL folder
        source_type_id=2,
        source_metadata=metadata,
        extraction_status=DocumentStatus.PENDING,
        indexing_status=DocumentStatus.PENDING,
      )
      session.add(parent_doc)
      await session.commit()
      await session.refresh(parent_doc)
      self.logger.info(f"Parent document created successfully with ID: {parent_doc.id}")

      # Send initial WebSocket notification for URL processing start
      await self.ws_manager.broadcast(
        org_id,
        {
          "id": str(parent_doc.id),
          "type": "url_processing_started",
          "operation": document_data.operation,
          "url": document_data.url,
          "title": document_data.title,
          "status": "pending",
          "folder_id": str(url_folder_id),
        },
        "kb",
        "write",
      )

      # Create jobs for URL processing and indexing
      async with async_session() as task_session:
        # Create processing job
        process_job = JobModel(
          name="URL Document Processing",
          description=f"Process URL document {document_data.title}",
          status=JobStatus.PENDING,
          created_by=UUID(user["id"]),
          context={
            "source_type_name": source_type_model.name,
            "source_type_config": source_type_model.config_schema,
            "doc_id": str(parent_doc.id),
            "kb_id": str(kb_id),
            "charge_id": str(charge.transaction_id),
            "complete_charge": False,
            "folder_id": str(url_folder_id),
          },
        )
        task_session.add(process_job)

        # Create indexing job
        index_job = JobModel(
          name="URL Document Indexing",
          description=f"Index URL document {document_data.title}",
          status=JobStatus.PENDING,
          created_by=UUID(user["id"]),
          parent_job_id=process_job.id,
          context={
            "doc_ids": [str(parent_doc.id)],
            "kb_id": str(kb_id),
            "initial_charge_id": str(charge.transaction_id),
            "user_id": str(user["id"]),
            "folder_id": str(url_folder_id),
          },
        )
        task_session.add(index_job)
        await task_session.commit()

      # Submit the URL processing task to Celery
      submit_process_document_task(
        job_id=process_job.id,
        source_type_name=source_type_model.name,
        source_type_config=source_type_model.config_schema,
        doc_id=parent_doc.id,
        org_id=org_id,
        user_id=UUID(user["id"]),
        charge_id=str(charge.transaction_id),
        complete_charge=False,
      )
      self.logger.info(f"Document processing task submitted to Celery for document {parent_doc.id}")

      # Return response with overall status
      self.logger.info(f"URL document addition completed successfully - document_id={parent_doc.id}, transaction_id={charge.transaction_id}")
      response_dict = parent_doc.__dict__.copy()
      response_dict["status"] = "pending"  # Initial status for newly created document
      return KBDocumentResponse.model_validate(response_dict)

    except Exception as e:
      # If we have a charge and encounter any error, release the hold
      self.logger.error(f"Error occurred during URL document addition: {str(e)}")
      if "charge" in locals():
        try:
          await charge.delete(reason=f"Error in URL document upload: {str(e)}")
          self.logger.info(f"Released charge for transaction {charge.transaction_id} due to error")
        except Exception as release_error:
          self.logger.error(f"Failed to release charge: {str(release_error)}")

      # Re-raise the original exception
      self.logger.error(f"Re-raising exception for URL document addition: {str(e)}")
      raise

  async def get_get_document_chunks(
    self,
    org_id: UUID,
    kb_id: UUID,
    doc_id: UUID,
    limit: int = 10,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ) -> KBDocumentChunksResponse:
    """
    Get document chunks from the vector store.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        doc_id: Document ID
        session: Database session
        user: User information

    Returns:
        KBDocumentChunksResponse containing the document chunks
    """
    # TODO: create a query Dependency
    if limit < 1:
      raise HTTPException(status_code=400, detail="Limit must be greater than 0")
    if offset < 0:
      raise HTTPException(status_code=400, detail="Offset must be non-negative")

    offset = offset * limit
    # Get document and verify access
    doc_model = await self._get_document(doc_id, org_id, session)
    if not doc_model:
      raise HTTPException(status_code=404, detail="Document not found")

    # Get knowledge base
    kb_model = await self._get_kb(kb_id, org_id, session)
    if not kb_model:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    try:
      # First, get the total count of chunks for this document
      count_query = text("""
        SELECT COUNT(*)
        FROM langchain_pg_embedding
        WHERE collection_id = :collection_id
        AND cmetadata->>'doc_id' = :doc_id
      """)
      count_result = await session.execute(count_query, {"collection_id": kb_model.collection_id, "doc_id": str(doc_id)})
      total_chunks: int = int(count_result.scalar() or 0)

      # Query vector store for chunks with matching doc_id, sorted by chunk_id as integer
      query = text("""
        SELECT id, document, embedding, cmetadata
        FROM langchain_pg_embedding
        WHERE collection_id = :collection_id
        AND cmetadata->>'doc_id' = :doc_id
        ORDER BY CAST(cmetadata->>'chunk_id' AS INTEGER)
        LIMIT :limit OFFSET :offset
      """)
      result = await session.execute(query, {"collection_id": kb_model.collection_id, "doc_id": str(doc_id), "limit": limit, "offset": offset})

      chunks = []
      for row in result:
        chunk = DocumentChunk(id=row.id, chunk_id=row.cmetadata.get("chunk_id"), content=row.document, metadata=row.cmetadata)
        chunks.append(chunk)

      return KBDocumentChunksResponse(document_id=doc_id, title=doc_model.title, chunks=chunks, total_chunks=total_chunks)

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to fetch document chunks: {str(e)}")

  async def delete_remove_document(
    self,
    org_id: UUID,
    doc_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "delete")),
  ) -> dict:
    """Delete a document."""
    # Get document and verify organization
    db_doc = await self._get_document(doc_id, org_id, session)
    if not db_doc:
      raise HTTPException(status_code=404, detail="Document not found")

    # Delete file from S3
    if db_doc.source_metadata.get("s3_key"):
      try:
        await s3_client.delete_file(db_doc.source_metadata["s3_key"])
      except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    await session.delete(db_doc)
    await session.commit()
    return {"message": "Document deleted successfully"}

  async def post_index_documents(
    self,
    org_id: UUID,
    kb_id: UUID,
    doc_ids: Annotated[List[UUID], Body(..., description="List of document IDs to index")],
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> JSONResponse:
    """Index documents."""
    # Create indexing job
    index_job = JobModel(
      name="Manual Document Indexing",
      description=f"Index {len(doc_ids)} documents",
      status=JobStatus.PENDING,
      created_by=UUID(user["id"]),
      context={"doc_ids": [str(doc_id) for doc_id in doc_ids], "kb_id": str(kb_id), "user_id": str(user["id"])},
    )
    session.add(index_job)
    await session.commit()
    await session.refresh(index_job)

    # Submit the indexing task to Celery
    submit_index_documents_task(
      job_id=index_job.id,
      doc_ids=doc_ids,
      org_id=org_id,
      user_id=UUID(user["id"]),
      kb_id=kb_id,
      initial_charge_id=None,
    )

    return JSONResponse(content={"message": "Documents are being indexed", "job_id": str(index_job.id)})

  async def put_update_document_chunk(
    self,
    org_id: UUID,
    kb_id: UUID,
    doc_chunk_data: DocumentChunkUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> DocumentChunk:
    """
    Update a document chunk and its embedding.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        chunk_id: Chunk ID to update
        content: New content for the chunk
        session: Database session
        user: User information

    Returns:
        Updated DocumentChunk
    """
    try:
      # Get knowledge base for embedding model
      kb_model = await self._get_kb(kb_id, org_id, session)
      if not kb_model:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

      # Get current chunk to verify it exists and get metadata
      query = text("""
        SELECT id, cmetadata
        FROM langchain_pg_embedding
        WHERE id = :chunk_id
        AND collection_id = :collection_id
      """)

      result = await session.execute(query, {"chunk_id": doc_chunk_data.chunk_id, "collection_id": kb_model.collection_id})

      chunk = result.first()

      if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

      # Generate new embedding
      embeddings = OpenAIEmbeddings(model=kb_model.embedding_model)
      new_embedding = await embeddings.aembed_query(doc_chunk_data.content)

      # Update chunk in database
      update_query = text(f"""
        UPDATE langchain_pg_embedding
        SET
            document = '{doc_chunk_data.content}',
            embedding = '{str(new_embedding)}'::vector
        WHERE id = '{doc_chunk_data.chunk_id}'
        AND collection_id = '{kb_model.collection_id}'
        RETURNING id, document, embedding, cmetadata
      """)

      result = await session.execute(update_query)

      updated_chunk = result.first()
      if not updated_chunk:
        raise HTTPException(status_code=404, detail="Failed to update chunk")
      await session.commit()

      return DocumentChunk(
        id=updated_chunk.id, chunk_id=updated_chunk.cmetadata.get("chunk_id"), content=updated_chunk.document, metadata=updated_chunk.cmetadata
      )

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to update document chunk: {str(e)}")

  async def delete_delete_document_chunks(
    self,
    org_id: UUID,
    kb_id: UUID,
    doc_chunk_data: DocumentChunkDelete,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "delete")),
  ) -> dict:
    """
    Delete document chunks from the vector store.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        chunk_ids: List of chunk IDs to delete
        session: Database session
        user: User information

    Returns:
        Confirmation message
    """
    try:
      # Get knowledge base
      kb_model = await self._get_kb(kb_id, org_id, session)
      if not kb_model:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

      # Delete chunks
      delete_query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE id = ANY(:chunk_ids)
        AND collection_id = :collection_id
        RETURNING id
      """)

      result = await session.execute(delete_query, {"chunk_ids": doc_chunk_data.chunk_ids, "collection_id": kb_model.collection_id})

      deleted_chunks = result.fetchall()
      await session.commit()

      if not deleted_chunks:
        raise HTTPException(status_code=404, detail="No chunks found to delete")

      return {"message": f"Successfully deleted {len(deleted_chunks)} chunks", "deleted_chunks": [str(row[0]) for row in deleted_chunks]}

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to delete document chunks: {str(e)}")

  async def post_search_chunks(
    self,
    org_id: UUID,
    kb_id: UUID,
    query: str = Body(..., description="Search query text"),
    limit: int = Body(default=5, description="Maximum number of results to return"),
    score_threshold: float = Body(default=0.0, description="Minimum similarity score threshold"),
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ) -> List[DocumentChunk]:
    """
    Search for similar document chunks using vector similarity.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        query: Search query text
        limit: Maximum number of results to return (default: 5)
        score_threshold: Minimum similarity score threshold (default: 0.7)
        session: Database session
        user: User information

    Returns:
        List of similar DocumentChunks with their similarity scores
    """
    try:
      # Get knowledge base
      kb_model = await self._get_kb(kb_id, org_id, session)
      if not kb_model:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

      # TODO: make a generic vector lib.
      # Initialize vector store
      vectorstore = PGVector(
        collection_name=str(kb_model.organization_id) + "_" + kb_model.name,
        connection=async_engine,
        embeddings=OpenAIEmbeddings(model=kb_model.embedding_model),
        use_jsonb=True,
        create_extension=False,
      )

      # Search for similar documents
      results = await vectorstore.asimilarity_search_with_relevance_scores(
        query,
        k=limit,
        score_threshold=score_threshold,
      )

      # Format results
      chunks = []
      for doc, score in results:
        chunk = DocumentChunk(
          chunk_id=int(doc.metadata.get("chunk_id", 0)),  # Convert to int with default 0
          content=doc.page_content,
          metadata=doc.metadata,
          score=score,
        )
        chunks.append(chunk)

      return chunks

    except Exception:
      raise

  async def get_get_document_content(
    self,
    org_id: UUID,
    doc_id: UUID,
    session: AsyncSession = Depends(get_db),
  ) -> KBDocumentResponse:
    """Get the content of a document."""
    doc_model = await self._get_document(doc_id, org_id, session)
    if not doc_model:
      raise HTTPException(status_code=404, detail="Document not found")

    # Convert status enums to strings
    response_dict = doc_model.__dict__.copy()
    response_dict["status"] = self._get_document_overall_status(doc_model.extraction_status, doc_model.indexing_status)
    return KBDocumentResponse.model_validate(response_dict)

  async def post_add_document_chunk(
    self,
    org_id: UUID,
    kb_id: UUID,
    doc_id: UUID,
    chunk_data: Annotated[DocumentChunkCreate, Body(...)],
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> DocumentChunk:
    """
    Add a new chunk to a document and index it in the vector store.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        doc_id: Document ID
        chunk_data: New chunk data
        session: Database session
        user: User information

    Returns:
        Created DocumentChunk with its metadata
    """
    # Get document and verify access
    doc_model = await self._get_document(doc_id, org_id, session)
    if not doc_model:
      raise HTTPException(status_code=404, detail="Document not found")

    # Get knowledge base for embedding model
    kb_model = await self._get_kb(kb_id, org_id, session)
    if not kb_model:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Get current max chunk index
    query = text("""
          SELECT COUNT(*) as max_index
          FROM langchain_pg_embedding
          WHERE collection_id = :collection_id
          AND cmetadata->>'doc_id' = :doc_id
      """)
    result = await session.execute(query, {"collection_id": kb_model.collection_id, "doc_id": str(doc_id)})
    max_index = result.scalar() or -1
    new_chunk_index = max_index + 1

    # Create metadata for the new chunk
    metadata = {**doc_model.source_metadata, "doc_id": str(doc_id), "chunk_id": new_chunk_index}

    # Initialize vector store
    vectorstore = PGVector(
      collection_name=str(kb_model.organization_id) + "_" + kb_model.name,
      connection=async_engine,
      embeddings=OpenAIEmbeddings(model=kb_model.embedding_model),
      use_jsonb=True,
      create_extension=False,
    )

    # Create document and add to vector store
    doc = Document(page_content=chunk_data.content, metadata=metadata)
    chunk_ids = await vectorstore.aadd_documents([doc])

    # Return the created chunk
    return DocumentChunk(id=UUID(chunk_ids[0]), chunk_id=new_chunk_index, content=chunk_data.content, metadata=metadata)

  async def post_create_folder(
    self,
    org_id: UUID,
    kb_id: UUID,
    folder_data: Annotated[KBFolderCreate, Body(...)],
    parent_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> FolderItem:
    """
    Create a new folder in the knowledge base at any level.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        folder_data: Folder creation data
        parent_id: Parent folder ID (can be passed as query parameter or in body)
        session: Database session
        user: User information

    Returns:
        Created folder information
    """

    # Use parent_id from query parameter if provided, otherwise use from folder_data
    effective_parent_id = parent_id if parent_id is not None else folder_data.parent_id

    # Check if knowledge base exists
    kb_model = await self._get_kb(kb_id, org_id, session)
    if not kb_model:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Check parent folder if specified
    if effective_parent_id:
      # Check if parent folder exists and belongs to this KB
      parent_query = select(KBFolder).where(KBFolder.id == effective_parent_id, KBFolder.kb_id == kb_id)
      result = await session.execute(parent_query)
      parent_folder = result.scalar_one_or_none()

      if not parent_folder:
        raise HTTPException(status_code=404, detail="Parent folder not found")

      # Check if folder with same name already exists in this parent
      existing_query = select(KBFolder).where(KBFolder.kb_id == kb_id, KBFolder.parent_id == effective_parent_id, KBFolder.name == folder_data.name)
    else:
      # Check at root level
      existing_query = select(KBFolder).where(KBFolder.kb_id == kb_id, KBFolder.parent_id.is_(None), KBFolder.name == folder_data.name)

    # Check for existing folder with the same name
    result = await session.execute(existing_query)
    existing_folder = result.scalar_one_or_none()
    if existing_folder:
      raise HTTPException(status_code=400, detail=f"A folder named '{folder_data.name}' already exists")

    # Create new folder
    new_folder = KBFolder(name=folder_data.name, parent_id=effective_parent_id, kb_id=kb_id, folder_info=folder_data.folder_info)

    session.add(new_folder)
    await session.commit()
    await session.refresh(new_folder)

    # Count items in the folder (will be 0 for a new folder)
    item_count = 0

    # Format response using FolderItem type
    folder_response = FolderItem(
      id=str(new_folder.id),
      name=new_folder.name,
      folder_info=new_folder.folder_info,
      created_at=new_folder.created_at.isoformat() if hasattr(new_folder, "created_at") else None,
      updated_at=new_folder.updated_at.isoformat(),
      item_count=item_count,
    )

    return folder_response

  async def get_list_knowledge(
    self,
    org_id: UUID,
    kb_id: UUID,
    folder_id: Optional[UUID] = None,  # If None, list root level
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ) -> KnowledgeResponse:
    """
    List folders and files at a specific level in the folder hierarchy.

    If folder_id is not provided, returns top-level folders and files.
    If folder_id is provided, returns folders and files inside that folder.
    """
    # First check if KB exists and belongs to the organization
    kb_model = await self._get_kb(kb_id, org_id, session)
    if not kb_model:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Get folders at the specified level
    folders_query = (
      select(KBFolder)
      .where(
        KBFolder.kb_id == kb_id,
        KBFolder.parent_id == folder_id,  # When folder_id is None, this finds root folders
      )
      .order_by(KBFolder.name)
    )

    folders_result = await session.execute(folders_query)
    folders = folders_result.scalars().all()

    # Get files at the specified level
    files_query = (
      select(KBDocumentModel)
      .where(
        KBDocumentModel.kb_id == kb_id,
        KBDocumentModel.folder_id == folder_id,  # When folder_id is None, this finds root files
      )
      .order_by(KBDocumentModel.title)
    )

    files_result = await session.execute(files_query)
    files = files_result.scalars().all()

    # Format the response
    response: KnowledgeResponse = {"folders": [], "files": []}

    # Add folders to response
    for folder in folders:
      # Count items inside the folder
      child_folders_count = await session.execute(select(func.count()).where(KBFolder.kb_id == kb_id, KBFolder.parent_id == folder.id))

      child_files_count = await session.execute(select(func.count()).where(KBDocumentModel.kb_id == kb_id, KBDocumentModel.folder_id == folder.id))

      total_items = child_folders_count.scalar_one() + child_files_count.scalar_one()

      folder_item: FolderItem = {
        "id": str(folder.id),
        "name": folder.name,
        "folder_info": folder.folder_info or {},  # Ensure folder_info is always a dictionary
        "created_at": folder.created_at.isoformat() if hasattr(folder, "created_at") else None,
        "updated_at": folder.updated_at.isoformat(),
        "item_count": total_items,
      }
      response["folders"].append(folder_item)

    # Add files to response with download URLs
    for file in files:
      file_data: FileItem = {
        "id": str(file.id),
        "title": file.title,
        "description": file.description,
        "file_type": file.source_metadata.get("file_type", ""),
        "size": file.source_metadata.get("size", 0),
        "status": self._get_document_overall_status(file.extraction_status, file.indexing_status),
        "created_at": file.created_at.isoformat() if hasattr(file, "created_at") else None,
        "updated_at": file.updated_at.isoformat(),
        "download_url": None,
      }

      # Generate pre-signed URL if file has s3_key
      if file.source_metadata.get("s3_key"):
        try:
          download_url = await s3_client.get_presigned_url(
            file.source_metadata["s3_key"],
            expires_in=3600,
            operation="get_object",
          )
          file_data["download_url"] = download_url
        except Exception:
          file_data["download_url"] = None

      response["files"].append(file_data)

    # Add current folder info if we're not at root level
    if folder_id:
      current_folder = await session.get(KBFolder, folder_id)
      if current_folder:
        current_folder_info: CurrentFolder = {
          "id": str(current_folder.id),
          "name": current_folder.name,
          "parent_id": str(current_folder.parent_id) if current_folder.parent_id else None,
        }
        response["current_folder"] = current_folder_info

        # Get folder breadcrumbs (path)
        if current_folder.parent_id:
          breadcrumbs: List[BreadcrumbItem] = []
          parent_id = current_folder.parent_id
          while parent_id:
            parent = await session.get(KBFolder, parent_id)
            if not parent:
              break
            breadcrumb_item: BreadcrumbItem = {"id": str(parent.id), "name": parent.name}
            breadcrumbs.insert(0, breadcrumb_item)
            parent_id = parent.parent_id
          response["breadcrumbs"] = breadcrumbs

    return response

  async def _get_kb(
    self,
    kb_id: UUID,
    org_id: UUID,
    session: AsyncSession,
  ) -> KnowledgeBaseModel | None:
    """Get knowledge base by ID and organization ID."""
    query = select(KnowledgeBaseModel).where(KnowledgeBaseModel.id == kb_id, KnowledgeBaseModel.organization_id == org_id)
    result = await session.execute(query)
    return result.scalar_one_or_none()

  async def _get_document(
    self,
    doc_id: UUID,
    org_id: UUID,
    session: AsyncSession,
  ) -> KBDocumentModel | None:
    """Get document by ID and verify organization."""
    query = select(KBDocumentModel).join(KnowledgeBaseModel).where(KBDocumentModel.id == doc_id, KnowledgeBaseModel.organization_id == org_id)
    result = await session.execute(query)
    return result.scalar_one_or_none()

  async def _get_documents(
    self,
    doc_ids: List[UUID],
    org_id: UUID,
    session: AsyncSession,
  ) -> List[KBDocumentModel]:
    """Get documents by IDs and verify organization."""
    query = select(KBDocumentModel).join(KnowledgeBaseModel).where(KBDocumentModel.id.in_(doc_ids), KnowledgeBaseModel.organization_id == org_id)
    result = await session.execute(query)
    return list(result.scalars().all())
