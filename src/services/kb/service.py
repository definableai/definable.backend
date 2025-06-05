from datetime import datetime
from io import BytesIO
from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified
from typing_extensions import TypedDict

from database import async_engine, get_db
from dependencies.security import RBAC
from dependencies.usage import Usage
from libs.s3.v1 import s3_client
from libs.vectorstore.v1 import create_vectorstore
from models import DocumentStatus, KBDocumentModel, KBFolder, KnowledgeBaseModel, SourceTypeModel
from services.__base.acquire import Acquire
from utils.charge import Charge

from .loaders import DocumentProcessor
from .schema import (
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
from .source_handlers import get_source_handler


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
  extraction_status: DocumentStatus
  indexing_status: DocumentStatus
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
    background_tasks: BackgroundTasks,
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

      # Calculate proper charge quantity based on file size or page count
      file_content = await document_data.file.read()
      file_size_mb = len(file_content) / (1024 * 1024)
      # TODO : we need to get page counts in case of PDF, docs, excel ,etc
      charge_qty = max(1, int(file_size_mb))
      file_metadata["charge_qty"] = charge_qty

      # Update charge with actual quantity and file information
      await charge.calculate_and_update(
        metadata={
          "file_name": document_data.file.filename,
          "file_size_mb": file_size_mb,
          "file_type": document_data.file.content_type,
          "pages": file_metadata.get("pages", 1),  # Include page count if available
          "sheet_count": file_metadata.get("sheet_count", 1),  # Include sheet count if available
          "charge_qty": charge_qty,
        },
        status="processing",  # It's still being processed at this point
      )

      # Process folder path if provided
      folder_id = None
      folder_path = document_data.folder_path if hasattr(document_data, "folder_path") and document_data.folder_path else None

      if folder_path and isinstance(folder_path, list) and len(folder_path) > 0:
        current_parent_id = None

        # Process each folder in the path
        # TODO: can we optimize this to not do a loop on a query?
        for i, folder_name in enumerate(folder_path):
          # Check if folder with same name already exists at this level
          query = select(KBFolder).where(KBFolder.kb_id == kb_id, KBFolder.name == folder_name, KBFolder.parent_id == current_parent_id)
          result = await session.execute(query)
          folder = result.scalar_one_or_none()

          if not folder:
            # Create new folder
            folder = KBFolder(
              name=folder_name,
              parent_id=current_parent_id,
              kb_id=kb_id,
              folder_info={},  # Empty default info
            )
            session.add(folder)
            await session.flush()  # To get the ID

          # Move to next level
          current_parent_id = folder.id

        # Set the final folder ID for the document
        folder_id = current_parent_id

        # Check if a file with the same name already exists in this folder
        existing_file_query = select(KBDocumentModel).where(
          KBDocumentModel.kb_id == kb_id, KBDocumentModel.folder_id == folder_id, KBDocumentModel.title == document_data.title
        )
        existing_file = await session.execute(existing_file_query)
        if existing_file.scalar_one_or_none():
          raise HTTPException(status_code=400, detail=f"A file with name '{document_data.title}' already exists in this folder")

      # Upload file to s3
      config_schema = source_type_model.config_schema
      storage = config_schema.get("storage")
      if not storage:
        # Release charge if storage config is invalid
        await charge.delete(reason="Storage configuration not found")
        raise HTTPException(status_code=400, detail="Storage not found in source type config")

      if folder_path:
        joined_folder_path = "/".join(folder_path)
        s3_key = f"{storage['bucket']}/{storage['path']}/{org_id}-{kb_id}/{joined_folder_path}/{document_data.file.filename}"
      else:
        s3_key = f"{storage['bucket']}/{storage['path']}/{org_id}-{kb_id}/{document_data.file.filename}"

      try:
        # Upload to S3 (using already read file_content)
        await s3_client.upload_file(BytesIO(file_content), s3_key, content_type=document_data.file.content_type)
      except Exception as e:
        # Release charge if file upload fails
        await charge.delete(reason=f"S3 upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

      # Update file metadata with s3 key and folder info
      file_metadata["s3_key"] = s3_key
      if folder_id:
        file_metadata["folder_id"] = str(folder_id)

      # Store the transaction ID in metadata for tracking
      file_metadata["billing_transaction_id"] = str(charge.transaction_id)

      # Create document
      db_doc = KBDocumentModel(
        title=document_data.title,
        description=document_data.description,
        kb_id=kb_id,
        folder_id=folder_id,  # Associate with folder
        source_type_id=1,  # File type
        source_metadata=file_metadata,
        extraction_status=DocumentStatus.PENDING,
        indexing_status=DocumentStatus.PENDING,
      )

      session.add(db_doc)
      await session.commit()
      await session.refresh(db_doc)

      # For the first task, pass the charge but don't complete it
      background_tasks.add_task(
        self._process_document_task,
        source_type=source_type_model,
        doc=db_doc,
        session=session,
        org_id=org_id,
        charge=charge,
        complete_charge=False,  # Add a parameter to indicate not to complete the charge
      )

      # For the second task, create a separate related charge
      background_tasks.add_task(
        self._indexing_documents_task, [db_doc.id], org_id, kb_id, session, initial_charge_id=charge.transaction_id, user_id=UUID(user["id"])
      )

      # If everything is successful, return the response
      return KBDocumentResponse.model_validate(db_doc)

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
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> KBDocumentResponse:
    """Add URL document to knowledge base."""
    try:
      # get source type model
      source_type_model = await session.get(SourceTypeModel, 2)
      if not source_type_model:
        raise HTTPException(status_code=404, detail="Source type not found")
      # Parse URLs and config
      metadata = document_data.get_metadata()
      metadata["is_parent"] = True
      metadata["parent_id"] = None
      # Create parent document to track crawl job
      parent_doc = KBDocumentModel(
        title=document_data.title,
        description=document_data.description,
        kb_id=kb_id,
        source_type_id=2,
        source_id=document_data.source_id,
        source_metadata=metadata,
        extraction_status=DocumentStatus.PENDING,
        indexing_status=DocumentStatus.PENDING,
      )
      session.add(parent_doc)
      await session.commit()
      await session.refresh(parent_doc)

      background_tasks.add_task(self._process_document_task, source_type=source_type_model, doc=parent_doc, session=session, org_id=org_id)
      return KBDocumentResponse.model_validate(parent_doc)

    except Exception as e:
      raise HTTPException(status_code=400, detail=str(e))

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
      # Query vector store for all chunks with matching doc_id
      query = text("""
        SELECT id, document, embedding, cmetadata
        FROM langchain_pg_embedding
        WHERE collection_id = :collection_id
        AND cmetadata->>'doc_id' = :doc_id
        ORDER BY cmetadata->>'chunk_id'
        LIMIT :limit OFFSET :offset
      """)
      result = await session.execute(query, {"collection_id": kb_model.collection_id, "doc_id": str(doc_id), "limit": limit, "offset": offset})

      chunks = []
      for row in result:
        chunk = DocumentChunk(id=row.id, chunk_id=row.cmetadata.get("chunk_id"), content=row.document, metadata=row.cmetadata)
        chunks.append(chunk)

      return KBDocumentChunksResponse(document_id=doc_id, title=doc_model.title, chunks=chunks, total_chunks=len(chunks))

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
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> JSONResponse:
    """Index documents."""
    background_tasks.add_task(self._indexing_documents_task, doc_ids, org_id, kb_id, session)
    return JSONResponse(content={"message": "Documents are getting indexed"})

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
    return KBDocumentResponse.model_validate(doc_model)

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
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> FolderItem:
    """
    Create a new folder in the knowledge base at any level.

    Args:
        org_id: Organization ID
        kb_id: Knowledge Base ID
        folder_data: Folder creation data
        session: Database session
        user: User information

    Returns:
        Created folder information
    """
    # Check if knowledge base exists
    kb_model = await self._get_kb(kb_id, org_id, session)
    if not kb_model:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Check parent folder if specified
    if folder_data.parent_id:
      # Check if parent folder exists and belongs to this KB
      parent_query = select(KBFolder).where(KBFolder.id == folder_data.parent_id, KBFolder.kb_id == kb_id)
      result = await session.execute(parent_query)
      parent_folder = result.scalar_one_or_none()

      if not parent_folder:
        raise HTTPException(status_code=404, detail="Parent folder not found")

      # Check if folder with same name already exists in this parent
      existing_query = select(KBFolder).where(KBFolder.kb_id == kb_id, KBFolder.parent_id == folder_data.parent_id, KBFolder.name == folder_data.name)
    else:
      # Check at root level
      existing_query = select(KBFolder).where(KBFolder.kb_id == kb_id, KBFolder.parent_id.is_(None), KBFolder.name == folder_data.name)

    # Check for existing folder with the same name
    result = await session.execute(existing_query)
    existing_folder = result.scalar_one_or_none()
    if existing_folder:
      raise HTTPException(status_code=400, detail=f"A folder named '{folder_data.name}' already exists")

    # Create new folder
    new_folder = KBFolder(name=folder_data.name, parent_id=folder_data.parent_id, kb_id=kb_id, folder_info=folder_data.folder_info)

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
        "extraction_status": file.extraction_status,
        "indexing_status": file.indexing_status,
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

  ### PRIVATE METHODS ###

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

  async def _indexing_documents_task(
    self,
    doc_ids: List[UUID],
    org_id: UUID,
    kb_id: UUID,
    session: AsyncSession,
    initial_charge_id: Optional[str] = None,
    user_id: Optional[UUID] = None,
  ) -> None:
    """Background task to process document with billing."""
    self.logger.info(f"Indexing documents: {doc_ids}")
    doc_models = await self._get_documents(doc_ids, org_id, session)
    if not doc_models:
      raise HTTPException(status_code=404, detail="Documents not found")
    self.logger.info(f"Documents found: {doc_models}")

    # Create a new charge specifically for embedding operations
    embedding_charge = None
    embedding_charge_created = False  # Add this flag to track if charge was created successfully
    try:
      # Use the provided user_id directly instead of trying to get it from documents
      if user_id:
        self.logger.info("Creating embedding charge")
        # Create a charge for embedding generation
        embedding_charge = Charge(name="o1-small-text-indexing", user_id=user_id, org_id=org_id, session=session, service="Knowledge Base")
        self.logger.info(f"Embedding charge: {embedding_charge}")

        def convert_uuids_to_strings(data):
          if isinstance(data, dict):
            return {k: str(v) if isinstance(v, UUID) else convert_uuids_to_strings(v) if isinstance(v, (dict, list)) else v for k, v in data.items()}
          elif isinstance(data, list):
            return [
              str(item) if isinstance(item, UUID) else convert_uuids_to_strings(item) if isinstance(item, (dict, list)) else item for item in data
            ]
          return data

        # Create metadata with all UUIDs converted to strings
        metadata = {
          "operation": "document_embedding",
          "kb_id": str(kb_id),
          "doc_count": len(doc_ids),
          "related_charge_id": initial_charge_id,
          "user_id": str(user_id),
        }

        # Convert any potential nested UUIDs
        metadata = convert_uuids_to_strings(metadata)

        await embedding_charge.create(
          qty=1,
          metadata=metadata,
        )
      embedding_charge_created = True
    except Exception as e:
      self.logger.error(f"Failed to create embedding charge: {str(e)}")
      await session.rollback()
      embedding_charge = None  # Add this line to ensure it's None after failed creation
      # Continue processing without billing

    # Process documents without depending on a successful charge creation
    self.logger.info("Processing documents")
    # Process documents one by one
    for doc_model in doc_models:
      self.logger.info(f"Processing document: {doc_model}")
      doc_model.indexing_status = DocumentStatus.PROCESSING
      session.add(doc_model)
      await session.commit()
      self.logger.info("Document added to session")

      try:
        # Get KB model
        kb_model = await self._get_kb(kb_id, org_id, session)
        if not kb_model:
          raise HTTPException(status_code=404, detail="Knowledge base not found")
        self.logger.info("Knowledge base found")
        # Estimate embedding cost based on document content
        if doc_model.content:
          token_count = len(doc_model.content.split())
          estimated_chunks = max(1, token_count // kb_model.settings["max_chunk_size"])
          self.logger.info(f"Estimated chunks: {estimated_chunks}")
          # Update charge with document-specific information if we have a charge
          if embedding_charge and embedding_charge_created and embedding_charge.transaction_id:
            try:
              await embedding_charge.calculate_and_update(
                content=doc_model.content,
                metadata={"doc_id": str(doc_model.id), "estimated_chunks": estimated_chunks, "token_count": token_count},
                status="processing",  # Still processing at this point
              )
              self.logger.info("Embedding charge updated")
            except Exception as e:
              self.logger.error(f"Failed to update embedding charge: {str(e)}")
              embedding_charge_created = False
        self.logger.info("Initializing processor")
        # Initialize processor
        processor = DocumentProcessor(
          embedding_model=kb_model.embedding_model,
          chunk_size=kb_model.settings["max_chunk_size"],
          chunk_overlap=kb_model.settings["chunk_overlap"],
        )
        self.logger.info("Processor initialized")
        # Process document
        await processor.process_document(
          kb_id=kb_model.id, documents=[doc_model], collection_name=str(kb_model.organization_id) + "_" + kb_model.name
        )
        self.logger.info("Document processed")
        # Get actual chunk count from metadata or estimate based on token count
        token_count = len(doc_model.content.split()) if doc_model.content else 0
        estimated_chunks = max(1, token_count // kb_model.settings["max_chunk_size"])
        actual_chunks = estimated_chunks  # Use estimate as actual since we can't get exact count
        self.logger.info("Actual chunks: {actual_chunks}")
        # Update status to completed
        doc_model.indexing_status = DocumentStatus.COMPLETED
        doc_model.indexing_completed_at = datetime.now()
        self.logger.info("Document status updated to completed")
        # Update metadata with chunk information
        metadata = doc_model.source_metadata or {}
        metadata["actual_chunks"] = actual_chunks
        flag_modified(doc_model, "source_metadata")
        doc_model.source_metadata = metadata
        self.logger.info("Document metadata updated")
        session.add(doc_model)
        await session.commit()
        self.logger.info("Document added to session")

        # Update charge with actual chunk information
        if embedding_charge and embedding_charge_created and embedding_charge.transaction_id:
          try:
            await embedding_charge.calculate_and_update(metadata={"actual_chunks": actual_chunks}, status="completed")
          except Exception as e:
            self.logger.error(f"Failed to update embedding charge: {str(e)}")
            embedding_charge_created = False
        self.logger.info("Embedding charge updated")
      except Exception as e:
        self.logger.info(f"Error processing document: {str(e)}")
        # Update status to failed
        doc_model.indexing_status = DocumentStatus.FAILED
        doc_model.error_message = str(e)
        session.add(doc_model)
        await session.commit()
        self.logger.info("Document added to session")
        # Update charge with failure information
        if embedding_charge and embedding_charge_created and embedding_charge.transaction_id:
          try:
            await embedding_charge.delete(reason=f"Error in document indexing: {str(e)}")
            self.logger.info(f"Released embedding charge for transaction {embedding_charge.transaction_id} due to error in indexing")
          except Exception as charge_error:
            self.logger.error(f"Failed to release embedding charge: {str(charge_error)}")
        self.logger.info("Embedding charge updated")
      # Broadcast status update
      await self.ws_manager.broadcast(
        org_id,
        {
          "id": str(doc_model.id),
          "indexing_status": doc_model.indexing_status.value,
        },
        "kb",
        "write",
      )

  async def _process_document_task(
    self,
    source_type: SourceTypeModel,
    doc: KBDocumentModel,
    session: AsyncSession,
    org_id: UUID,
    charge: Optional[Charge] = None,
    complete_charge: bool = True,
    **kwargs,
  ) -> None:
    self.logger.info(f"Starting document processing task for document ID: {doc.id}")

    try:
      # Get appropriate handler
      handler = get_source_handler(source_type.name, source_type.config_schema)
      self.logger.info(f"Handler obtained for source type: {source_type.name}")

      # Validate metadata
      await handler.validate_metadata(doc.source_metadata, **kwargs)
      self.logger.info(f"Metadata validated for document ID: {doc.id}")

      # Preprocess
      await handler.preprocess(doc, **kwargs)
      self.logger.info(f"Preprocessing completed for document ID: {doc.id}")

      # Extract content
      content = await handler.extract_content(
        doc,
        session=session,
        ws_manager=self.ws_manager,
        **kwargs,
      )
      self.logger.info(f"Content extracted for document ID: {doc.id}")

      # Update document with content
      doc.content = content
      doc.extraction_status = DocumentStatus.COMPLETED
      doc.extraction_completed_at = datetime.now()
      metadata = doc.source_metadata
      metadata["source_extraction_status"] = True
      flag_modified(doc, "source_metadata")
      doc.source_metadata = metadata
      session.add(doc)
      await session.commit()
      self.logger.info(f"Document updated and committed for document ID: {doc.id}")

      self.logger.info(f"Charge: {charge}")
      self.logger.info(f"Complete charge: {complete_charge}")
      # Finalize the charge for document extraction if it exists
      if charge:
        try:
          await charge.calculate_and_update(content=content, metadata=doc.source_metadata, status="completed")
          self.logger.info(f"Charge finalized for document ID: {doc.id}")
        except Exception as charge_error:
          self.logger.error(f"Failed to update charge for document ID: {doc.id}: {str(charge_error)}")

    except Exception as e:
      doc.extraction_status = DocumentStatus.FAILED
      doc.error_message = str(e)
      session.add(doc)
      await session.commit()
      self.logger.error(f"Error processing document ID: {doc.id}: {str(e)}")

      # If we have a charge and extraction failed, release part of the hold
      if charge:
        try:
          await charge.delete(reason=f"Error in document processing: {str(e)}")
          self.logger.info(f"Released charge for transaction {charge.transaction_id} due to error in processing")
        except Exception as charge_error:
          self.logger.error(f"Failed to release charge for document ID: {doc.id}: {str(charge_error)}")

    finally:
      await handler.cleanup(doc)
      self.logger.info(f"Cleanup completed for document ID: {doc.id}")
      await self.ws_manager.broadcast(
        kwargs.get("org_id"),
        {"id": str(doc.id), "extraction_status": doc.extraction_status, "source_extraction_status": True},
        "kb",
        "write",
      )
      self.logger.info(f"Broadcast completed for document ID: {doc.id}")
