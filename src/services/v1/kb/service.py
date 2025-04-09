import uuid
from datetime import datetime
from io import BytesIO
from typing import Annotated, List
from uuid import UUID

from fastapi import BackgroundTasks, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from database import async_engine, get_db
from dependencies.security import RBAC
from libs.s3.v1 import s3_client
from libs.vectorstore.v1 import create_vectorstore
from models import DocumentStatus, KBDocumentModel, KnowledgeBaseModel, SourceTypeModel
from services.__base.acquire import Acquire

from .loaders import DocumentProcessor
from .schema import (
  DocumentChunk,
  DocumentChunkCreate,
  DocumentChunkDelete,
  DocumentChunkUpdate,
  FileDocumentData,
  KBDocumentChunksResponse,
  KBDocumentResponse,
  KnowledgeBaseCreate,
  KnowledgeBaseDetailResponse,
  KnowledgeBaseResponse,
  KnowledgeBaseUpdate,
  URLDocumentData,
  validate_file_document_data,
)
from .source_handlers import get_source_handler


class KnowledgeBaseService:
  """Knowledge base service."""

  http_exposed = [
    "post=create",
    "put=update",
    "delete=remove",
    "get=get",
    "get=list",
    "delete=remove_document",
    "get=get_document_chunks",
    "put=update_document_chunk",
    "delete=delete_document_chunks",
    "post=search_chunks",
    "post=add_url_document",
    "post=add_file_document",
    "get=get_document_content",
    "post=index_documents",
    "post=add_document_chunk",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.settings = acquire.settings
    self.ws_manager = acquire.ws_manager
    self.utils = acquire.utils

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
  ) -> KBDocumentResponse:
    """Add file document to knowledge base."""
    try:
      # TODO: switch this logic to pre-process in handler
      # get source type model
      print(document_data)
      print(document_data.source_id)
      source_type_model = await session.get(SourceTypeModel, 1)
      if not source_type_model:
        raise HTTPException(status_code=404, detail="Source type not found")
      # extract file metadata
      file_metadata = document_data.get_metadata()
      # upload file to s3
      config_schema = source_type_model.config_schema
      storage = config_schema.get("storage")
      if not storage:
        raise HTTPException(status_code=400, detail="Storage not found in source type config")
      s3_key = f"{storage['bucket']}/{storage['path']}/{org_id}/{kb_id}/{str(uuid.uuid4())}.{file_metadata['file_type']}"
      try:
        file_content = await document_data.file.read()
        await s3_client.upload_file(BytesIO(file_content), s3_key, content_type=document_data.file.content_type)
      except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

      # update file metadata with s3 key
      file_metadata["s3_key"] = s3_key
      print(file_metadata)
      # Create document
      db_doc = KBDocumentModel(
        title=document_data.title,
        description=document_data.description,
        kb_id=kb_id,
        source_type_id=1,  # File type
        source_id=document_data.source_id,
        source_metadata=file_metadata,
        extraction_status=DocumentStatus.PENDING,
        indexing_status=DocumentStatus.PENDING,
      )

      session.add(db_doc)
      await session.commit()
      await session.refresh(db_doc)

      # Start background processing
      background_tasks.add_task(self._process_document_task, source_type=source_type_model, doc=db_doc, session=session, org_id=org_id)

      return KBDocumentResponse.model_validate(db_doc)

    except Exception as e:
      raise HTTPException(status_code=400, detail=str(e))

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
        ORDER BY cmetadata->>'chunk_index'
        LIMIT :limit OFFSET :offset
      """)
      result = await session.execute(query, {"collection_id": kb_model.collection_id, "doc_id": str(doc_id), "limit": limit, "offset": offset})

      chunks = []
      for row in result:
        chunk = DocumentChunk(id=row.id, chunk_id=row.cmetadata.get("chunk_index"), content=row.document, metadata=row.cmetadata)
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
        id=updated_chunk.id, chunk_id=updated_chunk.cmetadata.get("chunk_index"), content=updated_chunk.document, metadata=updated_chunk.cmetadata
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
          chunk_id=int(doc.metadata.get("chunk_index", 0)),  # Convert to int with default 0
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
    new_chunk_index = max_index

    # Create metadata for the new chunk
    metadata = {**doc_model.source_metadata, "doc_id": str(doc_id), "chunk_index": new_chunk_index}

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
  ) -> None:
    """Background task to process document."""
    doc_models = await self._get_documents(doc_ids, org_id, session)
    if not doc_models:
      raise HTTPException(status_code=404, detail="Documents not found")

    # start processing documents one by one
    for doc_model in doc_models:
      doc_model.indexing_status = DocumentStatus.PROCESSING
      session.add(doc_model)

      await session.commit()
      try:
        # get kb model
        kb_model = await self._get_kb(kb_id, org_id, session)
        if not kb_model:
          raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Initialize processor
        processor = DocumentProcessor(
          embedding_model=kb_model.embedding_model,
          chunk_size=kb_model.settings["max_chunk_size"],
          chunk_overlap=kb_model.settings["chunk_overlap"],
        )
        # Process document
        await processor.process_document(
          kb_id=kb_model.id, documents=[doc_model], collection_name=str(kb_model.organization_id) + "_" + kb_model.name
        )

        # Update status to completed
        doc_model.indexing_status = DocumentStatus.COMPLETED
        # TODO: need to check what is right? using timezone aware datetime or not?
        doc_model.indexing_completed_at = datetime.now()
        session.add(doc_model)
        await session.commit()

      except Exception:
        # Update status to failed
        doc_model.indexing_status = DocumentStatus.FAILED
        session.add(doc_model)
        await session.commit()

      await self.ws_manager.broadcast(
        org_id,
        {
          "id": str(doc_model.id),
          "indexing_status": doc_model.indexing_status.value,
        },
        "kb",
        "write",
      )

  async def _process_document_task(self, source_type: SourceTypeModel, doc: KBDocumentModel, session: AsyncSession, **kwargs) -> None:
    # Get appropriate handler
    handler = get_source_handler(source_type.name, source_type.config_schema)

    try:
      # Validate metadata
      await handler.validate_metadata(doc.source_metadata, **kwargs)

      # Preprocess
      await handler.preprocess(doc, **kwargs)

      # Extract content
      content = await handler.extract_content(
        doc,
        session=session,  # incase we have children documents to create i.e. urlhandler crawl jobs
        ws_manager=self.ws_manager,  # to send the notifications to the user via websocket for children documents
        **kwargs,
      )
      # TODO: transfer this logic to the handler
      doc.content = content
      doc.extraction_status = DocumentStatus.COMPLETED
      doc.extraction_completed_at = datetime.now()
      metadata = doc.source_metadata
      metadata["source_extraction_status"] = True
      flag_modified(doc, "source_metadata")
      doc.source_metadata = metadata
      session.add(doc)  # Mark as modified
      await session.commit()

    except Exception as e:
      doc.extraction_status = DocumentStatus.FAILED
      doc.error_message = str(e)
      session.add(doc)  # Mark as modified
      await session.commit()

    finally:
      await handler.cleanup(doc)
      # send the notification to the user via websocket
      await self.ws_manager.broadcast(
        kwargs.get("org_id"),
        {"id": str(doc.id), "extraction_status": doc.extraction_status, "source_extraction_status": True},
        "kb",
        "write",
      )
