from datetime import datetime
from io import BytesIO
from typing import Annotated, List
from uuid import UUID

from fastapi import BackgroundTasks, Body, Depends, File, Form, HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_engine, get_db
from dependencies.security import RBAC
from libs.s3 import s3_client
from libs.vectorstore import create_vectorstore
from services.__base.acquire import Acquire

from .loaders import DocumentProcessor
from .model import KBDocumentModel, KnowledgeBaseModel, ProcessingStatus
from .schema import (
  DocumentChunk,
  DocumentChunkDelete,
  DocumentChunkUpdate,
  KBDocumentChunksResponse,
  KBDocumentResponse,
  KnowledgeBaseCreate,
  KnowledgeBaseDetailResponse,
  KnowledgeBaseResponse,
  KnowledgeBaseUpdate,
  validate_file_extension,
)


class KnowledgeBaseService:
  """Knowledge base service."""

  http_exposed = [
    "post=create",
    "put=update",
    "delete=remove",
    "get=get",
    "get=list",
    "post=add_document",
    "delete=remove_document",
    "get=get_document_chunks",
    "put=update_document_chunk",
    "delete=delete_document_chunks",
    "post=search_chunks",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
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
    return list(result.scalars().all())

  async def post_add_document(
    self,
    org_id: UUID,
    kb_id: UUID,
    title: Annotated[str, Form(min_length=1, max_length=200, description="Document title", examples=["My Document", "Project Report"])],
    doc_type: Annotated[
      int,  # Changed from DocumentType to int for form data
      Form(description="Document type (0: FILE, 1: URL)", examples=[0, 1]),
    ],
    file: Annotated[UploadFile, File(description="Document file to upload")],
    background_tasks: BackgroundTasks,
    description: Annotated[str, Form(description="Optional document description")] = "",
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "write")),
  ) -> KBDocumentResponse:
    """Add a document to a knowledge base."""
    match doc_type:
      case 0:
        # Validate file extension
        if file.filename:
          validate_file_extension(file.filename)
        else:
          raise HTTPException(status_code=400, detail="File name is required")

        file_type = file.filename.split(".")[-1]
        s3_key = f"kb/{kb_id}/{self.utils.generate_unique_filename(file.filename)}"

      case 1:
        # TODO: add url to the knowledge base
        pass

      # Verify knowledge base exists
    db_kb = await self._get_kb(kb_id, org_id, session)
    if not db_kb:
      raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Upload file to S3
    try:
      file_content = await file.read()
      await s3_client.upload_file(BytesIO(file_content), s3_key, content_type=file.content_type)
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

    # Create document
    db_doc = KBDocumentModel(
      kb_id=kb_id,
      s3_key=s3_key,
      original_filename=file.filename if doc_type == 0 else None,
      file_size=file.size if doc_type == 0 else None,
      processing_status=ProcessingStatus.PENDING,
      last_processed_at=None,
      title=title,
      description=description,
      doc_type=doc_type,
      file_type=file_type if doc_type == 0 else None,
    )
    session.add(db_doc)
    await session.commit()
    await session.refresh(db_doc)

    # create a background task to process
    background_tasks.add_task(self._process_document_task, db_doc.id, org_id, kb_id)

    return KBDocumentResponse.model_validate(db_doc)

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
        chunk = DocumentChunk(chunk_id=row.cmetadata.get("chunk_index"), content=row.document, metadata=row.cmetadata)
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
    if db_doc.s3_key:
      try:
        await s3_client.delete_file(db_doc.s3_key)
      except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    await session.delete(db_doc)
    await session.commit()
    return {"message": "Document deleted successfully"}

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
      print(new_embedding)

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
      print(update_query)

      result = await session.execute(update_query)

      updated_chunk = result.first()
      if not updated_chunk:
        raise HTTPException(status_code=404, detail="Failed to update chunk")

      await session.commit()

      return DocumentChunk(chunk_id=updated_chunk.id, content=updated_chunk.document, metadata=updated_chunk.cmetadata)

    except Exception as e:
      print(e)
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

  async def _process_document_task(
    self,
    doc_id: UUID,
    org_id: UUID,
    kb_id: UUID,
  ) -> None:
    """Background task to process document."""
    try:
      # Create dedicated session for background task
      async with self.acquire.db_session() as session:
        doc_model = await self._get_document(doc_id, org_id, session)
        if not doc_model:
          raise HTTPException(status_code=404, detail="Document not found")
        # set document to processing
        doc_model.processing_status = ProcessingStatus.PROCESSING
        session.add(doc_model)
        await session.commit()
        try:
          # get kb model
          kb_model = await self._get_kb(kb_id, org_id, session)
          if not kb_model:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

          # Initialize processor
          processor = DocumentProcessor(embedding_model=kb_model.embedding_model)
          # TODO : optimizations, can we not just pass the collection_name, and file in bytes becuase its just vectorizing 1 file data.
          # Process document
          await processor.process_document(kb_id=kb_model.id, document=doc_model, collection_name=str(kb_model.organization_id) + "_" + kb_model.name)

          # Update status to completed
          doc_model.processing_status = ProcessingStatus.COMPLETED
          # TODO: need to check what is right? using timezone aware datetime or not?
          doc_model.last_processed_at = datetime.now()
          session.add(doc_model)
          await session.commit()

        except Exception:
          # Update status to failed
          doc_model.processing_status = ProcessingStatus.FAILED
          session.add(doc_model)
          await session.commit()
          import traceback

          traceback.print_exc()

    except Exception:
      raise
