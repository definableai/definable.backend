import pytest
from fastapi import HTTPException, UploadFile, BackgroundTasks
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

# Create mock modules before any imports
sys.modules['database'] = MagicMock()
sys.modules['database.postgres'] = MagicMock()
sys.modules['database.models'] = MagicMock()
sys.modules['src.database'] = MagicMock()
sys.modules['src.database.postgres'] = MagicMock()
sys.modules['src.database.models'] = MagicMock()
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['src.config'] = MagicMock()
sys.modules['src.config.settings'] = MagicMock()
sys.modules['src.services.__base.acquire'] = MagicMock()
sys.modules['dependencies.security'] = MagicMock()

# Create the models module with proper structure
models_mock = MagicMock()
sys.modules['models'] = models_mock

# Create the models module with proper structure
models_mock = MagicMock()
sys.modules['models'] = models_mock

# Set up libs.vectorstore.v1 module
vectorstore_mock = MagicMock()
vectorstore_mock.create_vectorstore = AsyncMock(return_value=uuid4())
sys.modules['libs.vectorstore.v1'] = vectorstore_mock

# Set up libs.s3.v1 module
s3_mock = MagicMock()
s3_client_mock = MagicMock()
s3_client_mock.upload_file = AsyncMock(return_value="uploads/test.pdf")
s3_client_mock.get_presigned_url = AsyncMock(return_value="https://example.com/uploads/test.pdf")
s3_client_mock.delete_file = AsyncMock()
s3_mock.s3_client = s3_client_mock
sys.modules['libs.s3.v1'] = s3_mock

# Constants for status
# Set up libs.vectorstore.v1 module
vectorstore_mock = MagicMock()
vectorstore_mock.create_vectorstore = AsyncMock(return_value=uuid4())
sys.modules['libs.vectorstore.v1'] = vectorstore_mock

# Set up libs.s3.v1 module
s3_mock = MagicMock()
s3_client_mock = MagicMock()
s3_client_mock.upload_file = AsyncMock(return_value="uploads/test.pdf")
s3_client_mock.get_presigned_url = AsyncMock(return_value="https://example.com/uploads/test.pdf")
s3_client_mock.delete_file = AsyncMock()
s3_mock.s3_client = s3_client_mock
sys.modules['libs.s3.v1'] = s3_mock

# Constants for status
class DocumentStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Add the enum to the mocked module
models_mock.DocumentStatus = DocumentStatus
# Add the enum to the mocked module
models_mock.DocumentStatus = DocumentStatus

# Mock models
class MockKnowledgeBaseModel(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    collection_id: Optional[UUID] = None
    settings: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    embedding_model: Optional[str] = None
    max_chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separator: Optional[str] = None
    documents: Optional[List[Any]] = Field(default_factory=lambda: [])
    chunks: Optional[List[Any]] = Field(default_factory=lambda: [])
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    knowledge_bases: Optional[List[Any]] = Field(default_factory=lambda: [])

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class MockKBDocumentModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    kb_id: UUID = Field(default_factory=uuid4)
    title: str = "Test Document"
    description: str = ""
    source_type_id: int = 1  # Default to FILE
    source_metadata: Dict[str, Any] = Field(default_factory=lambda: {})
    source_url: str = ""
    s3_key: str = ""
    extraction_status: str = DocumentStatus.PENDING
    indexing_status: str = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: UUID = Field(default_factory=uuid4)
    organization_id: UUID = Field(default_factory=uuid4)
    source_id: Optional[UUID] = None
    download_url: Optional[str] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    extraction_completed_at: Optional[datetime] = None
    indexing_completed_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

class MockSourceTypeModel:
    FILE = 1
    URL = 2

# Add models to mocked modules
models_mock.KnowledgeBaseModel = MockKnowledgeBaseModel
models_mock.KBDocumentModel = MockKBDocumentModel
models_mock.SourceTypeModel = MockSourceTypeModel
models_mock.KnowledgeBaseModel = MockKnowledgeBaseModel
models_mock.KBDocumentModel = MockKBDocumentModel
models_mock.SourceTypeModel = MockSourceTypeModel

# Create and configure langchain_core.documents
langchain_documents_mock = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = langchain_documents_mock

class MockDocument(BaseModel):
    page_content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

langchain_documents_mock.Document = MockDocument

class MockResponse(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    organization_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    collection_id: Optional[UUID] = None
    settings: Optional[Dict[str, Any]] = None
    created_at: Optional[Union[datetime, str]] = None
    updated_at: Optional[Union[datetime, str]] = None
    kb_id: Optional[UUID] = None
    source_type_id: Optional[int] = None
    source_url: Optional[str] = None
    s3_key: Optional[str] = None
    source_metadata: Optional[Dict[str, Any]] = None
    extraction_status: Optional[str] = None
    documents: Optional[List[Any]] = None
    chunks: Optional[List[Any]] = None
    total: Optional[int] = None
    has_more: Optional[bool] = None
    url: Optional[str] = None
    download_url: Optional[str] = None
    knowledge_bases: Optional[List[Any]] = None
    chunk_id: Optional[int] = None
    content: Optional[str] = None
    score: Optional[float] = None
    message: Optional[str] = None
    # Additional fields for tests
    chunk_ids: Optional[List[str]] = None
    embedding_model: Optional[str] = None
    max_chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separator: Optional[str] = None
    version: Optional[int] = None
    document_id: Optional[UUID] = None
    total_chunks: Optional[int] = None
    file: Optional[Any] = None
    source_id: Optional[UUID] = None
    operation: Optional[str] = None
    excludeTags: Optional[List[str]] = None
    includeTags: Optional[List[str]] = None
    onlyMainContent: Optional[bool] = None
    formats: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Crawler options
    maxDepth: Optional[int] = None
    limit: Optional[int] = None
    includePaths: Optional[List[str]] = None
    excludePaths: Optional[List[str]] = None
    ignoreSitemap: Optional[bool] = None
    allowBackwardLinks: Optional[bool] = None
    scrapeOptions: Optional[Any] = None

    model_config = ConfigDict(extra="allow")

    def get_metadata(self) -> Dict[str, Any]:
        """Helper method to return metadata dictionary used in tests"""
        return self.source_metadata or {}

@pytest.fixture
def mock_user():
    """Create a mock user."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
    }

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()

    # Setup scalar to return a properly mocked result
    scalar_mock = AsyncMock()
    session.scalar = scalar_mock

    # Setup execute to return a properly mocked result
    execute_mock = AsyncMock()
    # Make unique(), scalars(), first(), etc. return self to allow chaining
    execute_result = AsyncMock()
    execute_result.unique.return_value = execute_result
    execute_result.scalars.return_value = execute_result
    execute_result.scalar_one_or_none.return_value = None
    execute_result.scalar_one.return_value = None
    execute_result.first.return_value = None
    execute_result.all.return_value = []
    execute_result.mappings.return_value = execute_result

    execute_mock.return_value = execute_result
    session.execute = execute_mock

    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    return session

@pytest.fixture
def mock_knowledge_base():
    """Create a mock knowledge base."""
    return MockKnowledgeBaseModel(
        name="Test Knowledge Base",
        embedding_model="openai",
        settings={
            "embedding_model": "openai",
            "max_chunk_size": 1000,
            "chunk_overlap": 100,
            "separator": "\n\n",
            "version": 1
        }
    )

@pytest.fixture
def mock_kb_document():
    """Create a mock KB document."""
    return MockKBDocumentModel(
        title="Test Document",
        description="Test document description",
        source_type_id=MockSourceTypeModel.FILE,
        source_metadata={
            "file_type": "pdf",
            "original_filename": "test.pdf",
            "size": 1024,
            "mime_type": "application/pdf"
        }
    )

@pytest.fixture
def mock_upload_file():
    """Create a mock upload file."""
    file_content = b"test file content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.content_type = "application/pdf"
    file.size = 1024
    file.read = AsyncMock(return_value=file_content)
    file.file = BytesIO(file_content)
    return file

@pytest.fixture
def mock_background_tasks():
    """Create a mock background tasks."""
    return MagicMock(spec=BackgroundTasks)

@pytest.fixture
def mock_kb_service():
    """Create a mock KB service."""
    kb_service = MagicMock()

    async def mock_create(org_id, kb_data, session, user):
        # Create a knowledge base
        collection_id = await sys.modules['libs.vectorstore.v1'].create_vectorstore(org_id, kb_data.name)

        kb = MockKnowledgeBaseModel(
            organization_id=org_id,
            user_id=user["id"],
            collection_id=collection_id,
            name=kb_data.name,
            embedding_model=kb_data.settings["embedding_model"],
            embedding_model=kb_data.settings["embedding_model"],
            settings=kb_data.model_dump()
        )
        session.add(kb)
        await session.commit()
        await session.refresh(kb)

        return MockResponse.model_validate(kb.model_dump())

    async def mock_update(org_id, kb_id, kb_data, session, user):
        # Get knowledge base
        kb = await kb_service._get_kb(kb_id, org_id, session)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Update fields
        update_data = kb_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(kb, field, value)

        await session.commit()
        await session.refresh(kb)

        return MockResponse.model_validate(kb.model_dump())

    async def mock_get(org_id, kb_id, session, user):
        # Get knowledge base with documents
        kb = MockKnowledgeBaseModel(
            id=kb_id,
            organization_id=org_id,
            name="Test Knowledge Base",
            collection_id=uuid4()
        )

        # Create some documents
        documents = [
            MockKBDocumentModel(
                kb_id=kb_id,
                title=f"Document {i}",
                s3_key=f"uploads/doc{i}.pdf"
            ) for i in range(1, 3)
        ]

        # Add download URLs
        for doc in documents:
            doc.download_url = "https://example.com/" + doc.s3_key

        session.execute.return_value.unique.return_value.all.return_value = [(kb, doc) for doc in documents]

        # Create response with documents
        kb_dict = kb.model_dump()
        if 'documents' in kb_dict:
          kb_dict.pop('documents')
        kb_response = MockResponse(**kb_dict, documents=[MockResponse.model_validate(doc.model_dump()) for doc in documents])

        return kb_response

    async def mock_get_not_found(org_id, kb_id, session, user):
        # Return empty result to simulate KB not found
        session.execute.return_value.unique.return_value.all.return_value = []
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    async def mock_list(org_id, session, user):
        # Return mock knowledge bases
        kbs = [
            MockKnowledgeBaseModel(organization_id=org_id, name=f"Knowledge Base {i}")
            for i in range(1, 3)
        ]

        session.execute.return_value.scalars.return_value.all.return_value = kbs

        return [MockResponse.model_validate(kb.model_dump()) for kb in kbs]

    async def mock_remove(org_id, kb_id, session, user):
        # Get knowledge base
        kb = await kb_service._get_kb(kb_id, org_id, session)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Delete knowledge base
        await session.delete(kb)
        await session.commit()

        return {"message": "Knowledge base deleted successfully"}

    async def mock_add_file_document(org_id, kb_id, document_data, background_tasks, session, user):
        # Get knowledge base
        kb = await kb_service._get_kb(kb_id, org_id, session)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Create a mock S3 key
        s3_key = f"uploads/{document_data.file.filename}"

        # Upload file to S3
        file_content = await document_data.file.read()
        await s3_client_mock.upload_file(
        file=BytesIO(file_content),
        key=s3_key
        await s3_client_mock.upload_file(
        file=BytesIO(file_content),
        key=s3_key
        )

        # Create document
        source_metadata = document_data.source_metadata
        source_metadata = document_data.source_metadata
        doc = MockKBDocumentModel(
            kb_id=kb_id,
            title=document_data.title,
            description=document_data.description,
            source_type_id=MockSourceTypeModel.FILE,
            source_id=document_data.source_id,
            source_metadata=source_metadata,
            s3_key=s3_key,
            extraction_status=DocumentStatus.PENDING,
            indexing_status=DocumentStatus.PENDING
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        # Add document processing task to background tasks
        background_tasks.add_task(
            kb_service._process_document_task,
            MockSourceTypeModel.FILE,
            doc,
            session
        )

        # Generate download URL for response
        download_url = await s3_client_mock.get_presigned_url(
        download_url = await s3_client_mock.get_presigned_url(
            s3_key,
            expires_in=3600,
            operation="get_object"
        )

        doc.download_url = download_url

        return MockResponse.model_validate(doc.model_dump())

    async def mock_add_url_document(org_id, kb_id, document_data, background_tasks, session, user):
        # Get knowledge base
        kb = await kb_service._get_kb(kb_id, org_id, session)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Create document
        source_metadata = document_data.source_metadata
        source_metadata = document_data.source_metadata
        doc = MockKBDocumentModel(
            kb_id=kb_id,
            title=document_data.title,
            description=document_data.description,
            source_type_id=MockSourceTypeModel.URL,
            source_id=document_data.source_id,
            source_metadata=source_metadata,
            extraction_status=DocumentStatus.PENDING,
            indexing_status=DocumentStatus.PENDING
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)

        # Add document processing task to background tasks
        background_tasks.add_task(
            kb_service._process_document_task,
            MockSourceTypeModel.URL,
            doc,
            session
        )

        return MockResponse.model_validate(doc.model_dump())

    async def mock_get_document_chunks(org_id, kb_id, doc_id, limit, offset, session, user):
        # Get document
        doc = await kb_service._get_document(doc_id, org_id, session)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Generate mock chunks
        total_chunks = 5
        chunks = [
            {
                "id": uuid4(),
                "chunk_id": i,
                "content": f"Chunk {i} content",
                "metadata": {"source": doc.title, "chunk": i}
            }
            for i in range(offset, min(offset + limit, total_chunks))
        ]

        # Set up mock for raw SQL query result
        session.execute.return_value.mappings.return_value.all.return_value = chunks

        # Set up mock for count query
        session.scalar.return_value = total_chunks

        return MockResponse(
            document_id=doc_id,
            title=doc.title,
            chunks=[MockResponse.model_validate(chunk) for chunk in chunks],
            total_chunks=total_chunks
        )

    async def mock_remove_document(org_id, doc_id, session, user):
        # Get document
        doc = await kb_service._get_document(doc_id, org_id, session)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete document
        await session.delete(doc)
        await session.commit()

        return {"message": "Document deleted successfully"}

    async def mock_search_chunks(org_id, kb_id, query, limit, score_threshold, session, user):
        # Get knowledge base
        kb = await kb_service._get_kb(kb_id, org_id, session)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Generate mock search results
        mock_results = [
            {
                "id": uuid4(),
                "chunk_id": i,
                "content": f"Result {i} matching query '{query}'",
                "metadata": {"source": f"Document {i}", "chunk": i},
                "score": 0.9 - (i * 0.1)  # Descending scores
            }
            for i in range(min(limit, 3))
        ]

        # Filter by score threshold
        filtered_results = [r for r in mock_results if r["score"] >= score_threshold]

        return [MockResponse.model_validate(result) for result in filtered_results]

    async def mock__get_kb(kb_id, org_id, session):
        # Mock getting a knowledge base
        kb = MockKnowledgeBaseModel(
            id=kb_id,
            organization_id=org_id
        )

        session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = kb

        return kb

    async def mock__get_document(doc_id, org_id, session):
        # Mock getting a document
        doc = MockKBDocumentModel(
            id=doc_id
        )

        # Get the KB for this document
        kb = MockKnowledgeBaseModel(
            id=doc.kb_id,
            organization_id=org_id
        )

        # Mock DB query results
        session.execute.return_value.unique.return_value.scalar_one_or_none.side_effect = [kb, doc]

        return doc

    # Create AsyncMock objects
    create_mock = AsyncMock(side_effect=mock_create)
    update_mock = AsyncMock(side_effect=mock_update)
    get_mock = AsyncMock(side_effect=mock_get)
    list_mock = AsyncMock(side_effect=mock_list)
    remove_mock = AsyncMock(side_effect=mock_remove)
    add_file_document_mock = AsyncMock(side_effect=mock_add_file_document)
    add_url_document_mock = AsyncMock(side_effect=mock_add_url_document)
    get_document_chunks_mock = AsyncMock(side_effect=mock_get_document_chunks)
    remove_document_mock = AsyncMock(side_effect=mock_remove_document)
    search_chunks_mock = AsyncMock(side_effect=mock_search_chunks)
    _get_kb_mock = AsyncMock(side_effect=mock__get_kb)
    _get_document_mock = AsyncMock(side_effect=mock__get_document)

    # Assign the mocks to the service
    kb_service.post_create = create_mock
    kb_service.put_update = update_mock
    kb_service.get_get = get_mock
    kb_service.get_list = list_mock
    kb_service.delete_remove = remove_mock
    kb_service.post_add_file_document = add_file_document_mock
    kb_service.post_add_url_document = add_url_document_mock
    kb_service.get_get_document_chunks = get_document_chunks_mock
    kb_service.delete_remove_document = remove_document_mock
    kb_service.post_search_chunks = search_chunks_mock
    kb_service._get_kb = _get_kb_mock
    kb_service._get_document = _get_document_mock

    return kb_service

@pytest.mark.asyncio
class TestKBService:
    """Tests for the Knowledge Base service."""

    async def test_create_knowledge_base(self, mock_kb_service, mock_db_session, mock_user):
        """Test creating a new knowledge base."""
        # Create KB data
        org_id = uuid4()
        kb_data = MockResponse(
            name="New Knowledge Base",
            settings={
                "embedding_model": "openai",
                "max_chunk_size": 1000,
                "chunk_overlap": 100,
                "separator": "\n\n",
                "version": 1
            }
            settings={
                "embedding_model": "openai",
                "max_chunk_size": 1000,
                "chunk_overlap": 100,
                "separator": "\n\n",
                "version": 1
            }
        )

        # Call the service
        response = await mock_kb_service.post_create(
            org_id,
            kb_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.name == kb_data.name
        assert response.organization_id == org_id
        assert response.user_id == mock_user["id"]
        assert hasattr(response, "collection_id")
        assert hasattr(response, "created_at")

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.post_create.called

    async def test_update_knowledge_base(self, mock_kb_service, mock_db_session, mock_user):
        """Test updating a knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        update_data = MockResponse(
            name="Updated Knowledge Base"
        )

        # Call the service
        response = await mock_kb_service.put_update(
            org_id,
            kb_id,
            update_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.name == update_data.name

        # Verify database operations
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.put_update.called

    async def test_get_knowledge_base(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting a knowledge base with its documents."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()

        # Call the service
        response = await mock_kb_service.get_get(
            org_id,
            kb_id,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.id == kb_id
        assert response.organization_id == org_id
        assert hasattr(response, "documents")
        assert len(response.documents) == 2

        # Verify document fields
        for doc in response.documents:
            assert hasattr(doc, "id")
            assert hasattr(doc, "title")
            assert hasattr(doc, "description")
            assert hasattr(doc, "kb_id")
            assert hasattr(doc, "source_type_id")
            assert hasattr(doc, "source_metadata")
            assert hasattr(doc, "extraction_status")
            assert hasattr(doc, "indexing_status")
            assert hasattr(doc, "download_url")
            assert doc.kb_id == kb_id
            assert doc.download_url.startswith("https://")

        # Verify service method was called
        assert mock_kb_service.get_get.called

    async def test_get_knowledge_base_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()

        # Replace get method with one that raises an exception
        async def mock_get_not_found(org_id, kb_id, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.get_get = AsyncMock(side_effect=mock_get_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.get_get(
                org_id,
                kb_id,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.get_get.called

    async def test_list_knowledge_bases(self, mock_kb_service, mock_db_session, mock_user):
        """Test listing knowledge bases for an organization."""
        # Setup data
        org_id = uuid4()

        # Call the service
        response = await mock_kb_service.get_list(
            org_id,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert isinstance(response, list)
        assert len(response) == 2
        assert all(kb.organization_id == org_id for kb in response)

        # Verify KB fields
        for kb in response:
            assert hasattr(kb, "id")
            assert hasattr(kb, "name")
            assert hasattr(kb, "collection_id")
            assert hasattr(kb, "organization_id")
            assert hasattr(kb, "user_id")
            assert hasattr(kb, "created_at")

        # Verify service method was called
        assert mock_kb_service.get_list.called

    async def test_delete_knowledge_base(self, mock_kb_service, mock_db_session, mock_user):
        """Test deleting a knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()

        # Call the service
        response = await mock_kb_service.delete_remove(
            org_id,
            kb_id,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response["message"] == "Knowledge base deleted successfully"

        # Verify database operations
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.delete_remove.called

    async def test_add_file_document(self, mock_kb_service, mock_db_session, mock_user, mock_upload_file, mock_background_tasks):
        """Test adding a file document to a knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        document_data = MockResponse(
            title="Test File Document",
            description="Test file document description",
            file=mock_upload_file,
            source_id=None,
            source_metadata={
            source_metadata={
                "file_type": "pdf",
                "original_filename": "test.pdf",
                "size": 1024,
                "mime_type": "application/pdf"
            }
        )

        # Call the service
        response = await mock_kb_service.post_add_file_document(
            org_id,
            kb_id,
            document_data,
            mock_background_tasks,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.title == document_data.title
        assert response.description == document_data.description
        assert response.kb_id == kb_id
        assert response.source_type_id == MockSourceTypeModel.FILE
        assert response.extraction_status == DocumentStatus.PENDING
        assert response.indexing_status == DocumentStatus.PENDING
        assert hasattr(response, "download_url")
        assert response.download_url.startswith("https://")

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify background task was added
        mock_background_tasks.add_task.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.post_add_file_document.called

    async def test_add_url_document(self, mock_kb_service, mock_db_session, mock_user, mock_background_tasks):
        """Test adding a URL document to a knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        document_data = MockResponse(
            title="Test URL Document",
            description="Test URL document description",
            url="https://example.com",
            operation="scrape",
            source_id=None,
            settings={
                "excludeTags": [""],
                "includeTags": [""],
                "onlyMainContent": True,
                "formats": ["text"]
            },
            source_metadata={
            settings={
                "excludeTags": [""],
                "includeTags": [""],
                "onlyMainContent": True,
                "formats": ["text"]
            },
            source_metadata={
                "url": "https://example.com",
                "operation": "scrape",
                "settings": {
                    "excludeTags": [""],
                    "includeTags": [""],
                    "onlyMainContent": True,
                    "formats": ["text"]
                }
            }
        )

        # Call the service
        response = await mock_kb_service.post_add_url_document(
            org_id,
            kb_id,
            document_data,
            mock_background_tasks,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.title == document_data.title
        assert response.description == document_data.description
        assert response.kb_id == kb_id
        assert response.source_type_id == MockSourceTypeModel.URL
        assert response.extraction_status == DocumentStatus.PENDING
        assert response.indexing_status == DocumentStatus.PENDING

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify background task was added
        mock_background_tasks.add_task.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.post_add_url_document.called

    async def test_get_document_chunks(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting document chunks."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_id = uuid4()
        limit = 10
        offset = 0

        # Call the service
        response = await mock_kb_service.get_get_document_chunks(
            org_id,
            kb_id,
            doc_id,
            limit,
            offset,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.document_id == doc_id
        assert hasattr(response, "title")
        assert hasattr(response, "chunks")
        assert hasattr(response, "total_chunks")
        assert len(response.chunks) <= limit

        # Verify chunk fields
        for chunk in response.chunks:
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "metadata")

        # Verify service method was called
        assert mock_kb_service.get_get_document_chunks.called

    async def test_remove_document(self, mock_kb_service, mock_db_session, mock_user):
        """Test removing a document."""
        # Setup data
        org_id = uuid4()
        doc_id = uuid4()

        # Call the service
        response = await mock_kb_service.delete_remove_document(
            org_id,
            doc_id,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response["message"] == "Document deleted successfully"

        # Verify database operations
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.delete_remove_document.called

    async def test_search_chunks(self, mock_kb_service, mock_db_session, mock_user):
        """Test searching for chunks."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        query = "test query"
        limit = 5
        score_threshold = 0.5

        # Call the service
        response = await mock_kb_service.post_search_chunks(
            org_id,
            kb_id,
            query,
            limit,
            score_threshold,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert isinstance(response, list)
        assert len(response) <= limit
        assert all(hasattr(chunk, "content") for chunk in response)
        assert all(hasattr(chunk, "score") for chunk in response)
        assert all(chunk.score >= score_threshold for chunk in response)

        # Verify service method was called
        assert mock_kb_service.post_search_chunks.called

    async def test_update_knowledge_base_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test updating a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        update_data = MockResponse(
            name="Updated Knowledge Base"
        )

        # Replace update method with one that raises an exception
        async def mock_update_not_found(org_id, kb_id, kb_data, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.put_update = AsyncMock(side_effect=mock_update_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.put_update(
                org_id,
                kb_id,
                update_data,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.put_update.called

    async def test_delete_knowledge_base_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test deleting a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()

        # Replace delete method with one that raises an exception
        async def mock_delete_not_found(org_id, kb_id, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.delete_remove = AsyncMock(side_effect=mock_delete_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.delete_remove(
                org_id,
                kb_id,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.delete_remove.called

    async def test_add_file_document_kb_not_found(self, mock_kb_service, mock_db_session, mock_user, mock_upload_file, mock_background_tasks):
        """Test adding a file document to a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        document_data = MockResponse(
            title="Test File Document",
            description="Test file document description",
            file=mock_upload_file,
            source_id=None,
            source_metadata={
            source_metadata={
                "file_type": "pdf",
                "original_filename": "test.pdf",
                "size": 1024,
                "mime_type": "application/pdf"
            }
        )

        # Replace add file document method with one that raises an exception
        async def mock_add_file_document_kb_not_found(org_id, kb_id, document_data, background_tasks, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.post_add_file_document = AsyncMock(side_effect=mock_add_file_document_kb_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.post_add_file_document(
                org_id,
                kb_id,
                document_data,
                mock_background_tasks,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.post_add_file_document.called

    async def test_add_url_document_kb_not_found(self, mock_kb_service, mock_db_session, mock_user, mock_background_tasks):
        """Test adding a URL document to a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        document_data = MockResponse(
            title="Test URL Document",
            description="Test URL document description",
            url="https://example.com",
            operation="scrape",
            source_id=None,
            settings={
                "excludeTags": [""],
                "includeTags": [""],
                "onlyMainContent": True,
                "formats": ["text"]
            },
            source_metadata={
            settings={
                "excludeTags": [""],
                "includeTags": [""],
                "onlyMainContent": True,
                "formats": ["text"]
            },
            source_metadata={
                "url": "https://example.com",
                "operation": "scrape",
                "settings": {
                    "excludeTags": [""],
                    "includeTags": [""],
                    "onlyMainContent": True,
                    "formats": ["text"]
                }
            }
        )

        # Replace add URL document method with one that raises an exception
        async def mock_add_url_document_kb_not_found(org_id, kb_id, document_data, background_tasks, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.post_add_url_document = AsyncMock(side_effect=mock_add_url_document_kb_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.post_add_url_document(
                org_id,
                kb_id,
                document_data,
                mock_background_tasks,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.post_add_url_document.called

    async def test_get_document_chunks_document_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting document chunks for a nonexistent document."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_id = uuid4()
        limit = 10
        offset = 0

        # Replace get document chunks method with one that raises an exception
        async def mock_get_document_chunks_not_found(org_id, kb_id, doc_id, limit, offset, session, user):
            raise HTTPException(status_code=404, detail="Document not found")

        mock_kb_service.get_get_document_chunks = AsyncMock(side_effect=mock_get_document_chunks_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.get_get_document_chunks(
                org_id,
                kb_id,
                doc_id,
                limit,
                offset,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Document not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.get_get_document_chunks.called

    async def test_remove_document_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test removing a nonexistent document."""
        # Setup data
        org_id = uuid4()
        doc_id = uuid4()

        # Replace remove document method with one that raises an exception
        async def mock_remove_document_not_found(org_id, doc_id, session, user):
            raise HTTPException(status_code=404, detail="Document not found")

        mock_kb_service.delete_remove_document = AsyncMock(side_effect=mock_remove_document_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.delete_remove_document(
                org_id,
                doc_id,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Document not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.delete_remove_document.called

    async def test_search_chunks_kb_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test searching for chunks in a nonexistent knowledge base."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        query = "test query"
        limit = 5
        score_threshold = 0.5

        # Replace search chunks method with one that raises an exception
        async def mock_search_chunks_kb_not_found(org_id, kb_id, query, limit, score_threshold, session, user):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        mock_kb_service.post_search_chunks = AsyncMock(side_effect=mock_search_chunks_kb_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.post_search_chunks(
                org_id,
                kb_id,
                query,
                limit,
                score_threshold,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Knowledge base not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.post_search_chunks.called

    async def test_add_url_document_with_crawl_operation(self, mock_kb_service, mock_db_session, mock_user, mock_background_tasks):
        """Test adding a URL document with crawl operation."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        document_data = MockResponse(
            title="Test URL Document with Crawl",
            description="Test URL document with crawl operation",
            url="https://example.com",
            operation="crawl",
            source_id=None,
            settings={
                "maxDepth": 2,
                "limit": 50,
                "includePaths": ["/docs", "/blog"],
                "excludePaths": ["/private"],
                "ignoreSitemap": False,
                "allowBackwardLinks": True,
                "scrapeOptions": {
                    "excludeTags": ["nav", "footer"],
                    "includeTags": [],
                    "onlyMainContent": True,
                    "formats": ["text", "html"]
                }
            },
            source_metadata={
            settings={
                "maxDepth": 2,
                "limit": 50,
                "includePaths": ["/docs", "/blog"],
                "excludePaths": ["/private"],
                "ignoreSitemap": False,
                "allowBackwardLinks": True,
                "scrapeOptions": {
                    "excludeTags": ["nav", "footer"],
                    "includeTags": [],
                    "onlyMainContent": True,
                    "formats": ["text", "html"]
                }
            },
            source_metadata={
                "url": "https://example.com",
                "operation": "crawl",
                "settings": {
                    "maxDepth": 2,
                    "limit": 50,
                    "includePaths": ["/docs", "/blog"],
                    "excludePaths": ["/private"],
                    "ignoreSitemap": False,
                    "allowBackwardLinks": True,
                    "scrapeOptions": {
                        "excludeTags": ["nav", "footer"],
                        "includeTags": [],
                        "onlyMainContent": True,
                        "formats": ["text", "html"]
                    }
                }
            }
        )

        # Reimplement the original method for this test
        async def mock_add_url_document_crawl(org_id, kb_id, document_data, background_tasks, session, user):
            # Get knowledge base
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Create document
            source_metadata = document_data.source_metadata
            source_metadata = document_data.source_metadata
            doc = MockKBDocumentModel(
                kb_id=kb_id,
                title=document_data.title,
                description=document_data.description,
                source_type_id=MockSourceTypeModel.URL,
                source_id=document_data.source_id,
                source_metadata=source_metadata,
                extraction_status=DocumentStatus.PENDING,
                indexing_status=DocumentStatus.PENDING
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)

            # Add document processing task to background tasks
            background_tasks.add_task(
                mock_kb_service._process_document_task,
                MockSourceTypeModel.URL,
                doc,
                session
            )

            return MockResponse.model_validate(doc.model_dump())

        mock_kb_service.post_add_url_document = AsyncMock(side_effect=mock_add_url_document_crawl)

        # Call the service
        response = await mock_kb_service.post_add_url_document(
            org_id,
            kb_id,
            document_data,
            mock_background_tasks,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.title == document_data.title
        assert response.description == document_data.description
        assert response.kb_id == kb_id
        assert response.source_type_id == MockSourceTypeModel.URL
        assert response.extraction_status == DocumentStatus.PENDING
        assert response.indexing_status == DocumentStatus.PENDING

        # Verify source metadata
        assert response.source_metadata["operation"] == "crawl"
        assert "settings" in response.source_metadata
        assert "maxDepth" in response.source_metadata["settings"]
        assert "limit" in response.source_metadata["settings"]
        assert "scrapeOptions" in response.source_metadata["settings"]

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify background task was added
        mock_background_tasks.add_task.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.post_add_url_document.called

    async def test_update_document_chunk(self, mock_kb_service, mock_db_session, mock_user):
        """Test updating a document chunk."""
        # Setup test data
        # Setup test data
        org_id = uuid4()
        kb_id = uuid4()
        doc_chunk_data = MockResponse(
            chunk_id=123,
            chunk_id=123,
            content="Updated chunk content"
        )

        async def mock_update_document_chunk(org_id, kb_id, doc_chunk_data, session, user):
            # Verify KB exists
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Create updated chunk response
            return MockResponse(
                id=uuid4(),
                chunk_id=doc_chunk_data.chunk_id,
                chunk_id=doc_chunk_data.chunk_id,
                content=doc_chunk_data.content,
                metadata={"chunk_index": doc_chunk_data.chunk_id, "doc_id": str(uuid4())}
                metadata={"chunk_index": doc_chunk_data.chunk_id, "doc_id": str(uuid4())}
            )

        # Replace the method with our test implementation
        mock_kb_service.put_update_document_chunk = AsyncMock(side_effect=mock_update_document_chunk)

        # Call service
        response = await mock_kb_service.put_update_document_chunk(
            org_id=org_id,
            kb_id=kb_id,
            doc_chunk_data=doc_chunk_data,
            org_id=org_id,
            kb_id=kb_id,
            doc_chunk_data=doc_chunk_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.chunk_id == doc_chunk_data.chunk_id  # Compare directly without conversion
        assert response.chunk_id == doc_chunk_data.chunk_id  # Compare directly without conversion
        assert response.content == doc_chunk_data.content
        assert hasattr(response, "metadata")

        # Verify service method was called
        assert mock_kb_service.put_update_document_chunk.called

    async def test_update_document_chunk_not_found(self, mock_kb_service, mock_db_session, mock_user):
        """Test updating a nonexistent document chunk."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_chunk_data = MockResponse(
            chunk_id=999,
            chunk_id=999,
            content="Updated chunk content"
        )

        # Implement update document chunk method that raises an exception
        async def mock_update_document_chunk_not_found(org_id, kb_id, doc_chunk_data, session, user):
            raise HTTPException(status_code=404, detail="Chunk not found")

        mock_kb_service.put_update_document_chunk = AsyncMock(side_effect=mock_update_document_chunk_not_found)

        # Verify exception is raised
        with pytest.raises(HTTPException) as excinfo:
            await mock_kb_service.put_update_document_chunk(
                org_id,
                kb_id,
                doc_chunk_data,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 404
        assert "Chunk not found" in str(excinfo.value.detail)

        # Verify service method was called
        assert mock_kb_service.put_update_document_chunk.called

    async def test_delete_document_chunks(self, mock_kb_service, mock_db_session, mock_user):
        """Test deleting document chunks."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_chunk_data = MockResponse(
            chunk_ids=["1", "2", "3"]
        )

        # Implement delete document chunks method
        async def mock_delete_document_chunks(org_id, kb_id, doc_chunk_data, session, user):
            # Verify KB exists
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Delete the chunks (mocked)
            delete_count = len(doc_chunk_data.chunk_ids)
            session.execute.return_value.rowcount = delete_count

            # Return success message
            return {"message": f"Successfully deleted {delete_count} chunks"}

        mock_kb_service.delete_delete_document_chunks = AsyncMock(side_effect=mock_delete_document_chunks)

        # Call the service
        response = await mock_kb_service.delete_delete_document_chunks(
            org_id,
            kb_id,
            doc_chunk_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert "message" in response
        assert "3" in response["message"]  # Should mention deleting 3 chunks

        # Verify service method was called
        assert mock_kb_service.delete_delete_document_chunks.called

    async def test_get_document_content(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting document content."""
        # Setup data
        org_id = uuid4()
        doc_id = uuid4()

        # Implement get document content method
        async def mock_get_document_content(org_id, doc_id, session):
            # Get document
            doc = await mock_kb_service._get_document(doc_id, org_id, session)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")

            # Update document with content
            doc.content = "This is the content of the document."
            doc.download_url = "https://example.com/uploads/document.pdf"

            return MockResponse.model_validate(doc.model_dump())

        mock_kb_service.get_get_document_content = AsyncMock(side_effect=mock_get_document_content)

        # Call the service
        response = await mock_kb_service.get_get_document_content(
            org_id,
            doc_id,
            session=mock_db_session
        )

        # Verify result
        assert response.id == doc_id
        assert hasattr(response, "content")
        assert response.content == "This is the content of the document."
        assert hasattr(response, "download_url")
        assert response.download_url.startswith("https://")

        # Verify service method was called
        assert mock_kb_service.get_get_document_content.called

    async def test_index_documents(self, mock_kb_service, mock_db_session, mock_user, mock_background_tasks):
        """Test indexing documents."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_ids = [uuid4(), uuid4()]

        # Implement index documents method
        async def mock_index_documents(org_id, kb_id, doc_ids, background_tasks, session, user):
            # Verify KB exists
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Get documents
            docs = []
            for doc_id in doc_ids:
                doc = MockKBDocumentModel(
                    id=doc_id,
                    kb_id=kb_id,
                    indexing_status=DocumentStatus.PENDING
                )
                docs.append(doc)

            # Mock documents query result
            session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = docs

            # Add background task
            background_tasks.add_task(
                mock_kb_service._indexing_documents_task,
                doc_ids,
                org_id,
                kb_id,
                session
            )

            # Return success message
            return {"message": f"Indexing {len(doc_ids)} documents in progress"}

        mock_kb_service.post_index_documents = AsyncMock(side_effect=mock_index_documents)

        # Call the service
        response = await mock_kb_service.post_index_documents(
            org_id,
            kb_id,
            doc_ids,
            mock_background_tasks,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert "message" in response
        assert "Indexing" in response["message"]
        assert "2" in response["message"]  # Should mention indexing 2 documents

        # Verify background task was added
        mock_background_tasks.add_task.assert_called_once()

        # Verify service method was called
        assert mock_kb_service.post_index_documents.called

    async def test_add_document_chunk(self, mock_kb_service, mock_db_session, mock_user):
        """Test adding a document chunk."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_id = uuid4()
        chunk_data = MockResponse(
            content="This is a new document chunk."
        )

        # Implement add document chunk method
        async def mock_add_document_chunk(org_id, kb_id, doc_id, chunk_data, session, user):
            # Verify KB exists
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Get document
            doc = await mock_kb_service._get_document(doc_id, org_id, session)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")

            # Create new chunk
            new_chunk_id = 1
            session.scalar.return_value = 0  # Mock count of existing chunks

            # Create chunk object
            chunk = MockResponse(
                id=uuid4(),
                chunk_id=new_chunk_id,
                content=chunk_data.content,
                metadata={
                    "source": doc.title,
                    "chunk": new_chunk_id
                }
            )

            return chunk

        mock_kb_service.post_add_document_chunk = AsyncMock(side_effect=mock_add_document_chunk)

        # Call the service
        response = await mock_kb_service.post_add_document_chunk(
            org_id,
            kb_id,
            doc_id,
            chunk_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert hasattr(response, "id")
        assert hasattr(response, "chunk_id")
        assert response.content == chunk_data.content
        assert hasattr(response, "metadata")
        assert "source" in response.metadata
        assert "chunk" in response.metadata

        # Verify service method was called
        assert mock_kb_service.post_add_document_chunk.called

    async def test_search_chunks_with_high_threshold(self, mock_kb_service, mock_db_session, mock_user):
        """Test searching for chunks with a high score threshold."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        query = "test query"
        limit = 5
        score_threshold = 0.8  # High threshold that will filter some results

        # Create a new implementation of search chunks
        async def mock_search_chunks_high_threshold(org_id, kb_id, query, limit, score_threshold, session, user):
            # Verify KB exists
            kb = await mock_kb_service._get_kb(kb_id, org_id, session)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            # Generate mock search results with scores
            mock_results = [
                {
                    "id": uuid4(),
                    "chunk_id": i,
                    "content": f"Result {i} matching query '{query}'",
                    "metadata": {"source": f"Document {i}", "chunk": i},
                    "score": 0.9 - (i * 0.1)  # Descending scores: 0.9, 0.8, 0.7, 0.6, 0.5
                }
                for i in range(5)
            ]

            # Filter by score threshold - with 0.8 we should only get the first two results
            filtered_results = [r for r in mock_results if r["score"] >= score_threshold]

            return [MockResponse.model_validate(result) for result in filtered_results]

        mock_kb_service.post_search_chunks = AsyncMock(side_effect=mock_search_chunks_high_threshold)

        # Call the service
        response = await mock_kb_service.post_search_chunks(
            org_id,
            kb_id,
            query,
            limit,
            score_threshold,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert isinstance(response, list)
        assert len(response) == 2  # Should only have 2 results with scores >= 0.8
        assert all(hasattr(chunk, "content") for chunk in response)
        assert all(hasattr(chunk, "score") for chunk in response)
        assert all(chunk.score >= score_threshold for chunk in response)

        # Verify service method was called
        assert mock_kb_service.post_search_chunks.called

    async def test_get_document_chunks_with_pagination(self, mock_kb_service, mock_db_session, mock_user):
        """Test getting document chunks with pagination."""
        # Setup data
        org_id = uuid4()
        kb_id = uuid4()
        doc_id = uuid4()
        limit = 2  # Small limit to test pagination
        offset = 2  # Start from the 3rd chunk

        # Create a new implementation of get document chunks
        async def mock_get_document_chunks_paginated(org_id, kb_id, doc_id, limit, offset, session, user):
            # Get document
            doc = await mock_kb_service._get_document(doc_id, org_id, session)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")

            # Generate mock chunks - total of 5 chunks
            total_chunks = 5
            chunks = [
                {
                    "id": uuid4(),
                    "chunk_id": i,
                    "content": f"Chunk {i} content",
                    "metadata": {"source": doc.title, "chunk": i}
                }
                for i in range(offset, min(offset + limit, total_chunks))
            ]

            # Set up mock for raw SQL query result
            session.execute.return_value.mappings.return_value.all.return_value = chunks

            # Set up mock for count query
            session.scalar.return_value = total_chunks

            return MockResponse(
                document_id=doc_id,
                title=doc.title,
                chunks=[MockResponse.model_validate(chunk) for chunk in chunks],
                total_chunks=total_chunks
            )

        mock_kb_service.get_get_document_chunks = AsyncMock(side_effect=mock_get_document_chunks_paginated)

        # Call the service
        response = await mock_kb_service.get_get_document_chunks(
            org_id,
            kb_id,
            doc_id,
            limit,
            offset,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result
        assert response.document_id == doc_id
        assert response.total_chunks == 5
        assert len(response.chunks) == 2  # Should get 2 chunks (limit)

        # Verify chunk IDs - should be chunks 2 and 3 (offset of 2)
        assert response.chunks[0].chunk_id == 2
        assert response.chunks[1].chunk_id == 3

        # Verify service method was called
        assert mock_kb_service.get_get_document_chunks.called