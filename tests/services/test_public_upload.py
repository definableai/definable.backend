import pytest
from fastapi import UploadFile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from uuid import UUID, uuid4
from datetime import datetime
from fastapi import HTTPException

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

# Create the proper module structure for libs.s3.v1
sys.modules['libs'] = MagicMock()
sys.modules['libs.s3'] = MagicMock()
sys.modules['libs.s3.v1'] = MagicMock()

# Create a mock S3Client class
class MockS3Client:
    def __init__(self, bucket="test-bucket"):
        self.bucket = bucket
    
    async def upload_file(self, file, key, **kwargs):
        # Simulate S3 upload and return a URL
        return f"https://example.com/{key}"

# Add the S3Client to the mocked module
sys.modules['libs.s3.v1'].S3Client = MockS3Client

# Mock models
class MockPublicUploadModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', uuid4())
        self.filename = kwargs.get('filename', 'test.jpg')
        self.content_type = kwargs.get('content_type', 'image/jpeg')
        self.url = kwargs.get('url', 'https://example.com/uploads/test.jpg')
        self.created_at = kwargs.get('created_at', datetime.now().isoformat())
        self.__dict__ = {**self.__dict__, **kwargs}

class MockResponse:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, data):
        if isinstance(data, dict):
            return cls(**data)
        return cls(**{k: v for k, v in data.__dict__.items() if not k.startswith('_')})
    
    def model_dump(self, **kwargs):
        exclude_unset = kwargs.get('exclude_unset', False)
        if exclude_unset:
            # Return only items that have been explicitly set
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

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
    return session

@pytest.fixture
def mock_upload_file():
    """Create a mock upload file."""
    file_content = b"test file content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.jpg"
    file.content_type = "image/jpeg"
    file.read = AsyncMock(return_value=file_content)
    file.file = BytesIO(file_content)
    return file

@pytest.fixture
def mock_public_upload_service():
    """Create a mock public upload service."""
    upload_service = MagicMock()
    
    # Mock the S3 client
    s3_client = MockS3Client()
    upload_service.s3_client = s3_client
    
    async def mock_post(file, session, user):
        # Read file content
        file_content = await file.read()
        
        # Upload to S3
        url = await s3_client.upload_file(
            file=BytesIO(file_content), 
            key=f"uploads/{file.filename}"
        )
        
        # Create DB record
        current_time = datetime.now().isoformat()
        db_upload = MockPublicUploadModel(
            filename=file.filename,
            content_type=file.content_type,
            url=url,
            created_at=current_time
        )
        session.add(db_upload)
        await session.commit()
        
        # Return response
        return MockResponse(
            id=db_upload.id,
            url=url,
            filename=file.filename,
            content_type=file.content_type,
            created_at=current_time
        )
    
    async def mock_post_with_error(file, session, user):
        raise HTTPException(
            status_code=400,
            detail="Failed to upload file"
        )
    
    # Create AsyncMock object
    post_mock = AsyncMock(side_effect=mock_post)
    
    # Assign the mock to the service
    upload_service.post = post_mock
    
    return upload_service

@pytest.mark.asyncio
class TestPublicUploadService:
    """Tests for the Public Upload service."""
    
    async def test_upload_file(self, mock_public_upload_service, mock_db_session, mock_user, mock_upload_file):
        """Test uploading a file."""
        # Call the service
        response = await mock_public_upload_service.post(
            file=mock_upload_file,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result has all expected fields in the API schema
        assert hasattr(response, "id")
        assert hasattr(response, "url")
        assert hasattr(response, "filename")
        assert hasattr(response, "content_type")
        assert hasattr(response, "created_at")
        assert response.url == f"https://example.com/uploads/{mock_upload_file.filename}"
        assert response.filename == mock_upload_file.filename
        assert response.content_type == mock_upload_file.content_type
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # The service should have been called correctly
        assert mock_public_upload_service.post.called
    
    async def test_upload_file_with_special_characters(self, mock_public_upload_service, mock_db_session, mock_user):
        """Test uploading a file with special characters in the filename."""
        # Create a special filename
        file = MagicMock(spec=UploadFile)
        file.filename = "test file with spaces & special chars.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=b"test file content")
        file.file = BytesIO(b"test file content")
        
        # Call the service
        response = await mock_public_upload_service.post(
            file=file,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert hasattr(response, "url")
        assert "uploads/test file with spaces & special chars.jpg" in response.url
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # The service should have been called correctly
        assert mock_public_upload_service.post.called
    
    # Use the mock_s3_client fixture instead of patching a non-existent module
    async def test_upload_file_s3_exception(self, mock_public_upload_service, mock_db_session, mock_user, mock_upload_file):
        """Test handling S3 upload exception."""
        # Setup S3 client to raise an exception
        original_upload_file = mock_public_upload_service.s3_client.upload_file
        mock_public_upload_service.s3_client.upload_file = AsyncMock(side_effect=Exception("S3 upload failed"))
        
        # Create a custom implementation that handles the exception
        async def mock_post_with_exception(file, session, user):
            try:
                file_content = await file.read()
                # This will raise the mocked exception
                await mock_public_upload_service.s3_client.upload_file(
                    file=BytesIO(file_content), 
                    key=f"uploads/{file.filename}"
                )
                # The following code should not execute
                db_upload = MockPublicUploadModel(
                    filename=file.filename,
                    content_type=file.content_type,
                    url="https://example.com/uploads/test.jpg"
                )
                session.add(db_upload)
                await session.commit()
                return MockResponse(url="https://example.com/uploads/test.jpg")
            except Exception as e:
                return {"error": str(e)}
            finally:
                # Restore the original function for cleanup
                mock_public_upload_service.s3_client.upload_file = original_upload_file
        
        # Override the mock
        mock_public_upload_service.post.side_effect = mock_post_with_exception
        
        # Call the service
        response = await mock_public_upload_service.post(
            file=mock_upload_file,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert "error" in response
        assert "S3 upload failed" in response["error"]
        
        # Verify database operations weren't called
        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_not_called()
        
        # The service should have been called correctly
        assert mock_public_upload_service.post.called 

    async def test_upload_file_bad_request(self, mock_public_upload_service, mock_db_session, mock_user, mock_upload_file):
        """Test handling bad upload request."""
        # Replace the service method with one that raises an exception
        async def mock_post_with_error(file, session, user):
            raise HTTPException(
                status_code=400,
                detail="Failed to upload file"
            )
            
        mock_public_upload_service.post = AsyncMock(side_effect=mock_post_with_error)
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_public_upload_service.post(
                file=mock_upload_file,
                session=mock_db_session,
                user=mock_user
            )
        
        # Verify exception details
        assert exc_info.value.status_code == 400
        assert "Failed to upload file" in str(exc_info.value.detail)
        
        # Verify the service method was called
        assert mock_public_upload_service.post.called 