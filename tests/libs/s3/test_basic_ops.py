import pytest
import os
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import HTTPException, UploadFile

from src.libs.s3.v1.basic_ops import S3Client


# Define a function to check if we're running integration tests
def is_integration_test():
    """Check if we're running in integration test mode.
    This is controlled by the INTEGRATION_TEST environment variable.
    Set it to 1 or true to run integration tests.
    """
    integration_env = os.environ.get("INTEGRATION_TEST", "").lower()
    return integration_env in ("1", "true", "yes")


# Mock settings for S3
@pytest.fixture
def mock_settings():
    with patch("src.libs.s3.v1.basic_ops.settings") as mock_settings:
        mock_settings.s3_bucket = "test-bucket"
        mock_settings.s3_endpoint = "http://localhost:9000"
        mock_settings.s3_access_key = "test-access-key"
        mock_settings.s3_secret_key = "test-secret-key"
        yield mock_settings


# Create AsyncMock classes for each S3 operation we test
class MockS3Client:
    """Mock S3 client that properly handles async context manager."""

    def __init__(self, success=True):
        self.success = success
        self.put_object = AsyncMock(return_value={"ETag": "test-etag"})
        self.get_object = AsyncMock()
        self.delete_object = AsyncMock(return_value={"DeleteMarker": True})
        self.generate_presigned_url = AsyncMock(return_value="http://presigned-url.com/test-bucket/test.txt?expiry=3600")

        # Configure error scenarios if needed
        if not success:
            self.put_object.side_effect = Exception("Upload failed")
            self.get_object.side_effect = Exception("Download failed")
            self.delete_object.side_effect = Exception("Delete failed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_body():
    """Create a mock for response body stream."""
    mock_body = AsyncMock()
    mock_body.__aenter__ = AsyncMock(return_value=mock_body)
    mock_body.__aexit__ = AsyncMock(return_value=None)
    mock_body.read = AsyncMock(return_value=b"downloaded content")
    return mock_body


@pytest.fixture
def s3_client(mock_settings):
    """Create a test S3Client instance with mocked settings."""
    return S3Client(bucket="test-bucket")


@pytest.fixture
def test_file_bytesio():
    """Create a test file as BytesIO."""
    content = b"test file content"
    return BytesIO(content)


@pytest.fixture
def test_file_bytes():
    """Create a test file as bytes."""
    return b"test file content"


@pytest.fixture
def test_upload_file():
    """Create a mock UploadFile."""
    mock_file = MagicMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=b"test file content")
    mock_file.filename = "test_file.txt"
    return mock_file


@pytest.mark.asyncio
class TestS3Client:
    """Unit tests for S3Client."""

    async def test_initialization(self, mock_settings):
        """Test S3Client initialization."""
        # Test with explicit bucket
        client = S3Client(bucket="my-bucket")
        assert client.bucket == "my-bucket"
        assert client.endpoint_url == mock_settings.s3_endpoint
        assert client.aws_access_key_id == mock_settings.s3_access_key
        assert client.aws_secret_access_key == mock_settings.s3_secret_key

        # Test with default bucket
        client = S3Client()
        assert client.bucket == mock_settings.s3_bucket

    async def test_upload_file_uploadfile(self, s3_client, test_upload_file):
        """Test uploading a file from UploadFile."""
        # Setup mock S3 client
        mock_client = MockS3Client()

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.upload_file(
                file=test_upload_file,
                key="uploads/test_file.txt",
                content_type="text/plain"
            )

            # Assert
            assert result == "http://localhost:9000/test-bucket/uploads/test_file.txt"
            mock_client.put_object.assert_awaited_once()
            # Check the Bucket parameter was passed correctly
            args, kwargs = mock_client.put_object.await_args
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == "uploads/test_file.txt"
            assert kwargs["ContentType"] == "text/plain"
            assert kwargs["Body"] == b"test file content"

    async def test_upload_file_bytesio(self, s3_client, test_file_bytesio):
        """Test uploading a file from BytesIO."""
        # Setup mock S3 client
        mock_client = MockS3Client()

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.upload_file(
                file=test_file_bytesio,
                key="uploads/test_file.txt"
            )

            # Assert
            assert result == "http://localhost:9000/test-bucket/uploads/test_file.txt"
            mock_client.put_object.assert_awaited_once()
            # Check the Bucket parameter was passed correctly
            args, kwargs = mock_client.put_object.await_args
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == "uploads/test_file.txt"
            assert kwargs["Body"] == b"test file content"

    async def test_upload_file_bytes(self, s3_client, test_file_bytes):
        """Test uploading a file from bytes."""
        # Setup mock S3 client
        mock_client = MockS3Client()

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.upload_file(
                file=test_file_bytes,
                key="uploads/test_file.txt"
            )

            # Assert
            assert result == "http://localhost:9000/test-bucket/uploads/test_file.txt"
            mock_client.put_object.assert_awaited_once()
            # Check the Bucket parameter was passed correctly
            args, kwargs = mock_client.put_object.await_args
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == "uploads/test_file.txt"
            assert kwargs["Body"] == test_file_bytes

    async def test_upload_file_error(self, s3_client):
        """Test error handling when uploading a file."""
        # Setup mock S3 client with error
        mock_client = MockS3Client(success=False)

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute and Assert
            with pytest.raises(HTTPException) as excinfo:
                await s3_client.upload_file(
                    file=b"test",
                    key="test.txt"
                )

            assert excinfo.value.status_code == 500
            assert "Failed to upload file" in str(excinfo.value.detail)
            assert "Upload failed" in str(excinfo.value.detail)

    async def test_download_file(self, s3_client, mock_body):
        """Test downloading a file."""
        # Setup mock S3 client
        mock_client = MockS3Client()
        # Set up the response for get_object
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.download_file(key="test/download.txt")

            # Assert
            assert isinstance(result, BytesIO)
            assert result.getvalue() == b"downloaded content"
            mock_client.get_object.assert_awaited_once()
            # Check the Bucket parameter was passed correctly
            args, kwargs = mock_client.get_object.await_args
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == "test/download.txt"

    async def test_download_file_error(self, s3_client):
        """Test error handling when downloading a file."""
        # Setup mock S3 client with error
        mock_client = MockS3Client(success=False)

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute and Assert
            with pytest.raises(Exception) as excinfo:
                await s3_client.download_file(key="nonexistent.txt")

            assert "Download failed" in str(excinfo.value)

    async def test_delete_file(self, s3_client):
        """Test deleting a file."""
        # Setup mock S3 client
        mock_client = MockS3Client()

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.delete_file(key="test/delete.txt")

            # Assert
            assert result is True
            mock_client.delete_object.assert_awaited_once()
            # Check the Bucket parameter was passed correctly
            args, kwargs = mock_client.delete_object.await_args
            assert kwargs["Bucket"] == "test-bucket"
            assert kwargs["Key"] == "test/delete.txt"

    async def test_delete_file_error(self, s3_client):
        """Test error handling when deleting a file."""
        # Setup mock S3 client with error
        mock_client = MockS3Client(success=False)

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute and Assert
            with pytest.raises(Exception) as excinfo:
                await s3_client.delete_file(key="test.txt")

            assert "Delete failed" in str(excinfo.value)

    async def test_get_presigned_url(self, s3_client):
        """Test generating a presigned URL."""
        # Setup mock S3 client
        mock_client = MockS3Client()

        with patch.object(s3_client, "_get_client", return_value=mock_client):
            # Execute
            result = await s3_client.get_presigned_url(
                key="test/presigned.txt",
                expires_in=3600,
                operation="get_object"
            )

            # Assert
            assert result == "http://presigned-url.com/test-bucket/test.txt?expiry=3600"
            mock_client.generate_presigned_url.assert_awaited_once()
            # Check the parameters are passed correctly
            args, kwargs = mock_client.generate_presigned_url.await_args
            assert kwargs["ClientMethod"] == "get_object"
            assert kwargs["Params"]["Bucket"] == "test-bucket"
            assert kwargs["Params"]["Key"] == "test/presigned.txt"
            assert kwargs["ExpiresIn"] == 3600

    async def test_get_key_from_url(self):
        """Test extracting key from URL."""
        # Test with bucket in path
        url = "http://localhost:9000/test-bucket/uploads/test_file.txt"
        key = S3Client.get_key_from_url(url)
        assert key == "uploads/test_file.txt"

        # Test without bucket in path (just filename)
        url = "http://localhost:9000/test-file.txt"
        key = S3Client.get_key_from_url(url)
        assert key == "test-file.txt"

@pytest.mark.asyncio
class TestS3ClientErrorHandling:
    """Test error handling in S3Client."""

    async def test_missing_credentials(self):
        """Test error handling with missing credentials."""
        with patch("src.libs.s3.v1.basic_ops.settings") as mock_settings:
            # Simulate missing credentials
            mock_settings.s3_access_key = ""
            mock_settings.s3_secret_key = ""
            mock_settings.s3_bucket = "test-bucket"
            mock_settings.s3_endpoint = "http://localhost:9000"

            client = S3Client()

            # Test upload with missing credentials
            with pytest.raises(Exception):
                await client.upload_file(
                    file=BytesIO(b"test"),
                    key="test.txt"
                )

    async def test_nonexistent_bucket(self, s3_client):
        """Test error handling with nonexistent bucket."""
        # Setup mock client that raises NoSuchBucket exception
        error_client = MockS3Client(success=False)
        error_client.put_object.side_effect = Exception("NoSuchBucket")

        with patch.object(s3_client, "_get_client", return_value=error_client):
            # Test upload to nonexistent bucket
            with pytest.raises(HTTPException) as excinfo:
                await s3_client.upload_file(
                    file=BytesIO(b"test"),
                    key="test.txt"
                )

            assert excinfo.value.status_code == 500
            assert "Failed to upload file" in str(excinfo.value.detail)
            assert "NoSuchBucket" in str(excinfo.value.detail)