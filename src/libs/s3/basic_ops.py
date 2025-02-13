from io import BytesIO
from typing import Optional, Union
from urllib.parse import urlparse

import aioboto3
from fastapi import HTTPException, UploadFile

from config.settings import settings


class S3Client:
  """S3 client for basic operations."""

  def __init__(self):
    """Initialize S3 client."""
    self.session = aioboto3.Session()
    self.bucket = settings.s3_bucket
    self.endpoint_url = settings.s3_endpoint
    self.aws_access_key_id = settings.s3_access_key
    self.aws_secret_access_key = settings.s3_secret_key

  async def _get_client(self):
    """Get S3 client."""
    return self.session.client(
      "s3", endpoint_url=self.endpoint_url, aws_access_key_id=self.aws_access_key_id, aws_secret_access_key=self.aws_secret_access_key
    )

  async def upload_file(self, file: Union[UploadFile, BytesIO, bytes], key: str, content_type: Optional[str] = None) -> str:
    """
    Upload a file to S3.

    Args:
        file: File to upload (UploadFile, BytesIO, or bytes)
        key: S3 key (path/filename)
        content_type: Optional content type

    Returns:
        str: S3 URL of uploaded file
    """
    try:
      async with await self._get_client() as client:
        extra_args = {}
        if content_type:
          extra_args["ContentType"] = content_type

        if isinstance(file, UploadFile):
          contents = await file.read()
          await client.put_object(Bucket=self.bucket, Key=key, Body=contents, **extra_args)
        elif isinstance(file, (BytesIO, bytes)):
          await client.put_object(Bucket=self.bucket, Key=key, Body=file if isinstance(file, bytes) else file.getvalue(), **extra_args)

        url = f"{self.endpoint_url}/{self.bucket}/{key}"
        return url

    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

  async def download_file(self, key: str) -> BytesIO:
    """
    Download a file from S3.

    Args:
        key: S3 key (path/filename)

    Returns:
        BytesIO: File contents
    """
    try:
      async with await self._get_client() as client:
        response = await client.get_object(Bucket=self.bucket, Key=key)
        async with response["Body"] as stream:
          contents = await stream.read()
          return BytesIO(contents)

    except Exception:
      raise

  async def delete_file(self, key: str) -> bool:
    """
    Delete a file from S3.

    Args:
        key: S3 key (path/filename)

    Returns:
        bool: True if successful
    """
    try:
      async with await self._get_client() as client:
        await client.delete_object(Bucket=self.bucket, Key=key)
        return True

    except Exception:
      raise

  async def get_presigned_url(self, key: str, expires_in: int = 3600, operation: str = "get_object") -> str:
    """
    Generate a presigned URL for an S3 object.

    Args:
        key: S3 key (path/filename)
        expires_in: URL expiration time in seconds (default: 1 hour)
        operation: S3 operation ('get_object' or 'put_object')

    Returns:
        str: Presigned URL
    """
    try:
      async with await self._get_client() as client:
        url = await client.generate_presigned_url(ClientMethod=operation, Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in)
        return url

    except Exception:
      raise

  @staticmethod
  def get_key_from_url(url: str) -> str:
    """
    Extract the S3 key from a URL.

    Args:
        url: S3 URL

    Returns:
        str: S3 key
    """
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    # Remove bucket name from path if present
    parts = path.split("/", 1)
    return parts[1] if len(parts) > 1 else parts[0]


# Create a singleton instance
s3_client = S3Client()
