from io import BytesIO

from fastapi import Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import JWTBearer
from libs.s3.basic_ops import S3Client
from services.__base.acquire import Acquire

from .model import PublicUploadModel
from .schema import FileUploadResponse


class PublicUploadService:
  """Service for handling public file uploads."""

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.s3_client = S3Client(bucket=acquire.settings.public_s3_bucket)

  async def post(
    self,
    file: UploadFile,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(JWTBearer()),
  ) -> FileUploadResponse:
    """Upload a file to the public S3 bucket."""
    file_content = await file.read()
    url = await self.s3_client.upload_file(file=BytesIO(file_content), key=f"uploads/{file.filename}")
    db_upload = PublicUploadModel(filename=file.filename, content_type=file.content_type, url=url)
    session.add(db_upload)
    await session.commit()

    return FileUploadResponse(url=url)
