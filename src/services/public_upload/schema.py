from pydantic import BaseModel, HttpUrl


class FileUploadResponse(BaseModel):
  """Response schema for file uploads."""

  url: HttpUrl


class FileUploadRequest(BaseModel):
  """Request schema for file uploads."""

  filename: str
  content_type: str
