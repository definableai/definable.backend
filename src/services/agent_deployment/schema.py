from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class WebhookLogPayload(BaseModel):
  """Schema for incoming webhook log payload."""

  agent_id: UUID
  deployment_id: str = Field(..., max_length=255)
  log_type: str = Field(..., pattern="^(INIT|BUILD|DEPLOY|FINAL)$")
  log_level: str = Field(..., pattern="^(INFO|WARN|ERROR|DEBUG)$")
  message: str
  metadata: Dict[str, Any] = Field(default_factory=dict)
  timestamp: datetime
  source: str = Field(..., max_length=100)


class WebhookTracePayload(BaseModel):
  """Schema for incoming webhook trace payload."""

  agent_id: UUID
  deployment_id: str = Field(..., max_length=255)
  trace_id: str = Field(..., max_length=255)
  span_id: str = Field(..., max_length=255)
  parent_span_id: Optional[str] = Field(None, max_length=255)
  operation_name: str = Field(..., max_length=255)
  start_time: datetime
  end_time: Optional[datetime] = None
  duration_ms: Optional[int] = None
  status: str = Field(..., pattern="^(STARTED|SUCCESS|ERROR|TIMEOUT)$")
  tags: Dict[str, Any] = Field(default_factory=dict)
  logs: Dict[str, Any] = Field(default_factory=dict)


class LogResponse(BaseModel):
  """Response schema for deployment logs."""

  id: UUID
  agent_id: UUID
  organization_id: UUID
  user_id: UUID
  api_key_id: UUID
  deployment_id: str
  log_type: str
  log_level: str
  message: str
  log_metadata: Dict[str, Any]
  timestamp: datetime
  source: str
  created_at: datetime

  class Config:
    from_attributes = True


class TraceResponse(BaseModel):
  """Response schema for deployment traces."""

  id: UUID
  agent_id: UUID
  organization_id: UUID
  user_id: UUID
  api_key_id: UUID
  deployment_id: str
  trace_id: str
  span_id: str
  parent_span_id: Optional[str]
  operation_name: str
  start_time: datetime
  end_time: Optional[datetime]
  duration_ms: Optional[int]
  status: str
  tags: Dict[str, Any]
  trace_logs: Dict[str, Any]
  created_at: datetime

  class Config:
    from_attributes = True


class LogFilter(BaseModel):
  """Filter schema for querying logs."""

  agent_id: Optional[UUID] = None
  user_id: Optional[UUID] = None
  api_key_id: Optional[UUID] = None
  deployment_id: Optional[str] = None
  log_type: Optional[str] = Field(None, pattern="^(INIT|BUILD|DEPLOY|FINAL)$")
  log_level: Optional[str] = Field(None, pattern="^(INFO|WARN|ERROR|DEBUG)$")
  source: Optional[str] = None
  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  skip: int = Field(0, ge=0)
  limit: int = Field(50, ge=1, le=1000)


class TraceFilter(BaseModel):
  """Filter schema for querying traces."""

  agent_id: Optional[UUID] = None
  user_id: Optional[UUID] = None
  api_key_id: Optional[UUID] = None
  deployment_id: Optional[str] = None
  trace_id: Optional[str] = None
  status: Optional[str] = Field(None, pattern="^(STARTED|SUCCESS|ERROR|TIMEOUT)$")
  operation_name: Optional[str] = None
  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  skip: int = Field(0, ge=0)
  limit: int = Field(50, ge=1, le=1000)


class PaginatedLogResponse(BaseModel):
  """Paginated response for logs."""

  logs: List[LogResponse]
  total: int
  skip: int
  limit: int


class PaginatedTraceResponse(BaseModel):
  """Paginated response for traces."""

  traces: List[TraceResponse]
  total: int
  skip: int
  limit: int


class WebhookResponse(BaseModel):
  """Response for successful webhook processing."""

  success: bool = True
  message: str = "Webhook processed successfully"
  processed_at: datetime = Field(default_factory=datetime.now)
