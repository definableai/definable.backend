from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl


class ChatStatus(str, Enum):
  ACTIVE = "ACTIVE"
  ARCHIVED = "ARCHIVED"
  DELETED = "DELETED"


class MessageRole(str, Enum):
  USER = "USER"
  MODEL = "MODEL"
  AGENT = "AGENT"


class MessageCreate(BaseModel):
  """Create message schema."""

  content: str
  file_uploads: Optional[List[str]] = None


class MessageResponse(BaseModel):
  """Message response schema."""

  id: UUID
  content: str
  role: MessageRole
  chat_session_id: UUID
  parent_message_id: Optional[UUID] = None
  model_id: Optional[UUID] = None
  agent_id: Optional[UUID] = None
  metadata: Dict[str, Any] = {}
  created_at: str

  class Config:
    from_attributes = True


class ChatSessionBase(BaseModel):
  """Base chat session schema."""

  title: str
  status: Optional[ChatStatus] = ChatStatus.ACTIVE


class ChatSessionCreate(ChatSessionBase):
  """Create chat session schema."""

  pass


class ChatSessionUpdate(BaseModel):
  """Update chat session schema."""

  title: Optional[str] = None
  status: Optional[ChatStatus] = None


class ChatSessionResponse(ChatSessionBase):
  """Chat session response schema."""

  id: UUID
  org_id: UUID
  user_id: UUID
  metadata: Dict[str, Any] = {}
  created_at: str
  updated_at: str

  class Config:
    from_attributes = True


class ChatSessionWithMessages(ChatSessionResponse):
  """Chat session with messages schema."""

  messages: List[MessageResponse] = []


class BulkDeleteRequest(BaseModel):
  """Bulk delete request schema."""

  chat_ids: List[UUID]


class ChatFileUploadResponse(BaseModel):
  """Response schema for file uploads."""

  id: UUID
  url: HttpUrl
