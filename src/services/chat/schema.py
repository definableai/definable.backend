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

class PromptData(BaseModel):
  """Prompt response data"""

  id: UUID
  title: str
  description: str
  content: str

  class Config:
    from_attributes = True

class MessageResponse(BaseModel):
  """Message response schema."""

  id: UUID
  content: str
  role: MessageRole
  chat_session_id: UUID
  parent_message_id: Optional[UUID] = None
  prompt_data: Optional[PromptData] = None
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


class ChatUploadData(BaseModel):
  """Upload data model."""

  id: UUID
  message_id: UUID
  filename: str
  file_size: int
  content_type: str
  url: str


class AllUploads(BaseModel):
  """Model for all uploads in a chat session."""

  uploads: List[ChatUploadData] = []


class UploadsByMessage(BaseModel):
  """Model for uploads organized by message ID."""

  # Key is message_id as string, value is list of uploads
  message_uploads: Dict[str, List[ChatUploadData]] = {}


class ChatSessionWithMessages(ChatSessionResponse):
  """Chat session with messages schema."""

  messages: List[MessageResponse] = []
  uploads: List[ChatUploadData] = []


class BulkDeleteRequest(BaseModel):
  """Bulk delete request schema."""

  chat_ids: List[UUID]


class ChatFileUploadResponse(BaseModel):
  """Response schema for file uploads."""

  id: UUID
  url: HttpUrl

class Model(str, Enum):
  CHAT = "chat"
  REASON = "reason"

class TextInput(BaseModel):
  text: str
  num_prompts: int = 1
  prompt_type: str = "task" # creative, question, continuation, task
  model: Model = Model.CHAT
