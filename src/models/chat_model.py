import enum
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import BigInteger, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class Chat_Status(enum.Enum):
  ACTIVE = "ACTIVE"
  ARCHIVED = "ARCHIVED"
  DELETED = "DELETED"


class Message_Role(enum.Enum):
  USER = "USER"
  AGENT = "AGENT"
  MODEL = "MODEL"


class ChatModel(CRUD):
  __tablename__ = "chats"

  org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id"), nullable=False)
  user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
  title: Mapped[str] = mapped_column(String(255), nullable=False)
  status: Mapped[Chat_Status] = mapped_column(Enum(Chat_Status), default=Chat_Status.ACTIVE, nullable=False)
  _metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, default={})
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  def __repr__(self) -> str:
    return f"<Chat(id={self.id}, title={self.title})>"


class MessageModel(CRUD):
  __tablename__ = "messages"

  chat_session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chats.id"), nullable=False, index=True)
  parent_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)
  role: Mapped[Message_Role] = mapped_column(Enum(Message_Role), nullable=False)
  content: Mapped[str] = mapped_column(Text, nullable=False)
  model_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("models.id"), nullable=True)
  agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("agents.id"), nullable=True)
  prompt_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("prompts.id"), nullable=True)
  _metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, default={})

  def __repr__(self) -> str:
    return f"<Message(id={self.id}, role={self.role})>"


class ChatUploadModel(CRUD):
  __tablename__ = "chat_uploads"

  message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)
  filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
  content_type: Mapped[str] = mapped_column(String(150), nullable=False)
  file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
  url: Mapped[str] = mapped_column(String(500), nullable=False)
  _metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, default={})
  updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now(), nullable=False)

  def __repr__(self) -> str:
    return f"<ChatUpload(id={self.id}, filename={self.filename})>"
