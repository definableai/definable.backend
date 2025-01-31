from enum import Enum
from uuid import UUID

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import CRUD


class ChatSessionStatus(str, Enum):
  """Chat session status enum."""

  ACTIVE = "active"
  INACTIVE = "inactive"


class MessageRole(str, Enum):
  """Message role enum."""

  USER = "USER"
  AGENT = "AGENT"
  SYSTEM = "SYSTEM"


class ConversationModel(CRUD):
  """Conversation model."""

  __tablename__ = "conversations"

  organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
  user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
  title: Mapped[str] = mapped_column(String(255), nullable=False)
  description: Mapped[str] = mapped_column(Text, nullable=True)
  is_archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)


class ChatSessionModel(CRUD):
  """Chat session model."""

  __tablename__ = "chat_sessions"

  conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
  agent_id: Mapped[UUID] = mapped_column(ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
  model_id: Mapped[UUID] = mapped_column(ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
  status: Mapped[ChatSessionStatus] = mapped_column(SQLEnum(ChatSessionStatus), nullable=False)
  settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)


class MessageModel(CRUD):
  """Message model."""

  __tablename__ = "messages"

  chat_session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
  role: Mapped[MessageRole] = mapped_column(SQLEnum(MessageRole), nullable=False)
  content: Mapped[str] = mapped_column(Text, nullable=False)
  token_used: Mapped[int] = mapped_column(Integer, nullable=True)
  _metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)
