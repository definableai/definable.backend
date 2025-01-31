from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from libs.chats.models import LLMFactory
from services.__base.acquire import Acquire

from .model import ChatSessionModel, ConversationModel, MessageModel, MessageRole
from .schema import (
  ChatMessageCreate,
  ChatSessionCreate,
  ChatSessionResponse,
  ConversationCreate,
  ConversationResponse,
  ConversationUpdate,
  MessageResponse,
)


class ConversationService:
  """Conversation service."""

  http_exposed = [
    "post=create",
    "put=update",
    "delete=remove",
    "post=create_session",
    "post=chat",
    "get=stream_chat",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.llm_factory = LLMFactory()

  async def post_create(
    self,
    conversation_data: ConversationCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("conversations", "write")),
  ) -> ConversationResponse:
    """Create a new conversation."""
    db_conversation = ConversationModel(organization_id=user["organization_id"], user_id=user["id"], **conversation_data.model_dump())
    session.add(db_conversation)
    await session.commit()
    await session.refresh(db_conversation)
    return ConversationResponse.model_validate(db_conversation)

  async def put_update(
    self,
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("conversations", "write")),
  ) -> ConversationResponse:
    """Update a conversation."""
    query = select(ConversationModel).where(ConversationModel.id == conversation_id, ConversationModel.organization_id == user["organization_id"])
    result = await session.execute(query)
    db_conversation = result.scalar_one_or_none()

    if not db_conversation:
      raise HTTPException(status_code=404, detail="Conversation not found")

    update_data = conversation_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(db_conversation, field, value)

    await session.commit()
    await session.refresh(db_conversation)
    return ConversationResponse.model_validate(db_conversation)

  async def post_create_session(
    self,
    session_data: ChatSessionCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ChatSessionResponse:
    """Create a new chat session."""
    # Verify conversation exists and belongs to user's organization
    query = select(ConversationModel).where(
      ConversationModel.id == session_data.conversation_id, ConversationModel.organization_id == user["organization_id"]
    )
    result = await session.execute(query)
    if not result.scalar_one():
      raise HTTPException(status_code=404, detail="Conversation not found")

    db_session = ChatSessionModel(status="active", **session_data.model_dump())
    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)
    return ChatSessionResponse.model_validate(db_session)

  async def post_chat(
    self,
    chat_data: ChatMessageCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> MessageResponse:
    """Send a chat message and get response."""
    # Get chat session
    chat_session = await self._get_chat_session(chat_data.chat_session_id, session)
    if not chat_session:
      raise HTTPException(status_code=404, detail="Chat session not found")

    # Save user message
    user_message = MessageModel(chat_session_id=chat_session.id, role=MessageRole.USER, content=chat_data.message)
    session.add(user_message)
    await session.flush()

    # Get AI response
    response = await self.llm_factory.chat(llm=str(chat_session.model_id), chat_session_id=str(chat_session.id), message=chat_data.message)

    # Save AI response
    ai_message = MessageModel(
      chat_session_id=chat_session.id,
      role=MessageRole.AGENT,
      content=response.content if hasattr(response, "content") else str(response),
      token_used=len(response.content) if hasattr(response, "content") else len(str(response)),
    )
    session.add(ai_message)
    await session.commit()

    return MessageResponse.model_validate(ai_message)

  async def get_stream_chat(
    self,
    chat_session_id: UUID,
    message: str,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> StreamingResponse:
    """Stream chat response."""

    async def generate_response() -> AsyncGenerator[str, None]:
      chat_session = await self._get_chat_session(chat_session_id, session)
      if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

      # Save user message
      user_message = MessageModel(chat_session_id=chat_session.id, role=MessageRole.USER, content=message)
      session.add(user_message)
      await session.flush()

      try:
        full_response = ""
        async for token in self.llm_factory.stream_chat(llm=str(chat_session.model_id), chat_session_id=str(chat_session.id), message=message):
          full_response += token
          yield f"data: {token}\n\n"

        # Save AI response after streaming
        ai_message = MessageModel(chat_session_id=chat_session.id, role=MessageRole.AGENT, content=full_response, token_used=len(full_response))
        session.add(ai_message)
        await session.commit()

      except Exception as e:
        yield f"error: {str(e)}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")

  async def _get_chat_session(self, chat_session_id: UUID, session: AsyncSession) -> ChatSessionModel | None:
    """Get chat session."""
    return await session.get(ChatSessionModel, chat_session_id)
