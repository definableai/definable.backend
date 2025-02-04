import json
from typing import AsyncGenerator, List
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from database import get_db
from dependencies import get_llm
from dependencies.security import RBAC
from libs.chats.streaming import LLMFactory
from services.__base.acquire import Acquire
from services.llm.model import AgentsModel, LLMModel

from .model import ChatSessionModel, ConversationModel, Message_Role, MessageModel
from .schema import (
  ChatMessageCreate,
  ChatSessionCreate,
  ChatSessionDetailResponse,
  ChatSessionResponse,
  ConversationCreate,
  ConversationResponse,
  ConversationUpdate,
  ConversationWithSessionsResponse,
)


class ConversationService:
  """Conversation service."""

  http_exposed = [
    "post=create",
    "put=update",
    "delete=remove",
    "post=create_session",
    "post=chat",
    "post=stream_chat",
    "get=get_with_sessions",
    "get=list",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.chunk_size = 10
    self.models = acquire.models

  async def post_create(
    self,
    org_id: UUID,
    conversation_data: ConversationCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ConversationResponse:
    """Create a new conversation."""
    db_conversation = ConversationModel(organization_id=org_id, user_id=user["id"], **conversation_data.model_dump())
    session.add(db_conversation)
    await session.commit()
    await session.refresh(db_conversation)
    return ConversationResponse.model_validate(db_conversation)

  async def put_update(
    self,
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
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
    org_id: UUID,
    session_data: ChatSessionCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ChatSessionResponse:
    """Create a new chat session."""
    # Verify conversation exists and belongs to user's organization
    print(session_data)
    query = select(ConversationModel).where(ConversationModel.id == session_data.conversation_id, ConversationModel.organization_id == org_id)
    result = await session.execute(query)
    if not result.scalar_one():
      raise HTTPException(status_code=404, detail="Conversation not found")

    db_session = ChatSessionModel(status="active", **session_data.model_dump())
    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)
    return ChatSessionResponse.model_validate(db_session)

  async def post_stream_chat(
    self,
    org_id: UUID,
    chat_data: ChatMessageCreate,
    session: AsyncSession = Depends(get_db),
    llm: LLMFactory = Depends(get_llm),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> StreamingResponse:
    """Stream chat response."""

    async def generate_response() -> AsyncGenerator[str, None]:
      try:
        data = await self._get_chat_session(chat_data.chat_session_id, session)
        if not data:
          raise Exception("Chat session not found")

        chat_session, model_name = data
        full_response = ""
        buffer: list[str] = []
        # Consume the generator from LLMFactory
        for token in llm.chat(llm=model_name, chat_session_id=str(chat_session.id), message=chat_data.message):
          buffer.append(token)
          full_response += token
          if len(buffer) >= self.chunk_size:
            d = {"message": "".join(buffer)}
            yield f"data: {json.dumps(d)}\n\n"
            buffer = []

        if buffer:
          d = {"message": "".join(buffer)}
          yield f"data: {json.dumps(d)}\n\n"

        yield f"data: {json.dumps({'message': 'DONE'})}\n\n"
        # Save AI response after streaming
        ai_message = MessageModel(chat_session_id=chat_session.id, role=Message_Role.AGENT, content=full_response, token_used=len(full_response))
        session.add(ai_message)
        await session.commit()

      except Exception as e:
        yield f"data: {json.dumps({'message': 'ERROR', 'error': str(e)})}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")

  async def get_get_with_sessions(
    self,
    org_id: UUID,
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "read")),
  ) -> ConversationWithSessionsResponse:
    """Get conversation with all its chat sessions."""
    # Query conversation with chat sessions and related models/agents
    query = (
      select(ConversationModel, ChatSessionModel, LLMModel.name.label("model_name"), AgentsModel.name.label("agent_name"))
      .outerjoin(ChatSessionModel, ConversationModel.id == ChatSessionModel.conversation_id)
      .outerjoin(LLMModel, ChatSessionModel.model_id == LLMModel.id)
      .outerjoin(AgentsModel, ChatSessionModel.agent_id == AgentsModel.id)
      .where(ConversationModel.id == conversation_id, ConversationModel.organization_id == org_id)
    )

    result = await session.execute(query)
    rows = result.unique().all()

    if not rows:
      raise HTTPException(status_code=404, detail="Conversation not found")

    # Process results
    conversation = rows[0].ConversationModel
    chat_sessions = []

    for row in rows:
      if row.ChatSessionModel:
        session_dict = ChatSessionDetailResponse.model_validate({
          **row.ChatSessionModel.__dict__,
          "model_name": row.model_name,
          "agent_name": row.agent_name,
        })
        chat_sessions.append(session_dict)

    return ConversationWithSessionsResponse.model_validate({**conversation.__dict__, "chat_sessions": chat_sessions})

  async def get_list(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "read")),
  ) -> List[ConversationResponse]:
    """Get all conversations for an organization."""
    query = select(ConversationModel).where(ConversationModel.organization_id == org_id).order_by(ConversationModel.created_at.desc())

    result = await session.execute(query)
    conversations = result.scalars().all()

    return [ConversationResponse.model_validate(conv) for conv in conversations]

  ### Private methods ###
  async def _get_chat_session(self, chat_session_id: UUID, session: AsyncSession) -> tuple[ChatSessionModel, str] | None:
    """Get chat session."""
    query = (
      select(ChatSessionModel, LLMModel.name).join(LLMModel, ChatSessionModel.model_id == LLMModel.id).where(ChatSessionModel.id == chat_session_id)
    )

    result = await session.execute(query)
    chat_session, model_name = result.one()
    return chat_session, model_name
