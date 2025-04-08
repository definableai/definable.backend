import json
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db

# from dependencies import get_llm
from dependencies.security import RBAC
from libs.chats.v1.streaming import LLMFactory
from models import AgentModel, ChatSessionModel, ConversationModel, LLMModel, Message_Role, MessageModel
from services.__base.acquire import Acquire

from .schema import (
  ChatMessageCreate,
  ChatSessionCreate,
  ChatSessionDetailResponse,
  ChatSessionResponse,
  ConversationCreate,
  ConversationResponse,
  ConversationUpdate,
  ConversationWithSessionsResponse,
  MessageWithDetailsResponse,
  PaginatedMessagesResponse,
)

llm = LLMFactory()  # TODO: this is a tempered code don't touch it


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
    "get=messages",
  ]

  def __init__(self, acquire: Acquire):
    self.acquire = acquire
    self.chunk_size = 10
    self.models = acquire.models
    self.logger = acquire.logger

  async def post_create(
    self,
    org_id: UUID,
    conversation_data: ConversationCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ConversationResponse:
    """Create a new conversation."""
    self.logger.info("Creating new conversation", org_id=str(org_id), user_id=str(user["id"]))
    db_conversation = ConversationModel(organization_id=org_id, user_id=user["id"], **conversation_data.model_dump())
    session.add(db_conversation)
    await session.commit()
    await session.refresh(db_conversation)
    self.logger.debug("Conversation created successfully", conversation_id=str(db_conversation.id))
    return ConversationResponse.model_validate(db_conversation)

  async def put_update(
    self,
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ConversationResponse:
    """Update a conversation."""
    self.logger.info("Updating conversation", conversation_id=str(conversation_id))
    query = select(ConversationModel).where(ConversationModel.id == conversation_id, ConversationModel.organization_id == user["organization_id"])
    result = await session.execute(query)
    db_conversation = result.scalar_one_or_none()

    if not db_conversation:
      self.logger.error("Conversation not found", conversation_id=str(conversation_id))
      raise HTTPException(status_code=404, detail="Conversation not found")

    update_data = conversation_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(db_conversation, field, value)

    await session.commit()
    await session.refresh(db_conversation)
    self.logger.debug("Conversation updated successfully", conversation_id=str(conversation_id))
    return ConversationResponse.model_validate(db_conversation)

  async def post_create_session(
    self,
    org_id: UUID,
    session_data: ChatSessionCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ChatSessionResponse:
    """Create a new chat session."""
    self.logger.info("Creating new chat session", conversation_id=str(session_data.conversation_id), user_id=str(user["id"]))
    # Verify conversation exists and belongs to user's organization
    query = select(ConversationModel).where(ConversationModel.id == session_data.conversation_id, ConversationModel.organization_id == org_id)
    result = await session.execute(query)
    if not result.unique().scalar_one_or_none():
      self.logger.error("Conversation not found", conversation_id=str(session_data.conversation_id))
      raise HTTPException(status_code=404, detail="Conversation not found")

    db_session = ChatSessionModel(status="active", **session_data.model_dump())
    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)
    self.logger.debug("Chat session created successfully", session_id=str(db_session.id), conversation_id=str(session_data.conversation_id))
    return ChatSessionResponse.model_validate(db_session)

  async def post_stream_chat(
    self,
    org_id: UUID,
    chat_data: ChatMessageCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> StreamingResponse:
    """Stream chat response."""

    async def generate_response() -> AsyncGenerator[str, None]:
      try:
        self.logger.info("Starting chat stream", chat_session_id=str(chat_data.chat_session_id), user_id=str(user["id"]))
        data = await self._get_chat_session(chat_data.chat_session_id, session)
        if not data:
          raise Exception("Chat session not found")
        # add user's message to the chat session
        user_message = MessageModel(
          chat_session_id=chat_data.chat_session_id,
          role=Message_Role.USER,
          content=chat_data.message,
          token_used=len(chat_data.message.split(" ")),
          created_at=datetime.now(timezone.utc),
        )
        session.add(user_message)
        await session.commit()

        chat_session, model_name = data
        self.logger.debug("Using model for chat", model_name=model_name)
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
        self.logger.debug("Saving AI response", chat_session_id=str(chat_session.id), tokens=len(full_response.split(" ")))
        ai_message = MessageModel(
          chat_session_id=chat_session.id,
          role=Message_Role.AGENT,
          content=full_response,
          token_used=len(full_response.split(" ")),
          created_at=datetime.now(timezone.utc),
        )
        session.add(ai_message)
        await session.commit()

      except Exception as e:
        self.logger.exception("Error in chat stream", exc_info=e, chat_session_id=str(chat_data.chat_session_id))
        import traceback

        traceback.print_exc()
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
    self.logger.info("Fetching conversation with sessions", conversation_id=str(conversation_id), org_id=str(org_id))
    # Query conversation with chat sessions and related models/agents
    query = (
      select(ConversationModel, ChatSessionModel, LLMModel.name.label("model_name"), AgentModel.name.label("agent_name"))
      .outerjoin(ChatSessionModel, ConversationModel.id == ChatSessionModel.conversation_id)
      .outerjoin(LLMModel, ChatSessionModel.model_id == LLMModel.id)
      .outerjoin(AgentModel, ChatSessionModel.agent_id == AgentModel.id)
      .where(ConversationModel.id == conversation_id, ConversationModel.organization_id == org_id)
    )

    result = await session.execute(query)
    rows = result.unique().all()

    if not rows:
      self.logger.error("Conversation not found", conversation_id=str(conversation_id), org_id=str(org_id))
      raise HTTPException(status_code=404, detail="Conversation not found")

    # Process results
    conversation = rows[0].ConversationModel
    chat_sessions = []

    self.logger.debug("Processing conversation sessions", conversation_id=str(conversation_id), session_count=len(rows))

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
    self.logger.info("Fetching conversations list", org_id=str(org_id))
    query = select(ConversationModel).where(ConversationModel.organization_id == org_id).order_by(ConversationModel.created_at.desc())

    result = await session.execute(query)
    conversations = result.scalars().all()

    self.logger.debug("Retrieved conversations", org_id=str(org_id), count=len(conversations))
    return [ConversationResponse.model_validate(conv) for conv in conversations]

  async def get_messages(
    self,
    org_id: UUID,
    conversation_id: UUID,
    cursor: Optional[datetime] = None,
    offset: int = 0,
    limit: int = 50,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "read")),
  ) -> PaginatedMessagesResponse:
    """Get paginated messages for a conversation."""
    self.logger.info(
      "Fetching messages", conversation_id=str(conversation_id), cursor=cursor.isoformat() if cursor else None, offset=offset, limit=limit
    )
    # Base query for total count
    count_query = (
      select(func.count(MessageModel.id))
      .join(ChatSessionModel, MessageModel.chat_session_id == ChatSessionModel.id)
      .join(
        ConversationModel,
        and_(
          ChatSessionModel.conversation_id == ConversationModel.id,
          ConversationModel.organization_id == org_id,
          ConversationModel.id == conversation_id,
        ),
      )
    )

    # Main query with joins for details
    query = (
      select(
        MessageModel,
        ChatSessionModel,
        LLMModel.id.label("model_id"),
        LLMModel.name.label("model_name"),
        AgentModel.id.label("agent_id"),
        AgentModel.name.label("agent_name"),
      )
      .join(ChatSessionModel, MessageModel.chat_session_id == ChatSessionModel.id)
      .join(
        ConversationModel,
        and_(
          ChatSessionModel.conversation_id == ConversationModel.id,
          ConversationModel.organization_id == org_id,
          ConversationModel.id == conversation_id,
        ),
      )
      .outerjoin(LLMModel, ChatSessionModel.model_id == LLMModel.id)
      .outerjoin(AgentModel, ChatSessionModel.agent_id == AgentModel.id)
      .order_by(MessageModel.created_at.desc())
    )

    # Add cursor pagination
    if cursor:
      query = query.where(MessageModel.created_at < cursor)

    # Add offset and limit
    query = query.offset(offset * limit).limit(limit + 1)  # Get one extra to check if there are more

    # Execute queries
    total = await session.scalar(count_query)
    result = await session.execute(query)
    rows = result.unique().all()

    # Process results
    messages = []
    for row in rows[:limit]:  # Don't include the extra item in response
      message_dict = {
        **row.MessageModel.__dict__,
        "model_id": row.model_id,
        "model_name": row.model_name,
        "agent_id": row.agent_id,
        "agent_name": row.agent_name,
      }
      messages.append(MessageWithDetailsResponse.model_validate(message_dict))

    return PaginatedMessagesResponse(messages=messages, total=total if total is not None else 0, has_more=len(rows) > limit)

  ### Private methods ###
  async def _get_chat_session(self, chat_session_id: UUID, session: AsyncSession) -> tuple[ChatSessionModel, str] | None:
    """Get chat session."""
    self.logger.info("Fetching chat session", chat_session_id=str(chat_session_id))

    query = (
      select(ChatSessionModel, LLMModel.name).join(LLMModel, ChatSessionModel.model_id == LLMModel.id).where(ChatSessionModel.id == chat_session_id)
    )

    try:
      result = await session.execute(query)
      chat_session, model_name = result.one()
      return chat_session, model_name
    except Exception as e:
      self.logger.exception("Failed to fetch chat session", exc_info=e, chat_session_id=str(chat_session_id))
      return None
