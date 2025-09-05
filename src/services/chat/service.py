import json
import mimetypes
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID

import httpx
from agno.media import File, Image
from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database import get_db
from dependencies.security import RBAC, JWTBearer
from libs.chats.v1 import LLMFactory, extract_file_content, generate_chat_name, generate_prompts_stream
from libs.s3.v1 import S3Client
from libs.speech.v1 import transcribe
from models import ChatModel, ChatUploadModel, LLMModel, MessageModel
from models.agent_model import AgentModel
from models.prompt_model import PromptModel
from services.__base.acquire import Acquire
from utils.charge import Charge

from .schema import (
  AllUploads,
  BulkDeleteRequest,
  ChatFileUploadResponse,
  ChatSessionCreate,
  ChatSessionResponse,
  ChatSessionUpdate,
  ChatSessionWithMessages,
  ChatSettings,
  ChatStatus,
  ChatUploadData,
  MessageCreate,
  MessageResponse,
  MessageRole,
  PromptData,
  TextInput,
)


class ChatService:
  """Chat service for managing chat sessions and messages."""

  http_exposed = [
    "get=list",
    "post=send_message",
    "post=chat_with_model",
    "delete=delete_session",
    "post=bulk_delete_sessions",
    "post=upload_file",
    "post=transcribe",
    "post=generate_prompts",
    "post=prompt",
    "get=available_knowledge_bases",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.llm_factory = LLMFactory()
    self.chunk_size = 10
    self.s3_client = S3Client(bucket="chats")
    self.logger = acquire.logger
    self.ws_manager = acquire.ws_manager

  async def post(
    self,
    org_id: UUID,
    data: ChatSessionCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> JSONResponse:
    """Create a new chat session."""
    # Prepare metadata with settings if provided
    metadata = {}
    if data.settings:
      metadata["settings"] = data.settings.dict(exclude_none=True)

    db_session = ChatModel(
      title=data.title or "New Chat",
      status=data.status,
      user_id=user["id"],
      org_id=user["org_id"],
      _metadata=metadata,
    )

    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)

    return JSONResponse(
      status_code=status.HTTP_201_CREATED,
      content={"id": str(db_session.id), "message": "Chat session created successfully"},
    )

  async def put(
    self,
    chat_id: UUID,
    org_id: UUID,
    data: ChatSessionUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> JSONResponse:
    """Update an existing chat session."""
    # Get session
    query = select(ChatModel).where(
      and_(
        ChatModel.id == chat_id,
        ChatModel.user_id == user["id"],
        ChatModel.org_id == org_id,
      )
    )
    result = await session.execute(query)
    db_session = result.scalar_one_or_none()

    if not db_session:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    # Update fields
    if data.title is not None:
      db_session.title = data.title
    if data.status is not None:
      db_session.status = data.status

    # Update settings if provided
    if data.settings is not None:
      if not db_session._metadata:
        db_session._metadata = {}
      db_session._metadata["settings"] = data.settings.dict(exclude_none=True)

    await session.commit()
    await session.refresh(db_session)

    return JSONResponse(
      status_code=status.HTTP_200_OK,
      content={"id": str(db_session.id), "message": "Chat session updated successfully"},
    )

  async def get(
    self,
    chat_id: UUID,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "read")),
  ) -> ChatSessionWithMessages:
    """Get a single chat session with its messages."""
    query = select(ChatModel).where(
      and_(
        ChatModel.id == chat_id,
        ChatModel.user_id == user["id"],
        ChatModel.org_id == user["org_id"],
      )
    )
    result = await session.execute(query)
    db_session = result.scalar_one_or_none()

    if not db_session:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    # Get messages
    query = select(MessageModel).where(MessageModel.chat_session_id == chat_id).order_by(MessageModel.created_at)
    result = await session.execute(query)
    messages = result.scalars().all()

    message_responses = []

    # Get all message IDs for this chat session
    message_ids = [msg.id for msg in messages]

    # Get all uploads for this chat session in a single query
    all_uploads_query = select(
      ChatUploadModel.id,
      ChatUploadModel.message_id,
      ChatUploadModel.filename,
      ChatUploadModel.file_size,
      ChatUploadModel.content_type,
      ChatUploadModel.url,
    ).where(ChatUploadModel.message_id.in_(message_ids))
    all_uploads_result = await session.execute(all_uploads_query)

    # Create a dictionary to hold uploads by message_id
    uploads_by_message: Dict[str, List[ChatUploadData]] = {}
    all_upload_items = []

    for row in all_uploads_result.mappings():
      upload_data = ChatUploadData(
        id=row.id,
        message_id=row.message_id,
        filename=row.filename,
        file_size=row.file_size,
        content_type=row.content_type,
        url=row.url,
      )

      # Add to the list of all uploads
      all_upload_items.append(upload_data)

      # Organize by message_id for message-specific uploads
      msg_id = row.message_id
      if msg_id not in uploads_by_message:
        uploads_by_message[msg_id] = []
      uploads_by_message[msg_id].append(upload_data)

    # Process each message and include prompt_data if prompt_id exists
    for msg in messages:
      prompt_data = None
      if msg.prompt_id:
        prompt_model = await self._get_prompt(msg.prompt_id, session)
        prompt_data = PromptData(
          id=prompt_model.id,
          title=prompt_model.title,
          description=prompt_model.description,
          content=prompt_model.content,
        )

      message_responses.append(
        MessageResponse(
          id=msg.id,
          content=msg.content,
          role=msg.role,
          chat_session_id=msg.chat_session_id,
          parent_message_id=msg.parent_message_id,
          model_id=msg.model_id,
          agent_id=msg.agent_id,
          prompt_data=prompt_data,  # Include the prompt_data here
          metadata=msg._metadata,
          created_at=msg.created_at.isoformat(),
        )
      )

    # Create the uploads model
    uploads_model = AllUploads(uploads=all_upload_items)

    return ChatSessionWithMessages(
      id=db_session.id,
      title=db_session.title,
      status=db_session.status,
      org_id=db_session.org_id,
      user_id=db_session.user_id,
      metadata=db_session._metadata,
      settings=self._get_chat_settings(db_session),
      created_at=db_session.created_at.isoformat(),
      updated_at=db_session.updated_at.isoformat(),
      messages=message_responses,
      uploads=uploads_model.uploads,
    )

  async def get_list(
    self,
    org_id: UUID,
    status: Optional[ChatStatus] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "read")),
  ) -> List[ChatSessionResponse]:
    """Get list of chat sessions for the user."""
    if status:
      query = (
        select(ChatModel)
        .where(
          and_(
            ChatModel.user_id == user["id"],
            ChatModel.org_id == user["org_id"],
            ChatModel.status == status,
          )
        )
        .order_by(ChatModel.updated_at.desc())
      )
    else:
      query = (
        select(ChatModel)
        .where(
          and_(
            ChatModel.user_id == user["id"],
            ChatModel.org_id == user["org_id"],
          )
        )
        .order_by(ChatModel.updated_at.desc())
      )

    result = await session.execute(query)
    db_sessions = result.scalars().all()

    return [
      ChatSessionResponse(
        id=db_session.id,
        title=db_session.title,
        status=db_session.status,
        org_id=db_session.org_id,
        user_id=db_session.user_id,
        metadata=db_session._metadata,
        settings=self._get_chat_settings(db_session),
        created_at=db_session.created_at.isoformat(),
        updated_at=db_session.updated_at.isoformat(),
      )
      for db_session in db_sessions
    ]

  async def post_send_message(
    self,
    message_data: MessageCreate,
    org_id: UUID,
    agent_id: Optional[UUID] = None,
    chat_id: Optional[UUID] = None,
    instruction_id: Optional[UUID] = None,
    model_id: Optional[UUID] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> StreamingResponse:
    """Send a message to a chat session."""
    is_new_chat = False
    db_session = None
    user_id = user["id"]
    prompt = None
    charge = None

    self.logger.info(f"Processing message request for model={model_id}, agent={agent_id}, chat={chat_id}")

    try:
      if not model_id and not agent_id:
        raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Either model_id or agent_id must be provided",
        )

      # Handle chat session creation/verification
      if not chat_id:
        db_session = ChatModel(
          title="New Chat",
          status=ChatStatus.ACTIVE,
          user_id=user_id,
          org_id=org_id,
        )
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        chat_id = db_session.id
        is_new_chat = True
        self.logger.info(f"Created new chat: {chat_id}")
      else:
        query = select(ChatModel).where(
          and_(
            ChatModel.id == chat_id,
            ChatModel.user_id == user["id"],
            ChatModel.org_id == user["org_id"],
          )
        )
        result = await session.execute(query)
        db_session = result.scalar_one_or_none()
        if not db_session:
          raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
          )
        # Flag to update the chat title if it's "New Chat"
        is_new_chat = db_session.title == "New Chat"

      # Get effective settings (saved + provided) and save any new settings
      effective_temp, effective_max, effective_top_p = self._get_effective_settings(db_session, temperature, max_tokens, top_p)

      # Save any new settings provided
      if any(x is not None for x in [temperature, max_tokens, top_p]):
        self._save_settings_to_chat(db_session, temperature, max_tokens, top_p)
        await session.commit()

      # if instruction_id is provided, check if instruction exists
      if instruction_id:
        instruction = await self._get_prompt(instruction_id, session)
        if not instruction:
          raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instruction not found")
        # get the prompt from the instruction
        prompt = instruction.content

      # get the parent id from the last message of the chat session
      query = (
        select(MessageModel)
        .where(
          MessageModel.chat_session_id == chat_id,
          MessageModel.role != MessageRole.USER,
        )
        .order_by(MessageModel.created_at.desc())
      )
      result = await session.execute(query)
      last_message = result.scalars().first()
      parent_id = last_message.id if last_message else None

      # LLM processing logic
      if model_id:
        # Get LLM model
        query = select(LLMModel).where(LLMModel.id == model_id)
        result = await session.execute(query)
        llm_model = result.scalar_one_or_none()
        if not llm_model:
          raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM model not found",
          )

        # Check if model is active
        if not llm_model.is_active:
          raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"The model '{llm_model.name}' is no longer available. "
            f"It has been deprecated or deactivated. Please choose an active model from the marketplace.",
          )

        # Initialize billing - simple HOLD with qty=1
        try:
          # Create a more descriptive transaction message
          charge_description = f"Chat with {llm_model.name}: {message_data.content[:30]}..."

          charge = Charge(name=llm_model.name, user_id=user_id, org_id=org_id, session=session, service="chat")
          await charge.create(
            qty=1,
            metadata={"chat_id": str(chat_id), "model": llm_model.name, "provider": llm_model.provider},
            description=charge_description,  # Pass custom description
          )
        except Exception as billing_error:
          self.logger.error(f"Billing initialization failed: {str(billing_error)}")
          # Don't continue execution when billing fails - throw an appropriate error
          if "Insufficient credits" in str(billing_error):
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Insufficient credits to use this model.")

          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Billing error: {str(billing_error)}")

        # Create user message
        user_message = MessageModel(
          chat_session_id=chat_id,
          parent_message_id=parent_id,
          model_id=model_id,
          content=message_data.content,
          prompt_id=instruction_id or None,
          role=MessageRole.USER,
          _metadata={"knowledge_base_ids": message_data.knowledge_base_ids or []},
          created_at=datetime.now(timezone.utc),
        )
        session.add(user_message)
        await session.commit()
        await session.refresh(user_message)
        self.logger.info(f"Created user message: {user_message.id}")

        # Handle file uploads if any
        files: List[Union[File, Image]] = []
        file_content_for_prompt = ""

        if message_data.file_uploads:
          # Fetch the file uploads from the database
          query = select(ChatUploadModel).filter(ChatUploadModel.id.in_(message_data.file_uploads))
          result = await session.execute(query)
          file_uploads = result.scalars().all()

          for file_upload in file_uploads:
            # Create a new ChatUploadModel linking to this message
            new_upload = ChatUploadModel(
              message_id=user_message.id,
              filename=file_upload.filename,
              content_type=file_upload.content_type,
              file_size=file_upload.file_size,
              url=file_upload.url,
              _metadata=file_upload._metadata,
            )
            session.add(new_upload)

            # Handle file processing based on provider
            if llm_model.provider == "deepseek":
              # For DeepSeek: Extract text content using Agno readers
              try:
                extracted_text = await extract_file_content(file_upload.url, file_upload.filename, file_upload.content_type)
                if extracted_text:
                  file_content_for_prompt += f"\n\n--- File: {file_upload.filename} ---\n{extracted_text}\n"
                  self.logger.info(f"Extracted content from {file_upload.filename} for DeepSeek")
              except Exception as e:
                self.logger.error(f"Failed to extract content from {file_upload.filename}: {str(e)}")
                file_content_for_prompt += f"\n\n--- File: {file_upload.filename} ---\n[Error processing file: {str(e)}]\n"
            else:
              # For other providers: Use existing file handling
              if file_upload.content_type.startswith("image/"):
                files.append(Image(url=file_upload.url))
              else:
                files.append(File(url=file_upload.url, mime_type=file_upload.content_type))

          # Commit the new file uploads
          await session.commit()

          if llm_model.provider == "deepseek":
            self.logger.info(f"Processed {len(file_uploads)} files for DeepSeek with text extraction")
          else:
            self.logger.info(f"Processed {len(files)} file uploads for {llm_model.provider}")

        # Store the chat_id and is_new_chat at a higher scope for access in the async generator
        stored_chat_id = chat_id

        # Search knowledge bases if provided
        enhanced_prompt = prompt if prompt is not None else ""
        if hasattr(message_data, "knowledge_base_ids") and message_data.knowledge_base_ids:
          try:
            from services.kb.service import KnowledgeBaseService

            kb_service = KnowledgeBaseService(self.acquire)

            kb_context_parts = []
            for kb_id in message_data.knowledge_base_ids:
              try:
                chunks = await kb_service.post_search_chunks(
                  org_id=org_id,
                  kb_id=UUID(kb_id),
                  query=message_data.content,
                  limit=getattr(message_data, "kb_search_limit", 10),
                  score_threshold=0.1,
                  session=session,
                  user=user,
                )

                for chunk in chunks:
                  kb_context_parts.append(f"[Knowledge Base Context]: {chunk.content}")

              except Exception as e:
                self.logger.error(f"Error searching KB {kb_id}: {str(e)}")
                continue

            if kb_context_parts:
              kb_context = "\n\n".join(kb_context_parts)
              base_prompt = prompt if prompt is not None else ""
              enhanced_prompt = f"""{base_prompt}

KNOWLEDGE BASE CONTEXT:
{kb_context}

Use the above context to answer the user's question when relevant. If the context doesn't contain relevant information, use your high IQ to answer

"""
              self.logger.info(f"Enhanced prompt with context from {len(message_data.knowledge_base_ids)} knowledge bases")

          except Exception as e:
            self.logger.error(f"Error processing knowledge bases: {str(e)}")
            enhanced_prompt = prompt if prompt is not None else ""

        # Add file content to prompt for DeepSeek
        if file_content_for_prompt:
          enhanced_prompt += f"\n\nFILE CONTENT:{file_content_for_prompt}\n"
          self.logger.info("Added extracted file content to prompt")

        # generate a streaming response
        async def generate_model_response() -> AsyncGenerator[str, None]:
          full_response = ""
          buffer: list[str] = []
          token_count = 0
          input_tokens = 0

          # Stream the response
          self.logger.debug(f"Sending message to {llm_model.provider} {llm_model.version}")

          # Analyze user intent to determine if image generation is needed
          from libs.chats.v1.intent_analysis import UserIntent, get_intent_service

          intent_service = get_intent_service()

          user_intent = await intent_service.analyze_intent(message_data.content)

          # Choose appropriate agent based on detected intent
          if user_intent == UserIntent.IMAGE_GENERATION:
            self.logger.info(f"Detected image generation intent for message: {message_data.content}")
            chat_method = self.llm_factory.image_chat
          else:
            self.logger.info(f"Detected normal chat intent for message: {message_data.content}")
            chat_method = self.llm_factory.chat

          async for token in chat_method(
            provider=llm_model.provider,
            llm=llm_model.version,
            chat_session_id=chat_id,
            message=message_data.content,
            assets=files,
            prompt=enhanced_prompt,  # Use enhanced prompt with KB context
            temperature=effective_temp,
            max_tokens=effective_max,
            top_p=effective_top_p,
            thinking=getattr(message_data, "thinking", False),
          ):
            # Handle streaming
            if token.content is not None:
              # Check if this is a reasoning step by checking token type or content type
              if hasattr(token, "type") and token.type == "reasoning":
                # This is a reasoning step - send separately so frontend can show/hide it
                reasoning_data = {"type": "reasoning", "content": str(token.content)}
                yield f"data: {json.dumps(reasoning_data)}\n\n"
              elif "ReasoningStep" in str(type(token.content)):
                reasoning_data = {"type": "reasoning", "content": str(token.content)}
                yield f"data: {json.dumps(reasoning_data)}\n\n"
              else:
                buffer.append(token.content)
                full_response += token.content
                token_count += 1  # Simple token counting

            if len(buffer) >= self.chunk_size:
              d = {"message": "".join(buffer)}
              yield f"data: {json.dumps(d)}\n\n"
              buffer = []

          # Send remaining buffer
          if buffer:
            d = {"message": "".join(buffer)}
            yield f"data: {json.dumps(d)}\n\n"

          yield f"data: {json.dumps({'message': 'DONE'})}\n\n"

          # Process generated images if this was an image generation request
          processed_response = full_response
          if user_intent == UserIntent.IMAGE_GENERATION:
            # Create AI message first to get the ID
            ai_message = MessageModel(
              chat_session_id=chat_id,
              parent_message_id=user_message.id,
              model_id=model_id,
              content=full_response,  # Temporary content
              role=MessageRole.MODEL,
              created_at=datetime.now(timezone.utc),
            )
            session.add(ai_message)
            await session.commit()
            await session.refresh(ai_message)

            # Process images and get updated response text
            processed_response = await self._process_generated_images(
              response_text=full_response,
              chat_id=chat_id,
              org_id=org_id,
              ai_message_id=ai_message.id,
              session=session,
            )

            # Update the AI message with processed content
            ai_message.content = processed_response
            await session.commit()

            self.logger.info(f"Created AI image message: {ai_message.id} with processed images")
          else:
            # Create AI message normally for regular chat
            ai_message = MessageModel(
              chat_session_id=chat_id,
              parent_message_id=user_message.id,
              model_id=model_id,
              content=full_response,
              role=MessageRole.MODEL,
              created_at=datetime.now(timezone.utc),
            )
            session.add(ai_message)
            await session.commit()
            await session.refresh(ai_message)
            self.logger.info(f"Created AI response message: {ai_message.id}")

          # Update chat title if needed
          response_message = processed_response
          if is_new_chat:
            await self._update_chat_name(
              response_message,
              is_new_chat,
              org_id,
              user_id,
              session,
              stored_chat_id,
            )

          # IMPORTANT: Finalize the billing by converting HOLD to DEBIT
          if charge:
            try:
              # Estimate input tokens based on message length (simple approximation)
              input_tokens = len(message_data.content.split())
              # Use token_count as output tokens
              output_tokens = token_count

              # Get pricing from model (with null check)
              pricing = {"input": 1, "output": 1}  # Default fallback values
              if hasattr(llm_model, "model_metadata"):
                pricing = llm_model.model_metadata.get("credits_per_1000_tokens", {"input": 1, "output": 1})

              # Calculate total tokens with model-specific weights
              input_ratio = pricing.get("input", 1)
              output_ratio = pricing.get("output", 1)
              weighted_total = (input_tokens * input_ratio) + (output_tokens * output_ratio)

              # Finalize billing with token metrics
              await charge.calculate_and_update(
                metadata={
                  "input_tokens": input_tokens,
                  "output_tokens": output_tokens,
                  "total_tokens": int(weighted_total),
                  "input_ratio": input_ratio,
                  "output_ratio": output_ratio,
                  "user_message_id": str(user_message.id),
                  "ai_message_id": str(ai_message.id),
                },
                status="completed",
              )
              self.logger.info(f"Successfully finalized charge for chat {chat_id}")
            except Exception as e:
              self.logger.error(f"Error finalizing LLM chat billing: {str(e)}")
              # Attempt to complete billing anyway with basic info
              try:
                await charge.update(additional_metadata={"billing_error": str(e), "fallback_billing": True})
              except Exception as charge_error:
                self.logger.error(f"Failed to finalize charge: {str(charge_error)}")

        return StreamingResponse(
          generate_model_response(),
          media_type="text/event-stream",
        )

      # if chatting with an agent
      elif agent_id:
        # check if agent exists
        query = select(AgentModel).where(
          and_(
            AgentModel.id == agent_id,
            AgentModel.is_active,
          )
        )
        result = await session.execute(query)
        agent = result.scalar_one_or_none()
        if not agent:
          raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
          )

        if not agent.is_active:
          raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not active",
          )

        agent_version = agent.version
        # Create the user's message
        user_message = MessageModel(
          chat_session_id=chat_id,
          parent_message_id=parent_id,
          agent_id=agent_id,
          content=message_data.content,
          role=MessageRole.USER,
          _metadata={"knowledge_base_ids": message_data.knowledge_base_ids or []},
          created_at=datetime.now(timezone.utc),
        )
        session.add(user_message)
        await session.commit()
        await session.refresh(user_message)

        request_id = user_message.id
        agent_base_url = settings.agent_base_url

        # Function to generate streaming response
        async def generate_agent_response() -> AsyncGenerator[str, None]:
          full_response = ""
          buffer: list[str] = []
          try:
            # Send a message to the agent with a higher timeout
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:  # Set timeout to 30 seconds
              slug = agent.name.lower().replace(" ", "-")
              url = f"{agent_base_url}/{slug}/{agent_version}/invoke"
              print(f"Agent URL: {url}")
              params: dict = {
                "user_id": str(user_id),
                "org_id": str(org_id),
              }
              payload = {"query": message_data.content}  # Adjust payload to match the agent's expected input
              headers = {"x-request-id": str(request_id)}
              async with client.stream("POST", url, json=payload, headers=headers, params=params) as response:
                self.logger.debug(f"Agent response: {response}")
                if response.status_code != 200:
                  error_detail = f"Agent returned an error: {response.status_code}"
                  self.logger.error(error_detail)
                  yield f"data: {json.dumps({'error': error_detail})}\n\n"
                  return

                # Simplified streaming - pass through chunks as-is
                async for chunk in response.aiter_text():
                  if "DONE" in chunk:
                    break
                  try:
                    # Parse the JSON chunk to extract the "message" content
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    message = chunk_data.get("message", "")
                    full_response += message  # Append the message content to full_response
                    yield chunk
                  except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse chunk: {chunk}")
                    yield f"data: {json.dumps({'error': 'Invalid chunk format'})}\n\n"

            # Yield any remaining data in the buffer
            if buffer:
              d = {"message": "".join(buffer)}
              yield f"data: {json.dumps(d)}\n\n"

            # Signal the end of the stream
            yield f"data: {json.dumps({'message': 'DONE'})}\n\n"

            # Create the agent's message
            agent_message = MessageModel(
              chat_session_id=chat_id,
              parent_message_id=user_message.id,
              agent_id=agent_id,
              content=full_response,
              role=MessageRole.AGENT,
              created_at=datetime.now(timezone.utc),
            )
            session.add(agent_message)
            await session.commit()
            await session.refresh(agent_message)

          except httpx.ReadTimeout as e:
            self.logger.error(f"Timeout while waiting for agent response: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'Timeout while waiting for agent response'})}\n\n"
          except Exception as e:
            self.logger.error(f"Error during streaming response: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'An error occurred during streaming'})}\n\n"
          # Get token counts from Agno session data
          if charge:
            try:
              # Query Agno session data
              query = text("""
                SELECT memory FROM __agno_chat_sessions
                WHERE session_id = :session_id
                ORDER BY created_at DESC LIMIT 1
              """)
              result = await session.execute(query, {"session_id": str(chat_id)})
              memory_data = result.scalar_one_or_none()
              if memory_data and "runs" in memory_data:
                # Get the runs - handle both single object and array cases
                runs = memory_data["runs"]

                # Get the last run (most recent interaction)
                last_run = runs[-1] if isinstance(runs, list) else runs

                if "response" in last_run and "metrics" in last_run["response"]:
                  metrics = last_run["response"]["metrics"]

                  # Get the current run's index in the history
                  run_index = len(metrics.get("input_tokens", [])) - 1 if isinstance(metrics.get("input_tokens", []), list) else 0

                  # Extract tokens for the current exchange using the correct index
                  input_tokens = (
                    metrics.get("input_tokens", [0])[run_index]
                    if isinstance(metrics.get("input_tokens", []), list)
                    else metrics.get("input_tokens", 0)
                  )
                  output_tokens = (
                    metrics.get("output_tokens", [0])[run_index]
                    if isinstance(metrics.get("output_tokens", []), list)
                    else metrics.get("output_tokens", 0)
                  )

                  self.logger.info(f"Token usage: input={input_tokens}, output={output_tokens}")

                  # Get pricing from model (with null check)
                  pricing = {"input": 1, "output": 1}  # Default fallback values
                  if llm_model and hasattr(llm_model, "model_metadata"):
                    pricing = llm_model.model_metadata.get("credits_per_1000_tokens", {"input": 1, "output": 1})

                  # Calculate total tokens with model-specific weights
                  input_ratio = pricing.get("input", 1)
                  output_ratio = pricing.get("output", 1)
                  weighted_total = (input_tokens * input_ratio) + (output_tokens * output_ratio)

                  # Pass the pricing info to the charge calculation
                  await charge.calculate_and_update(
                    metadata={
                      "input_tokens": input_tokens,
                      "output_tokens": output_tokens,
                      "total_tokens": int(weighted_total),  # Use weighted total
                      "input_ratio": input_ratio,
                      "output_ratio": output_ratio,
                      "user_message_id": str(user_message.id),
                    },
                    status="completed",
                  )
                else:
                  # Fallback if metrics not found
                  await charge.update(additional_metadata={"billing_error": "No metrics in response"})
              else:
                await charge.update(additional_metadata={"billing_error": "No runs in memory"})

            except Exception as e:
              self.logger.error(f"Error finalizing chat billing: {str(e)}")
              # Attempt to complete billing anyway
              try:
                await charge.update(additional_metadata={"billing_error": str(e), "fallback_billing": True})
              except Exception as charge_error:
                self.logger.error(f"Failed to finalize charge: {str(charge_error)}")

        return StreamingResponse(generate_agent_response(), media_type="text/event-stream")

      # Handle agent-based processing (if not using model_id)
      if agent_id and not model_id:
        # Return a placeholder error for now
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Agent-based chat not implemented yet")

      # Default fallback in case no specific processing was handled
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid processing method available for the provided parameters")

    except Exception as e:
      from traceback import print_exc

      print_exc()

      # Release charge if error occurs and it was successfully created
      if "charge" in locals() and charge and hasattr(charge, "transaction_id") and charge.transaction_id:
        try:
          await charge.delete(reason=f"Error processing message: {str(e)}")
        except Exception as release_error:
          self.logger.error(f"Failed to release charge: {str(release_error)}")

      # Re-raise HTTP exceptions with their original status code and message
      if isinstance(e, HTTPException):
        raise e

      # Only convert non-HTTP exceptions to a 500 error
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error sending message: {str(e)}")

  async def delete_session(
    self,
    session_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "delete")),
  ) -> Dict[str, str]:
    """Delete a chat session."""
    # Check if chat session exists
    query = select(ChatModel).where(
      and_(
        ChatModel.id == session_id,
        ChatModel.user_id == user["id"],
        ChatModel.org_id == user["org_id"],
      )
    )
    result = await session.execute(query)
    db_session = result.scalar_one_or_none()

    if not db_session:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    # Delete the chat session
    await session.delete(db_session)
    await session.commit()

    return {"message": "Chat session deleted successfully"}

  async def post_bulk_delete_sessions(
    self,
    delete_data: BulkDeleteRequest,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "delete")),
  ) -> Dict[str, str]:
    """Delete multiple chat sessions."""
    # Check if any of the requested chat sessions exist
    query = select(ChatModel).where(
      and_(
        ChatModel.id.in_(delete_data.chat_ids),
        ChatModel.user_id == user["id"],
        ChatModel.org_id == user["org_id"],
      )
    )
    result = await session.execute(query)
    db_sessions = result.scalars().all()

    if not db_sessions:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No chat sessions found to delete")

    for db_session in db_sessions:
      await session.delete(db_session)

    await session.commit()

    return {"message": f"{len(db_sessions)} chat sessions deleted successfully"}

  async def post_upload_file(
    self,
    org_id: UUID,
    file: UploadFile,
    chat_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ChatFileUploadResponse:
    """Upload a file to the public S3 bucket."""
    file_content = await file.read()
    salt = str(uuid.uuid4())
    file_name = None
    if file.filename:
      _, extension = file.filename.rsplit(".", 1)
      file_name = f"{salt}.{extension}"
    else:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Filename is missing",
      )
    if chat_id:
      key = f"{org_id}/{chat_id}/{file_name}"
    else:
      key = f"chats/{org_id}/{file_name}"
    effective_content_type = (
      (file.content_type or "").strip()
      or (mimetypes.guess_type(file.filename)[0] if file.filename else None)
      or ("application/pdf" if (file.filename or "").lower().endswith(".pdf") else "application/octet-stream")
    )
    await self.s3_client.upload_file(file=BytesIO(file_content), key=key, content_type=effective_content_type)
    url = await self.s3_client.get_presigned_url(key=key, expires_in=3600 * 24 * 30)
    metadata = {
      "org_id": str(org_id),
      "filename": file.filename,
      "chat_id": str(chat_id) if chat_id else None,
    }
    db_upload = ChatUploadModel(
      filename=file_name,
      content_type=effective_content_type,
      file_size=file.size,
      url=url,
      _metadata=metadata,
    )
    session.add(db_upload)
    await session.commit()

    return ChatFileUploadResponse(id=db_upload.id, url=url)

  async def post_transcribe(
    self,
    audio_data: Optional[bytes] = None,
    content_type: Optional[str] = None,
    language: str = "en-US",
    file: Optional[UploadFile] = None,
    user: dict = Depends(JWTBearer()),
  ) -> Dict[str, str]:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio data in bytes (optional if file is provided)
        content_type: MIME type of the audio (e.g., "audio/mp3", "audio/wav")
        language: Language code for transcription (default: "en-US")
        file: Uploaded audio file (optional if audio_data is provided)

    Returns:
        Dictionary with transcribed text and status
    """
    try:
      # Handle file upload case
      if file and not audio_data:
        file_content = await file.read()
        file_size = len(file_content)

        if file_size == 0:
          raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

        # Use tempfile module instead of hardcoded path
        # Create temporary file with proper extension
        temp_dir = tempfile.gettempdir()
        file_name = file.filename or "temp_audio_file"
        file_name = file_name.replace(" ", "_")  # Replace spaces with underscores
        temp_path = os.path.join(temp_dir, file_name)

        try:
          with open(temp_path, "wb") as f:
            f.write(file_content)
        except Exception as write_error:
          print(f"Warning: Could not save debug file: {str(write_error)}")
          # Continue even if we can't save the debug file

        # Pass correct arguments to transcribe function
        transcribed_text = await transcribe(source=file_content, content_type=file.content_type, language=language)
      # Handle raw bytes case
      elif audio_data and content_type:
        transcribed_text = await transcribe(source=audio_data, content_type=content_type, language=language)
      else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either audio_data with content_type or file must be provided")

      return {"text": transcribed_text, "status": "success"}
    except Exception as e:
      from traceback import print_exc

      print_exc()

      # More detailed error message
      error_message = str(e)
      error_type = type(e).__name__

      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error transcribing audio: {error_type}: {error_message}")

  async def post_prompt(
    self,
    data: TextInput,
    user: dict = Depends(JWTBearer()),
  ) -> StreamingResponse:
    """Generate prompts for a given text with streaming response."""
    try:

      async def content_generator():
        try:
          async for token in generate_prompts_stream(data.text, data.prompt_type, data.num_prompts, data.model):
            yield f"data: {json.dumps({'content': token})}\n\n"

          # Send a DONE message when complete
          yield f"data: {json.dumps({'content': 'DONE'})}\n\n"
        except Exception as e:
          yield f"data: {json.dumps({'error': str(e)})}\n\n"

      return StreamingResponse(content_generator(), media_type="text/event-stream")
    except Exception as e:
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating prompts: {str(e)}")

  ### PRIVATE METHODS ###

  def _get_effective_settings(
    self, chat: ChatModel, temperature: Optional[float], max_tokens: Optional[int], top_p: Optional[float]
  ) -> tuple[Optional[float], Optional[int], Optional[float]]:
    """Get effective settings: query params override saved settings."""
    saved_settings = chat._metadata.get("settings", {}) if chat._metadata else {}

    effective_temp = temperature if temperature is not None else saved_settings.get("temperature")
    effective_max = max_tokens if max_tokens is not None else saved_settings.get("max_tokens")
    effective_top_p = top_p if top_p is not None else saved_settings.get("top_p")

    return effective_temp, effective_max, effective_top_p

  def _save_settings_to_chat(self, chat: ChatModel, temperature: Optional[float], max_tokens: Optional[int], top_p: Optional[float]) -> None:
    """Save provided settings to chat metadata."""
    if not chat._metadata:
      chat._metadata = {}
    if "settings" not in chat._metadata:
      chat._metadata["settings"] = {}

    if temperature is not None:
      chat._metadata["settings"]["temperature"] = temperature
    if max_tokens is not None:
      chat._metadata["settings"]["max_tokens"] = max_tokens
    if top_p is not None:
      chat._metadata["settings"]["top_p"] = top_p

  def _get_chat_settings(self, chat: ChatModel) -> Optional[ChatSettings]:
    """Extract ChatSettings from chat metadata."""
    if not chat._metadata or "settings" not in chat._metadata:
      return None

    settings_data = chat._metadata["settings"]
    return ChatSettings(temperature=settings_data.get("temperature"), max_tokens=settings_data.get("max_tokens"), top_p=settings_data.get("top_p"))

  async def _update_chat_name(
    self,
    response_text: str,
    is_new_chat: bool,
    org_id: UUID,
    user_id: UUID,
    session: AsyncSession,
    chat_id: Optional[UUID] = None,
  ):
    """Background task to update the chat name after the response is sent."""

    try:
      if not is_new_chat:
        return
      if not chat_id:
        return

      # Get the chat
      query = select(ChatModel).where(ChatModel.id == chat_id)
      result = await session.execute(query)
      db_session = result.scalar_one_or_none()

      if not db_session or db_session.title != "New Chat":
        return

      # Generate a name based on the AI response
      chat_name = await generate_chat_name(response_text)

      # Update the chat title
      db_session.title = chat_name
      await session.commit()

      # Broadcast the updated chat info via WebSocket
      response_data = {
        "id": str(chat_id),
        "title": chat_name,
        "user_id": str(user_id),
        "org_id": str(org_id),
      }
      await self.ws_manager.broadcast(
        org_id,
        {
          "data": response_data,
        },
        "chats",
        "write",
      )

      self.logger.debug(f"Updated chat {chat_id} title to: {chat_name}")

    except Exception as e:
      self.logger.error(f"Error updating chat name: {str(e)}")

  async def _get_prompt(self, prompt_id: UUID, session: AsyncSession) -> PromptModel:
    """Get a prompt from the database."""
    try:
      query = select(PromptModel).where(PromptModel.id == prompt_id)
      result = await session.execute(query)
      prompt = result.scalar_one_or_none()
      if not prompt:
        self.logger.error(f"Prompt not found: {prompt_id}")
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail="Prompt not found",
        )
      return prompt

    except Exception as e:
      self.logger.error(f"Error getting prompt: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Error getting prompt",
      )

  async def _process_generated_images(
    self,
    response_text: str,
    chat_id: UUID,
    org_id: UUID,
    ai_message_id: UUID,
    session: AsyncSession,
  ) -> str:
    """Process DALL-E URLs in response text and replace with our S3 URLs."""
    try:
      # Find all DALL-E URLs in the response using regex
      dalle_url_pattern = r"https://oaidalleapiprodscus\.blob\.core\.windows\.net/[^\s\)]+\.png[^\s\)]*"
      dalle_urls = re.findall(dalle_url_pattern, response_text)

      if not dalle_urls:
        return response_text

      processed_text = response_text

      for dalle_url in dalle_urls:
        try:
          self.logger.info(f"Processing DALL-E image URL: {dalle_url}")

          # Download the image from DALL-E URL
          async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(dalle_url)
            response.raise_for_status()
            image_bytes = response.content

          # Generate our S3 key
          image_id = str(uuid.uuid4())
          s3_key = f"{org_id}/{chat_id}/generated-{image_id}.png"

          # Upload to our S3
          our_image_url = await self.s3_client.upload_file(file=BytesIO(image_bytes), key=s3_key, content_type="image/png")
          # Create upload record in database
          upload_record = ChatUploadModel(
            message_id=ai_message_id,
            filename=f"generated-{image_id}.png",
            content_type="image/png",
            file_size=len(image_bytes),
            url=our_image_url,
            _metadata={"generated": True, "type": "image_generation", "original_dalle_url": dalle_url},
          )
          session.add(upload_record)

          # Replace DALL-E URL with our URL in the response text
          processed_text = processed_text.replace(dalle_url, our_image_url)

          self.logger.info(f"Successfully processed image: {dalle_url} -> {our_image_url}")

        except Exception as e:
          self.logger.error(f"Error processing DALL-E URL {dalle_url}: {str(e)}")
          # Continue with other URLs if one fails
          continue

      # Commit all upload records
      await session.commit()

      return processed_text

    except Exception as e:
      self.logger.error(f"Error in _process_generated_images: {str(e)}")
      return response_text  # Return original text if processing fails

  async def get_available_knowledge_bases(
    self,
    org_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("kb", "read")),
  ):
    """Get available knowledge bases for chat."""
    try:
      from services.kb.service import KnowledgeBaseService

      kb_service = KnowledgeBaseService(self.acquire)
      return await kb_service.get_list(org_id=org_id, session=session, user=user)
    except Exception as e:
      self.logger.error(f"Error getting available knowledge bases: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Error getting available knowledge bases",
      )
