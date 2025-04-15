import json
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from typing import AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID

from agno.media import File, Image
from datetime import datetime, timezone
from io import BytesIO

from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db

from dependencies.security import RBAC, JWTBearer
from libs.chats.v1 import LLMFactory, generate_prompts_stream
from libs.s3.v1 import S3Client
from libs.speech.v1 import transcribe
from dependencies.security import RBAC

from models import ChatModel, ChatUploadModel, LLMModel, MessageModel
from services.__base.acquire import Acquire

from .schema import (
  BulkDeleteRequest,
  ChatFileUploadResponse,
  ChatSessionCreate,
  ChatSessionResponse,
  ChatSessionUpdate,
  ChatSessionWithMessages,
  ChatStatus,
  MessageCreate,
  MessageResponse,
  MessageRole,
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
    "get=prompt",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.llm_factory = LLMFactory()
    self.chunk_size = 10
    self.s3_client = S3Client(bucket="chats")

  async def post(
    self,
    org_id: UUID,
    data: ChatSessionCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> JSONResponse:
    """Create a new chat session."""
    db_session = ChatModel(
      title=data.title,
      status=data.status,
      user_id=user["id"],
      org_id=user["org_id"],
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
    for msg in messages:
      message_responses.append(
        MessageResponse(
          id=msg.id,
          content=msg.content,
          role=msg.role,
          chat_session_id=msg.chat_session_id,
          parent_message_id=msg.parent_message_id,
          model_id=msg.model_id,
          agent_id=msg.agent_id,
          metadata=msg._metadata,
          created_at=msg.created_at.isoformat(),
        )
      )

    return ChatSessionWithMessages(
      id=db_session.id,
      title=db_session.title,
      status=db_session.status,
      org_id=db_session.org_id,
      user_id=db_session.user_id,
      metadata=db_session._metadata,
      created_at=db_session.created_at.isoformat(),
      updated_at=db_session.updated_at.isoformat(),
      messages=message_responses,
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
        created_at=db_session.created_at.isoformat(),
        updated_at=db_session.updated_at.isoformat(),
      )
      for db_session in db_sessions
    ]

  async def post_send_message(
    self,
    chat_id: UUID,
    org_id: UUID,
    message_data: MessageCreate,
    model_id: Optional[UUID] = None,
    agent_id: Optional[UUID] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> StreamingResponse:
    """Send a message to a chat session."""
    try:
      if not model_id and not agent_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either model_id or agent_id must be provided")

      # Check if chat session exists
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

      # get the parent id from the last message of the chat session
      query = select(MessageModel).where(MessageModel.chat_session_id == chat_id).order_by(MessageModel.created_at.desc())
      result = await session.execute(query)
      last_message = result.scalars().first()
      parent_id = last_message.id if last_message else None

      # if chatting with a LLM model
      if model_id:
        # check if LLM model exists
        query = select(LLMModel).where(LLMModel.id == model_id)
        result = await session.execute(query)
        llm_model = result.scalar_one_or_none()
        if not llm_model:
          raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM model not found")

        # Create the users message
        user_message = MessageModel(
          chat_session_id=chat_id,
          parent_message_id=parent_id,
          model_id=model_id,
          content=message_data.content,
          role=MessageRole.USER,
          created_at=datetime.now(timezone.utc),
        )
        session.add(user_message)
        await session.commit()
        await session.refresh(user_message)

        files: List[Union[File, Image]] = []
        if message_data.file_uploads:
          # Fetch the file uploads from the database
          query = select(ChatUploadModel).filter(ChatUploadModel.id.in_(message_data.file_uploads))
          result = await session.execute(query)
          file_uploads = result.scalars().all()

          for file_upload in file_uploads:
            # Create the appropriate File object based on mimetype
            if file_upload.content_type.startswith("image/"):
              files.append(Image(url=file_upload.url))
            else:
              files.append(File(url=file_upload.url, mime_type=file_upload.content_type))

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

          # Commit the new file uploads
          await session.commit()

        # generate a streaming response
        async def generate_response() -> AsyncGenerator[str, None]:
          full_response = ""
          buffer: list[str] = []
          print(f"message_data.content: {message_data.content}")
          async for token in self.llm_factory.chat(
            provider=llm_model.provider,
            llm=llm_model.version,
            chat_session_id=chat_id,
            message=message_data.content,
            assets=files,
          ):
            buffer.append(token.content)
            full_response += token.content
            if len(buffer) >= self.chunk_size:
              d = {"message": "".join(buffer)}
              yield f"data: {json.dumps(d)}\n\n"
              buffer = []

          if buffer:
            d = {"message": "".join(buffer)}
            yield f"data: {json.dumps(d)}\n\n"

          yield f"data: {json.dumps({'message': 'DONE'})}\n\n"

          # Create ai message
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

    except Exception:
      from traceback import print_exc

      print_exc()
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error sending message")

    return StreamingResponse(generate_response(), media_type="text/event-stream")

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
    chat_id: UUID,
    file: UploadFile,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("chats", "write")),
  ) -> ChatFileUploadResponse:
    """Upload a file to the public S3 bucket."""
    file_content = await file.read()
    key = f"{org_id}/{chat_id}/{file.filename}"
    await self.s3_client.upload_file(file=BytesIO(file_content), key=key)
    url = await self.s3_client.get_presigned_url(key=key, expires_in=3600 * 24 * 30)
    metadata = {
      "org_id": str(org_id),
      "chat_id": str(chat_id),
    }
    db_upload = ChatUploadModel(filename=file.filename, content_type=file.content_type, file_size=file.size, url=url, _metadata=metadata)
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

  async def get_prompt(
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
