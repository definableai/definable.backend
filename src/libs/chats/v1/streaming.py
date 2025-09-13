import asyncio
from typing import Any, AsyncGenerator, List, Optional, Sequence, Type, Union
from uuid import UUID

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.media import File, Image
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from agno.tools.dalle import DalleTools
from agno.tools.reasoning import ReasoningTools

from config.settings import settings

# Define a type alias for model classes
ModelClass = Union[Type[OpenAIChat], Type[Claude], Type[DeepSeek]]


class LLMFactory:
  """Factory for creating Agno agents with different LLM providers."""

  # Map provider names to their model classes
  PROVIDER_MODELS: dict[str, ModelClass] = {
    "openai": OpenAIChat,
    "anthropic": Claude,
    "deepseek": DeepSeek,
  }

  def __init__(self):
    # Configure storage for conversation persistence
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    self.storage = PostgresDb(session_table="__agno_chat_sessions", db_url=db_url, db_schema="public")

  def get_model_class(self, provider: str) -> ModelClass:
    """Get the model class for a given provider name."""
    if provider not in self.PROVIDER_MODELS:
      raise ValueError(f"Unsupported provider: {provider}")
    return self.PROVIDER_MODELS[provider]

  async def chat(
    self,
    chat_session_id: str | UUID,
    llm: str,
    provider: str,
    message: str,
    prompt: Optional[str] = None,
    assets: Sequence[Union[File, Image]] = [],
    memory_size: int = 100,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    thinking: bool = False,
  ) -> AsyncGenerator[Any, None]:
    """Stream chat responses using Agno agent.

    Args:
        llm: The LLM model identifier
        chat_session_id: Unique ID for the chat session
        message: User's input message

    Yields:
        Streaming response tokens
    """

    images: List[Image] = []
    files: List[File] = []

    for asset in assets:
      if isinstance(asset, Image):
        images.append(asset)
      elif isinstance(asset, File):
        files.append(asset)

    model_class = self.get_model_class(provider)

    # Anthropic requires max_tokens;
    effective_max_tokens = max_tokens
    if provider == "anthropic" and effective_max_tokens is None:
      effective_max_tokens = 1024
    tools: List[Any] = []
    if thinking:
      tools.append(ReasoningTools(add_instructions=True))

    # Create agent with storage for memory retention
    agent = Agent(
      model=model_class(
        id=llm,
        temperature=temperature,
        max_tokens=effective_max_tokens,
        top_p=top_p,
      ),  # type: ignore
      tools=tools or None,  # type: ignore[arg-type]
      db=self.storage,
      markdown=True,
      stream=True,
      stream_intermediate_steps=thinking,
      add_history_to_context=True,
      read_chat_history=True,
      session_id=str(chat_session_id),
      num_history_runs=memory_size,
      instructions=prompt,
    )
    async for token in agent.arun(message, files=files or None, images=images or None, stream=True):
      yield token

  async def image_chat(
    self,
    chat_session_id: str | UUID,
    llm: str,
    provider: str,
    message: str,
    prompt: Optional[str] = None,
    assets: Sequence[Union[File, Image]] = [],
    memory_size: int = 100,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
  ) -> AsyncGenerator[Any, None]:
    """Stream image generation responses using Agno agent with DalleTools.

    Args:
        llm: The LLM model identifier
        chat_session_id: Unique ID for the chat session
        message: User's input message for image generation
        prompt: Optional system prompt/instructions
        assets: Optional files/images for context
        memory_size: Number of previous messages to include
        temperature: Model temperature parameter
        max_tokens: Maximum tokens to generate
        top_p: Top-p parameter for nucleus sampling

    Yields:
        Streaming response tokens including image generation results
    """

    images: List[Image] = []
    files: List[File] = []

    for asset in assets:
      if isinstance(asset, Image):
        images.append(asset)
      elif isinstance(asset, File):
        files.append(asset)

    model_class = self.get_model_class(provider)

    # Create agent with DalleTools for image generation
    agent = Agent(
      model=model_class(
        id=llm,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
      ),  # type: ignore
      tools=[DalleTools()],  # Enable image generation
      db=self.storage,
      markdown=True,
      stream=True,
      add_history_to_context=True,
      read_chat_history=True,
      session_id=str(chat_session_id),
      num_history_runs=memory_size,
      instructions=prompt
      or (
        "You are an AI assistant that can generate images. When users ask for images, "
        "use the image generation tool to create them based on their descriptions."
      ),
    )

    async for token in agent.arun(message, files=files or None, images=images or None, stream=True):
      yield token


if __name__ == "__main__":

  async def main():
    llm_factory: LLMFactory = LLMFactory()

    # Example usage with streaming
    async for token in llm_factory.chat(
      provider="openai",
      llm="o1",
      chat_session_id="234234234",
      message="who are you and what is your purpose? tell me you model and version",
    ):
      print(token.content, end="", flush=True)  # Print tokens as they come

  # Run the async main function
  asyncio.run(main())
