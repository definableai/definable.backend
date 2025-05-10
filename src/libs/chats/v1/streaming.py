import asyncio
from typing import AsyncGenerator, List, Optional, Sequence, Type, Union
from uuid import UUID

from agno.agent import Agent, RunResponse
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.media import File, Image


from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek

from agno.models.deepseek import DeepSeek

from agno.storage.postgres import PostgresStorage

from config.settings import settings

# Define a type alias for model classes
ModelClass = Union[Type[OpenAIChat], Type[Claude], Type[DeepSeek]]
ModelClass = Union[Type[OpenAIChat], Type[Claude], Type[DeepSeek]]


class LLMFactory:
  """Factory for creating Agno agents with different LLM providers."""

  # Map provider names to their model classes
  PROVIDER_MODELS: dict[str, ModelClass] = {
    "openai": OpenAIChat,
    "anthropic": Claude,
    "deepseek": DeepSeek,
    "deepseek": DeepSeek,
  }

  def __init__(self):
    # Configure storage for conversation persistence
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    self.storage = PostgresStorage(table_name="__agno_chat_sessions", db_url=db_url, schema="public")
    self.storage.create()

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
    assets: Sequence[Union[File, Image]] = [],
    knowledge_base: Optional[LangChainKnowledgeBase] = None,
    memory_size: int = 100,
  ) -> AsyncGenerator[RunResponse, None]:
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
    # Create agent with storage for memory retention
    agent = Agent(
      model=model_class(id=llm),  # type: ignore
      storage=self.storage,
      markdown=True,
      stream=True,
      add_history_to_messages=True,
      session_id=str(chat_session_id),
      num_history_responses=memory_size,
      knowledge=knowledge_base,
      search_knowledge=True
    )
    async for token in await agent.arun(message, files=files or None, images=images or None):
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
