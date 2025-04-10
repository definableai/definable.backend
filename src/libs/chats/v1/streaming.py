import asyncio
from typing import AsyncGenerator

from agno.agent import Agent, RunResponse
from agno.media import File
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage

from config.settings import settings


class LLMFactory:
  """Factory for creating Agno agents with different LLM providers."""

  def __init__(self):
    # Initialize model configurations
    self.model_configs = {
      "openai": {
        "models": {
          "gpt-4o": {"class": OpenAIChat, "id": "gpt-4o"},
          "gpt-4o-mini": {"class": OpenAIChat, "id": "gpt-4o-mini"},
          "o1-preview": {"class": OpenAIChat, "id": "o1-preview"},
          "gpt-3.5-turbo": {"class": OpenAIChat, "id": "gpt-3.5-turbo"},
          "o1": {"class": OpenAIChat, "id": "o1"},
        },
      },
      "anthropic": {
        "models": {
          "claude-3.7-sonnet": {"class": Claude, "id": "claude-3-7-sonnet-latest"},
          "claude-3.5-sonnet": {"class": Claude, "id": "claude-3-5-sonnet-latest"},
          "claude-3.5-haiku": {"class": Claude, "id": "claude-3-5-haiku-latest"},
        },
      },
    }

    # Configure storage for conversation persistence
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    self.storage = PostgresStorage(table_name="chat_sessions", db_url=db_url, schema="public")
    self.storage.create()

  async def chat(
    self, provider: str, llm: str, chat_session_id: str, message: str, memory_size: int = 100, files: list[File] = []
  ) -> AsyncGenerator[RunResponse, None]:
    """Stream chat responses using Agno agent.

    Args:
        llm: The LLM model identifier
        chat_session_id: Unique ID for the chat session
        message: User's input message

    Yields:
        Streaming response tokens
    """
    # Create agent with storage for memory retention
    agent = Agent(
      model=self.model_configs[provider]["models"][llm]["class"](id=self.model_configs[provider]["models"][llm]["id"]),  # type: ignore
      storage=self.storage,
      markdown=True,
      stream=True,
      add_history_to_messages=True,
      session_id=chat_session_id,
      num_history_responses=memory_size,
    )
    async for token in await agent.arun(message, files=files):
      yield token


if __name__ == "__main__":

  async def main():
    llm_factory = LLMFactory()

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
