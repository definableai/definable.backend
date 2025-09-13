import contextlib
from typing import AsyncGenerator, Type, Union
from uuid import UUID

from agno.agent import Agent, RunOutput
from agno.db.postgres import PostgresDb
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.tools.mcp import MultiMCPTools

from config.settings import settings

# Define a type alias for model classes
ModelClass = Union[Type[OpenAIChat], Type[Claude], Type[DeepSeek]]


class MCPPlaygroundFactory:
  """Factory for creating MCP playground sessions with conversation persistence."""

  # Map provider names to their model classes (same as chat service)
  PROVIDER_MODELS: dict[str, ModelClass] = {
    "openai": OpenAIChat,
    "anthropic": Claude,
    "deepseek": DeepSeek,
  }

  def __init__(self):
    # Configure storage for conversation persistence (same as chat service)
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    self.storage = PostgresDb(session_table="__agno_chat_sessions", db_url=db_url, db_schema="public")

  def get_model_class(self, provider: str) -> ModelClass:
    """Get the model class for a given provider name."""
    if provider not in self.PROVIDER_MODELS:
      raise ValueError(f"Unsupported provider: {provider}")
    return self.PROVIDER_MODELS[provider]

  async def chat(
    self,
    session_id: str | UUID,
    mcp_urls: list[str],
    message: str,
    llm: str,
    provider: str,
    memory_size: int = 50,
  ) -> AsyncGenerator[RunOutput, None]:
    """
    Stream chat responses with MCP tools integration.

    Args:
        session_id: Unique ID for the chat session
        mcp_urls: List of MCP server URLs to connect to
        message: User's input message
        llm: The LLM model identifier
        provider: The model provider (openai, anthropic, deepseek)
        memory_size: Number of previous messages to include

    Yields:
        Streaming response tokens
    """

    mcp_tools = MultiMCPTools(
      urls=mcp_urls,
      urls_transports=["streamable-http"] * len(mcp_urls),
      timeout_seconds=30,
    )

    try:
      await mcp_tools.connect()

      model_class = self.get_model_class(provider)

      agent = Agent(
        name="MCP Playground Assistant",
        model=model_class(id=llm),  # type: ignore
        tools=[mcp_tools],
        db=self.storage,
        markdown=True,
        stream=True,
        add_history_to_context=True,
        session_id=str(session_id),
        read_chat_history=True,
        num_history_runs=memory_size,
        stream_intermediate_steps=True,
        instructions="""
  You are an intelligent assistant with access to various tools through MCP (Model Context Protocol).

  IMPORTANT GUIDELINES:
  - Understand user requests in natural language
  - Analyze available tools carefully before choosing which one to use
  - Always choose the most appropriate tool for the user's request
  - Be precise in your tool selection - avoid using similar-sounding tools incorrectly
  - If a user wants to "send an email", use tools specifically for sending, not drafting
  - If a user wants to "read emails", use tools for reading/listing, not composing
  - Be conversational and helpful in your responses
  - Explain what you're doing when using tools
  - Ask for clarification if the user's request is ambiguous
  - Provide clear feedback about the results of tool usage

  When uncertain about which tool to use:
  1. List the available relevant tools
  2. Ask the user to clarify their intent
  3. Explain the difference between similar tools

  Always prioritize accuracy over speed when selecting tools.
                """,
      )

      # Stream the response
      async for token in agent.arun(message, stream=True):
        yield token

    finally:
      # Clean up MCP connection
      with contextlib.suppress(Exception):
        await mcp_tools.close()
