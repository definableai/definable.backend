import contextlib
from typing import AsyncGenerator, Optional
from uuid import UUID

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.tools.mcp import MCPTools

from config.settings import settings


class MCPPlaygroundFactory:
  """Factory for creating MCP playground sessions with conversation persistence."""

  def __init__(self):
    # Configure storage for conversation persistence (same as chat service)
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    self.storage = PostgresStorage(table_name="__agno_chat_sessions", db_url=db_url, schema="public")
    self.storage.create()

  async def chat(
    self,
    session_id: str | UUID,
    mcp_url: str,
    message: str,
    openai_api_key: str,
    memory_size: int = 50,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
  ) -> AsyncGenerator[RunResponse, None]:
    """
    Stream chat responses with MCP tools integration.

    Args:
        session_id: Unique ID for the chat session
        mcp_url: The MCP server URL to connect to
        message: User's input message
        openai_api_key: OpenAI API key for the model
        memory_size: Number of previous messages to include
        temperature: Model temperature parameter
        max_tokens: Maximum tokens to generate

    Yields:
        Streaming response tokens
    """

    # Initialize MCP tools with the provided URL
    mcp_tools = MCPTools(
      url=mcp_url,
      transport="streamable-http",  # Server-Sent Events transport for HTTP
      timeout_seconds=30,
    )

    try:
      # Connect to the MCP server
      await mcp_tools.connect()

      # Create agent with MCP tools and storage for memory retention
      agent = Agent(
        name="MCP Playground Assistant",
        model=OpenAIChat(
          id="gpt-4o-mini",
          api_key=openai_api_key,
          temperature=temperature,
          max_tokens=max_tokens,
        ),
        tools=[mcp_tools],
        storage=self.storage,
        markdown=True,
        stream=True,
        add_history_to_messages=True,
        session_id=str(session_id),
        num_history_responses=memory_size,
        show_tool_calls=True,
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
      async for token in await agent.arun(message):
        yield token

    finally:
      # Clean up MCP connection
      with contextlib.suppress(Exception):
        await mcp_tools.close()
