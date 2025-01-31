from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from psycopg_pool import ConnectionPool

from config.settings import settings


class LLMFactory:
  def __init__(self):
    self.llms = {
      "gpt-4o": ChatOpenAI(model="chatgpt-4o-latest"),
      "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
      "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo"),
      "o1-preview": ChatOpenAI(model="o1-mini"),
      "o1": ChatOpenAI(model="o1-preview"),
    }

    # Setup connection pool without context manager
    self.pool = ConnectionPool(
      conninfo=settings.database_url,
      max_size=20,
      kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
      },
    )
    self.checkpointer = PostgresSaver(self.pool)  # type: ignore
    self.checkpointer.setup()

  async def chat(self, llm: str, chat_session_id: str, message: str):
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "model")
    builder.add_node("model", lambda state: self.llms[llm].invoke(state["messages"]))

    # Create proper RunnableConfig
    config = RunnableConfig(configurable={"thread_id": chat_session_id})

    graph = builder.compile(checkpointer=self.checkpointer)
    res = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    print(res)

  def __del__(self):
    """Cleanup connection pool when object is destroyed."""
    if hasattr(self, "pool"):
      self.pool.close()
