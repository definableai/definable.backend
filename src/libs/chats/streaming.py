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

    # Fix connection string format
    connection_kwargs = {
      "autocommit": True,
      "prepare_threshold": 0,
      "application_name": "llm_factory",
    }

    # Format: postgresql://user:password@host:port/dbname
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    self.pool = ConnectionPool(
      conninfo=db_url,
      max_size=5,
      min_size=1,
      kwargs=connection_kwargs,
    )
    self.checkpointer = PostgresSaver(self.pool)  # type: ignore
    self.checkpointer.setup()

  def chat(self, llm: str, chat_session_id: str, message: str):
    model = self.llms[llm]
    config = RunnableConfig(configurable={"thread_id": chat_session_id})

    def call_model(state: MessagesState, config: RunnableConfig):
      response = model.invoke(state["messages"], config)
      return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_edge(START, "model")
    builder.add_node("model", call_model)

    graph = builder.compile(checkpointer=self.checkpointer)
    for message, metadata in graph.stream({"messages": [HumanMessage(content=message)]}, config, stream_mode="messages"):
      if message.content and not isinstance(message, HumanMessage):  # type: ignore
        yield message.content  # type: ignore

  def __del__(self):
    print("Closing pool")
    self.pool.close()
