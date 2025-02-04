from app import app


def get_llm():
  return app.state.llm
