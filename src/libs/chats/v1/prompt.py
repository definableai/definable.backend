from typing import AsyncGenerator

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from config.settings import settings

def map_model_to_deepseek(model_name: str) -> str:
    """Maps the generic model names to DeepSeek model IDs."""
    model_mapping = {
        "chat": "deepseek-chat",
        "reason": "deepseek-reason"
    }
    return model_mapping.get(model_name, "deepseek-chat")

async def generate_prompts_stream(text: str, prompt_type: str = "task", num_prompts: int = 1, model: str = "chat") -> AsyncGenerator[str, None]:
  """Generate prompts using DeepSeek V3 model with streaming response"""

  # Create system message based on prompt type
  system_prompts = {
    "creative": "You are a creative writing assistant. Generate imaginative and thought-provoking prompts based on the user's input.",
    "question": "You are a critical thinking assistant. Generate thoughtful questions based on the user's input.",
    "continuation": "You are a narrative assistant. Generate natural continuations based on the user's input.",
    "task": "You are an AI system design expert. Generate detailed task instructions for AI systems based on the user's input.",
  }

  # Get the appropriate system prompt
  system_prompt = system_prompts.get(prompt_type, system_prompts["task"])

  # Create a temporary agent with the correct system message
  temp_agent = Agent(
    model=DeepSeek(id=map_model_to_deepseek(model), api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com"),
    markdown=True,
  )

  # Create the user prompt
  if num_prompts > 1:
    user_prompt = f"""{system_prompt}\n\nGenerate {num_prompts} different {prompt_type} prompts based on this text:
"{text}"

The prompts should be clear, engaging, and directly related to the text.
Each prompt should be on a separate line with no numbering or additional formatting."""
  else:
    user_prompt = f"""{system_prompt}\n\nGenerate a {prompt_type} prompt based on this text:
"{text}"

The prompt should be clear, engaging, and directly related to the text."""

  # Get the async iterator from the agent's run method
  run_response = await temp_agent.arun(user_prompt, stream=True)

  # Buffer to accumulate tokens
  buffer = []
  buffer_size = settings.prompt_buffer_size  # Adjust this number to control chunk size

  # Now we can use async for on the iterator
  async for token in run_response:
    buffer.append(token.content)

    # When buffer reaches the target size, yield the content and reset
    if len(buffer) >= buffer_size:
      yield "".join(buffer)
      buffer = []

  # Don't forget any remaining content
  if buffer:
    yield "".join(buffer)
