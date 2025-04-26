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
    "task": (
      "You are an AI system design expert. Generate detailed task instructions for AI systems based on the user's input."
      "Now you have to remember that first you have to read the prompt thoroughly and learn whether the prompt is a task"
      "or just a simple message. You have to use your brain to understand the user's intent and then you have to generate the prompt accordingly.",  # noqa: E501
      "If the prompt does not appears to be a task then DO NOT GENERATE ANY PROMPT, JUST RETURN THE TEXT AS IT IS WITH A LITTLE BIT OF ENHANCEMENT OF THE TEXT WITH SOME EXAMPLES IF POSSIBLE."  # noqa: E501
    )
  }

  extension_prompt = (  # noqa: F841
      "Always remember that the prompts you are generating will be used by an AI system to generate a response. "
      "So, you have to generate the prompts in a way that is easy for an AI system to understand."
  )

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

  extended_prompt = f"{user_prompt}\n\n{extension_prompt}"

  # Get the async iterator from the agent's run method
  run_response = await temp_agent.arun(extended_prompt, stream=True)

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
