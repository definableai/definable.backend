from typing import Any, Dict, List


def parse_configuration(config_items: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Parse configuration items into kwargs for tool initialization"""
  kwargs = {}

  for item in config_items:
    name = item.get("name")
    if not name:
      continue

    # Otherwise use default if available
    value = item.get("value")
    if value is not None:
      kwargs[name] = value

  return kwargs


boilerplate = """import asyncio
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from {module_name} import {class_name}

if __name__ == "__main__":

  async def main():
    async_agent = Agent(
      model={model_class_name}(
        id="{model_name}",
        api_key="{api_key}",
      ),
      tools=[{class_name}(**{config_items})],
      show_tool_calls=True,
      instructions=\"\"\"{instructions}\"\"\",
      markdown=True,
      stream=True,
    )

    async for token in await async_agent.arun("{input_prompt}"):
      print(token.content, end="", flush=True)

  asyncio.run(main())
"""


def generate_boilerplate(
  class_name: str,
  input_prompt: str,
  model_name: str,
  provider: str,
  api_key: str,
  module_name: str = "tool",
  config_items: List[Dict[str, Any]] = [],
  instructions: str = "",
) -> str:
  if provider == "openai":
    model_class_name = "OpenAIChat"
  elif provider == "anthropic":
    model_class_name = "Claude"
  else:
    raise ValueError(f"Invalid provider: {provider}")

  return boilerplate.format(
    module_name=module_name,
    class_name=class_name,
    model_name=model_name,
    api_key=api_key,
    input_prompt=input_prompt,
    config_items=parse_configuration(config_items),
    instructions=instructions,
    model_class_name=model_class_name,
  )


if __name__ == "__main__":
  module_name = "tests.basic_tool_agent"
  class_name = "WeatherTool"
  model_name = "gpt-4o-mini"
  api_key = (
    "sk-proj-AcQ-UrB-g7MWrsnXzb-LPYhD_Su_wEMtvP9fnayCCljDtLxGg6Ta8JWXptFp1710Rv3OIV"
    "-dtqT3BlbkFJYIvG2Xy-MLdbuOw9kkkPJY3r-cNY7BBPsfnOI0TNu3fS4VnwSLMUYPMc1exe6ci1_454lNIjQA"
  )
  input_prompt = "How is the weather in Tokyo tomorrow?"
  config_items = parse_configuration([
    {"name": "api_key", "type": "str", "description": "The API key for the weather service", "required": True, "default": None},
  ])

  print(
    boilerplate.format(
      module_name=module_name, class_name=class_name, model_name=model_name, api_key=api_key, input_prompt=input_prompt, config_items=config_items
    )
  )
