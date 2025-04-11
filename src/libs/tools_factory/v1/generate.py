import json
from typing import Any, Dict

from agno.agent import Agent
from agno.models.anthropic import Claude

from config.settings import settings


class ToolGenerator:
  def __init__(self):
    self.agent = Agent(model=Claude(id="claude-3-7-sonnet-latest", api_key=settings.anthropic_api_key), show_tool_calls=True, markdown=True)

  async def generate_toolkit_from_json(self, tool_json: Dict[str, Any]) -> str:
    """Generate an Agno toolkit from a JSON file using the Claude 3.7 model."""

    # Prepare the prompt for the agent
    prompt = self._create_prompt(tool_json)

    # Generate the toolkit code using the agent
    toolkit_code = await self.agent.arun(prompt, stream=False)

    # remove ```python and ``` from the code
    return toolkit_code.content.replace("```python", "").replace("```", "")

  def _create_prompt(self, tool_json: Dict[str, Any]) -> str:
    """Create a prompt for the agent to generate the toolkit code."""

    # Create the prompt
    prompt = f"""
        Generate an Agno toolkit class based on the following JSON input.
        Ensure the code has 2-space indentation, proper types, function descriptions,
        and returns. The toolkit should be syntactically accurate and functionally working.

        JSON Input:
        {json.dumps(tool_json, indent=2)}

        Requirements:
        1. The toolkit class should extend the `Toolkit` class from Agno.
        2. The class name should be derived from the tool name in the JSON (e.g., "Weather" -> "WeatherToolkit").
        3. The `__init__` method should include any configuration parameters from the JSON.
        4. The `run` method should be generated from the function code in the JSON.
        5. Ensure proper imports and function descriptions are included.
        6. The code should be formatted with 2-space indentation.
        7. For logging and logger please use from agno.utils.log import log_info, logger, if required.
        8. In output, please just return the code, no other text or comments.

        Example Input:
        {{
          "info": {{"name": "Weather", "version": "1", "description": "Tool to extract weather of a specific location"}},
          "function_info": {{
            "name": "run",
            "is_async": True,
            "description": "Get the current weather for a location",
            "code": 'import aiohttp\n\nasync def run(self, location: str, units: str = "metric") -> str:\n  \"\"\"Get the current weather for a ' \
                    'location\n\n  Args:\n      location (str): The city name or ZIP code\n      units (str): ' \
                    'The unit system (metric or imperial)\n\n ' \
                    'Returns:\n      str: A string with the weather ' \
                    'information\n  \"\"\"\n  url = "https://api.openweathermap.org/data/2.5/weather"\n ' \
                    'params = {{"q": location, "units": units, "appid": self.api_key}}\n\n  async with ' \
                    'aiohttp.ClientSession() as session:\n    async ' \
                    'with session.get(url, params=params) as response:\n      if response.status != 200:\n        return f"Error: Could not get ' \
                    'weather data (Status {{response.status}})"\n\n      data = await response.json()\n      temp = data["main"]["temp"]\n ' \
                    'desc = data["weather"][0]["description"]\n      city = data["name"]\n\n      return f"Weather in {{city}}: {{desc}}, ' \
                    'Temperature: {{temp}}°{{"C" if units == "metric" else "F"}}"',
          }},
          "configuration": [{{"name":"api_key","type":"str","description":"The API key for the weather service","required":True,"default":None}}],
        }}

        Example Output:
        ```python
        from agno.tools.toolkit import Toolkit
        import aiohttp

        class WeatherToolkit(Toolkit):
          \"\"\"Tool to extract weather of a specific location\"\"\"
          def __init__(self, api_key: str):
            super().__init__(name="weather")
            self.api_key = api_key
            self.register(self.run)


          async def run(self, location: str, units: str = "metric") -> str:
            \"\"\"Get the current weather for a location

            Args:
                location (str): The city name or ZIP code
                units (str): The unit system (metric or imperial)

            Returns:
                str: A string with the weather information
            \"\"\"
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {{"q": location, "units": units, "appid": self.api_key}}

            async with aiohttp.ClientSession() as session:
              async with session.get(url, params=params) as response:
                if response.status != 200:
                  return f"Error: Could not get weather data (Status {{response.status}})"

                data = await response.json()
                temp = data["main"]["temp"]
                desc = data["weather"][0]["description"]
                city = data["name"]

                return f"Weather in {{city}}: {{desc}}, Temperature: {{temp}}°{{"C" if units == "metric" else "F"}}"

        ```

        Now, generate the toolkit code based on the provided JSON input.
        """

    return prompt
