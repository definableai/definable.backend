import json
import logging
from typing import Any, Dict, List, Optional


class PromptBuilder:
  """Builds prompts for Claude to generate agent code."""

  def __init__(self, logger=None):
    self.logger = logger or logging.getLogger(__name__)

  # Main template - moved from agent_template.py
  AGENT_PROMPT_TEMPLATE = """
# Agent Code Generation Task

Generate three critical files for an agent named '{agent_name}' (version {version}):
1. agent.py - The main agent implementation using the Agno framework
2. Dockerfile - For containerizing the agent
3. compose.yml - Docker Compose configuration for deployment
4. requirements.txt - Essential dependencies

The agent will integrate with pre-existing tool files that implement various functionalities. Focus on properly importing and initializing these tools
,but DO NOT generate the tool code itself.

## Agent Specifications
Agent Name: {agent_name}
Version: {version}
Provider: {provider}

## Model Information
Model Name: {model_name}
Model Provider: {model_provider}

Description: {description}
System Prompt: {system_prompt}
Instructions: {instructions}
"""

  # Import instructions section
  IMPORT_INSTRUCTIONS = """
EXTREMELY IMPORTANT: You MUST follow these precise instructions for imports:

1. Import Agent correctly:
   ✅ from agno.agent import Agent
   ❌ from agno import Agent

2. Import tools using this pattern:
   ✅ from .tools import WeatherToolToolkit
   ❌ from .tools.weather_tool_toolkit import WeatherToolToolkit

   Tools should be imported directly from the tools package, not from individual files.
"""

  # FastAPI requirements section
  FASTAPI_REQUIREMENTS = """
## REQUIRED API INTERFACE:
Every agent.py file MUST include a FastAPI interface with these components:

1. FastAPI imports and app creation:
```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(title=f"{agent_name} API", version=version)

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[Any, Any]] = None
    session_id: Optional[str] = None
```

2. API endpoint to process queries:
```python
@app.post(f"/{version}/{agent_name}")
async def process_query(request: QueryRequest):
    try:
        response = await agent.arun(message=request.query)
        return JSONResponse(content=response.content)
    except Exception as e:
        print("error : ", e)
        raise HTTPException(status_code=500, detail=str(e))
```

3. Health check endpoint:
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": version}
```

The API interface is MANDATORY, not optional.
"""

  # Code structure requirements section
  CODE_STRUCTURE_REQUIREMENTS = """
## CRITICAL CODE STRUCTURE REQUIREMENTS:

1. Use EXACT class names:
   - When importing tools, use the ACTUAL class name from the tool file
   - Example: If the tool file has `class WeatherTool`, import it as `WeatherTool`
   - DO NOT append 'Toolkit' to class names unless that's actually in the class definition

2. Handle API keys properly:
   - Use provided API key values exactly as shown
   - For environment variables, provide suitable defaults: `os.environ.get("KEY_NAME") or "default-key"`

3. Agent initialization:
   - Use `Agent` from `agno.agent`
   - Use `AgentMemory()` without max_tokens parameter
   - Correct parameter names:
     - Use `memory` not `memory_config`
     - Use `num_history_runs` not `num_responses`

4. FastAPI interface:
   - Create the agent BEFORE using it in routes
   - Use string constants for routes, not f-strings with agent.name
   - Add proper type hints: `Optional[Dict[Any, Any]]` for nullable parameters
   - Use agent.run() not agent.process()
"""

  # Agent initialization examples section
  AGENT_INITIALIZATION_EXAMPLES = """
## CORRECT AGENT INITIALIZATION EXAMPLES

### Basic Agent
```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    name="my_agent",
    model=OpenAIChat(id="gpt-4o"),
    description="A helpful assistant",
    system_message="You are a helpful assistant that provides clear and concise answers.",
)
```

### Agent with Memory
```python
from agno.agent import Agent
from agno.memory.agent import AgentMemory
from agno.models.openai import OpenAIChat

agent = Agent(
    name="my_agent",
    model=OpenAIChat(id="gpt-4o"),
    description="A helpful assistant with memory",
    system_message="You are a helpful assistant that remembers previous conversations.",
    memory=AgentMemory(),  # No max_tokens parameter
)
```

### Agent with Tools
```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from .tools.weather_tool import WeatherTool  # Note: exact class name from file

agent = Agent(
    name="my_agent",
    model=OpenAIChat(id="gpt-4o"),
    description="An assistant with weather capabilities",
    system_message="You are a helpful assistant with access to weather data.",
    tools=[
        WeatherTool(api_key=os.environ.get("WEATHER_API_KEY") or "default-key")
    ],
)
```

### Agent with Knowledge Base
```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.agent import AgentKnowledge

agent = Agent(
    name="research_agent",
    model=OpenAIChat(id="gpt-4o"),
    description="A research assistant with knowledge access",
    system_message="You are a helpful research assistant with access to a knowledge base.",
    knowledge=AgentKnowledge(),  # Initialize knowledge base
    add_references=True,  # Add references from knowledge to prompts
    search_knowledge=True,  # Enable search_knowledge_base tool
)
```

IMPORTANT MODEL INITIALIZATION NOTES:
1. For OpenAI models, use: `OpenAIChat(id="gpt-4o")`
2. For Anthropic models, use: `Claude(id="claude-3-opus-20240229")`
3. For model IDs, use the actual model identifier string, not UUIDs
4. Do not use `provider` as a parameter to Agent
5. Do not use `version` as a parameter to Agent
"""

  # Input mapping requirements section
  INPUT_MAPPING_REQUIREMENTS = """
## INPUT MAPPING REQUIREMENTS

When creating an agent from API input, follow these CRITICAL transformations:

1. Knowledge Base Mapping:
   ```json
   // Input format
   "knowledge_base": {
     "enabled": true,
     "sources": ["pubmed", "arxiv"]
   }
   ```

   ```python
   # Correct Agno format
   knowledge=AgentKnowledge(sources=knowledge_base["sources"]) if knowledge_base and knowledge_base.get("enabled") else None,
   add_references=True,  # Add if knowledge is present
   search_knowledge=True  # Add if knowledge is present
   ```

2. Memory Configuration:
   ```json
   // Input format
   "memory_config": {
     "memory_type": "conversation",
     "max_tokens": 4000
   }
   ```

   ```python
   # Correct Agno format - NEVER pass max_tokens directly
   memory=AgentMemory(),  # Initialize without parameters
   ```

3. Expected Output:
   ```json
   // Input format
   "expected_output": {
     "format": "markdown",
     "sections": ["summary", "key_findings"]
   }
   ```

   ```python
   # Correct Agno format - Pass as-is
   expected_output=expected_output,  # Pass the dictionary as-is
   markdown=True if expected_output.get("format") == "markdown" else False,
   ```

4. Model Initialization:
   ```json
   // Input format using UUID
   "model_id": "3c57ba08-0e0e-4cc3-b496-e5da23abe1de",
   "provider": "openai"
   ```

   ```python
   # Correct Agno format - Use actual model names, not UUIDs
   model=OpenAIChat(id="gpt-4o"),  # For OpenAI models
   # OR
   model=Claude(id="claude-3-opus-20240229"),  # For Anthropic models
   ```

5. Tool Parameters:
   CRITICAL: Only pass parameters that the tool __init__ method actually accepts!
   Before passing any parameter to a tool, check the tool's __init__ method parameters.

   ```python
   # WRONG - passing version when not accepted
   WeatherToolToolkit(api_key="key", version=3)

   # CORRECT - only passing accepted parameters
   WeatherToolToolkit(api_key="key")
   ```
"""

  # Example template section
  EXAMPLE_TEMPLATE = """
## Example Code Templates

### Example agent.py with direct API keys
```python
# Tool initialization with provided API keys
tools = [
    WeatherToolkit(api_key="ACTUAL_API_KEY_HERE"),  # Direct approach when key is provided
    SearchToolkit(api_key=os.environ.get("SEARCH_API_KEY")),  # Environment variable approach
]
```
"""

  def create_prompt(
    self,
    agent_name: str,
    version: str,
    provider: str,
    model_details: Dict[str, str],
    description: str,
    system_prompt: str,
    instructions: str,
    expected_output: Dict[str, Any],
    memory_config: Optional[Dict[str, Any]],
    knowledge_base: Optional[Dict[str, Any]],
    tools: List[Dict[str, Any]],
  ) -> str:
    """Create a prompt for Claude to generate agent files."""
    try:
      # Start with main prompt template
      prompt = self.AGENT_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        version=version,
        provider=provider,
        model_name=model_details["name"],
        model_provider=model_details["provider"],
        description=description,
        system_prompt=system_prompt,
        instructions=instructions,
      )

      # Add JSON objects
      prompt += "Expected Output: " + json.dumps(expected_output, indent=2) + "\n\n"
      prompt += "Memory Configuration: " + (json.dumps(memory_config, indent=2) if memory_config else "None") + "\n\n"
      prompt += "Knowledge Base: " + (json.dumps(knowledge_base, indent=2) if knowledge_base else "None") + "\n\n"

      # Add tools information
      prompt += self._format_tools_section(tools)

      # Add all other sections
      prompt += self._add_example_sections()

      self.logger.info("Claude prompt created successfully")
      return prompt
    except Exception as e:
      self.logger.error(f"Error creating Claude prompt: {str(e)}")
      raise

  def _format_tools_section(self, tools: List[Dict[str, Any]]) -> str:
    """Format the tools section of the prompt."""
    section = "## Tools Integration\n"
    section += f"The agent will use {len(tools)} pre-existing tools. Your task is to correctly import and initialize these tools.\n\n"

    for i, tool in enumerate(tools):
      section += f"### Tool {i + 1}: {tool['file_name']}\n"
      section += f"- Class name: {tool['class_name']}\n"
      section += f"- API key: {tool['api_key_handling']}  # Use this exact API key initialization\n"
      section += f"- Version: {tool['version']}\n"
      section += f"- Description: {tool['description']}\n\n"

    return section

  def _add_example_sections(self) -> str:
    """Add all example sections to the prompt."""
    sections = [
      self.EXAMPLE_TEMPLATE,
      self.IMPORT_INSTRUCTIONS,
      self.FASTAPI_REQUIREMENTS,
      self.CODE_STRUCTURE_REQUIREMENTS,
      self.AGENT_INITIALIZATION_EXAMPLES,
      self.INPUT_MAPPING_REQUIREMENTS,
    ]

    return "".join(sections)
