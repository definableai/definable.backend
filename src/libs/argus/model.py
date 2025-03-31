from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMConfig(BaseModel):
    provider: LLMProvider
    model_name: str
    api_key: str
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})

class ToolConfig(BaseModel):
    name: str
    tool_type: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})

class AgentConfig(BaseModel):
    name: str
    description: str
    instructions: Optional[str] = None
    input_model: str
    output_model: str
    llm: LLMConfig
    tools: List[ToolConfig]

class AgentResponse(BaseModel):
    status: str
    agent_path: str
    message: Optional[str] = None