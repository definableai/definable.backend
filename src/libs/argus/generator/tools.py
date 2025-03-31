from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolMetadata(BaseModel):
  class_name: str
  import_path: str
  configurable_params: Dict[str, Any]
  function_name_mapping: Dict[str, str] = {}  # Maps parameter names to function names


class ToolRegistry:
  """Registry to manage available tools and their configurations"""

  _tools: Dict[str, ToolMetadata] = {
    "youtube": ToolMetadata(
      class_name="YouTubeTools",
      import_path="agno.tools.youtube",
      configurable_params={"max_results": {"type": "int", "required": False, "default": 10}},
    ),
    "newspaper": ToolMetadata(
      class_name="NewspaperTools",
      import_path="agno.tools.newspaper",
      configurable_params={"timeout": {"type": "int", "required": False, "default": 30}},
    ),
    "newspaper4k": ToolMetadata(
      class_name="Newspaper4kTools",
      import_path="agno.tools.newspaper4k",
      configurable_params={
        "read_article": {"type": "bool", "required": False, "default": True},
        "include_summary": {"type": "bool", "required": False, "default": False},
        "article_length": {"type": "int", "required": False, "default": None},
        "cache_results": {"type": "bool", "required": False, "default": False},
        "cache_ttl": {"type": "int", "required": False, "default": 3600},
      },
    ),
    "duckduckgo": ToolMetadata(
      class_name="DuckDuckGoTools",
      import_path="agno.tools.duckduckgo",
      configurable_params={
        "search": {"type": "bool", "required": False, "default": True},
        "news": {"type": "bool", "required": False, "default": True},
        "fixed_max_results": {"type": "int", "required": False, "default": 5},
        "timeout": {"type": "int", "required": False, "default": 10},
        "cache_results": {"type": "bool", "required": False, "default": False},
      },
    ),
    "exa": ToolMetadata(
      class_name="ExaTools",
      import_path="agno.tools.exa",
      configurable_params={
        "start_published_date": {"type": "str", "required": False, "default": ""},
        "type": {"type": "str", "required": False, "default": "keyword"},
        "api_key": {"type": "str", "required": True, "default": ""},
      },
    ),
    "spider": ToolMetadata(
      class_name="SpiderTools", import_path="agno.tools.spider", configurable_params={"max_depth": {"type": "int", "required": False, "default": 2}}
    ),
    "file": ToolMetadata(class_name="FileTools", import_path="agno.tools.file", configurable_params={}),
    "arxiv": ToolMetadata(
      class_name="ArxivTools", import_path="agno.tools.arxiv", configurable_params={"max_results": {"type": "int", "required": False, "default": 5}}
    ),
    "pubmed": ToolMetadata(
      class_name="PubMedTools", import_path="agno.tools.pubmed", configurable_params={"max_results": {"type": "int", "required": False, "default": 5}}
    ),
    "searxng": ToolMetadata(
      class_name="SearxNGTools",
      import_path="agno.tools.searxng",
      configurable_params={"max_results": {"type": "int", "required": False, "default": 5}},
    ),
    "serpapi": ToolMetadata(
      class_name="SerpAPITools",
      import_path="agno.tools.serpapi",
      configurable_params={"engine": {"type": "str", "required": False, "default": "google"}},
    ),
    "wikipedia": ToolMetadata(
      class_name="WikipediaTools",
      import_path="agno.tools.wikipedia",
      configurable_params={"max_results": {"type": "int", "required": False, "default": 5}},
    ),
    "website": ToolMetadata(
      class_name="WebsiteTools", import_path="agno.tools.website", configurable_params={"timeout": {"type": "int", "required": False, "default": 30}}
    ),
  }

  @classmethod
  def get_tool_metadata(cls, tool_type: str) -> Optional[ToolMetadata]:
    return cls._tools.get(tool_type)

  @classmethod
  def list_available_tools(cls) -> Dict[str, ToolMetadata]:
    return cls._tools

  @classmethod
  def validate_tool_config(cls, tool_type: str, config: Dict[str, Any]) -> bool:
    tool = cls.get_tool_metadata(tool_type)
    if not tool:
      raise ValueError(f"Unknown tool type: {tool_type}")

    # Validate each parameter against the allowed configurable params
    for param, value in config.items():
      if param not in tool.configurable_params:
        raise ValueError(f"Invalid parameter '{param}' for tool {tool_type}")

      param_spec = tool.configurable_params[param]
      param_type = param_spec["type"]

      # Type validation
      if param_type == "int" and not isinstance(value, int) and value is not None:
        raise ValueError(f"Parameter '{param}' must be an integer for tool {tool_type}")
      if param_type == "str" and not isinstance(value, str) and value is not None:
        raise ValueError(f"Parameter '{param}' must be a string for tool {tool_type}")
      if param_type == "bool" and not isinstance(value, bool) and value is not None:
        raise ValueError(f"Parameter '{param}' must be a boolean for tool {tool_type}")

      # Range validation for numeric types
      if param_type == "int" and "min" in param_spec and value is not None and value < param_spec["min"]:
        raise ValueError(f"Parameter '{param}' must be >= {param_spec['min']} for tool {tool_type}")
      if param_type == "int" and "max" in param_spec and value is not None and value > param_spec["max"]:
        raise ValueError(f"Parameter '{param}' must be <= {param_spec['max']} for tool {tool_type}")

    # Check required parameters
    for param, specs in tool.configurable_params.items():
      if specs.get("required", False) and param not in config:
        raise ValueError(f"Required parameter {param} missing for tool {tool_type}")

    return True
