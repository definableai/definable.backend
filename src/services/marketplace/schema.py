from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MarketplacePricing(BaseModel):
  """Marketplace pricing schema."""

  type: str = Field(..., description="Pricing type: free or paid")
  price: Optional[str] = Field(None, description="Price string if paid")


class MarketplaceSpecifications(BaseModel):
  """LLM model specifications schema."""

  context_window: Optional[int] = Field(None, description="Maximum context window size in tokens")
  max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
  pricing_per_token: Optional[Dict[str, Any]] = Field(None, description="Token pricing information")
  capabilities: Optional[List[str]] = Field(None, description="List of model capabilities")
  performance_tier: Optional[str] = Field(None, description="Performance tier (fast/high/highest/reasoning/fastest)")
  knowledge_cutoff: Optional[str] = Field(None, description="Knowledge cutoff date")
  training_cutoff: Optional[str] = Field(None, description="Training data cutoff date")
  model_version: Optional[str] = Field(None, description="Model version (e.g., DeepSeek-V3.1)")
  reasoning_score: Optional[int] = Field(None, ge=1, le=5, description="Reasoning capability score (1-5)")
  speed_score: Optional[int] = Field(None, ge=1, le=5, description="Speed performance score (1-5)")
  modalities: Optional[List[str]] = Field(None, description="Supported modalities (text, image, audio)")
  features: Optional[List[str]] = Field(None, description="Special features (streaming, vision, function_calling)")
  thinking_mode: Optional[bool] = Field(None, description="Whether model supports thinking mode")
  reasoning_token_support: Optional[bool] = Field(None, description="Whether model supports reasoning tokens")
  rate_limits: Optional[Dict[str, Any]] = Field(None, description="Rate limiting information")
  endpoints: Optional[List[str]] = Field(None, description="Supported API endpoints")


class MarketplaceOverviewFeature(BaseModel):
  """Marketplace overview feature schema."""

  title: str
  description: str


class MarketplaceOverviewDescription(BaseModel):
  """Marketplace overview main description schema."""

  title: str
  content: List[str]


class MarketplaceOverview(BaseModel):
  """Marketplace overview schema."""

  whats_new: List[str]
  main_description: MarketplaceOverviewDescription
  features: Optional[List[MarketplaceOverviewFeature]] = None


class MarketplaceReview(BaseModel):
  """Marketplace review schema."""

  id: str
  user_id: str
  user_name: str
  rating: int = Field(..., ge=1, le=5)
  title: Optional[str] = None
  content: Optional[str] = None
  created_at: str


class MarketplaceMCPServer(BaseModel):
  """MCP Server schema."""

  id: str
  name: str
  description: str
  icon: str
  status: str  # required, optional
  setup_instructions: str
  category: str


class MarketplaceTool(BaseModel):
  """Marketplace tool schema."""

  id: str
  name: str
  description: str
  icon: str
  status: str
  category: str


class MarketplaceToolCategory(BaseModel):
  """Marketplace tool category schema."""

  id: str
  name: str
  description: str
  icon: str
  color: str
  tools: List[MarketplaceTool]


class MarketplaceAssistantBase(BaseModel):
  """Base marketplace assistant schema."""

  id: str
  name: str
  description: str
  type: str  # 'core' | 'agent'
  provider: str
  developer: str
  verified: bool

  # Marketplace metadata
  rating: float
  review_count: int
  conversation_count: str
  category: str
  is_featured: bool
  is_new: bool
  is_popular: bool
  is_active: bool  # Whether the underlying model/agent is active
  tags: List[str]
  pricing: MarketplacePricing

  # Optional fields
  icon: Optional[str] = None
  screenshots: Optional[List[str]] = None
  marketplace_features: Optional[List[str]] = None
  integrations: Optional[List[str]] = None
  specifications: Optional[MarketplaceSpecifications] = None


class MarketplaceAssistantItem(MarketplaceAssistantBase):
  """Marketplace assistant list item schema."""

  pass


class MarketplaceAssistantDetail(MarketplaceAssistantBase):
  """Detailed marketplace assistant schema."""

  class Config:
    extra = "allow"  # Allow extra fields

  overview: Optional[MarketplaceOverview] = None
  reviews: Optional[List[MarketplaceReview]] = None
  mcp_servers: Optional[List[MarketplaceMCPServer]] = None
  tool_categories: Optional[List[MarketplaceToolCategory]] = None
  user_context: Optional[Dict[str, Any]] = None


class MarketplaceAssistantsResponse(BaseModel):
  """Marketplace assistants list response schema."""

  assistants: List[MarketplaceAssistantItem]
  total: int
  has_more: bool
  categories: Optional[List[str]] = None


class MarketplaceCategoriesResponse(BaseModel):
  """Marketplace categories response schema."""

  categories: List["MarketplaceCategory"]


class MarketplaceFeaturedItem(BaseModel):
  """Featured item schema."""

  type: str  # 'llm_model' | 'agent'
  id: str
  featured_reason: Optional[str] = None
  priority: Optional[int] = None
  trend_score: Optional[int] = None
  trend_reason: Optional[str] = None
  popularity_score: Optional[int] = None
  added_date: Optional[str] = None


class MarketplaceFeaturedResponse(BaseModel):
  """Marketplace featured response schema."""

  categories: Dict[str, List[Dict[str, Any]]]


class MarketplaceReviewsResponse(BaseModel):
  """Marketplace reviews response schema."""

  reviews: List[MarketplaceReview]
  total_reviews: int
  rating_breakdown: Dict[str, float]  # {"5": 40.0, "4": 40.0, "3": 13.0, "2": 0.0, "1": 7.0}
  average_rating: float


# New schemas for database-based marketplace


class MarketplacePublishRequest(BaseModel):
  """Request to publish assistant to marketplace."""

  assistant_type: str = Field(..., description="Type: llm_model or agent")
  assistant_id: str = Field(..., description="ID of the assistant to publish")
  pricing_type: str = Field(default="free", description="Pricing type: free or paid")


class MarketplaceReviewCreate(BaseModel):
  """Create marketplace review request."""

  rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
  title: Optional[str] = Field(None, max_length=200, description="Review title")
  content: Optional[str] = Field(None, description="Review content")


class MarketplaceUsageResponse(BaseModel):
  """Marketplace usage response."""

  assistant_id: str
  usage_count: int
  last_updated: str


class MarketplaceCategory(BaseModel):
  """Marketplace category with count."""

  id: str
  name: str
  count: int
