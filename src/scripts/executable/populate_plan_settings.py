#!/usr/bin/env python3
"""
Plan Settings Population Script

Populates the database with plan features and limits according to
the Free/Pro/Enterprise tier structure defined in the product specification.
"""

import os
import sys
from typing import Dict, List

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import log as logger
from scripts.core.base_script import BaseScript


class PopulatePlanSettingsScript(BaseScript):
  """Script to populate plan settings with comprehensive Free/Pro/Enterprise tiers."""

  def __init__(self):
    super().__init__("populate_plan_settings")

  def _define_categories(self) -> List[Dict]:
    """Define feature categories with sort order."""
    return [
      {"name": "text_models", "display_name": "Text Models", "description": "AI text generation models", "sort_order": 1},
      {"name": "chat_features", "display_name": "Chat Features", "description": "Chat interface features", "sort_order": 2},
      {"name": "image_models", "display_name": "Image Models", "description": "AI image generation models", "sort_order": 3},
      {"name": "image_editing", "display_name": "Image Editing", "description": "Image manipulation tools", "sort_order": 4},
      {"name": "knowledge_bases", "display_name": "Knowledge Bases", "description": "Document storage and retrieval", "sort_order": 5},
      {"name": "ai_agent_tools", "display_name": "AI Agent Tools", "description": "Tools for AI agents", "sort_order": 6},
      {"name": "agent_deployment", "display_name": "Agent Creation & Deployment", "description": "Agent development features", "sort_order": 7},
      {"name": "marketplace", "display_name": "Marketplace Features", "description": "Agent marketplace capabilities", "sort_order": 8},
      {"name": "collaboration", "display_name": "Collaboration & Team Features", "description": "Team and collaboration tools", "sort_order": 9},
    ]

  def _define_features(self) -> List[Dict]:
    """Define all features with their metadata."""
    return [
      # Text Models
      {
        "name": "gpt-5",
        "display_name": "GPT-5",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "OpenAI GPT-5 model",
        "sort_order": 1,
      },
      {
        "name": "gpt-4o",
        "display_name": "GPT-4o",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "OpenAI GPT-4o model",
        "sort_order": 2,
      },
      {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o-mini",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "OpenAI GPT-4o-mini model",
        "sort_order": 3,
      },
      {
        "name": "claude-opus-4.1",
        "display_name": "Claude Opus 4.1",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "Anthropic Claude Opus 4.1",
        "sort_order": 4,
      },
      {
        "name": "claude-sonnet-4",
        "display_name": "Claude Sonnet 4",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "Anthropic Claude Sonnet 4",
        "sort_order": 5,
      },
      {
        "name": "claude-3.5-sonnet",
        "display_name": "Claude 3.5 Sonnet",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "Anthropic Claude 3.5 Sonnet",
        "sort_order": 6,
      },
      {
        "name": "gemini-2.0-ultra",
        "display_name": "Gemini 2.0 Ultra",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "Google Gemini 2.0 Ultra",
        "sort_order": 7,
      },
      {
        "name": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "Google Gemini 1.5 Pro",
        "sort_order": 8,
      },
      {
        "name": "deepseek-v3",
        "display_name": "DeepSeek-V3",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "DeepSeek-V3 model",
        "sort_order": 9,
      },
      {
        "name": "o1-preview",
        "display_name": "o1-preview",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "OpenAI o1-preview model",
        "sort_order": 10,
      },
      {
        "name": "o1-mini",
        "display_name": "o1-mini",
        "category": "text_models",
        "feature_type": "model_access",
        "measurement_unit": "requests",
        "description": "OpenAI o1-mini model",
        "sort_order": 11,
      },
      # Chat Features
      {
        "name": "daily_chat_limit",
        "display_name": "Daily Chat Limit",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "messages",
        "description": "Messages per day",
        "sort_order": 1,
      },
      {
        "name": "context_window",
        "display_name": "Context Window",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "tokens",
        "description": "Maximum tokens per conversation",
        "sort_order": 2,
      },
      {
        "name": "file_uploads",
        "display_name": "File Uploads",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "files",
        "description": "Files per conversation",
        "sort_order": 3,
      },
      {
        "name": "voice_input_output",
        "display_name": "Voice Input/Output",
        "category": "chat_features",
        "feature_type": "feature_toggle",
        "measurement_unit": "minutes",
        "description": "Voice conversations",
        "sort_order": 4,
      },
      {
        "name": "web_search",
        "display_name": "Web Search",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "searches",
        "description": "Real-time web access",
        "sort_order": 5,
      },
      {
        "name": "deep_research",
        "display_name": "Deep Research Mode",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "sessions",
        "description": "Multi-step research with citations",
        "sort_order": 6,
      },
      {
        "name": "code_interpreter",
        "display_name": "Code Interpreter",
        "category": "chat_features",
        "feature_type": "feature_toggle",
        "measurement_unit": "boolean",
        "description": "Execute Python code",
        "sort_order": 7,
      },
      {
        "name": "multi_model_chat",
        "display_name": "Multi-Model Chat",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "models",
        "description": "Compare responses across models",
        "sort_order": 8,
      },
      {
        "name": "chat_history",
        "display_name": "Chat History",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "days",
        "description": "Saved conversation history",
        "sort_order": 9,
      },
      {
        "name": "custom_system_prompts",
        "display_name": "Custom System Prompts",
        "category": "chat_features",
        "feature_type": "usage_limit",
        "measurement_unit": "prompts",
        "description": "Pre-configured prompts",
        "sort_order": 10,
      },
      # Image Models
      {
        "name": "dall-e-3",
        "display_name": "DALL-E 3",
        "category": "image_models",
        "feature_type": "model_access",
        "measurement_unit": "images",
        "description": "OpenAI DALL-E 3",
        "sort_order": 1,
      },
      {
        "name": "dall-e-2",
        "display_name": "DALL-E 2",
        "category": "image_models",
        "feature_type": "model_access",
        "measurement_unit": "images",
        "description": "OpenAI DALL-E 2",
        "sort_order": 2,
      },
      {
        "name": "stable-diffusion-xl",
        "display_name": "Stable Diffusion XL",
        "category": "image_models",
        "feature_type": "model_access",
        "measurement_unit": "images",
        "description": "Stable Diffusion XL",
        "sort_order": 3,
      },
      {
        "name": "midjourney",
        "display_name": "Midjourney (via API)",
        "category": "image_models",
        "feature_type": "model_access",
        "measurement_unit": "images",
        "description": "Midjourney via API",
        "sort_order": 4,
      },
      {
        "name": "flux-1-pro",
        "display_name": "Flux.1 Pro",
        "category": "image_models",
        "feature_type": "model_access",
        "measurement_unit": "images",
        "description": "Flux.1 Pro model",
        "sort_order": 5,
      },
      {
        "name": "resolution_options",
        "display_name": "Resolution Options",
        "category": "image_models",
        "feature_type": "feature_toggle",
        "measurement_unit": "boolean",
        "description": "Available output sizes",
        "sort_order": 6,
      },
      {
        "name": "batch_generation",
        "display_name": "Batch Generation",
        "category": "image_models",
        "feature_type": "usage_limit",
        "measurement_unit": "variations",
        "description": "Multiple variations",
        "sort_order": 7,
      },
      # Image Editing
      {
        "name": "background_removal",
        "display_name": "Background Removal",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "operations",
        "description": "5 credits per operation",
        "sort_order": 1,
      },
      {
        "name": "image_upscaling",
        "display_name": "Image Upscaling",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "operations",
        "description": "10-20 credits per operation",
        "sort_order": 2,
      },
      {
        "name": "inpainting",
        "display_name": "Inpainting",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "operations",
        "description": "15 credits per operation",
        "sort_order": 3,
      },
      {
        "name": "outpainting",
        "display_name": "Outpainting",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "operations",
        "description": "20 credits per operation",
        "sort_order": 4,
      },
      {
        "name": "style_transfer",
        "display_name": "Style Transfer",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "operations",
        "description": "10 credits per operation",
        "sort_order": 5,
      },
      {
        "name": "batch_processing",
        "display_name": "Batch Processing",
        "category": "image_editing",
        "feature_type": "usage_limit",
        "measurement_unit": "images",
        "description": "Process multiple images",
        "sort_order": 6,
      },
      # Knowledge Bases
      {
        "name": "knowledge_base_count",
        "display_name": "Number of Knowledge Bases",
        "category": "knowledge_bases",
        "feature_type": "usage_limit",
        "measurement_unit": "bases",
        "description": "Separate knowledge repositories",
        "sort_order": 1,
      },
      {
        "name": "files_per_kb",
        "display_name": "Files per Knowledge Base",
        "category": "knowledge_bases",
        "feature_type": "usage_limit",
        "measurement_unit": "files",
        "description": "Maximum files",
        "sort_order": 2,
      },
      {
        "name": "total_storage",
        "display_name": "Total Storage",
        "category": "knowledge_bases",
        "feature_type": "storage_limit",
        "measurement_unit": "MB",
        "description": "Combined storage limit",
        "sort_order": 3,
      },
      {
        "name": "file_types",
        "display_name": "File Types Supported",
        "category": "knowledge_bases",
        "feature_type": "feature_toggle",
        "measurement_unit": "types",
        "description": "Allowed formats",
        "sort_order": 4,
      },
      {
        "name": "max_file_size",
        "display_name": "Max File Size",
        "category": "knowledge_bases",
        "feature_type": "storage_limit",
        "measurement_unit": "MB",
        "description": "Per file",
        "sort_order": 5,
      },
      {
        "name": "vector_embeddings",
        "display_name": "Vector Embeddings",
        "category": "knowledge_bases",
        "feature_type": "usage_limit",
        "measurement_unit": "chunks",
        "description": "Searchable chunks",
        "sort_order": 6,
      },
      {
        "name": "web_scraping",
        "display_name": "Web Scraping",
        "category": "knowledge_bases",
        "feature_type": "usage_limit",
        "measurement_unit": "pages",
        "description": "Import web content",
        "sort_order": 7,
      },
      # AI Agent Tools
      {
        "name": "web_search_tool",
        "display_name": "Web Search Tool",
        "category": "ai_agent_tools",
        "feature_type": "usage_limit",
        "measurement_unit": "searches",
        "description": "Internet access for agents",
        "sort_order": 1,
      },
      {
        "name": "calculator_tool",
        "display_name": "Calculator Tool",
        "category": "ai_agent_tools",
        "feature_type": "feature_toggle",
        "measurement_unit": "boolean",
        "description": "Mathematical computations",
        "sort_order": 2,
      },
      {
        "name": "code_execution",
        "display_name": "Code Execution",
        "category": "ai_agent_tools",
        "feature_type": "feature_toggle",
        "measurement_unit": "languages",
        "description": "Run code",
        "sort_order": 3,
      },
      {
        "name": "api_calling",
        "display_name": "API Calling",
        "category": "ai_agent_tools",
        "feature_type": "usage_limit",
        "measurement_unit": "calls",
        "description": "External API integration",
        "sort_order": 4,
      },
      {
        "name": "custom_tools",
        "display_name": "Custom Tools",
        "category": "ai_agent_tools",
        "feature_type": "usage_limit",
        "measurement_unit": "tools",
        "description": "Build your own tools",
        "sort_order": 5,
      },
      # Agent Creation & Deployment
      {
        "name": "agent_count",
        "display_name": "Number of Agents",
        "category": "agent_deployment",
        "feature_type": "usage_limit",
        "measurement_unit": "agents",
        "description": "Total agents you can create",
        "sort_order": 1,
      },
      {
        "name": "deployment_limit",
        "display_name": "One-Click Deployment",
        "category": "agent_deployment",
        "feature_type": "usage_limit",
        "measurement_unit": "deployments",
        "description": "Push to cloud",
        "sort_order": 2,
      },
      {
        "name": "api_endpoints",
        "display_name": "API Endpoints",
        "category": "agent_deployment",
        "feature_type": "usage_limit",
        "measurement_unit": "calls",
        "description": "REST API access",
        "sort_order": 3,
      },
      # Marketplace Features
      {
        "name": "marketplace_listings",
        "display_name": "List Agents in Marketplace",
        "category": "marketplace",
        "feature_type": "usage_limit",
        "measurement_unit": "agents",
        "description": "Sell your agents",
        "sort_order": 1,
      },
      {
        "name": "revenue_share",
        "display_name": "Revenue Share",
        "category": "marketplace",
        "feature_type": "feature_toggle",
        "measurement_unit": "percentage",
        "description": "Your earnings",
        "sort_order": 2,
      },
      # Collaboration & Team Features
      {
        "name": "team_members",
        "display_name": "Team Members",
        "category": "collaboration",
        "feature_type": "usage_limit",
        "measurement_unit": "users",
        "description": "Number of users",
        "sort_order": 1,
      },
      {
        "name": "api_rate_limits",
        "display_name": "API Rate Limits",
        "category": "collaboration",
        "feature_type": "usage_limit",
        "measurement_unit": "requests",
        "description": "Requests per second",
        "sort_order": 2,
      },
    ]

  def _define_limits(self) -> List[Dict]:
    """Define plan limits for each feature across Free/Pro/Enterprise tiers."""
    return [
      # Text Models - Free Plan (starter)
      {"plan": "starter", "feature": "gpt-4o", "available": True, "limit": 50, "credits_per_1k": 10, "reset_period": "monthly"},
      {"plan": "starter", "feature": "gpt-4o-mini", "available": True, "limit": 250, "credits_per_1k": 2, "reset_period": "monthly"},
      {"plan": "starter", "feature": "claude-sonnet-4", "available": True, "limit": 33, "credits_per_1k": 15, "reset_period": "monthly"},
      {"plan": "starter", "feature": "claude-3.5-sonnet", "available": True, "limit": 41, "credits_per_1k": 12, "reset_period": "monthly"},
      {"plan": "starter", "feature": "gemini-1.5-pro", "available": True, "limit": 62, "credits_per_1k": 8, "reset_period": "monthly"},
      {"plan": "starter", "feature": "deepseek-v3", "available": True, "limit": 100, "credits_per_1k": 5, "reset_period": "monthly"},
      # Text Models - Pro Plan
      {"plan": "pro", "feature": "gpt-5", "available": True, "limit": 200, "credits_per_1k": 25, "reset_period": "monthly"},
      {"plan": "pro", "feature": "gpt-4o", "available": True, "limit": 500, "credits_per_1k": 10, "reset_period": "monthly"},
      {"plan": "pro", "feature": "gpt-4o-mini", "available": True, "limit": 2500, "credits_per_1k": 2, "reset_period": "monthly"},
      {"plan": "pro", "feature": "claude-opus-4.1", "available": True, "limit": 166, "credits_per_1k": 30, "reset_period": "monthly"},
      {"plan": "pro", "feature": "claude-sonnet-4", "available": True, "limit": 333, "credits_per_1k": 15, "reset_period": "monthly"},
      {"plan": "pro", "feature": "claude-3.5-sonnet", "available": True, "limit": 416, "credits_per_1k": 12, "reset_period": "monthly"},
      {"plan": "pro", "feature": "gemini-2.0-ultra", "available": True, "limit": 250, "credits_per_1k": 20, "reset_period": "monthly"},
      {"plan": "pro", "feature": "gemini-1.5-pro", "available": True, "limit": 625, "credits_per_1k": 8, "reset_period": "monthly"},
      {"plan": "pro", "feature": "deepseek-v3", "available": True, "limit": 1000, "credits_per_1k": 5, "reset_period": "monthly"},
      {"plan": "pro", "feature": "o1-preview", "available": True, "limit": 50, "credits_per_request": 100, "reset_period": "monthly"},
      {"plan": "pro", "feature": "o1-mini", "available": True, "limit": 166, "credits_per_request": 30, "reset_period": "monthly"},
      # Text Models - Enterprise Plan
      {"plan": "enterprise", "feature": "gpt-5", "available": True, "limit": 800, "credits_per_1k": 25, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "gpt-4o", "available": True, "limit": 2000, "credits_per_1k": 10, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "gpt-4o-mini", "available": True, "limit": 10000, "credits_per_1k": 2, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "claude-opus-4.1", "available": True, "limit": 666, "credits_per_1k": 30, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "claude-sonnet-4", "available": True, "limit": 1333, "credits_per_1k": 15, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "claude-3.5-sonnet", "available": True, "limit": 1666, "credits_per_1k": 12, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "gemini-2.0-ultra", "available": True, "limit": 1000, "credits_per_1k": 20, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "gemini-1.5-pro", "available": True, "limit": 2500, "credits_per_1k": 8, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "deepseek-v3", "available": True, "limit": 4000, "credits_per_1k": 5, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "o1-preview", "available": True, "limit": 200, "credits_per_request": 100, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "o1-mini", "available": True, "limit": 666, "credits_per_request": 30, "reset_period": "monthly"},
      # Chat Features - All Plans
      {"plan": "starter", "feature": "daily_chat_limit", "available": True, "limit": 30, "reset_period": "daily"},
      {"plan": "pro", "feature": "daily_chat_limit", "available": True, "limit": 500, "reset_period": "daily"},
      {"plan": "enterprise", "feature": "daily_chat_limit", "available": True, "limit": 2000, "reset_period": "daily"},
      {"plan": "starter", "feature": "context_window", "available": True, "limit": 8000, "reset_period": "never"},
      {"plan": "pro", "feature": "context_window", "available": True, "limit": 128000, "reset_period": "never"},
      {"plan": "enterprise", "feature": "context_window", "available": True, "limit": 200000, "reset_period": "never"},
      {"plan": "starter", "feature": "file_uploads", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "file_uploads", "available": True, "limit": 10, "reset_period": "never"},
      {"plan": "enterprise", "feature": "file_uploads", "available": True, "limit": 50, "reset_period": "never"},
      {"plan": "starter", "feature": "voice_input_output", "available": False, "limit": 0, "reset_period": "monthly"},
      {"plan": "pro", "feature": "voice_input_output", "available": True, "limit": 60, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "voice_input_output", "available": True, "limit": 300, "reset_period": "monthly"},
      {"plan": "starter", "feature": "web_search", "available": True, "limit": 5, "reset_period": "daily"},
      {"plan": "pro", "feature": "web_search", "available": True, "limit": 100, "reset_period": "daily"},
      {"plan": "enterprise", "feature": "web_search", "available": True, "limit": 500, "reset_period": "daily"},
      {"plan": "starter", "feature": "deep_research", "available": False, "limit": 0, "reset_period": "monthly"},
      {"plan": "pro", "feature": "deep_research", "available": True, "limit": 10, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "deep_research", "available": True, "limit": 50, "reset_period": "monthly"},
      {"plan": "starter", "feature": "code_interpreter", "available": False, "limit": 0, "reset_period": "never"},
      {"plan": "pro", "feature": "code_interpreter", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "enterprise", "feature": "code_interpreter", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "starter", "feature": "multi_model_chat", "available": False, "limit": 0, "reset_period": "never"},
      {"plan": "pro", "feature": "multi_model_chat", "available": True, "limit": 3, "reset_period": "never"},
      {"plan": "enterprise", "feature": "multi_model_chat", "available": True, "limit": 5, "reset_period": "never"},
      {"plan": "starter", "feature": "chat_history", "available": True, "limit": 7, "reset_period": "never"},
      {"plan": "pro", "feature": "chat_history", "available": True, "limit": 90, "reset_period": "never"},
      {"plan": "enterprise", "feature": "chat_history", "available": True, "limit": 365, "reset_period": "never"},
      {"plan": "starter", "feature": "custom_system_prompts", "available": False, "limit": 0, "reset_period": "never"},
      {"plan": "pro", "feature": "custom_system_prompts", "available": True, "limit": 5, "reset_period": "never"},
      {"plan": "enterprise", "feature": "custom_system_prompts", "available": True, "limit": 20, "reset_period": "never"},
      # Image Models
      {"plan": "starter", "feature": "dall-e-3", "available": True, "limit": 20, "credits_per_image": 25, "reset_period": "monthly"},
      {"plan": "pro", "feature": "dall-e-3", "available": True, "limit": 200, "credits_per_image": 25, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "dall-e-3", "available": True, "limit": 800, "credits_per_image": 25, "reset_period": "monthly"},
      {"plan": "starter", "feature": "dall-e-2", "available": True, "limit": 50, "credits_per_image": 10, "reset_period": "monthly"},
      {"plan": "pro", "feature": "dall-e-2", "available": True, "limit": 500, "credits_per_image": 10, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "dall-e-2", "available": True, "limit": 2000, "credits_per_image": 10, "reset_period": "monthly"},
      {"plan": "starter", "feature": "stable-diffusion-xl", "available": True, "limit": 100, "credits_per_image": 5, "reset_period": "monthly"},
      {"plan": "pro", "feature": "stable-diffusion-xl", "available": True, "limit": 1000, "credits_per_image": 5, "reset_period": "monthly"},
      {"plan": "enterprise", "feature": "stable-diffusion-xl", "available": True, "limit": 4000, "credits_per_image": 5, "reset_period": "monthly"},
      # Knowledge Bases
      {"plan": "starter", "feature": "knowledge_base_count", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "knowledge_base_count", "available": True, "limit": 10, "reset_period": "never"},
      {"plan": "enterprise", "feature": "knowledge_base_count", "available": True, "limit": 50, "reset_period": "never"},
      {"plan": "starter", "feature": "files_per_kb", "available": True, "limit": 5, "reset_period": "never"},
      {"plan": "pro", "feature": "files_per_kb", "available": True, "limit": 100, "reset_period": "never"},
      {"plan": "enterprise", "feature": "files_per_kb", "available": True, "limit": 1000, "reset_period": "never"},
      {"plan": "starter", "feature": "total_storage", "available": True, "limit": 100, "reset_period": "never"},  # MB
      {"plan": "pro", "feature": "total_storage", "available": True, "limit": 10240, "reset_period": "never"},  # 10 GB
      {"plan": "enterprise", "feature": "total_storage", "available": True, "limit": 102400, "reset_period": "never"},  # 100 GB
      # AI Agent Tools
      {"plan": "starter", "feature": "web_search_tool", "available": True, "limit": 10, "reset_period": "daily"},
      {"plan": "pro", "feature": "web_search_tool", "available": True, "limit": 1000, "reset_period": "daily"},
      {"plan": "enterprise", "feature": "web_search_tool", "available": True, "limit": 10000, "reset_period": "daily"},
      {"plan": "starter", "feature": "calculator_tool", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "calculator_tool", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "enterprise", "feature": "calculator_tool", "available": True, "limit": 1, "reset_period": "never"},
      # Agent Creation & Deployment
      {"plan": "starter", "feature": "agent_count", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "agent_count", "available": True, "limit": 10, "reset_period": "never"},
      {"plan": "enterprise", "feature": "agent_count", "available": True, "limit": 100, "reset_period": "never"},
      {"plan": "starter", "feature": "deployment_limit", "available": True, "limit": 1, "reset_period": "daily"},
      {"plan": "pro", "feature": "deployment_limit", "available": True, "limit": 50, "reset_period": "daily"},
      {"plan": "enterprise", "feature": "deployment_limit", "available": True, "limit": 500, "reset_period": "daily"},
      # Team Features
      {"plan": "starter", "feature": "team_members", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "team_members", "available": True, "limit": 5, "reset_period": "never"},
      {"plan": "enterprise", "feature": "team_members", "available": True, "limit": 25, "reset_period": "never"},
      {"plan": "starter", "feature": "api_rate_limits", "available": True, "limit": 1, "reset_period": "never"},
      {"plan": "pro", "feature": "api_rate_limits", "available": True, "limit": 10, "reset_period": "never"},
      {"plan": "enterprise", "feature": "api_rate_limits", "available": True, "limit": 100, "reset_period": "never"},
    ]

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting plan settings population...")

    try:
      await self._insert_categories(db)
      await self._insert_features(db)
      await self._insert_limits(db)

      await db.commit()
      logger.info("✅ Plan settings populated successfully!")

    except Exception as e:
      await db.rollback()
      logger.error(f"❌ Error populating plan settings: {str(e)}")
      raise

  async def _insert_categories(self, db: AsyncSession) -> None:
    """Insert plan feature categories."""
    logger.info("Inserting feature categories...")
    categories = self._define_categories()

    for category in categories:
      await db.execute(
        text("""
          INSERT INTO plan_feature_categories (id, name, display_name, description, sort_order)
          VALUES (gen_random_uuid(), :name, :display_name, :description, :sort_order)
          ON CONFLICT (name) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            description = EXCLUDED.description,
            sort_order = EXCLUDED.sort_order
        """),
        category,
      )

  async def _insert_features(self, db: AsyncSession) -> None:
    """Insert plan features."""
    logger.info("Inserting features...")
    features = self._define_features()

    for feature in features:
      await db.execute(
        text("""
          INSERT INTO plan_features (id, category_id, name, display_name, feature_type, measurement_unit, description, sort_order)
          SELECT gen_random_uuid(), c.id, :name, :display_name, CAST(:feature_type AS feature_type_enum),
                 :measurement_unit, :description, :sort_order
          FROM plan_feature_categories c
          WHERE c.name = :category
          ON CONFLICT (name) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            feature_type = EXCLUDED.feature_type,
            measurement_unit = EXCLUDED.measurement_unit,
            description = EXCLUDED.description,
            sort_order = EXCLUDED.sort_order
        """),
        feature,
      )

  async def _insert_limits(self, db: AsyncSession) -> None:
    """Insert plan feature limits."""
    logger.info("Inserting plan limits...")
    limits = self._define_limits()

    for limit in limits:
      # Build metadata JSON
      metadata = {}
      if "credits_per_1k" in limit:
        metadata["credits_per_1k_tokens"] = limit["credits_per_1k"]
      if "credits_per_image" in limit:
        metadata["credits_per_image"] = limit["credits_per_image"]
      if "credits_per_request" in limit:
        metadata["credits_per_request"] = limit["credits_per_request"]

      metadata_json = None
      if metadata:
        import json

        metadata_json = json.dumps(metadata)

      await db.execute(
        text("""
          INSERT INTO plan_feature_limits (billing_plan_id, feature_id, is_available, limit_value, limit_metadata, reset_period)
          SELECT bp.id, pf.id, :available, :limit, CAST(:metadata AS jsonb), CAST(:reset_period AS reset_period_enum)
          FROM billing_plans bp
          CROSS JOIN plan_features pf
          WHERE bp.name = :plan
            AND bp.cycle = 'MONTHLY'
            AND pf.name = :feature
          ON CONFLICT (billing_plan_id, feature_id) DO UPDATE SET
            is_available = EXCLUDED.is_available,
            limit_value = EXCLUDED.limit_value,
            limit_metadata = EXCLUDED.limit_metadata,
            reset_period = EXCLUDED.reset_period
        """),
        {
          "plan": limit["plan"],
          "feature": limit["feature"],
          "available": limit["available"],
          "limit": limit["limit"],
          "metadata": metadata_json,
          "reset_period": limit["reset_period"],
        },
      )

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback the script execution by deleting created plan data."""
    logger.info("Rolling back plan settings...")

    # Delete in reverse order due to foreign key constraints
    # Delete limits first
    limits_result = await db.execute(text("DELETE FROM plan_feature_limits"))
    limits_deleted = getattr(limits_result, "rowcount", 0)

    # Delete features
    features_result = await db.execute(text("DELETE FROM plan_features"))
    features_deleted = getattr(features_result, "rowcount", 0)

    # Delete categories
    categories_result = await db.execute(text("DELETE FROM plan_feature_categories"))
    categories_deleted = getattr(categories_result, "rowcount", 0)

    await db.commit()

    logger.info(f"Deleted {limits_deleted} limits, {features_deleted} features, and {categories_deleted} categories during rollback")

  async def verify(self, db: AsyncSession) -> bool:
    """Verify script execution was successful."""
    # Check if categories were created
    categories_result = await db.execute(text("SELECT COUNT(*) FROM plan_feature_categories"))
    categories_count = categories_result.scalar()

    # Check if features were created
    features_result = await db.execute(text("SELECT COUNT(*) FROM plan_features"))
    features_count = features_result.scalar()

    # Check if limits were created
    limits_result = await db.execute(text("SELECT COUNT(*) FROM plan_feature_limits"))
    limits_count = limits_result.scalar()

    if not categories_count or not features_count or not limits_count:
      logger.error(f"Verification failed: categories={categories_count}, features={features_count}, limits={limits_count}")
      return False

    logger.info(f"Verification passed: {categories_count} categories, {features_count} features, {limits_count} limits created")
    return True


def main():
  """Entry point for backward compatibility."""
  script = PopulatePlanSettingsScript()
  script.main()


if __name__ == "__main__":
  script = PopulatePlanSettingsScript()
  script.run_cli()
