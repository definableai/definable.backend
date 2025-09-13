#!/usr/bin/env python3
"""
Update model specifications with comprehensive data from all-models.json.

This script updates the model_metadata.specifications field for all supported models
with detailed information including context windows, pricing, capabilities, and more.
"""

import os
import sys

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import log as logger
from scripts.core.base_script import BaseScript


class UpdateModelSpecificationsScript(BaseScript):
  """Update model specifications with comprehensive data from all-models.json."""

  def __init__(self):
    super().__init__("update_model_specifications")

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting model specifications update...")

    # Update OpenAI models
    await self._update_openai_models(db)

    # Update Anthropic models
    await self._update_anthropic_models(db)

    # Update DeepSeek models
    await self._update_deepseek_models(db)

    # Update xAI/Grok models
    await self._update_grok_models(db)

    # Update Google/Gemini models
    await self._update_gemini_models(db)

    logger.info("Model specifications update completed successfully!")

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback logic."""
    logger.info("Rolling back model specifications...")

    # Clear model specifications
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) - 'specifications'
            WHERE model_metadata ? 'specifications';
        """)
    )

    await db.commit()
    logger.info("Model specifications rollback completed!")

  async def verify(self, db: AsyncSession) -> bool:
    """Verification logic."""
    # Check if models have specifications
    result = await db.execute(
      text("""
            SELECT COUNT(*) FROM models
            WHERE model_metadata ? 'specifications'
            AND model_metadata->'specifications' ? 'context_window'
        """)
    )
    specs_count = result.scalar()

    logger.info(f"Verification: {specs_count} models have specifications")
    return (specs_count or 0) > 0

  async def _update_openai_models(self, db: AsyncSession) -> None:
    """Update OpenAI model specifications."""
    logger.info("Updating OpenAI model specifications...")

    # GPT-4o - Fast, intelligent, flexible model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": 16384,
                    "pricing_per_token": {
                        "input": 2.5,
                        "output": 10.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "streaming", "structured_outputs", "function_calling"],
                    "modalities": ["text", "image"],
                    "features": ["streaming", "structured_outputs", "function_calling"],
                    "endpoints": ["chat_completions", "responses"],
                    "provider": "OpenAI",
                    "performance_tier": "high",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": false,
                    "snapshots": ["gpt-4o", "gpt-4o-2024-08-06"],
                    "rate_limits": {
                        "tier_1": {"rpm": 500, "tpm": 200000},
                        "tier_2": {"rpm": 5000, "tpm": 2000000},
                        "tier_3": {"rpm": 5000, "tpm": 40000000},
                        "tier_4": {"rpm": 10000, "tpm": 80000000},
                        "tier_5": {"rpm": 30000, "tpm": 160000000}
                    },
                    "performance_score": 4,
                    "speed_score": 4
                }
            }'::jsonb
            WHERE name ILIKE '%gpt-4o%' AND name NOT ILIKE '%mini%'
                AND name NOT ILIKE '%search%' AND name NOT ILIKE '%transcribe%' AND name NOT ILIKE '%tts%';
        """)
    )

    # GPT-4o Mini - Alternative efficient version
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": 16384,
                    "pricing_per_token": {
                        "input": 0.15,
                        "output": 0.6,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "streaming", "structured_outputs", "function_calling"],
                    "modalities": ["text", "image"],
                    "features": ["streaming", "structured_outputs", "function_calling"],
                    "endpoints": ["chat_completions", "responses"],
                    "provider": "OpenAI",
                    "performance_tier": "fast",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": false,
                    "snapshots": ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
                    "rate_limits": {
                        "tier_1": {"rpm": 500, "tpm": 200000},
                        "tier_2": {"rpm": 5000, "tpm": 2000000},
                        "tier_3": {"rpm": 5000, "tpm": 40000000},
                        "tier_4": {"rpm": 10000, "tpm": 80000000},
                        "tier_5": {"rpm": 30000, "tpm": 160000000}
                    },
                    "performance_score": 3,
                    "speed_score": 5
                }
            }'::jsonb
            WHERE name ILIKE '%gpt-4o-mini%' AND name NOT ILIKE '%search%'
                AND name NOT ILIKE '%transcribe%' AND name NOT ILIKE '%tts%';
        """)
    )

    # o1 - Full reasoning model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 200000,
                    "max_output_tokens": 100000,
                    "pricing_per_token": {
                        "input": 15.0,
                        "output": 60.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "reasoning"],
                    "modalities": ["text"],
                    "features": ["reasoning"],
                    "endpoints": ["chat_completions"],
                    "provider": "OpenAI",
                    "performance_tier": "reasoning",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": true,
                    "snapshots": ["o1", "o1-2024-12-17"],
                    "rate_limits": {
                        "tier_1": {"rpm": 500, "tpm": 200000},
                        "tier_2": {"rpm": 5000, "tpm": 2000000},
                        "tier_3": {"rpm": 5000, "tpm": 40000000},
                        "tier_4": {"rpm": 10000, "tpm": 80000000},
                        "tier_5": {"rpm": 30000, "tpm": 160000000}
                    },
                    "reasoning_score": 5,
                    "speed_score": 2
                }
            }'::jsonb
            WHERE name ILIKE '%o1%' AND name NOT ILIKE '%mini%' AND name NOT ILIKE '%pro%' AND name NOT ILIKE '%preview%';
        """)
    )

    # o1-mini - Efficient reasoning model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": 65536,
                    "pricing_per_token": {
                        "input": 3.0,
                        "output": 12.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "reasoning"],
                    "modalities": ["text"],
                    "features": ["reasoning"],
                    "endpoints": ["chat_completions"],
                    "provider": "OpenAI",
                    "performance_tier": "reasoning",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": true,
                    "snapshots": ["o1-mini", "o1-mini-2024-09-12"],
                    "rate_limits": {
                        "tier_1": {"rpm": 500, "tpm": 200000},
                        "tier_2": {"rpm": 5000, "tpm": 2000000},
                        "tier_3": {"rpm": 5000, "tpm": 40000000},
                        "tier_4": {"rpm": 10000, "tpm": 80000000},
                        "tier_5": {"rpm": 30000, "tpm": 160000000}
                    },
                    "reasoning_score": 4,
                    "speed_score": 3
                }
            }'::jsonb
            WHERE name ILIKE '%o1-mini%';
        """)
    )

    # GPT-3.5 Turbo - Legacy conversational model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 16385,
                    "max_output_tokens": 4096,
                    "pricing_per_token": {
                        "input": 0.5,
                        "output": 1.5,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "function_calling", "json_output"],
                    "modalities": ["text"],
                    "features": ["function_calling", "json_output"],
                    "endpoints": ["chat_completions"],
                    "provider": "OpenAI",
                    "performance_tier": "fast",
                    "knowledge_cutoff": "Sep 2021",
                    "reasoning_token_support": false,
                    "snapshots": ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"],
                    "rate_limits": {
                        "tier_1": {"rpm": 3500, "tpm": 200000},
                        "tier_2": {"rpm": 3500, "tpm": 2000000},
                        "tier_3": {"rpm": 3500, "tpm": 40000000},
                        "tier_4": {"rpm": 5000, "tpm": 80000000},
                        "tier_5": {"rpm": 10000, "tpm": 160000000}
                    },
                    "performance_score": 2,
                    "speed_score": 5
                }
            }'::jsonb
            WHERE name ILIKE '%gpt-3.5-turbo%';
        """)
    )

    # o1-preview - Preview of first o-series reasoning model (Deprecated)
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": 32768,
                    "pricing_per_token": {
                        "input": 15.0,
                        "output": 60.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["reasoning"],
                    "modalities": ["text"],
                    "features": ["reasoning"],
                    "endpoints": ["chat_completions"],
                    "provider": "OpenAI",
                    "performance_tier": "reasoning",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": true,
                    "snapshots": ["o1-preview", "o1-preview-2024-09-12"],
                    "rate_limits": {
                        "tier_1": {"rpm": 500, "tpm": 200000},
                        "tier_2": {"rpm": 5000, "tpm": 2000000},
                        "tier_3": {"rpm": 5000, "tpm": 40000000},
                        "tier_4": {"rpm": 10000, "tpm": 80000000},
                        "tier_5": {"rpm": 30000, "tpm": 160000000}
                    },
                    "reasoning_score": 5,
                    "speed_score": 1
                }
            }'::jsonb
            WHERE name ILIKE '%o1-preview%';
        """)
    )

    # o3-mini - Small model alternative to o3
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 200000,
                    "max_output_tokens": 100000,
                    "pricing_per_token": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "capabilities": ["reasoning"],
                    "modalities": ["text"],
                    "features": ["reasoning"],
                    "endpoints": null,
                    "provider": "OpenAI",
                    "performance_tier": "reasoning",
                    "knowledge_cutoff": "Oct 01, 2023",
                    "reasoning_token_support": true,
                    "snapshots": ["o3-mini"],
                    "rate_limits": null,
                    "reasoning_score": 4,
                    "speed_score": 3
                }
            }'::jsonb
            WHERE name ILIKE '%o3-mini%';
        """)
    )

    await db.commit()

  async def _update_anthropic_models(self, db: AsyncSession) -> None:
    """Update Anthropic model specifications."""
    logger.info("Updating Anthropic model specifications...")

    # Claude Opus 4.1 - Most capable model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "description": "Our most capable and intelligent model yet. Sets new standards in complex reasoning and advanced coding",
                    "context_window": 200000,
                    "max_output_tokens": 32000,
                    "pricing_per_token": {
                        "input": 15.0,
                        "cached_input_5m": 18.75,
                        "cached_input_1h": 30.0,
                        "cache_hits": 1.50,
                        "output": 75.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "extended_thinking", "priority_tier", "streaming", "vision", "multilingual"],
                    "modalities": ["text", "image"],
                    "features": ["extended_thinking", "priority_tier", "streaming", "vision", "multilingual"],
                    "endpoints": ["messages", "anthropic_api", "aws_bedrock", "gcp_vertex"],
                    "provider": "Anthropic",
                    "performance_tier": "highest",
                    "knowledge_cutoff": "2025-03-31",
                    "reasoning_token_support": true,
                    "snapshots": ["claude-opus-4-1-20250805"],
                    "aliases": ["claude-opus-4-1"],
                    "aws_bedrock_id": "anthropic.claude-opus-4-1-20250805-v1:0",
                    "gcp_vertex_id": "claude-opus-4-1@20250805",
                    "latency": "moderately_fast",
                    "strengths": "Highest level of intelligence and capability",
                    "status": "active"
                }
            }'::jsonb
            WHERE name = 'claude-opus-4.1' OR name ILIKE '%claude-opus-4-1%';
        """)
    )

    # Claude Opus 4 - Previous flagship model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "description": "Our previous flagship model with very high intelligence and capability",
                    "context_window": 200000,
                    "max_output_tokens": 32000,
                    "pricing_per_token": {
                        "input": 15.0,
                        "cached_input_5m": 18.75,
                        "cached_input_1h": 30.0,
                        "cache_hits": 1.50,
                        "output": 75.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "extended_thinking", "priority_tier", "streaming", "vision", "multilingual"],
                    "modalities": ["text", "image"],
                    "features": ["extended_thinking", "priority_tier", "streaming", "vision", "multilingual"],
                    "endpoints": ["messages", "anthropic_api", "aws_bedrock", "gcp_vertex"],
                    "provider": "Anthropic",
                    "performance_tier": "highest",
                    "knowledge_cutoff": "2025-03-31",
                    "reasoning_token_support": true,
                    "snapshots": ["claude-opus-4-20250514"],
                    "aliases": ["claude-opus-4-0"],
                    "aws_bedrock_id": "anthropic.claude-opus-4-20250514-v1:0",
                    "gcp_vertex_id": "claude-opus-4@20250514",
                    "latency": "moderately_fast",
                    "strengths": "Very high intelligence and capability",
                    "status": "active"
                }
            }'::jsonb
            WHERE name = 'claude-opus-4';
        """)
    )

    # Claude Sonnet 4 - High-performance with exceptional reasoning
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "description": "High-performance model with exceptional reasoning and efficiency",
                    "context_window": 200000,
                    "context_window_beta": 1000000,
                    "max_output_tokens": 64000,
                    "pricing_per_token": {
                        "input": 3.0,
                        "cached_input_5m": 3.75,
                        "cached_input_1h": 6.0,
                        "cache_hits": 0.30,
                        "output": 15.0,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "extended_thinking", "priority_tier", "streaming", "vision", "multilingual", "1m_context_beta"],
                    "modalities": ["text", "image"],
                    "features": ["extended_thinking", "priority_tier", "streaming", "vision", "multilingual", "1m_context_beta"],
                    "endpoints": ["messages", "anthropic_api", "aws_bedrock", "gcp_vertex"],
                    "provider": "Anthropic",
                    "performance_tier": "high",
                    "knowledge_cutoff": "2025-03-31",
                    "reasoning_token_support": true,
                    "snapshots": ["claude-sonnet-4-20250514"],
                    "aliases": ["claude-sonnet-4-0"],
                    "aws_bedrock_id": "anthropic.claude-sonnet-4-20250514-v1:0",
                    "gcp_vertex_id": "claude-sonnet-4@20250514",
                    "latency": "fast",
                    "strengths": "High intelligence and balanced performance",
                    "beta_headers": ["context-1m-2025-08-07"],
                    "status": "active"
                }
            }'::jsonb
            WHERE name = 'claude-sonnet-4';
        """)
    )

    # Claude Haiku 3 - Fast and compact model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 200000,
                    "max_output_tokens": 4096,
                    "pricing_per_token": {
                        "input": 0.25,
                        "cached_input_5m": 0.30,
                        "cached_input_1h": 0.50,
                        "cache_hits": 0.03,
                        "output": 1.25,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["text", "image", "streaming", "vision", "multilingual"],
                    "modalities": ["text", "image"],
                    "features": ["streaming", "vision", "multilingual"],
                    "endpoints": ["messages", "anthropic_api", "aws_bedrock", "gcp_vertex"],
                    "provider": "Anthropic",
                    "performance_tier": "fast",
                    "knowledge_cutoff": "2023-08-31",
                    "reasoning_token_support": false,
                    "snapshots": ["claude-3-haiku-20240307"],
                    "aws_bedrock_id": "anthropic.claude-3-haiku-20240307-v1:0",
                    "gcp_vertex_id": "claude-3-haiku@20240307",
                    "latency": "fast",
                    "strengths": "Quick and accurate targeted performance",
                    "speed_score": 4
                }
            }'::jsonb
            WHERE name ILIKE '%claude-3-haiku%' AND name NOT ILIKE '%3-5%' AND name NOT ILIKE '%3.5%';
        """)
    )

    await db.commit()

  async def _update_deepseek_models(self, db: AsyncSession) -> None:
    """Update DeepSeek model specifications."""
    logger.info("Updating DeepSeek model specifications...")

    # DeepSeek Chat - Non-thinking mode for general chat and reasoning
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": {
                        "default": 4000,
                        "maximum": 8000
                    },
                    "pricing_per_token": {
                        "current": {
                            "valid_until": "2025-09-05T16:00:00Z",
                            "standard_hours": "UTC 00:30-16:30",
                            "discount_hours": "UTC 16:30-00:30",
                            "standard": {
                                "input_cache_hit": 0.07,
                                "input_cache_miss": 0.27,
                                "output": 1.10,
                                "unit": "per 1M tokens"
                            },
                            "discount": {
                                "input_cache_hit": 0.035,
                                "input_cache_miss": 0.135,
                                "output": 0.550,
                                "unit": "per 1M tokens"
                            }
                        },
                        "future": {
                            "valid_from": "2025-09-05T16:00:00Z",
                            "input_cache_hit": 0.07,
                            "input_cache_miss": 0.56,
                            "output": 1.68,
                            "unit": "per 1M tokens",
                            "notes": "Nighttime discount cancelled"
                        }
                    },
                    "capabilities": ["json_output", "function_calling", "chat_prefix_completion", "fim_completion"],
                    "modalities": ["text"],
                    "features": ["json_output", "function_calling", "chat_prefix_completion_beta", "fim_completion_beta", "streaming", "caching"],
                    "endpoints": ["chat_completions"],
                    "provider": "DeepSeek",
                    "performance_tier": "high",
                    "model_version": "DeepSeek-V3.1",
                    "reasoning_token_support": false,
                    "thinking_mode": false
                }
            }'::jsonb
            WHERE name ILIKE '%deepseek-chat%';
        """)
    )

    # DeepSeek Reasoner - Thinking mode for advanced reasoning
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 128000,
                    "max_output_tokens": {
                        "default": 32000,
                        "maximum": 64000
                    },
                    "pricing_per_token": {
                        "current": {
                            "valid_until": "2025-09-05T16:00:00Z",
                            "standard_hours": "UTC 00:30-16:30",
                            "discount_hours": "UTC 16:30-00:30",
                            "standard": {
                                "input_cache_hit": 0.14,
                                "input_cache_miss": 0.55,
                                "output": 2.19,
                                "unit": "per 1M tokens"
                            },
                            "discount": {
                                "input_cache_hit": 0.035,
                                "input_cache_miss": 0.135,
                                "output": 0.550,
                                "unit": "per 1M tokens"
                            }
                        },
                        "future": {
                            "valid_from": "2025-09-05T16:00:00Z",
                            "input_cache_hit": 0.07,
                            "input_cache_miss": 0.56,
                            "output": 1.68,
                            "unit": "per 1M tokens",
                            "notes": "Nighttime discount cancelled, same pricing as deepseek-chat"
                        }
                    },
                    "capabilities": ["json_output", "chat_prefix_completion", "advanced_reasoning", "thinking_mode"],
                    "modalities": ["text"],
                    "features": ["json_output", "chat_prefix_completion_beta", "streaming", "caching", "thinking_mode", "advanced_reasoning"],
                    "endpoints": ["chat_completions"],
                    "provider": "DeepSeek",
                    "performance_tier": "reasoning",
                    "model_version": "DeepSeek-V3.1",
                    "reasoning_token_support": true,
                    "thinking_mode": true,
                    "reasoning_score": 5,
                    "limitations": ["Function calling not supported directly", "FIM completion not supported"],
                    "special_behavior": {
                        "function_calling_fallback": "Requests with tools parameter automatically processed using deepseek-chat model"
                    }
                }
            }'::jsonb
            WHERE name ILIKE '%deepseek-reasoner%' OR name ILIKE '%deepseek-reason%';
        """)
    )

    await db.commit()

  async def _update_grok_models(self, db: AsyncSession) -> None:
    """Update xAI/Grok model specifications."""
    logger.info("Updating xAI/Grok model specifications...")

    # Grok-4-0709 - Advanced multimodal model with vision
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 256000,
                    "pricing_per_token": {
                        "input": 3.00,
                        "output": 15.00,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["function_calling", "structured_outputs", "reasoning", "vision"],
                    "modalities": ["text", "image"],
                    "features": ["streaming", "function_calling", "structured_outputs", "vision"],
                    "endpoints": ["chat_completions"],
                    "provider": "xAI",
                    "performance_tier": "high",
                    "reasoning_token_support": false,
                    "rate_limits": {
                        "tpm": 2000000,
                        "rpm": 480
                    }
                }
            }'::jsonb
            WHERE name ILIKE '%grok-4%' OR name ILIKE '%grok-4-0709%' OR name = 'grok-4-0709';
        """)
    )

    # Grok-code-fast-1 - Fast coding-focused model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 256000,
                    "pricing_per_token": {
                        "input": 0.20,
                        "output": 1.50,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["function_calling", "structured_outputs"],
                    "modalities": ["text"],
                    "features": ["streaming", "function_calling", "structured_outputs"],
                    "endpoints": ["chat_completions"],
                    "provider": "xAI",
                    "performance_tier": "fast",
                    "reasoning_token_support": false,
                    "rate_limits": {
                        "tpm": 2000000,
                        "rpm": 480
                    }
                }
            }'::jsonb
            WHERE name ILIKE '%grok-code-fast%' OR name = 'grok-code-fast-1';
        """)
    )

    # Grok-3 - General-purpose language model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 131072,
                    "pricing_per_token": {
                        "input": 3.00,
                        "output": 15.00,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["function_calling", "structured_outputs"],
                    "modalities": ["text"],
                    "features": ["streaming", "function_calling", "structured_outputs"],
                    "endpoints": ["chat_completions"],
                    "provider": "xAI",
                    "performance_tier": "high",
                    "reasoning_token_support": false,
                    "rate_limits": {
                        "rpm": 600
                    }
                }
            }'::jsonb
            WHERE (name ILIKE '%grok-3%' AND name NOT ILIKE '%mini%') OR name = 'grok-3';
        """)
    )

    # Grok-3-mini - Compact and efficient version
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 131072,
                    "pricing_per_token": {
                        "input": 0.30,
                        "output": 0.50,
                        "unit": "per 1M tokens"
                    },
                    "capabilities": ["function_calling", "structured_outputs"],
                    "modalities": ["text"],
                    "features": ["streaming", "function_calling", "structured_outputs"],
                    "endpoints": ["chat_completions"],
                    "provider": "xAI",
                    "performance_tier": "fast",
                    "reasoning_token_support": false,
                    "rate_limits": {
                        "rpm": 480
                    }
                }
            }'::jsonb
            WHERE name ILIKE '%grok-3-mini%' OR name = 'grok-3-mini';
        """)
    )

    await db.commit()

  async def _update_gemini_models(self, db: AsyncSession) -> None:
    """Update Google/Gemini model specifications."""
    logger.info("Updating Google/Gemini model specifications...")

    # Gemini 2.5 Pro - Most powerful thinking model
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 65536,
                    "reasoning_token_support": true,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio", "PDF"],
                    "features": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "search_grounding", "thinking", "batch_mode", "url_context"
                    ],
                    "capabilities": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "search_grounding", "thinking", "batch_mode", "url_context"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "knowledge_cutoff": "January 2025",
                    "performance_tier": "highest",
                    "reasoning_score": 5,
                    "speed_score": 3,
                    "snapshots": ["gemini-2.5-pro"]
                }
            }'::jsonb
            WHERE (name ILIKE '%gemini-2.5-pro%' AND name NOT ILIKE '%tts%' AND name NOT ILIKE '%preview%') OR name = 'gemini-2.5-pro';
        """)
    )

    # Gemini 2.5 Flash - Best price-performance with thinking
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 65536,
                    "reasoning_token_support": true,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "url_context", "search_grounding", "thinking", "batch_mode"
                    ],
                    "capabilities": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "search_grounding", "thinking", "batch_mode", "url_context"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "knowledge_cutoff": "January 2025",
                    "performance_tier": "high",
                    "reasoning_score": 4,
                    "speed_score": 4,
                    "snapshots": ["gemini-2.5-flash"]
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-2.5-flash%' AND name NOT ILIKE '%lite%'
                AND name NOT ILIKE '%preview%' AND name NOT ILIKE '%tts%'
                AND name NOT ILIKE '%live%' AND name NOT ILIKE '%image%' AND name NOT ILIKE '%audio%';
        """)
    )

    # Gemini 2.5 Flash Lite - Cost-efficient high throughput
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 65536,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio", "PDF"],
                    "features": ["structured_outputs", "caching", "function_calling", "code_execution", "search", "url_context"],
                    "capabilities": ["structured_outputs", "caching", "function_calling", "code_execution", "search", "url_context"],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "knowledge_cutoff": "January 2025",
                    "performance_tier": "fast",
                    "reasoning_score": 3,
                    "speed_score": 5,
                    "snapshots": ["gemini-2.5-flash-lite"]
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-2.5-flash-lite%';
        """)
    )

    # Gemini 2.0 Flash - Next-gen features and speed
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 8192,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": ["structured_outputs", "caching", "function_calling", "code_execution"],
                    "capabilities": ["structured_outputs", "caching", "function_calling", "code_execution", "search"],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "knowledge_cutoff": "August 2024",
                    "performance_tier": "high",
                    "reasoning_score": 4,
                    "speed_score": 4,
                    "snapshots": ["gemini-2.0-flash"]
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-2.0-flash%' AND name NOT ILIKE '%lite%' AND name NOT ILIKE '%preview%' AND name NOT ILIKE '%live%';
        """)
    )

    # Gemini 2.0 Flash Lite - Cost efficient and low latency
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 8192,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "search", "audio_generation", "url_context"
                    ],
                    "capabilities": [
                        "structured_outputs", "caching", "function_calling", "code_execution",
                        "search", "audio_generation", "url_context"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "knowledge_cutoff": "August 2024",
                    "performance_tier": "fast",
                    "reasoning_score": 3,
                    "speed_score": 5,
                    "snapshots": ["gemini-2.0-flash-lite"]
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-2.0-flash-lite%';
        """)
    )

    # Gemini 1.5 Flash - Fast and versatile (deprecated)
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 8192,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "tuning", "function_calling", "code_execution"
                    ],
                    "capabilities": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "tuning", "function_calling", "code_execution"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "performance_tier": "fast",
                    "reasoning_score": 3,
                    "speed_score": 4,
                    "snapshots": ["gemini-1.5-flash"],
                    "deprecation_date": "September 2025"
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-1.5-flash%' AND name NOT ILIKE '%8b%';
        """)
    )

    # Gemini 1.5 Flash 8B - Small model for lower intelligence tasks (deprecated)
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 1048576,
                    "max_output_tokens": 8192,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "function_calling", "code_execution"
                    ],
                    "capabilities": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "function_calling", "code_execution"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "performance_tier": "fast",
                    "reasoning_score": 2,
                    "speed_score": 5,
                    "snapshots": ["gemini-1.5-flash-8b"],
                    "deprecation_date": "September 2025"
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-1.5-flash-8b%';
        """)
    )

    # Gemini 1.5 Pro - Mid-size multimodal for reasoning (deprecated)
    await db.execute(
      text("""
            UPDATE models
            SET model_metadata = COALESCE(model_metadata::jsonb, '{}'::jsonb) || '{
                "specifications": {
                    "context_window": 2097152,
                    "max_output_tokens": 8192,
                    "reasoning_token_support": false,
                    "pricing": {
                        "input": null,
                        "output": null,
                        "unit": "TBD"
                    },
                    "modalities": ["text", "image", "video", "audio"],
                    "features": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "tuning", "function_calling", "code_execution"
                    ],
                    "capabilities": [
                        "system_instructions", "json_mode", "json_schema", "adjustable_safety_settings",
                        "caching", "tuning", "function_calling", "code_execution"
                    ],
                    "endpoints": ["generateContent"],
                    "provider": "Google",
                    "performance_tier": "high",
                    "reasoning_score": 4,
                    "speed_score": 3,
                    "snapshots": ["gemini-1.5-pro"],
                    "deprecation_date": "September 2025"
                }
            }'::jsonb
            WHERE name ILIKE '%gemini-1.5-pro%';
        """)
    )

    await db.commit()


def main():
  """Entry point for backward compatibility."""
  script = UpdateModelSpecificationsScript()
  script.main()


if __name__ == "__main__":
  script = UpdateModelSpecificationsScript()
  script.run_cli()
