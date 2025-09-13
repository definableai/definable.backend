#!/usr/bin/env python3
"""
Populate marketplace data including model specifications, categories, and marketplace entries.
Uses the script_run_tracker table to track execution.
Enhanced with click commands for rerun and rollback functionality.
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


class PopulateMarketplaceDataScript(BaseScript):
  """Populate marketplace data including model specifications, categories, and marketplace entries."""

  def __init__(self):
    super().__init__("populate_marketplace_data")

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting marketplace data population...")

    # Add missing models from all-models.json that don't exist in the database yet
    await self._insert_missing_models(db)

    # Create marketplace entries for ALL existing LLM models
    await self._create_marketplace_entries(db)

    # Assign categories to existing models
    await self._assign_model_categories(db)

    # Add tags to existing models
    await self._add_model_tags(db)

    logger.info("Marketplace data population completed successfully!")

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback logic."""
    logger.info("Rolling back marketplace data population...")

    # Remove newly inserted models
    await db.execute(
      text("""
            DELETE FROM models WHERE name IN (
                'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                'gemini-2.0-flash', 'gemini-2.0-flash-lite',
                'gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro',
                'grok-4-0709', 'grok-code-fast-1', 'grok-3', 'grok-3-mini',
                'claude-opus-4.1', 'claude-opus-4', 'claude-sonnet-4', 'claude-3-haiku'
            );
        """)
    )

    # Clear marketplace data
    await db.execute(text("DELETE FROM marketplace_usage;"))
    await db.execute(text("DELETE FROM marketplace_reviews;"))
    await db.execute(text("DELETE FROM marketplace_assistants;"))
    await db.execute(text("DELETE FROM llm_model_categories;"))

    # Clear tags from models
    await db.execute(text("UPDATE models SET tags = NULL;"))

    await db.commit()
    logger.info("Marketplace data rollback completed!")

  async def verify(self, db: AsyncSession) -> bool:
    """Verification logic."""
    # Check if marketplace entries were created
    result = await db.execute(text("SELECT COUNT(*) FROM marketplace_assistants WHERE assistant_type = 'llm_model'"))
    marketplace_count = result.scalar()

    # Check if models have categories
    result = await db.execute(text("SELECT COUNT(*) FROM llm_model_categories"))
    category_count = result.scalar()

    # Check if models have tags
    result = await db.execute(text("SELECT COUNT(*) FROM models WHERE tags IS NOT NULL AND array_length(tags, 1) > 0"))
    tags_count = result.scalar()

    logger.info(f"Verification: {marketplace_count} marketplace entries, {category_count} category assignments, {tags_count} models with tags")

    return (marketplace_count or 0) > 0 and (category_count or 0) > 0 and (tags_count or 0) > 0

  async def _insert_missing_models(self, db: AsyncSession) -> None:
    """Insert missing models from all-models.json."""
    logger.info("Inserting missing models...")

    # Insert missing Gemini models (Google)
    gemini_models = [
      ("gemini-2.5-pro", "google", "2.5", True),
      ("gemini-2.5-flash", "google", "2.5", True),
      ("gemini-2.5-flash-lite", "google", "2.5", True),
      ("gemini-2.0-flash", "google", "2.0", True),
      ("gemini-2.0-flash-lite", "google", "2.0", True),
      ("gemini-1.5-flash", "google", "1.5", False),
      ("gemini-1.5-flash-8b", "google", "1.5", False),
      ("gemini-1.5-pro", "google", "1.5", False),
    ]

    # Insert missing Grok models (xAI)
    grok_models = [
      ("grok-4-0709", "xai", "4.0", True),
      ("grok-code-fast-1", "xai", "1.0", True),
      ("grok-3", "xai", "3.0", True),
      ("grok-3-mini", "xai", "3.0", True),
    ]

    # Insert missing Claude models (Anthropic)
    claude_models = [
      ("claude-opus-4.1", "anthropic", "4.1", True),
      ("claude-opus-4", "anthropic", "4.0", True),
      ("claude-sonnet-4", "anthropic", "4.0", True),
      ("claude-3-haiku", "anthropic", "3.0", True),
    ]

    all_models = gemini_models + grok_models + claude_models

    for name, provider, version, is_active in all_models:
      await db.execute(
        text("""
                INSERT INTO models (name, provider, version, is_active, config, props, model_metadata)
                SELECT CAST(:name AS text), CAST(:provider AS text), CAST(:version AS text), CAST(:is_active AS boolean),
                       CAST('{}' AS jsonb), CAST('{}' AS jsonb), CAST('{}' AS jsonb)
                WHERE NOT EXISTS (SELECT 1 FROM models WHERE name = CAST(:name AS text));
            """),
        {"name": name, "provider": provider, "version": version, "is_active": is_active},
      )

    await db.commit()
    logger.info(f"Inserted {len(all_models)} missing models")

  async def _create_marketplace_entries(self, db: AsyncSession) -> None:
    """Create marketplace entries for ALL existing LLM models."""
    logger.info("Creating marketplace entries for existing LLM models...")

    await db.execute(
      text("""
            INSERT INTO marketplace_assistants (
                assistant_type,
                assistant_id,
                organization_id,
                is_published,
                pricing_type
            )
            SELECT
                'llm_model' as assistant_type,
                m.id as assistant_id,
                NULL as organization_id,  -- LLM models are system-wide
                m.is_active as is_published,
                'paid' as pricing_type  -- Most LLM models are paid
            FROM models m
            WHERE NOT EXISTS (
                SELECT 1 FROM marketplace_assistants ma
                WHERE ma.assistant_type = 'llm_model'
                AND ma.assistant_id = m.id
            );
        """)
    )

    await db.commit()
    logger.info("Created marketplace entries for existing LLM models")

  async def _assign_model_categories(self, db: AsyncSession) -> None:
    """Assign categories to existing models based on their names and capabilities."""
    logger.info("Assigning categories to models...")

    await db.execute(
      text("""
            WITH category_mappings AS (
                SELECT
                    m.id as model_id,
                    CASE
                        -- Reasoning models (DeepSeek reasoner, o1 series)
                        WHEN m.name ILIKE '%deepseek-reasoner%' OR m.name ILIKE '%deepseek-reason%' OR m.name ILIKE '%o1%' THEN
                            (SELECT id FROM llm_category WHERE name = 'reasoning-models')
                        -- Vision models (models with multimodal capabilities)
                        WHEN m.name ILIKE '%gpt-4o%' OR m.name ILIKE '%claude%' OR m.name ILIKE '%gemini%' OR m.name ILIKE '%grok-4%' THEN
                            (SELECT id FROM llm_category WHERE name = 'vision-models')
                        -- Code models (coding-focused models)
                        WHEN m.name ILIKE '%code%' OR m.name ILIKE '%codex%' OR m.name ILIKE '%grok-code%' THEN
                            (SELECT id FROM llm_category WHERE name = 'code-models')
                        -- Creative models (Claude models are good for creative tasks)
                        WHEN m.name ILIKE '%claude%' THEN
                            (SELECT id FROM llm_category WHERE name = 'creative-models')
                        -- Default to advanced language models
                        ELSE
                            (SELECT id FROM llm_category WHERE name = 'advanced-language-models')
                    END as category_id,
                    -- Set primary category based on model specialization
                    true as is_primary
                FROM models m
            )
            INSERT INTO llm_model_categories (model_id, category_id, is_primary)
            SELECT model_id, category_id, is_primary
            FROM category_mappings
            WHERE category_id IS NOT NULL
            ON CONFLICT (model_id, category_id) DO NOTHING;
        """)
    )

    # Add secondary categories for multi-capability models
    await db.execute(
      text("""
            INSERT INTO llm_model_categories (model_id, category_id, is_primary)
            SELECT
                m.id as model_id,
                (SELECT id FROM llm_category WHERE name = 'advanced-language-models') as category_id,
                false as is_primary
            FROM models m
            WHERE (m.name ILIKE '%gpt-4%' OR m.name ILIKE '%claude%' OR m.name ILIKE '%deepseek%' OR m.name ILIKE '%gemini%' OR m.name ILIKE '%grok%')
            AND NOT EXISTS (
                SELECT 1 FROM llm_model_categories lmc
                WHERE lmc.model_id = m.id
                AND lmc.category_id = (SELECT id FROM llm_category WHERE name = 'advanced-language-models')
            )
            ON CONFLICT (model_id, category_id) DO NOTHING;
        """)
    )

    await db.commit()
    logger.info("Assigned categories to models")

  async def _add_model_tags(self, db: AsyncSession) -> None:
    """Add tags to existing models based on exact capabilities."""
    logger.info("Adding tags to models...")

    await db.execute(
      text("""
            UPDATE models
            SET tags = CASE
                -- OpenAI Models
                WHEN name ILIKE '%gpt-4o%' AND name NOT ILIKE '%mini%' THEN
                    ARRAY['multimodal', 'vision', 'function-calling', 'structured-outputs', 'streaming', 'balanced']
                WHEN name ILIKE '%gpt-4o-mini%' THEN
                    ARRAY['efficient', 'multimodal', 'vision', 'function-calling', 'structured-outputs', 'streaming', 'cost-effective', 'fast']
                WHEN name ILIKE '%o1%' AND name NOT ILIKE '%mini%' AND name NOT ILIKE '%preview%' THEN
                    ARRAY['reasoning', 'advanced-reasoning', 'high-intelligence', 'complex-problems']
                WHEN name ILIKE '%o1-mini%' THEN ARRAY['reasoning', 'efficient-reasoning', 'cost-effective', 'fast-reasoning']
                WHEN name ILIKE '%o1-preview%' THEN ARRAY['reasoning', 'preview', 'deprecated', 'advanced-reasoning']
                WHEN name ILIKE '%o3-mini%' THEN ARRAY['reasoning', 'next-gen', 'efficient', 'tbd-pricing']
                WHEN name ILIKE '%gpt-3.5-turbo%' THEN ARRAY['legacy', 'conversational', 'function-calling', 'json-output', 'cost-effective', 'fast']

                -- Anthropic Models
                WHEN name ILIKE '%claude-opus-4.1%' THEN
                    ARRAY['flagship', 'highest-intelligence', 'extended-thinking', 'multimodal', 'vision', 'priority-tier', 'multilingual']
                WHEN name ILIKE '%claude-opus-4%' AND name NOT ILIKE '%4.1%' THEN
                    ARRAY['flagship', 'high-intelligence', 'extended-thinking', 'multimodal', 'vision', 'priority-tier', 'multilingual']
                WHEN name ILIKE '%claude-sonnet-4%' THEN
                    ARRAY['high-performance', 'extended-thinking', 'multimodal', 'vision',
                          'priority-tier', 'multilingual', '1m-context-beta', 'balanced']
                WHEN name ILIKE '%claude-3-haiku%' THEN ARRAY['fast', 'compact', 'multimodal', 'vision', 'multilingual', 'targeted-performance']

                -- DeepSeek Models
                WHEN name ILIKE '%deepseek-chat%' THEN
                    ARRAY['conversational', 'json-output', 'function-calling', 'caching', 'efficient', 'non-thinking', 'time-based-pricing']
                WHEN name ILIKE '%deepseek-reasoner%' OR name ILIKE '%deepseek-reason%' THEN
                    ARRAY['reasoning', 'thinking-mode', 'advanced-reasoning', 'problem-solving', 'caching', 'time-based-pricing']

                -- xAI/Grok Models
                WHEN name ILIKE '%grok-4%' THEN ARRAY['multimodal', 'vision', 'function-calling', 'structured-outputs', 'reasoning', 'advanced']
                WHEN name ILIKE '%grok-code-fast%' THEN ARRAY['coding', 'fast', 'function-calling', 'structured-outputs', 'development-focused']
                WHEN name ILIKE '%grok-3%' AND name NOT ILIKE '%mini%' THEN
                    ARRAY['general-purpose', 'function-calling', 'structured-outputs', 'conversational']
                WHEN name ILIKE '%grok-3-mini%' THEN ARRAY['compact', 'efficient', 'function-calling', 'structured-outputs', 'cost-effective']

                -- Google/Gemini Models
                WHEN name ILIKE '%gemini-2.5-pro%' THEN
                    ARRAY['thinking', 'reasoning', 'multimodal', 'highest-intelligence', 'code-execution', 'search-grounding', 'batch-mode']
                WHEN name ILIKE '%gemini-2.5-flash%' AND name NOT ILIKE '%lite%' THEN
                    ARRAY['thinking', 'reasoning', 'multimodal', 'price-performance', 'high-volume', 'low-latency', 'agentic']
                WHEN name ILIKE '%gemini-2.5-flash-lite%' THEN ARRAY['cost-efficient', 'high-throughput', 'multimodal', 'real-time', 'low-latency']
                WHEN name ILIKE '%gemini-2.0-flash%' AND name NOT ILIKE '%lite%' THEN
                    ARRAY['next-gen', 'fast', 'native-tool-use', 'multimodal', 'streaming']
                WHEN name ILIKE '%gemini-2.0-flash-lite%' THEN ARRAY['cost-efficient', 'low-latency', 'multimodal', 'audio-generation', 'fast']
                WHEN name ILIKE '%gemini-1.5-flash%' AND name NOT ILIKE '%8b%' THEN
                    ARRAY['deprecated', 'fast', 'versatile', 'multimodal', 'tuning-supported']
                WHEN name ILIKE '%gemini-1.5-flash-8b%' THEN ARRAY['deprecated', 'small', 'low-intelligence', 'high-volume', 'multimodal']
                WHEN name ILIKE '%gemini-1.5-pro%' THEN ARRAY['deprecated', 'reasoning', 'large-context', 'multimodal', 'complex-tasks']

                ELSE ARRAY['general-purpose']
            END
            WHERE tags IS NULL OR array_length(tags, 1) IS NULL;
        """)
    )

    await db.commit()
    logger.info("Added tags to models")


def main():
  """Entry point for backward compatibility."""
  script = PopulateMarketplaceDataScript()
  script.main()


if __name__ == "__main__":
  script = PopulateMarketplaceDataScript()
  script.run_cli()
