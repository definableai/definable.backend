#!/usr/bin/env python3
"""
Populate plans settings for service access
"""

import os
import sys

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from sqlalchemy.ext.asyncio import AsyncSession

from common.logger import log as logger
from scripts.core.base_script import BaseScript


class PopulatePlanSettingsScript(BaseScript):
  """
  Populate plans settings for service access
  """

  def __init__(self):
    super().__init__("populate_plan_settings")

  async def execute(self, db: AsyncSession) -> None:
    """
    Main script execution logic.
    Populate plan feature categories, features, and limits.
    """
    from sqlalchemy import text

    logger.info("Starting populate_plan_settings script execution...")

    # Insert categories
    logger.info("Inserting plan feature categories...")
    await db.execute(
      text("""
            INSERT INTO plan_feature_categories (name, display_name, description, sort_order)
            VALUES
            ('text_models', 'Text Models', 'AI language models for text generation and processing', 1),
            ('chat_features', 'Chat Features', 'Conversational AI features and limitations', 2),
            ('image_models', 'Image Models', 'AI models for image generation and editing', 3),
            ('image_editing', 'Image Editing', 'Advanced image editing and manipulation tools', 4)
            ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                sort_order = EXCLUDED.sort_order
        """)
    )

    await self._insert_text_model_features(db)
    await self._insert_chat_features(db)
    await self._insert_image_features(db)
    await self._insert_limits(db)

    await db.commit()
    logger.info("populate_plan_settings script execution completed.")

  async def _insert_text_model_features(self, db: AsyncSession) -> None:
    """Insert text model features using exact model names from ensure_model_props.py"""
    from sqlalchemy import text

    logger.info("Inserting text model features...")
    await db.execute(
      text("""
            INSERT INTO plan_features (category_id, name, display_name, feature_type, measurement_unit, description, sort_order)
            SELECT pfc.id, f.feature_name, f.feature_display_name, f.feature_type::feature_type_enum, f.measurement_unit, f.description, f.sort_order
            FROM plan_feature_categories pfc, (VALUES
                ('text_models', 'gpt-4.5-preview', 'GPT-4.5 Preview', 'model_access', 'requests', 'Access to GPT-4.5 Preview model (15 credits per 1K tokens)', 1),
                ('text_models', 'gpt-4.1', 'GPT-4.1', 'model_access', 'requests', 'Access to GPT-4.1 model (12 credits per 1K tokens)', 2),
                ('text_models', 'gpt-4o', 'GPT-4o', 'model_access', 'requests', 'Access to GPT-4o model (10 credits per 1K tokens)', 3),
                ('text_models', 'gpt-4o-mini', 'GPT-4o Mini', 'model_access', 'requests', 'Access to GPT-4o Mini model (2 credits per 1K tokens)', 4),
                ('text_models', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'model_access', 'requests', 'Access to GPT-3.5 Turbo model (1 credit per 1K tokens)', 5),
                ('text_models', 'o4-mini', 'O4 Mini', 'model_access', 'requests', 'Access to O4 Mini model (3 credits per 1K tokens)', 6),
                ('text_models', 'o3-mini', 'O3 Mini', 'model_access', 'requests', 'Access to O3 Mini model (2 credits per 1K tokens)', 7),
                ('text_models', 'o1-preview', 'O1 Preview', 'model_access', 'requests', 'Access to O1 Preview model (8 credits per 1K tokens)', 8),
                ('text_models', 'o1', 'O1', 'model_access', 'requests', 'Access to O1 model (6 credits per 1K tokens)', 9),
                ('text_models', 'claude-3-7-sonnet-latest', 'Claude 3.7 Sonnet', 'model_access', 'requests', 'Access to Claude 3.7 Sonnet model (15 credits per 1K tokens)', 10),
                ('text_models', 'claude-3-5-sonnet-latest', 'Claude 3.5 Sonnet', 'model_access', 'requests', 'Access to Claude 3.5 Sonnet model (12 credits per 1K tokens)', 11),
                ('text_models', 'claude-3-5-haiku-latest', 'Claude 3.5 Haiku', 'model_access', 'requests', 'Access to Claude 3.5 Haiku model (8 credits per 1K tokens)', 12),
                ('text_models', 'deepseek-chat', 'DeepSeek Chat', 'model_access', 'requests', 'Access to DeepSeek Chat model (4 credits per 1K tokens)', 13),
                ('text_models', 'deepseek-reason', 'DeepSeek Reason', 'model_access', 'requests', 'Access to DeepSeek Reason model (5 credits per 1K tokens)', 14)
            ) AS f(category_name, feature_name, feature_display_name, feature_type, measurement_unit, description, sort_order)
            WHERE pfc.name = f.category_name
            ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                sort_order = EXCLUDED.sort_order
        """)  # noqa: E501
    )

  async def _insert_chat_features(self, db: AsyncSession) -> None:
    """Insert chat features"""
    from sqlalchemy import text

    logger.info("Inserting chat features...")
    await db.execute(
      text("""
            INSERT INTO plan_features (category_id, name, display_name, feature_type, measurement_unit, description, sort_order)
            SELECT pfc.id, f.feature_name, f.feature_display_name, f.feature_type::feature_type_enum, f.measurement_unit, f.description, f.sort_order
            FROM plan_feature_categories pfc, (VALUES
                ('chat_features', 'daily_chat_limit', 'Daily Chat Limit', 'usage_limit', 'messages', 'Maximum messages per day', 1),
                ('chat_features', 'context_window', 'Context Window', 'usage_limit', 'tokens', 'Maximum tokens per conversation', 2),
                ('chat_features', 'web_search', 'Web Search', 'usage_limit', 'searches', 'Real-time web search access per day', 3)
            ) AS f(category_name, feature_name, feature_display_name, feature_type, measurement_unit, description, sort_order)
            WHERE pfc.name = f.category_name
            ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                sort_order = EXCLUDED.sort_order
        """)
    )

  async def _insert_image_features(self, db: AsyncSession) -> None:
    """Insert image model and editing features"""
    from sqlalchemy import text

    logger.info("Inserting image model features...")
    await db.execute(
      text("""
            INSERT INTO plan_features (category_id, name, display_name, feature_type, measurement_unit, description, sort_order)
            SELECT pfc.id, f.feature_name, f.feature_display_name, f.feature_type::feature_type_enum, f.measurement_unit, f.description, f.sort_order
            FROM plan_feature_categories pfc, (VALUES
                ('image_models', 'dalle_3', 'DALL-E 3', 'model_access', 'images', 'Access to DALL-E 3 model (25 credits per image)', 1),
                ('image_models', 'dalle_2', 'DALL-E 2', 'model_access', 'images', 'Access to DALL-E 2 model (10 credits per image)', 2),
                ('image_models', 'stable_diffusion_xl', 'Stable Diffusion XL', 'model_access', 'images', 'Access to Stable Diffusion XL (5 credits per image)', 3)
            ) AS f(category_name, feature_name, feature_display_name, feature_type, measurement_unit, description, sort_order)
            WHERE pfc.name = f.category_name
            ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                sort_order = EXCLUDED.sort_order
        """)  # noqa: E501
    )

    logger.info("Inserting image editing features...")
    await db.execute(
      text("""
            INSERT INTO plan_features (category_id, name, display_name, feature_type, measurement_unit, description, sort_order)
            SELECT pfc.id, f.feature_name, f.feature_display_name, f.feature_type::feature_type_enum, f.measurement_unit, f.description, f.sort_order
            FROM plan_feature_categories pfc, (VALUES
                ('image_editing', 'background_removal', 'Background Removal', 'usage_limit', 'operations', 'Automated background removal (5 credits per operation)', 1)
            ) AS f(category_name, feature_name, feature_display_name, feature_type, measurement_unit, description, sort_order)
            WHERE pfc.name = f.category_name
            ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                sort_order = EXCLUDED.sort_order
        """)  # noqa: E501
    )

  async def _insert_limits(self, db: AsyncSession) -> None:
    """Insert plan feature limits for text models and chat features"""
    from sqlalchemy import text

    logger.info("Inserting text model plan limits...")
    await db.execute(
      text("""
            INSERT INTO plan_feature_limits (billing_plan_id, feature_id, is_available, limit_value, limit_metadata, reset_period)
            SELECT bp.id, pf.id,
                   CASE
                 WHEN bp.name = 'starter' AND pf.name IN ('gpt-4o-mini', 'claude-3-5-haiku-latest', 'deepseek-chat', 'gpt-3.5-turbo') THEN true
                 WHEN bp.name = 'pro' AND pf.name IN ('gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest', 'deepseek-chat', 'deepseek-reason', 'gpt-3.5-turbo', 'o1', 'o3-mini') THEN true
                     WHEN bp.name = 'enterprise' THEN true
                     ELSE false
                   END as is_available,
                   CASE
                     -- Starter plan limits
                     WHEN bp.name = 'starter' AND pf.name = 'gpt-4o-mini' THEN 250
                     WHEN bp.name = 'starter' AND pf.name = 'claude-3-5-haiku-latest' THEN 62
                     WHEN bp.name = 'starter' AND pf.name = 'deepseek-chat' THEN 125
                     WHEN bp.name = 'starter' AND pf.name = 'gpt-3.5-turbo' THEN 500
                     -- Pro plan limits
                     WHEN bp.name = 'pro' AND pf.name = 'gpt-4o' THEN 500
                     WHEN bp.name = 'pro' AND pf.name = 'gpt-4o-mini' THEN 2500
                     WHEN bp.name = 'pro' AND pf.name = 'claude-3-5-sonnet-latest' THEN 416
                     WHEN bp.name = 'pro' AND pf.name = 'claude-3-5-haiku-latest' THEN 625
                     WHEN bp.name = 'pro' AND pf.name = 'deepseek-chat' THEN 1250
                     WHEN bp.name = 'pro' AND pf.name = 'deepseek-reason' THEN 1000
                     WHEN bp.name = 'pro' AND pf.name = 'gpt-3.5-turbo' THEN 5000
                     WHEN bp.name = 'pro' AND pf.name = 'o1' THEN 833
                     WHEN bp.name = 'pro' AND pf.name = 'o3-mini' THEN 2500
                     -- Enterprise plan limits
                     WHEN bp.name = 'enterprise' AND pf.name = 'gpt-4.5-preview' THEN 1333
                     WHEN bp.name = 'enterprise' AND pf.name = 'gpt-4.1' THEN 1666
                     WHEN bp.name = 'enterprise' AND pf.name = 'gpt-4o' THEN 2000
                     WHEN bp.name = 'enterprise' AND pf.name = 'gpt-4o-mini' THEN 10000
                     WHEN bp.name = 'enterprise' AND pf.name = 'claude-3-7-sonnet-latest' THEN 1333
                     WHEN bp.name = 'enterprise' AND pf.name = 'claude-3-5-sonnet-latest' THEN 1666
                     WHEN bp.name = 'enterprise' AND pf.name = 'claude-3-5-haiku-latest' THEN 2500
                     WHEN bp.name = 'enterprise' AND pf.name = 'deepseek-chat' THEN 5000
                     WHEN bp.name = 'enterprise' AND pf.name = 'deepseek-reason' THEN 4000
                     WHEN bp.name = 'enterprise' AND pf.name = 'gpt-3.5-turbo' THEN 20000
                     WHEN bp.name = 'enterprise' AND pf.name = 'o4-mini' THEN 6666
                     WHEN bp.name = 'enterprise' AND pf.name = 'o3-mini' THEN 10000
                     WHEN bp.name = 'enterprise' AND pf.name = 'o1-preview' THEN 2500
                     WHEN bp.name = 'enterprise' AND pf.name = 'o1' THEN 3333
                     ELSE NULL
                   END as limit_value,
                   ('{"credits_per_1k_tokens": ' ||
                     CASE pf.name
                       WHEN 'gpt-4.5-preview' THEN '15'
                       WHEN 'gpt-4.1' THEN '12'
                       WHEN 'gpt-4o' THEN '10'
                       WHEN 'gpt-4o-mini' THEN '2'
                       WHEN 'gpt-3.5-turbo' THEN '1'
                       WHEN 'o4-mini' THEN '3'
                       WHEN 'o3-mini' THEN '2'
                       WHEN 'o1-preview' THEN '8'
                       WHEN 'o1' THEN '6'
                       WHEN 'claude-3-7-sonnet-latest' THEN '15'
                       WHEN 'claude-3-5-sonnet-latest' THEN '12'
                       WHEN 'claude-3-5-haiku-latest' THEN '8'
                       WHEN 'deepseek-chat' THEN '4'
                       WHEN 'deepseek-reason' THEN '5'
                       ELSE '0'
                     END || '}')::jsonb as limit_metadata,
                   'monthly'::reset_period_enum as reset_period
            FROM billing_plans bp
            CROSS JOIN plan_features pf
            WHERE bp.cycle = 'MONTHLY'
              AND bp.name IN ('starter', 'pro', 'enterprise')
              AND pf.name IN ('gpt-4.5-preview', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini',
                              'gpt-3.5-turbo', 'o4-mini', 'o3-mini', 'o1-preview',
                              'o1', 'claude-3-7-sonnet-latest', 'claude-3-5-sonnet-latest',
                              'claude-3-5-haiku-latest', 'deepseek-chat', 'deepseek-reason')
            ON CONFLICT (billing_plan_id, feature_id) DO UPDATE SET
                is_available = EXCLUDED.is_available,
                limit_value = EXCLUDED.limit_value,
                limit_metadata = EXCLUDED.limit_metadata,
                reset_period = EXCLUDED.reset_period
        """)  # noqa: E501
    )

    logger.info("Inserting chat feature limits...")
    await db.execute(
      text("""
            INSERT INTO plan_feature_limits (billing_plan_id, feature_id, is_available, limit_value, reset_period)
            SELECT bp.id, pf.id, true,
                   CASE
                     WHEN bp.name = 'starter' AND pf.name = 'daily_chat_limit' THEN 30
                     WHEN bp.name = 'starter' AND pf.name = 'context_window' THEN 8000
                     WHEN bp.name = 'starter' AND pf.name = 'web_search' THEN 5
                     WHEN bp.name = 'pro' AND pf.name = 'daily_chat_limit' THEN 500
                     WHEN bp.name = 'pro' AND pf.name = 'context_window' THEN 128000
                     WHEN bp.name = 'pro' AND pf.name = 'web_search' THEN 100
                     WHEN bp.name = 'enterprise' AND pf.name = 'daily_chat_limit' THEN 2000
                     WHEN bp.name = 'enterprise' AND pf.name = 'context_window' THEN 200000
                     WHEN bp.name = 'enterprise' AND pf.name = 'web_search' THEN 500
                     ELSE NULL
                   END as limit_value,
                   CASE
                     WHEN pf.name IN ('daily_chat_limit', 'web_search') THEN 'daily'
                     WHEN pf.name = 'context_window' THEN 'never'
                     ELSE 'monthly'
                   END::reset_period_enum as reset_period
            FROM billing_plans bp
            CROSS JOIN plan_features pf
            WHERE bp.cycle = 'MONTHLY'
              AND bp.name IN ('starter', 'pro', 'enterprise')
              AND pf.name IN ('daily_chat_limit', 'context_window', 'web_search')
            ON CONFLICT (billing_plan_id, feature_id) DO UPDATE SET
                is_available = EXCLUDED.is_available,
                limit_value = EXCLUDED.limit_value,
                reset_period = EXCLUDED.reset_period
        """)
    )

  async def rollback(self, db: AsyncSession) -> None:
    """
    Rollback logic for the script.
    Remove all plan features and categories created by this script.
    """
    from sqlalchemy import text

    logger.info("Rolling back populate_plan_settings script...")

    await db.execute(text("DELETE FROM plan_feature_limits"))
    await db.execute(text("DELETE FROM plan_features"))
    await db.execute(text("DELETE FROM plan_feature_categories"))

    await db.commit()
    logger.info("populate_plan_settings script rollback completed.")

  async def verify(self, db: AsyncSession) -> bool:
    """
    Verify script execution was successful.
    Return True if everything is as expected, False otherwise.
    """
    from sqlalchemy import text

    logger.info("Verifying populate_plan_settings script execution...")

    # Check that categories were created
    result = await db.execute(text("SELECT COUNT(*) FROM plan_feature_categories"))
    category_count = result.scalar_one()
    if category_count < 4:
      logger.error(f"Expected at least 4 categories, found {category_count}")
      return False

    # Check that features were created
    result = await db.execute(text("SELECT COUNT(*) FROM plan_features"))
    feature_count = result.scalar_one()
    if feature_count < 10:
      logger.error(f"Expected at least 10 features, found {feature_count}")
      return False

    # Check that limits were created
    result = await db.execute(text("SELECT COUNT(*) FROM plan_feature_limits"))
    limit_count = result.scalar_one()
    if limit_count < 10:
      logger.error(f"Expected at least 10 feature limits, found {limit_count}")
      return False

    logger.info("Verification passed: all plan settings populated successfully")
    return True


def main():
  """Entry point for backward compatibility."""
  script = PopulatePlanSettingsScript()
  script.main()


if __name__ == "__main__":
  script = PopulatePlanSettingsScript()
  script.run_cli()
