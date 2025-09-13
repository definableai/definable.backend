"""024_add_marketplace_tables

Revision ID: bf55e877b87c
Revises: 94d74fb48263
Create Date: 2025-09-08 17:38:42.101549

"""

from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "bf55e877b87c"
down_revision: Union[str, None] = "94d74fb48263"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create llm_category table
  op.create_table(
    "llm_category",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(length=100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("display_order", sa.Integer(), server_default="0", nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name", name="uq_llm_category_name"),
  )

  # Create llm_model_categories junction table (many-to-many)
  op.create_table(
    "llm_model_categories",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("model_id", postgresql.UUID(), nullable=False),
    sa.Column("category_id", postgresql.UUID(), nullable=False),
    sa.Column("is_primary", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["category_id"], ["llm_category.id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("model_id", "category_id", name="uq_model_category"),
  )

  # Create marketplace_assistants table
  op.create_table(
    "marketplace_assistants",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("assistant_type", sa.String(length=20), nullable=False),
    sa.Column("assistant_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=True),
    sa.Column("is_published", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("pricing_type", sa.String(length=20), server_default="'free'", nullable=False),
    sa.Column("rating_avg", sa.Numeric(precision=3, scale=2), server_default="0", nullable=False),
    sa.Column("rating_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("conversation_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("assistant_type", "assistant_id", name="uq_marketplace_assistant_type_id"),
    sa.CheckConstraint("assistant_type IN ('llm_model', 'agent')", name="ck_assistant_type"),
    sa.CheckConstraint("pricing_type IN ('free', 'paid')", name="ck_pricing_type"),
    sa.CheckConstraint("rating_avg >= 0 AND rating_avg <= 5", name="ck_rating_avg_range"),
    sa.CheckConstraint("rating_count >= 0", name="ck_rating_count_positive"),
    sa.CheckConstraint("conversation_count >= 0", name="ck_conversation_count_positive"),
  )

  # Create indexes for llm_category
  op.create_index("idx_llm_category_active", "llm_category", ["is_active", "display_order"])
  op.create_index("idx_llm_category_name", "llm_category", ["name"])

  # Create indexes for llm_model_categories
  op.create_index("idx_llm_model_categories_model", "llm_model_categories", ["model_id"])
  op.create_index("idx_llm_model_categories_category", "llm_model_categories", ["category_id"])
  op.create_index("idx_llm_model_categories_primary", "llm_model_categories", ["model_id", "is_primary"])

  # Create indexes for marketplace_assistants
  op.create_index(
    "idx_marketplace_assistants_published",
    "marketplace_assistants",
    ["is_published", "assistant_type"],
    postgresql_where=sa.text("is_published = true"),
  )
  op.create_index(
    "idx_marketplace_assistants_rating",
    "marketplace_assistants",
    [sa.desc("rating_avg"), sa.desc("conversation_count")],
    postgresql_where=sa.text("is_published = true"),
  )

  # Create marketplace_reviews table
  op.create_table(
    "marketplace_reviews",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("marketplace_assistant_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("rating", sa.Integer(), nullable=False),
    sa.Column("title", sa.String(length=200), nullable=True),
    sa.Column("content", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(["marketplace_assistant_id"], ["marketplace_assistants.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id"),
    sa.CheckConstraint("rating >= 1 AND rating <= 5", name="ck_rating_range"),
  )

  # Create indexes for marketplace_reviews
  op.create_index("idx_marketplace_reviews_assistant", "marketplace_reviews", ["marketplace_assistant_id", "rating"])

  # Create marketplace_usage table
  op.create_table(
    "marketplace_usage",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("marketplace_assistant_id", postgresql.UUID(), nullable=False),
    sa.Column("usage_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(["marketplace_assistant_id"], ["marketplace_assistants.id"], ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("marketplace_assistant_id", name="uq_marketplace_usage_assistant"),
    sa.CheckConstraint("usage_count >= 0", name="ck_usage_count_positive"),
  )

  # Create indexes for marketplace_usage
  op.create_index("idx_marketplace_usage_assistant", "marketplace_usage", ["marketplace_assistant_id"])
  op.create_index("idx_marketplace_usage_count", "marketplace_usage", [sa.desc("usage_count")])

  # Add tags columns to existing tables
  op.add_column("models", sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True))
  op.add_column("agents", sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True))

  # Seed LLM categories
  op.execute("""
    INSERT INTO llm_category (name, description, display_order, is_active) VALUES
    ('advanced-language-models', 'Advanced Language Models for general conversation and text generation', 1, true),
    ('reasoning-models', 'Models specialized in complex reasoning and problem-solving', 2, true),
    ('vision-models', 'Models with vision and multimodal capabilities', 3, true),
    ('code-models', 'Models specialized in code generation and programming', 4, true),
    ('research-models', 'Models optimized for research and analysis tasks', 5, true),
    ('creative-models', 'Models focused on creative writing and content generation', 6, true);
  """)

  # Create marketplace permissions
  marketplace_permissions = [
    ("marketplace_read", "View marketplace assistants and reviews", "marketplace", "read"),
    ("marketplace_write", "Create reviews and manage marketplace content", "marketplace", "write"),
    ("marketplace_delete", "Delete marketplace reviews and content", "marketplace", "delete"),
  ]

  permission_ids = {}
  for name, desc, resource, action in marketplace_permissions:
    perm_id = str(uuid4())
    op.execute(f"""
      INSERT INTO permissions (id, name, description, resource, action, created_at)
      VALUES ('{perm_id}', '{name}', '{desc}', '{resource}', '{action}', CURRENT_TIMESTAMP)
    """)
    permission_ids[name] = perm_id

  # Assign marketplace permissions to existing roles
  # Get existing role IDs
  conn = op.get_bind()
  result = conn.execute(sa.text("SELECT id, name FROM roles WHERE is_system_role = true"))
  roles = {row[1]: row[0] for row in result.fetchall()}

  # Assign permissions to roles
  for role_name, role_id in roles.items():
    if role_name == "owner":
      # Owner gets all marketplace permissions (read, write, delete)
      for perm_id in permission_ids.values():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)
    elif role_name == "admin":
      # Admin gets all marketplace permissions (read, write, delete)
      for perm_id in permission_ids.values():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)
    elif role_name == "dev":
      # Dev gets only read and write permissions (no delete)
      for perm_name, perm_id in permission_ids.items():
        if "delete" not in perm_name:
          op.execute(f"""
            INSERT INTO role_permissions (id, role_id, permission_id, created_at)
            VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
          """)

  # Create database triggers for automatic cache updates

  # Trigger function to update rating cache when reviews are added/updated/deleted
  op.execute("""
    CREATE OR REPLACE FUNCTION update_marketplace_rating_cache()
    RETURNS TRIGGER AS $$
    DECLARE
      new_rating_avg DECIMAL(3,2);
      new_rating_count INTEGER;
      target_assistant_id UUID;
    BEGIN
      -- Get the assistant ID from either NEW or OLD record
      target_assistant_id := COALESCE(NEW.marketplace_assistant_id, OLD.marketplace_assistant_id);

      -- Calculate new rating statistics
      SELECT
        COALESCE(AVG(rating)::DECIMAL(3,2), 0),
        COALESCE(COUNT(*), 0)
      INTO new_rating_avg, new_rating_count
      FROM marketplace_reviews
      WHERE marketplace_assistant_id = target_assistant_id;

      -- Update the marketplace_assistants table
      UPDATE marketplace_assistants
      SET
        rating_avg = new_rating_avg,
        rating_count = new_rating_count,
        updated_at = CURRENT_TIMESTAMP
      WHERE id = target_assistant_id;

      RETURN COALESCE(NEW, OLD);
    END;
    $$ LANGUAGE plpgsql;
  """)

  # Create triggers for reviews
  op.execute("""
    CREATE TRIGGER trigger_update_rating_cache_insert
    AFTER INSERT ON marketplace_reviews
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_rating_cache();
  """)

  op.execute("""
    CREATE TRIGGER trigger_update_rating_cache_update
    AFTER UPDATE ON marketplace_reviews
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_rating_cache();
  """)

  op.execute("""
    CREATE TRIGGER trigger_update_rating_cache_delete
    AFTER DELETE ON marketplace_reviews
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_rating_cache();
  """)

  # Trigger function to update usage cache when usage is updated
  op.execute("""
    CREATE OR REPLACE FUNCTION update_marketplace_usage_cache()
    RETURNS TRIGGER AS $$
    DECLARE
      new_usage_count INTEGER;
      target_assistant_id UUID;
    BEGIN
      -- Get the assistant ID from either NEW or OLD record
      target_assistant_id := COALESCE(NEW.marketplace_assistant_id, OLD.marketplace_assistant_id);

      -- Get the usage count from marketplace_usage table
      SELECT COALESCE(usage_count, 0)
      INTO new_usage_count
      FROM marketplace_usage
      WHERE marketplace_assistant_id = target_assistant_id;

      -- Update the marketplace_assistants table
      UPDATE marketplace_assistants
      SET
        conversation_count = new_usage_count,
        updated_at = CURRENT_TIMESTAMP
      WHERE id = target_assistant_id;

      RETURN COALESCE(NEW, OLD);
    END;
    $$ LANGUAGE plpgsql;
  """)

  # Create triggers for usage
  op.execute("""
    CREATE TRIGGER trigger_update_usage_cache_insert
    AFTER INSERT ON marketplace_usage
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_usage_cache();
  """)

  op.execute("""
    CREATE TRIGGER trigger_update_usage_cache_update
    AFTER UPDATE ON marketplace_usage
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_usage_cache();
  """)

  op.execute("""
    CREATE TRIGGER trigger_update_usage_cache_delete
    AFTER DELETE ON marketplace_usage
    FOR EACH ROW EXECUTE FUNCTION update_marketplace_usage_cache();
  """)

  # LLM Sync Trigger - Auto-create and sync marketplace entries for LLM models
  op.execute("""
    CREATE OR REPLACE FUNCTION sync_llm_marketplace()
    RETURNS TRIGGER AS $$
    BEGIN
      IF TG_OP = 'INSERT' THEN
        -- Auto-create marketplace entry for new LLM models
        INSERT INTO marketplace_assistants (
          assistant_type,
          assistant_id,
          organization_id,
          is_published,
          pricing_type
        ) VALUES (
          'llm_model',
          NEW.id,
          NULL,  -- LLM models are system-wide
          NEW.is_active,
          'paid'  -- Most LLM models are paid
        ) ON CONFLICT (assistant_type, assistant_id) DO NOTHING;

        -- Auto-assign default category (advanced-language-models) for new models
        INSERT INTO llm_model_categories (model_id, category_id, is_primary)
        SELECT
          NEW.id,
          (SELECT id FROM llm_category WHERE name = 'advanced-language-models'),
          true
        WHERE EXISTS (SELECT 1 FROM llm_category WHERE name = 'advanced-language-models')
        ON CONFLICT (model_id, category_id) DO NOTHING;

        RETURN NEW;
      ELSIF TG_OP = 'UPDATE' AND OLD.is_active != NEW.is_active THEN
        -- Sync is_published when is_active changes
        UPDATE marketplace_assistants
        SET
          is_published = NEW.is_active,
          updated_at = CURRENT_TIMESTAMP
        WHERE assistant_type = 'llm_model' AND assistant_id = NEW.id;

        RETURN NEW;
      END IF;

      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
  """)

  # Create LLM sync triggers
  op.execute("""
    CREATE TRIGGER trigger_sync_llm_marketplace_insert
    AFTER INSERT ON models
    FOR EACH ROW EXECUTE FUNCTION sync_llm_marketplace();
  """)

  op.execute("""
    CREATE TRIGGER trigger_sync_llm_marketplace_update
    AFTER UPDATE OF is_active ON models
    FOR EACH ROW EXECUTE FUNCTION sync_llm_marketplace();
  """)

  # Auto-usage tracking trigger
  # Track usage when messages are created with agents or models
  op.execute("""
    CREATE OR REPLACE FUNCTION auto_track_marketplace_usage()
    RETURNS TRIGGER AS $$
    DECLARE
      marketplace_assistant_uuid UUID;
    BEGIN
      -- Check if this message has an agent_id
      IF NEW.agent_id IS NOT NULL THEN
        -- Find the marketplace assistant entry for this agent
        SELECT ma.id INTO marketplace_assistant_uuid
        FROM marketplace_assistants ma
        WHERE ma.assistant_type = 'agent'
        AND ma.assistant_id = NEW.agent_id
        AND ma.is_published = true;

        -- If found, increment usage count
        IF marketplace_assistant_uuid IS NOT NULL THEN
          INSERT INTO marketplace_usage (marketplace_assistant_id, usage_count)
          VALUES (marketplace_assistant_uuid, 1)
          ON CONFLICT (marketplace_assistant_id)
          DO UPDATE SET
            usage_count = marketplace_usage.usage_count + 1,
            updated_at = CURRENT_TIMESTAMP;
        END IF;

      -- Check if this message has a model_id (LLM)
      ELSIF NEW.model_id IS NOT NULL THEN
        -- Find the marketplace assistant entry for this LLM model
        SELECT ma.id INTO marketplace_assistant_uuid
        FROM marketplace_assistants ma
        WHERE ma.assistant_type = 'llm_model'
        AND ma.assistant_id = NEW.model_id
        AND ma.is_published = true;

        -- If found, increment usage count
        IF marketplace_assistant_uuid IS NOT NULL THEN
          INSERT INTO marketplace_usage (marketplace_assistant_id, usage_count)
          VALUES (marketplace_assistant_uuid, 1)
          ON CONFLICT (marketplace_assistant_id)
          DO UPDATE SET
            usage_count = marketplace_usage.usage_count + 1,
            updated_at = CURRENT_TIMESTAMP;
        END IF;
      END IF;

      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
  """)

  # Create trigger for auto-tracking usage when messages are created
  op.execute("""
    CREATE TRIGGER trigger_auto_track_marketplace_usage
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION auto_track_marketplace_usage();
  """)


# NOTE: Data population is handled by separate scripts:
# 1. Run: python src/scripts/populate_marketplace_data.py
# 2. Run: python src/scripts/update_model_specifications.py


def downgrade() -> None:
  # Drop triggers first
  op.execute("DROP TRIGGER IF EXISTS trigger_auto_track_marketplace_usage ON messages;")
  op.execute("DROP TRIGGER IF EXISTS trigger_sync_llm_marketplace_update ON models;")
  op.execute("DROP TRIGGER IF EXISTS trigger_sync_llm_marketplace_insert ON models;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_usage_cache_delete ON marketplace_usage;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_usage_cache_update ON marketplace_usage;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_usage_cache_insert ON marketplace_usage;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_rating_cache_delete ON marketplace_reviews;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_rating_cache_update ON marketplace_reviews;")
  op.execute("DROP TRIGGER IF EXISTS trigger_update_rating_cache_insert ON marketplace_reviews;")

  # Remove marketplace permissions
  # First remove role_permissions for marketplace
  conn = op.get_bind()
  result = conn.execute(sa.text("SELECT id FROM permissions WHERE resource = 'marketplace'"))
  marketplace_perm_ids = [row[0] for row in result.fetchall()]

  for perm_id in marketplace_perm_ids:
    op.execute(f"DELETE FROM role_permissions WHERE permission_id = '{perm_id}'")

  # Then remove the permissions themselves
  op.execute("DELETE FROM permissions WHERE resource = 'marketplace'")

  # Drop trigger functions
  op.execute("DROP FUNCTION IF EXISTS auto_track_marketplace_usage();")
  op.execute("DROP FUNCTION IF EXISTS sync_llm_marketplace();")
  op.execute("DROP FUNCTION IF EXISTS update_marketplace_usage_cache();")
  op.execute("DROP FUNCTION IF EXISTS update_marketplace_rating_cache();")

  # Drop indexes (use IF EXISTS to avoid errors)
  op.execute("DROP INDEX IF EXISTS idx_marketplace_usage_count")
  op.execute("DROP INDEX IF EXISTS idx_marketplace_usage_assistant")
  op.execute("DROP INDEX IF EXISTS idx_marketplace_reviews_assistant")
  op.execute("DROP INDEX IF EXISTS idx_marketplace_assistants_rating")
  op.execute("DROP INDEX IF EXISTS idx_marketplace_assistants_published")
  op.execute("DROP INDEX IF EXISTS idx_llm_model_categories_primary")
  op.execute("DROP INDEX IF EXISTS idx_llm_model_categories_category")
  op.execute("DROP INDEX IF EXISTS idx_llm_model_categories_model")
  op.execute("DROP INDEX IF EXISTS idx_llm_category_name")
  op.execute("DROP INDEX IF EXISTS idx_llm_category_active")

  # Drop tables (in correct order due to foreign keys, use IF EXISTS to avoid errors)
  op.execute("DROP TABLE IF EXISTS marketplace_usage")
  op.execute("DROP TABLE IF EXISTS marketplace_reviews")
  op.execute("DROP TABLE IF EXISTS marketplace_assistants")
  op.execute("DROP TABLE IF EXISTS llm_model_categories")
  op.execute("DROP TABLE IF EXISTS llm_category")

  # Remove tags columns from existing tables (use IF EXISTS to avoid errors)
  op.execute("ALTER TABLE agents DROP COLUMN IF EXISTS tags")
  op.execute("ALTER TABLE models DROP COLUMN IF EXISTS tags")
