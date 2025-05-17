"""013_prompts

Revision ID: 7364319dc095
Revises: aa9d2cf45c4d
Create Date: 2025-04-26 21:22:57.573727

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7364319dc095"
down_revision: Union[str, None] = "aa9d2cf45c4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create prompt_categories table
  op.create_table(
    "prompt_categories",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("icon_url", sa.String(255), nullable=True),
    sa.Column("display_order", sa.Integer(), server_default="0", nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), onupdate=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name"),
  )

  # Create prompts table
  op.create_table(
    "prompts",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("category_id", postgresql.UUID(), nullable=False),
    sa.Column("creator_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("title", sa.String(200), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("is_public", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("is_featured", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), onupdate=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["category_id"], ["prompt_categories.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["creator_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
  )

  # Create indexes for prompts
  op.create_index("idx_prompts_category_id", "prompts", ["category_id"])
  op.create_index("idx_prompts_creator_id", "prompts", ["creator_id"])
  op.create_index("idx_prompts_organization_id", "prompts", ["organization_id"])
  op.create_index("idx_prompts_is_public", "prompts", ["is_public"])
  op.create_index("idx_prompts_is_featured", "prompts", ["is_featured"])

  # Add trigram indexes for substring search
  op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
  op.execute("""
    CREATE INDEX idx_prompts_title_trgm ON prompts
    USING gin(title gin_trgm_ops)
  """)
  op.execute("""
    CREATE INDEX idx_prompts_content_trgm ON prompts
    USING gin(content gin_trgm_ops)
  """)

  # Add prompt_id column to messages table
  op.add_column("messages", sa.Column("prompt_id", postgresql.UUID(), nullable=True))
  op.create_foreign_key("fk_messages_prompt_id", "messages", "prompts", ["prompt_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
  # Drop the foreign key and column first
  op.drop_constraint("fk_messages_prompt_id", "messages", type_="foreignkey")
  op.drop_column("messages", "prompt_id")

  # Drop indexes first
  op.drop_index("idx_prompts_title_trgm", table_name="prompts")
  op.drop_index("idx_prompts_content_trgm", table_name="prompts")
  op.drop_index("idx_prompts_is_featured", table_name="prompts")
  op.drop_index("idx_prompts_is_public", table_name="prompts")
  op.drop_index("idx_prompts_organization_id", table_name="prompts")
  op.drop_index("idx_prompts_creator_id", table_name="prompts")
  op.drop_index("idx_prompts_category_id", table_name="prompts")

  # Drop tables
  op.drop_table("prompts")
  op.drop_table("prompt_categories")
