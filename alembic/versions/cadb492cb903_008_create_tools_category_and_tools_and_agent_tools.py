"""008_create_tools_category_and_tools_and_agent_tools

Revision ID: cadb492cb903
Revises: fb7256243eb1
Create Date: 2025-03-30 13:21:54.656095

"""

from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cadb492cb903"
down_revision: Union[str, None] = "fb7256243eb1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  """Create the tools_category table."""
  op.create_table(
    "tools_category",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False, unique=True),
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    sa.Column("updated_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"), onupdate=sa.text("CURRENT_TIMESTAMP")),
    sa.PrimaryKeyConstraint("id"),
  )

  # Create index on the 'name' column
  op.create_index("ix_tools_category_name", "tools_category", ["name"], unique=True)

  # Create tools table
  op.create_table(
    "tools",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False, unique=True),
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("category_id", postgresql.UUID(), nullable=False),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    sa.Column("input", postgresql.JSONB, nullable=False),
    sa.Column("output", postgresql.JSONB, nullable=False),
    sa.Column("config", postgresql.JSONB, nullable=False),
    sa.Column("settings", postgresql.JSONB, nullable=False),
    sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    sa.Column("updated_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP"), onupdate=sa.text("CURRENT_TIMESTAMP")),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
    sa.ForeignKeyConstraint(["category_id"], ["tools_category.id"]),
    sa.PrimaryKeyConstraint("id"),
  )

  # Create indexes for tools table
  op.create_index("ix_tools_name", "tools", ["name"], unique=True)
  op.create_index("ix_tools_category_id", "tools", ["category_id"])
  op.create_index("ix_tools_is_active", "tools", ["is_active"])

  # Create agent_tools table
  op.create_table(
    "agent_tools",
    sa.Column("agent_id", postgresql.UUID(), nullable=False),
    sa.Column("tool_id", postgresql.UUID(), nullable=False),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    sa.Column("added_at", sa.TIMESTAMP, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    sa.ForeignKeyConstraint(
      ["agent_id"],
      ["agents.id"],
    ),
    sa.ForeignKeyConstraint(
      ["tool_id"],
      ["tools.id"],
    ),
    sa.PrimaryKeyConstraint("agent_id", "tool_id"),
  )

  llm_models = [
    ("gpt-4.5", "OpenAI", "gpt-4.5-preview", True),
    ("o3", "OpenAI", "o3", True),
    ("o3-mini", "OpenAI", "o3-mini", True),
    ("claude-3.7-sonnet", "Anthropic", "claude-3-7-sonnet", True),
  ]

  for name, provider, version, is_active in llm_models:
    op.execute(f"""
      INSERT INTO models (id, name, provider, version, is_active, config, created_at, updated_at)
      VALUES ('{str(uuid4())}', '{name}', '{provider}', '{version}', {is_active}, '{{}}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """)


def downgrade() -> None:
  """Drop the tools_category and agent_tools tables."""
  # First remove the LLM model records
  op.execute("""
    DELETE FROM models
    WHERE name IN ('gpt-4.5', 'o3', 'o3-mini', 'claude-3.7-sonnet')
    AND provider IN ('OpenAI', 'Anthropic')
  """)

  # Drop agent_tools first since it depends on both agents and tools
  op.drop_table("agent_tools")

  # Drop tools table and its indexes
  op.drop_index("ix_tools_is_active", table_name="tools")
  op.drop_index("ix_tools_category_id", table_name="tools")
  op.drop_index("ix_tools_name", table_name="tools")
  op.drop_table("tools")

  # Finally drop tools_category and its index
  op.drop_index("ix_tools_category_name", table_name="tools_category")
  op.drop_table("tools_category")
