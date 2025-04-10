"""009_create_tools

Revision ID: ba6775663b8a
Revises: a8f51c3e9d12
Create Date: 2025-03-27 21:20:27.801368

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ba6775663b8a"
down_revision: Union[str, None] = "a8f51c3e9d12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create tools_category table first since it's referenced by tools
  op.create_table(
    "tools_category",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name"),
  )

  # Create tools table
  op.create_table(
    "tools",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("category_id", postgresql.UUID(), nullable=False),
    sa.Column("logo_url", sa.String(255), nullable=True),
    sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    sa.Column("version", sa.String(50), nullable=False),
    sa.Column("is_public", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    sa.Column("is_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    sa.Column("inputs", postgresql.JSONB(), nullable=False),
    sa.Column("outputs", postgresql.JSONB(), nullable=False),
    sa.Column("configuration", postgresql.JSONB(), nullable=True),
    sa.Column("settings", postgresql.JSONB(), nullable=False),
    sa.Column("generated_code", sa.Text(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name", "version"),
    sa.ForeignKeyConstraint(["category_id"], ["tools_category.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
  )

  # Create agent_tools table
  op.create_table(
    "agent_tools",
    sa.Column("agent_id", postgresql.UUID(), nullable=False),
    sa.Column("tool_id", postgresql.UUID(), nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    sa.Column("added_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.PrimaryKeyConstraint("agent_id", "tool_id"),
    sa.ForeignKeyConstraint(["tool_id"], ["tools.id"]),
    sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
  )

  # Create indexes
  op.create_index("idx_tools_category_id", "tools", ["category_id"])
  op.create_index("idx_tools_is_active", "tools", ["is_active"])

  for table in ["tools", "tools_category"]:
    op.execute(f"""
      CREATE TRIGGER update_{table}_updated_at
          BEFORE UPDATE ON {table}
          FOR EACH ROW
          EXECUTE PROCEDURE update_updated_at_column();
      """)


def downgrade() -> None:
  # Drop tables in reverse order to handle dependencies
  op.drop_table("agent_tools")
  op.drop_table("tools")
  op.drop_table("tools_category")
