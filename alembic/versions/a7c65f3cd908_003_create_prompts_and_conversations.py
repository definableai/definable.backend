"""003_create_prompts_and_conversations

Revision ID: a7c65f3cd908
Revises: 9f6b02ca3265
Create Date: 2025-01-29 09:18:59.049322

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c65f3cd908"
down_revision: Union[str, None] = "9f6b02ca3265"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create prompts table
  op.create_table(
    "prompts",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("category", sa.String(100), nullable=False),
    sa.Column("is_public", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("created_by_user", postgresql.UUID(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["created_by_user"], ["users.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("organization_id", "name"),
  )
  op.create_index("ix_prompts_org_name", "prompts", ["organization_id", "name"], unique=True)
  op.create_index("ix_prompts_category", "prompts", ["category"])
  op.create_index("ix_prompts_created_by_user", "prompts", ["created_by_user"])

  # Create conversations table
  op.create_table(
    "conversations",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("agent_id", postgresql.UUID(), nullable=False),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("is_archived", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
  )
  op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
  op.create_index("ix_conversations_agent_id", "conversations", ["agent_id"])
  op.create_index("ix_conversations_created_at", "conversations", ["created_at"])

  # Create updated_at triggers
  for table in ["prompts", "conversations"]:
    op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE PROCEDURE update_updated_at_column();
        """)


def downgrade():
  for table in ["prompts", "conversations"]:
    op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
  op.drop_table("conversations")
  op.drop_table("prompts")
