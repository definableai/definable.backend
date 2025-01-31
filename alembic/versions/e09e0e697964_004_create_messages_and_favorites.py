"""004_create_messages_and_favorites

Revision ID: e09e0e697964
Revises: a7c65f3cd908
Create Date: 2025-01-29 09:19:48.079207

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e09e0e697964"
down_revision: Union[str, None] = "a7c65f3cd908"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create messages table
  op.create_table(
    "messages",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("conversation_id", postgresql.UUID(), nullable=False),
    sa.Column("prompt_id", postgresql.UUID(), nullable=True),
    sa.Column("role", sa.Enum("user", "assistant", name="message_role"), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"], ondelete="SET NULL"),
  )
  op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
  op.create_index("ix_messages_prompt_id", "messages", ["prompt_id"])
  op.create_index("ix_messages_created_at", "messages", ["created_at"])

  # Create user_favorites table
  op.create_table(
    "user_favorites",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("entity_id", postgresql.UUID(), nullable=False),
    sa.Column("entity_type", sa.Enum("prompt", "agent", name="entity_type"), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("user_id", "entity_id", "entity_type"),
  )
  op.create_index("ix_user_favorites_user_entity", "user_favorites", ["user_id", "entity_id", "entity_type"], unique=True)


def downgrade():
  op.drop_table("user_favorites")
  op.execute("DROP TYPE entity_type")
  op.drop_table("messages")
  op.execute("DROP TYPE message_role")
