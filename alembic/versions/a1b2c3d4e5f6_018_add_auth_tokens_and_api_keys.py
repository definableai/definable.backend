"""018_add_auth_tokens_and_api_keys

Revision ID: a1b2c3d4e5f6
Revises: 65b320fc1dfa
Create Date: 2024-01-15

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "65b320fc1dfa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Skip adding auth_token to users - we'll use unified API keys table instead

  # Make model_id optional in agents table
  op.alter_column("agents", "model_id", nullable=True)

  # Create api_keys table
  op.create_table(
    "api_keys",
    sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column("token_type", sa.String(20), nullable=False, server_default="api"),
    sa.Column("api_key_token", sa.String(500), nullable=False),
    sa.Column("api_key_hash", sa.String(255), nullable=False),
    sa.Column("name", sa.String(100), nullable=True),
    sa.Column("permissions", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    sa.Column("expires_at", sa.DateTime(), nullable=True),
    sa.Column("last_used_at", sa.DateTime(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
  )

  # Create indexes for api_keys table
  op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
  op.create_index("ix_api_keys_agent_id", "api_keys", ["agent_id"])
  op.create_index("ix_api_keys_token_type", "api_keys", ["token_type"])
  op.create_index("ix_api_keys_api_key_token", "api_keys", ["api_key_token"], unique=True)
  op.create_index("ix_api_keys_api_key_hash", "api_keys", ["api_key_hash"], unique=True)
  op.create_index("idx_api_keys_user_active", "api_keys", ["user_id", "is_active"])
  op.create_index("idx_api_keys_agent_active", "api_keys", ["agent_id", "is_active"])
  op.create_index("idx_api_keys_hash_active", "api_keys", ["api_key_hash", "is_active"])
  op.create_index("idx_api_keys_user_type", "api_keys", ["user_id", "token_type"])


def downgrade() -> None:
  # Drop api_keys table (this will automatically drop its indexes)
  op.drop_table("api_keys")

  # Clean up agents with NULL model_id before making column NOT NULL
  op.execute("DELETE FROM agents WHERE model_id IS NULL")

  # Make model_id required again in agents table
  op.alter_column("agents", "model_id", nullable=False)

  # No need to drop auth_token columns - they were never added in unified approach
