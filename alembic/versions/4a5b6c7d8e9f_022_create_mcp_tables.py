"""
Revision ID: 4a5b6c7d8e9f
Revises: 3c4d5e6f7g8h
Create Date: 2025-08-29

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "4a5b6c7d8e9f"
down_revision = "3c4d5e6f7g8h"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade():
  op.create_table(
    "mcp_toolkits",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("slug", sa.String(), nullable=False),
    sa.Column("logo", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP")),
  )

  op.create_unique_constraint("uq_mcp_toolkits_name", "mcp_toolkits", ["name"])
  op.create_unique_constraint("uq_mcp_toolkits_slug", "mcp_toolkits", ["slug"])

  op.create_table(
    "mcp_servers",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("toolkits", postgresql.JSONB, nullable=True),
    sa.Column("auth_config_ids", postgresql.JSONB, nullable=True),
    sa.Column("auth_scheme", sa.String(), nullable=True),
    sa.Column("expected_input_fields", postgresql.JSONB, nullable=True),
    sa.Column("allowed_tools", postgresql.JSONB, nullable=True),
    sa.Column("server_instance_count", sa.Integer(), nullable=False, default=0),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP")),
  )

  op.create_table(
    "mcp_tools",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("slug", sa.String(), nullable=False),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("toolkit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("mcp_toolkits.id"), nullable=False),
  )

  op.create_table(
    "mcp_sessions",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("instance_id", sa.String(), nullable=False),
    sa.Column("mcp_server_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("mcp_servers.id"), nullable=False),
    sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
    sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
    sa.Column("connected_account_id", sa.String(), nullable=False),
    sa.Column("status", sa.String(), nullable=False, default="pending"),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP")),
  )


def downgrade():
  op.drop_table("mcp_sessions")
  op.drop_table("mcp_tools")
  op.drop_table("mcp_servers")
  op.drop_table("mcp_toolkits")
