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
    "mcp_servers",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("mcp_url", sa.String(), nullable=False),
    sa.Column("auth_config_ids", postgresql.JSONB, nullable=True),
    sa.Column("toolkits", postgresql.JSONB, nullable=True),
    sa.Column("allowed_tools", postgresql.JSONB, nullable=True),
    sa.Column("server_instance_count", sa.Integer(), nullable=False, default=0),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
  )

  op.create_table(
    "mcp_tools",
    sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("slug", sa.String(), nullable=False),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("toolkit", sa.String(), nullable=False),
  )

  op.create_table(
    "mcp_sessions",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("instance_id", sa.String(), nullable=False),
    sa.Column("mcp_server_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("mcp_servers.id"), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
  )


def downgrade():
  op.drop_table("mcp_sessions")
  op.drop_table("mcp_servers")
  op.drop_table("mcp_tools")
