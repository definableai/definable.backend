"""019_add_agent_deployment_tables

Revision ID: 1a2b3c4d5e6f
Revises: a1b2c3d4e5f6
Create Date: 2025-01-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create agent_deployment_logs table
  op.create_table(
    "agent_deployment_logs",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("agent_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("api_key_id", postgresql.UUID(), nullable=False),
    sa.Column("deployment_id", sa.String(255), nullable=False),
    sa.Column("log_type", sa.String(20), nullable=False),
    sa.Column("log_level", sa.String(10), nullable=False),
    sa.Column("message", sa.Text(), nullable=False),
    sa.Column("log_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
    sa.Column("timestamp", sa.DateTime(), nullable=False),
    sa.Column("source", sa.String(100), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
  )

  # Create agent_deployment_traces table
  op.create_table(
    "agent_deployment_traces",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("agent_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("api_key_id", postgresql.UUID(), nullable=False),
    sa.Column("deployment_id", sa.String(255), nullable=False),
    sa.Column("trace_id", sa.String(255), nullable=False),
    sa.Column("span_id", sa.String(255), nullable=False),
    sa.Column("parent_span_id", sa.String(255), nullable=True),
    sa.Column("operation_name", sa.String(255), nullable=False),
    sa.Column("start_time", sa.DateTime(), nullable=False),
    sa.Column("end_time", sa.DateTime(), nullable=True),
    sa.Column("duration_ms", sa.Integer(), nullable=True),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("tags", postgresql.JSONB(), nullable=False, server_default="{}"),
    sa.Column("trace_logs", postgresql.JSONB(), nullable=False, server_default="{}"),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
  )

  # Create indexes for performance
  op.create_index("idx_agent_deployment_logs_agent_timestamp", "agent_deployment_logs", ["agent_id", "timestamp"])
  op.create_index("idx_agent_deployment_logs_deployment_id", "agent_deployment_logs", ["deployment_id"])
  op.create_index("idx_agent_deployment_logs_org_timestamp", "agent_deployment_logs", ["organization_id", "timestamp"])
  op.create_index("idx_agent_deployment_logs_log_type", "agent_deployment_logs", ["log_type"])
  op.create_index("idx_agent_deployment_logs_user_id", "agent_deployment_logs", ["user_id"])
  op.create_index("idx_agent_deployment_logs_api_key_id", "agent_deployment_logs", ["api_key_id"])

  op.create_index("idx_agent_deployment_traces_agent_trace", "agent_deployment_traces", ["agent_id", "trace_id"])
  op.create_index("idx_agent_deployment_traces_deployment_id", "agent_deployment_traces", ["deployment_id"])
  op.create_index("idx_agent_deployment_traces_span_hierarchy", "agent_deployment_traces", ["trace_id", "parent_span_id"])
  op.create_index("idx_agent_deployment_traces_org_time", "agent_deployment_traces", ["organization_id", "start_time"])
  op.create_index("idx_agent_deployment_traces_status", "agent_deployment_traces", ["status"])
  op.create_index("idx_agent_deployment_traces_user_id", "agent_deployment_traces", ["user_id"])
  op.create_index("idx_agent_deployment_traces_api_key_id", "agent_deployment_traces", ["api_key_id"])

  # Create updated_at trigger for logs table
  op.execute("""
        CREATE TRIGGER update_agent_deployment_logs_updated_at
        BEFORE UPDATE ON agent_deployment_logs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)

  # Create updated_at trigger for traces table
  op.execute("""
        CREATE TRIGGER update_agent_deployment_traces_updated_at
        BEFORE UPDATE ON agent_deployment_traces
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
  # Drop triggers
  op.execute("DROP TRIGGER IF EXISTS update_agent_deployment_traces_updated_at ON agent_deployment_traces")
  op.execute("DROP TRIGGER IF EXISTS update_agent_deployment_logs_updated_at ON agent_deployment_logs")

  # Drop indexes
  op.drop_index("idx_agent_deployment_traces_api_key_id", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_user_id", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_status", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_org_time", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_span_hierarchy", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_deployment_id", "agent_deployment_traces")
  op.drop_index("idx_agent_deployment_traces_agent_trace", "agent_deployment_traces")

  op.drop_index("idx_agent_deployment_logs_api_key_id", "agent_deployment_logs")
  op.drop_index("idx_agent_deployment_logs_user_id", "agent_deployment_logs")
  op.drop_index("idx_agent_deployment_logs_log_type", "agent_deployment_logs")
  op.drop_index("idx_agent_deployment_logs_org_timestamp", "agent_deployment_logs")
  op.drop_index("idx_agent_deployment_logs_deployment_id", "agent_deployment_logs")
  op.drop_index("idx_agent_deployment_logs_agent_timestamp", "agent_deployment_logs")

  # Drop tables
  op.drop_table("agent_deployment_traces")
  op.drop_table("agent_deployment_logs")
