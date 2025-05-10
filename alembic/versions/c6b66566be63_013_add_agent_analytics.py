"""013_add_agent_analytics

Revision ID: c6b66566be63
Revises: aa9d2cf45c4d
Create Date: 2025-05-09 20:21:14.982566

"""

from typing import Sequence, Union

from sqlalchemy import Column, ForeignKeyConstraint, PrimaryKeyConstraint, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6b66566be63"
down_revision: Union[str, None] = "aa9d2cf45c4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create agent_analytics table
  op.create_table(
    "agent_analytics",
    Column("id", UUID(), server_default=text("gen_random_uuid()"), nullable=False, comment="Unique identifier for the analytics record"),
    Column("agent_id", UUID(), nullable=False, comment="Foreign key referencing the agent"),
    Column("session_id", UUID(), nullable=False, unique=True, comment="Foreign key referencing the chat session (messages.id)"),
    Column("user_id", UUID(), nullable=True, comment="Foreign key referencing the user"),
    Column("org_id", UUID(), nullable=True, comment="Foreign key referencing the organization"),
    Column("memory", JSONB(), nullable=True, comment="Memory data associated with the agent"),
    Column("agent_data", JSONB(), nullable=True, comment="Data specific to the agent"),
    Column("session_data", JSONB(), nullable=True, comment="Data specific to the session"),
    Column(
      "created_at",
      TIMESTAMP(timezone=True),
      server_default=text("CURRENT_TIMESTAMP"),
      nullable=False,
      comment="Timestamp when the record was created",
      onupdate=text("CURRENT_TIMESTAMP"),
    ),
    Column(
      "updated_at",
      TIMESTAMP(timezone=True),
      server_default=text("CURRENT_TIMESTAMP"),
      nullable=False,
      comment="Timestamp when the record was last updated",
      onupdate=text("CURRENT_TIMESTAMP"),
    ),
    PrimaryKeyConstraint("id"),
    ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
    ForeignKeyConstraint(["session_id"], ["messages.id"], ondelete="SET NULL"),
    ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
    ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="SET NULL"),
    comment="Table for storing analytics data related to agent interactions",
  )

  # Create indexes for frequently queried fields
  op.create_index("ix_agent_analytics_agent_id", "agent_analytics", ["agent_id"])
  op.create_index("ix_agent_analytics_session_id", "agent_analytics", ["session_id"])
  op.create_index("ix_agent_analytics_user_id", "agent_analytics", ["user_id"])
  op.create_index("ix_agent_analytics_org_id", "agent_analytics", ["org_id"])

  # Create trigger to update the updated_at column
  op.execute("""
        CREATE TRIGGER update_agent_analytics_updated_at
            BEFORE UPDATE ON agent_analytics
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column();
    """)
  # Insert the "analytics_read" permission
  op.execute("""
      INSERT INTO permissions (id, name, description, resource, action, created_at)
      VALUES (
          gen_random_uuid(),
          'analytics_read',
          'Read analytics data',
          'analytics',
          'read',
          CURRENT_TIMESTAMP
      );
  """)

  # Assign the permission to the "owner" and "admin" roles
  op.execute("""
      INSERT INTO role_permissions (id, role_id, permission_id, created_at)
      SELECT
          gen_random_uuid(),
          r.id,
          p.id,
          CURRENT_TIMESTAMP
      FROM
          roles r
      CROSS JOIN
          permissions p
      WHERE
          r.name IN ('owner', 'admin')
          AND p.name = 'analytics_read';
  """)


def downgrade() -> None:
  # Drop the trigger first
  op.execute("DROP TRIGGER IF EXISTS update_agent_analytics_updated_at ON agent_analytics")

  # Drop indexes
  op.drop_index("ix_agent_analytics_org_id", table_name="agent_analytics")
  op.drop_index("ix_agent_analytics_user_id", table_name="agent_analytics")
  op.drop_index("ix_agent_analytics_session_id", table_name="agent_analytics")
  op.drop_index("ix_agent_analytics_agent_id", table_name="agent_analytics")

  # Drop the table
  op.drop_table("agent_analytics")

  # Remove the "analytics_read" permission and its role assignments
  op.execute("""
      DELETE FROM role_permissions
      WHERE permission_id IN (
          SELECT id FROM permissions WHERE name = 'analytics_read'
      );
  """)
  op.execute("""
      DELETE FROM permissions
      WHERE name = 'analytics_read';
  """)
