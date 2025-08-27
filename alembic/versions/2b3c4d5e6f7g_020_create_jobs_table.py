"""020_create_jobs_table

Revision ID: 2b3c4d5e6f7g
Revises: 1a2b3c4d5e6f
Create Date: 2025-01-30 00:00:00.000000

"""

from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "2b3c4d5e6f7g"
down_revision: Union[str, None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create jobs table
  op.create_table(
    "jobs",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("created_by", postgresql.UUID(), nullable=False),
    sa.Column("status", sa.SmallInteger(), nullable=False, server_default="0"),
    sa.Column("message", sa.Text(), nullable=True),
    sa.Column("context", postgresql.JSONB(), nullable=True),
    sa.Column("parent_job_id", postgresql.UUID(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["parent_job_id"], ["jobs.id"], ondelete="SET NULL"),
  )

  # Create indexes for performance
  op.create_index("idx_jobs_created_by", "jobs", ["created_by"])
  op.create_index("idx_jobs_status", "jobs", ["status"])
  op.create_index("idx_jobs_created_at", "jobs", ["created_at"])
  op.create_index("idx_jobs_name_status", "jobs", ["name", "status"])
  op.create_index("idx_jobs_parent_job_id", "jobs", ["parent_job_id"])
  op.create_index("idx_jobs_context", "jobs", ["context"], postgresql_using="gin")

  # Create jobs permissions
  jobs_permissions = [
    ("jobs_read", "View jobs", "jobs", "read"),
    ("jobs_write", "Create/modify jobs", "jobs", "write"),
    ("jobs_delete", "Delete jobs", "jobs", "delete"),
  ]

  permission_ids = {}
  for name, desc, resource, action in jobs_permissions:
    perm_id = str(uuid4())
    op.execute(f"""
      INSERT INTO permissions (id, name, description, resource, action, created_at)
      VALUES ('{perm_id}', '{name}', '{desc}', '{resource}', '{action}', CURRENT_TIMESTAMP)
    """)
    permission_ids[name] = perm_id

  # Assign jobs permissions to existing roles
  # Get existing role IDs
  conn = op.get_bind()
  result = conn.execute(sa.text("SELECT id, name FROM roles WHERE is_system_role = true"))
  roles = {row[1]: row[0] for row in result.fetchall()}

  # Assign permissions to roles
  for role_name, role_id in roles.items():
    if role_name == "owner":
      # Owner gets all jobs permissions
      for perm_id in permission_ids.values():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)
    elif role_name == "admin":
      # Admin gets read and write permissions but not delete
      for perm_name, perm_id in permission_ids.items():
        if "delete" not in perm_name:
          op.execute(f"""
            INSERT INTO role_permissions (id, role_id, permission_id, created_at)
            VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
          """)
    elif role_name == "dev":
      # Dev gets read and write permissions for jobs
      for perm_name, perm_id in permission_ids.items():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)

  # Create updated_at trigger
  op.execute("""
        CREATE TRIGGER update_jobs_updated_at
        BEFORE UPDATE ON jobs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
  # Remove jobs permissions and role assignments
  op.execute("""
    DELETE FROM role_permissions
    WHERE permission_id IN (
      SELECT id FROM permissions
      WHERE resource = 'jobs'
    )
  """)

  op.execute("DELETE FROM permissions WHERE resource = 'jobs'")

  # Drop trigger
  op.execute("DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs")

  # Drop indexes
  op.drop_index("idx_jobs_context", "jobs")
  op.drop_index("idx_jobs_parent_job_id", "jobs")
  op.drop_index("idx_jobs_name_status", "jobs")
  op.drop_index("idx_jobs_created_at", "jobs")
  op.drop_index("idx_jobs_status", "jobs")
  op.drop_index("idx_jobs_created_by", "jobs")

  # Drop table
  op.drop_table("jobs")
