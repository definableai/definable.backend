"""004_add_script_executions_table

Revision ID: e09e0e697964
Revises: a7c65f3cd908
Create Date: 2025-01-29 09:19:48.079207

"""

import contextlib
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
  # Create the enum type with a unique name to avoid conflicts
  # Use try-except to handle the case where it might already exist
  with contextlib.suppress(Exception):
    op.execute(
      sa.text("""
      CREATE TYPE script_run_status_enum AS ENUM (
        'pending',
        'success',
        'failed',
        'rolled_back'
      )
    """)
    )

  op.create_table(
    "script_run_tracker",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("script_name", sa.String(255), nullable=False, unique=True),
    sa.Column(
      "status",
      # Use the new uniquely named enum
      postgresql.ENUM("pending", "success", "failed", "rolled_back", name="script_run_status_enum", create_type=False),
      nullable=False,
      server_default="pending",
    ),
    sa.Column("executed_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
  )


def downgrade():
  # Drop the table
  op.drop_table("script_run_tracker")

  # Try to drop the enum type if it exists
  with contextlib.suppress(Exception):
    op.execute(sa.text("DROP TYPE script_run_status_enum"))
