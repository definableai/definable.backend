"""008_create_invitations_table

Revision ID: a8f51c3e9d12
Revises: fb7256243eb1
Create Date: 2024-03-31 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a8f51c3e9d12"
down_revision: Union[str, None] = "fb7256243eb1"  # This should match the previous migration
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create invites table
  op.create_table(
    "invites",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("role_id", postgresql.UUID(), nullable=False),
    sa.Column("invitee_email", sa.String(255), nullable=False),
    sa.Column("invited_by", postgresql.UUID(), nullable=False),
    sa.Column("status", sa.SmallInteger(), server_default="0", nullable=False),
    sa.Column("expiry_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="CASCADE"),
  )

  # Create indexes
  op.create_index("ix_invites_organization_id", "invites", ["organization_id"])
  op.create_index("ix_invites_status", "invites", ["status"])
  op.create_index("ix_invites_org_email", "invites", ["organization_id", "invitee_email"])

  # Create updated_at trigger
  op.execute("""
        CREATE TRIGGER update_invites_updated_at
            BEFORE UPDATE ON invites
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column();
    """)


def downgrade() -> None:
  # Drop trigger
  op.execute("DROP TRIGGER IF EXISTS update_invites_updated_at ON invites")

  # Drop indexes
  op.execute("DROP INDEX IF EXISTS ix_invites_organization_id")
  op.execute("DROP INDEX IF EXISTS ix_invites_status")
  op.execute("DROP INDEX IF EXISTS ix_invites_org_email")

  # Drop table
  op.drop_table("invites")
