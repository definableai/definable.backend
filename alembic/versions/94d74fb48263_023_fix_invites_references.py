"""023_fix_invites_references

Revision ID: 94d74fb48263
Revises: 3c4d5e6f7g8h
Create Date: 2025-09-03 01:25:18.630638

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "94d74fb48263"
down_revision: Union[str, None] = "4a5b6c7d8e9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Make stytch_id nullable to support pre-creation of invited users
  op.alter_column("users", "stytch_id", nullable=True)

  # Add invite_id column to organization_members table to track invitation used for org membership
  op.add_column("organization_members", sa.Column("invite_id", sa.UUID(), nullable=True))

  # Add deleted_by column to track who deleted the user from the organization
  op.add_column("organization_members", sa.Column("deleted_by", sa.UUID(), nullable=True))

  # Create foreign key constraints
  op.create_foreign_key("fk_org_members_invite_id", "organization_members", "invites", ["invite_id"], ["id"], ondelete="SET NULL")
  op.create_foreign_key("fk_org_members_deleted_by", "organization_members", "users", ["deleted_by"], ["id"], ondelete="SET NULL")

  # Add indexes for better query performance
  op.create_index("ix_org_members_invite_id", "organization_members", ["invite_id"])
  op.create_index("ix_org_members_deleted_by", "organization_members", ["deleted_by"])

  # Add "deleted" to member_status enum (only if it doesn't exist)
  op.execute(
    "DO $$ BEGIN "
    "IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'deleted' AND "
    "enumtypid = (SELECT oid FROM pg_type WHERE typname = 'member_status')) "
    "THEN ALTER TYPE member_status ADD VALUE 'deleted'; END IF; END $$"
  )

  # Drop the invited_by column from invites table since we track this in organization_members
  op.drop_column("invites", "invited_by")


def downgrade() -> None:
  # Re-add invited_by column to invites table
  op.add_column("invites", sa.Column("invited_by", sa.UUID(), nullable=True))

  # Drop indexes
  op.drop_index("ix_org_members_deleted_by", table_name="organization_members")
  op.drop_index("ix_org_members_invite_id", table_name="organization_members")

  # Drop foreign key constraints
  op.drop_constraint("fk_org_members_deleted_by", "organization_members", type_="foreignkey")
  op.drop_constraint("fk_org_members_invite_id", "organization_members", type_="foreignkey")

  # Drop columns
  op.drop_column("organization_members", "deleted_by")
  op.drop_column("organization_members", "invite_id")
