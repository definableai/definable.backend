"""001_create_base_tables

Revision ID: d9ffb8ed17ea
Revises:
Create Date: 2025-01-29 09:17:22.825991

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d9ffb8ed17ea"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create updated_at trigger function
  op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

  # Create organizations table
  op.create_table(
    "organizations",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("slug", sa.String(255), nullable=False),
    sa.Column("settings", postgresql.JSONB(), nullable=True),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("slug"),
  )
  op.create_index("ix_organizations_slug", "organizations", ["slug"], unique=True)

  # Create users table
  op.create_table(
    "users",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("email", sa.String(255), nullable=False),
    sa.Column("password", sa.String(255), nullable=False),
    sa.Column("first_name", sa.String(255), nullable=False),
    sa.Column("last_name", sa.String(255), nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("email"),
  )
  op.create_index("ix_users_email", "users", ["email"], unique=True)

  # Create roles table
  op.create_table(
    "roles",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("is_system_role", sa.Boolean(), server_default="false", nullable=False),
    sa.Column("hierarchy_level", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("organization_id", "name"),
  )
  op.create_index("ix_roles_organization_id_name", "roles", ["organization_id", "name"], unique=True)
  op.create_index("ix_roles_hierarchy_level", "roles", ["hierarchy_level"])

  # Create permissions table
  op.create_table(
    "permissions",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("resource", sa.String(100), nullable=False),
    sa.Column("action", sa.String(50), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("resource", "action"),
  )
  op.create_index("ix_permissions_resource_action", "permissions", ["resource", "action"], unique=True)

  # Create role_permissions table
  op.create_table(
    "role_permissions",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("role_id", postgresql.UUID(), nullable=False),
    sa.Column("permission_id", postgresql.UUID(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("role_id", "permission_id"),
  )
  op.create_index("ix_role_permissions_role_id_permission_id", "role_permissions", ["role_id", "permission_id"], unique=True)

  # Create organization_members table
  op.create_table(
    "organization_members",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("role_id", postgresql.UUID(), nullable=True),
    sa.Column("default_role_id", postgresql.UUID(), nullable=True),
    sa.Column("invited_by", postgresql.UUID(), nullable=True),
    sa.Column("status", sa.Enum("invited", "active", "suspended", name="member_status"), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["invited_by"], ["users.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("organization_id", "user_id"),
  )
  op.create_index("ix_organization_members_org_user", "organization_members", ["organization_id", "user_id"], unique=True)
  op.create_index("ix_organization_members_role_id", "organization_members", ["role_id"])
  op.create_index("ix_organization_members_invited_by", "organization_members", ["invited_by"])
  # Create updated_at triggers
  for table in ["organizations", "users", "roles", "organization_members"]:
    op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE PROCEDURE update_updated_at_column();
        """)


def downgrade():
  # Drop triggers
  for table in ["organizations", "users", "roles", "organization_members"]:
    op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

  op.execute("DROP FUNCTION IF EXISTS update_updated_at_column")
  op.drop_table("organization_members")
  op.execute("DROP TYPE member_status")
  op.drop_table("role_permissions")
  op.drop_table("permissions")
  op.drop_table("roles")
  op.drop_table("users")
  op.drop_table("organizations")
