"""006_insert_default_roles_permissions

Revision ID: e080a53fbcbd
Revises: 0bb3780d56dd
Create Date: 2025-01-29 13:18:01.249505
"""

from typing import Sequence, Union
from uuid import uuid4

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e080a53fbcbd"
down_revision: Union[str, None] = "0bb3780d56dd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  permissions = [
    # Billing permissions
    ("billing_read", "Read billing information", "billing", "read"),
    ("billing_write", "Modify billing settings", "billing", "write"),
    ("billing_delete", "Delete billing records", "billing", "delete"),
    # User permissions
    ("users_read", "View users", "users", "read"),
    ("users_write", "Modify users", "users", "write"),
    ("users_delete", "Delete users", "users", "delete"),
    # Agent permissions
    ("agents_read", "View agents", "agents", "read"),
    ("agents_write", "Modify agents", "agents", "write"),
    ("agents_delete", "Delete agents", "agents", "delete"),
    # Prompt permissions
    ("prompts_read", "View prompts", "prompts", "read"),
    ("prompts_write", "Create/modify prompts", "prompts", "write"),
    ("prompts_delete", "Delete prompts", "prompts", "delete"),
    # Model permissions
    ("models_read", "View models", "models", "read"),
    ("models_write", "Configure models", "models", "write"),
    ("models_delete", "Delete models", "models", "delete"),
    # Chats permissions
    ("chats_read", "View chats", "chats", "read"),
    ("chats_write", "Create/modify chats", "chats", "write"),
    ("chats_delete", "Delete chats", "chats", "delete"),
    # Organization permissions
    ("org_read", "View organization", "org", "read"),
    ("org_write", "Modify organization", "org", "write"),
    ("org_delete", "Delete organization", "org", "delete"),
    # Role permissions
    ("role_read", "View roles", "roles", "read"),
    ("role_write", "Create/modify roles", "roles", "write"),
    ("role_delete", "Delete roles", "roles", "delete"),
    # Knowledge base permissions
    ("kb_read", "View knowledge bases", "kb", "read"),
    ("kb_write", "Create/modify knowledge bases", "kb", "write"),
    ("kb_delete", "Delete knowledge bases", "kb", "delete"),
  ]

  permission_ids = {}
  for name, desc, resource, action in permissions:
    perm_id = str(uuid4())
    op.execute(f"""
      INSERT INTO permissions (id, name, description, resource, action, created_at)
      VALUES ('{perm_id}', '{name}', '{desc}', '{resource}', '{action}', CURRENT_TIMESTAMP)
    """)
    permission_ids[name] = perm_id

  org_id = str(uuid4())
  op.execute(f"""
    INSERT INTO organizations (id, name, slug, created_at, updated_at)
    VALUES ('{org_id}', 'Default', 'default', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
  """)

  roles = [
    ("owner", "Organization owner with full access", True, 90),
    ("admin", "Administrator with management access", True, 60),
    ("dev", "Developer with limited access", True, 30),
  ]

  for role_name, desc, is_system, level in roles:
    role_id = str(uuid4())
    op.execute(f"""
      INSERT INTO roles (id, organization_id, name, description, is_system_role, hierarchy_level, created_at, updated_at)
      VALUES (
        '{role_id}',
        '{org_id}',
        '{role_name}',
        '{desc}',
        {is_system},
        {level},
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
      )
    """)
    if role_name == "owner":
      # Owner gets all permissions
      for perm_id in permission_ids.values():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)

    elif role_name == "admin":
      # Admin gets all read/write but no delete
      for perm_name, perm_id in permission_ids.items():
        if "delete" not in perm_name:
          op.execute(f"""
                    INSERT INTO role_permissions (id, role_id, permission_id, created_at)
                    VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
                """)

    elif role_name == "dev":
      # Developer gets only read permissions
      for perm_name, perm_id in permission_ids.items():
        if perm_name.endswith("_read"):
          op.execute(f"""
                    INSERT INTO role_permissions (id, role_id, permission_id, created_at)
                    VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
                """)


def downgrade() -> None:
  # Delete in reverse order to respect foreign key constraints

  op.execute("""
    DELETE FROM role_permissions
    WHERE role_id IN (
      SELECT id FROM roles
      WHERE is_system_role = true
    )
  """)

  op.execute("""
    DELETE FROM roles
    WHERE is_system_role = true
  """)

  op.execute("DELETE FROM permissions")
