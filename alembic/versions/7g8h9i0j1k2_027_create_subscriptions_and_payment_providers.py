"""027_create_subscriptions_and_payment_providers

Revision ID: 7g8h9i0j1k2
Revises: 6f7g8h9i0j1k
Create Date: 2025-01-30 14:00:00.000000

"""

from datetime import datetime, timezone
from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7g8h9i0j1k2"
down_revision: Union[str, None] = "6f7g8h9i0j1k"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create payment_providers table
  op.create_table(
    "payment_providers",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("name", sa.String(50), nullable=False, unique=True),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
  )

  # Create subscriptions table
  op.create_table(
    "subscriptions",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
    sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("provider_id", UUID(as_uuid=True), sa.ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=False),
    sa.Column("subscription_id", sa.String(255), nullable=True, index=True),
    sa.Column("settings", JSON, nullable=True),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
  )

  # Create indexes for better performance
  op.create_index("ix_payment_providers_name", "payment_providers", ["name"])
  op.create_index("ix_payment_providers_is_active", "payment_providers", ["is_active"])
  op.create_index("ix_subscriptions_organization_id", "subscriptions", ["organization_id"])
  op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])
  op.create_index("ix_subscriptions_provider_id", "subscriptions", ["provider_id"])

  # Create subscription_id index with conditional creation to handle existing indexes
  op.execute("""
    CREATE INDEX IF NOT EXISTS ix_subscriptions_subscription_id
    ON subscriptions (subscription_id)
  """)

  op.create_index("ix_subscriptions_is_active", "subscriptions", ["is_active"])
  op.create_index("ix_subscriptions_org_user", "subscriptions", ["organization_id", "user_id"])

  # Insert default payment providers
  op.bulk_insert(
    sa.table(
      "payment_providers",
      sa.Column("id", UUID(as_uuid=True)),
      sa.Column("name", sa.String()),
      sa.Column("is_active", sa.Boolean()),
      sa.Column("created_at", sa.DateTime()),
      sa.Column("updated_at", sa.DateTime()),
    ),
    [
      {
        "id": uuid4(),
        "name": "stripe",
        "is_active": True,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "id": uuid4(),
        "name": "razorpay",
        "is_active": True,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
    ],
  )

  # Create triggers for updated_at columns
  for table in ["payment_providers", "subscriptions"]:
    op.execute(f"""
      CREATE TRIGGER update_{table}_updated_at
          BEFORE UPDATE ON {table}
          FOR EACH ROW
          EXECUTE PROCEDURE update_updated_at_column();
    """)


def downgrade() -> None:
  # Drop triggers first
  for table in ["payment_providers", "subscriptions"]:
    op.execute(f"""
      DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
    """)

  # Drop indexes
  op.drop_index("ix_subscriptions_org_user", "subscriptions")
  op.drop_index("ix_subscriptions_is_active", "subscriptions")
  op.drop_index("ix_subscriptions_subscription_id", "subscriptions")
  op.drop_index("ix_subscriptions_provider_id", "subscriptions")
  op.drop_index("ix_subscriptions_user_id", "subscriptions")
  op.drop_index("ix_subscriptions_organization_id", "subscriptions")
  op.drop_index("ix_payment_providers_is_active", "payment_providers")
  op.drop_index("ix_payment_providers_name", "payment_providers")

  # Drop tables in reverse order
  op.drop_table("subscriptions")
  op.drop_table("payment_providers")
