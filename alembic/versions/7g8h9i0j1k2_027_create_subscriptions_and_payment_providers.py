"""027_create_subscriptions_and_payment_providers

Revision ID: 7g8h9i0j1k2
Revises: 6f7g8h9i0j1k
Create Date: 2025-01-30 14:00:00.000000

"""

from datetime import datetime, timezone
from typing import Sequence, Union

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
    sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("name", sa.String(50), nullable=False, unique=True),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
  )

  # Create subscriptions table
  op.create_table(
    "subscriptions",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
    sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("provider_id", UUID(as_uuid=True), sa.ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=False),
    sa.Column("plan_id", UUID(as_uuid=True), sa.ForeignKey("billing_plans.id", ondelete="RESTRICT"), nullable=False),
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
  op.create_index("ix_subscriptions_plan_id", "subscriptions", ["plan_id"])

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
        "name": "stripe",
        "is_active": True,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "name": "razorpay",
        "is_active": True,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
    ],
  )

  # Add provider_id FK to customers table
  op.add_column("customer", sa.Column("provider_id", UUID(as_uuid=True), sa.ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=True))

  # Update existing customer records to use provider_id FK using subquery
  op.execute("""
    UPDATE customer
    SET provider_id = (
      SELECT id FROM payment_providers
      WHERE name = customer.payment_provider
    )
    WHERE payment_provider IS NOT NULL
  """)

  # Make provider_id NOT NULL after data migration
  op.alter_column("customer", "provider_id", nullable=False)

  # Add index for provider_id
  op.create_index("ix_customer_provider_id", "customer", ["provider_id"])

  # Drop the old payment_provider column
  op.drop_column("customer", "payment_provider")

  # Add provider_id FK to transactions table
  op.add_column(
    "transactions", sa.Column("provider_id", UUID(as_uuid=True), sa.ForeignKey("payment_providers.id", ondelete="RESTRICT"), nullable=True)
  )

  # Update existing transaction records to use provider_id FK using subquery
  op.execute("""
    UPDATE transactions
    SET provider_id = (
      SELECT id FROM payment_providers
      WHERE name = transactions.payment_provider
    )
    WHERE payment_provider IS NOT NULL
  """)

  # Add index for provider_id
  op.create_index("ix_transactions_provider_id", "transactions", ["provider_id"])

  # Keep the old payment_provider column for backward compatibility for now
  # It can be dropped in a future migration after ensuring all code is updated

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

  # Restore payment_provider columns in customer and transactions tables
  op.add_column("customer", sa.Column("payment_provider", sa.String(20), nullable=True, index=True))

  # Restore data from provider_id FK to payment_provider string
  op.execute("""
    UPDATE customer
    SET payment_provider = pp.name
    FROM payment_providers pp
    WHERE customer.provider_id = pp.id
  """)

  # Make payment_provider NOT NULL and add index
  op.alter_column("customer", "payment_provider", nullable=False)

  # Drop provider_id column and its index
  op.drop_index("ix_customer_provider_id", "customer")
  op.drop_column("customer", "provider_id")

  # For transactions table, the payment_provider column is still there, just restore the data
  op.execute("""
    UPDATE transactions
    SET payment_provider = pp.name
    FROM payment_providers pp
    WHERE transactions.provider_id = pp.id
  """)

  # Drop provider_id column and its index from transactions
  op.drop_index("ix_transactions_provider_id", "transactions")
  op.drop_column("transactions", "provider_id")

  # Drop indexes
  op.drop_index("ix_subscriptions_org_user", "subscriptions")
  op.drop_index("ix_subscriptions_is_active", "subscriptions")
  op.drop_index("ix_subscriptions_subscription_id", "subscriptions")
  op.drop_index("ix_subscriptions_plan_id", "subscriptions")
  op.drop_index("ix_subscriptions_provider_id", "subscriptions")
  op.drop_index("ix_subscriptions_user_id", "subscriptions")
  op.drop_index("ix_subscriptions_organization_id", "subscriptions")
  op.drop_index("ix_payment_providers_is_active", "payment_providers")
  op.drop_index("ix_payment_providers_name", "payment_providers")

  # Drop tables in reverse order
  op.drop_table("subscriptions")
  op.drop_table("payment_providers")
