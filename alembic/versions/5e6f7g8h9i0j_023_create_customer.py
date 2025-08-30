"""023_create_customer

Revision ID: 5e6f7g8h9i0j
Revises: 4d5e6f7g8h9i
Create Date: 2025-01-30 10:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e6f7g8h9i0j"
down_revision: Union[str, None] = "4d5e6f7g8h9i"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create customer table for payment provider accounts
  op.create_table(
    "customer",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("payment_provider", sa.String(20), nullable=False),
    sa.Column("customer_id", sa.String(255), nullable=False),
    sa.Column("is_active", sa.Boolean(), default=True, server_default="true", nullable=False),
    sa.Column("provider_metadata", sa.JSON(), nullable=True),  # Store provider-specific data
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
  )

  # Create indexes for optimal performance
  # Unique constraint: one customer account per user per provider
  op.create_unique_constraint("uq_customer_user_provider", "customer", ["user_id", "payment_provider"])

  # Unique constraint: customer_id must be unique per provider
  op.create_unique_constraint("uq_customer_external_provider", "customer", ["customer_id", "payment_provider"])

  # Performance indexes
  op.create_index("idx_customer_user_id", "customer", ["user_id"])
  op.create_index("idx_customer_payment_provider", "customer", ["payment_provider"])
  op.create_index("idx_customer_external_id", "customer", ["customer_id"])
  op.create_index("idx_customer_active", "customer", ["is_active"])

  # Composite index for common lookup patterns
  op.create_index("idx_customer_user_provider_active", "customer", ["user_id", "payment_provider", "is_active"])

  # Add updated_at trigger
  op.execute("""
        CREATE TRIGGER update_customer_updated_at
            BEFORE UPDATE ON customer
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column();
    """)

  # Add payment_metadata JSON column for storing payment provider specific data
  op.add_column("transactions", sa.Column("payment_metadata", sa.JSON(), nullable=True))

  # Update the transaction_usage_stats view before dropping columns
  op.execute("""
    CREATE OR REPLACE VIEW transaction_usage_stats AS
    SELECT
    t.id,
    t.user_id,
    t.organization_id AS organization_id,
    u.email AS user_email,
    u.first_name AS user_first_name,
    u.last_name AS user_last_name,
    o.name AS organization_name,
    t.type,
    t.status,
    t.credits,
    t.description,
    t.created_at,
    t.transaction_metadata,
        t.amount AS amount_usd,
    -- Extract date parts for grouping
    DATE(t.created_at) AS usage_date,
    DATE_TRUNC('month', t.created_at)::date AS usage_month,
    -- Extract service from metadata
    COALESCE(t.transaction_metadata->>'service', 'Unknown Service') AS service,
    -- Extra metadata fields that are commonly used in UI
    COALESCE(t.transaction_metadata->>'action', '') AS action,
    COALESCE(t.transaction_metadata->>'charge_name', '') AS charge_name,
    -- Calculate cost in USD based on credits
    ROUND((t.credits::float / 1000)::numeric, 2) AS cost_usd
    FROM
    transactions t
    LEFT JOIN
    users u ON t.user_id = u.id
    LEFT JOIN
    organizations o ON t.organization_id = o.id
    LEFT JOIN
    transaction_types tt ON t.type = tt.type_name
    WHERE
    tt.is_tracked = TRUE
    -- Exclude credit purchase transactions (those with payment providers)
    AND t.payment_provider IS NULL;
  """)

  # Drop legacy payment provider columns
  op.drop_column("transactions", "stripe_payment_intent_id")
  op.drop_column("transactions", "stripe_customer_id")
  op.drop_column("transactions", "stripe_invoice_id")
  op.drop_column("transactions", "razorpay_invoice_id")
  op.drop_column("transactions", "razorpay_payment_id")
  op.drop_column("transactions", "razorpay_customer_id")


def downgrade() -> None:
  # Re-add the dropped columns
  op.add_column("transactions", sa.Column("stripe_payment_intent_id", sa.String(), nullable=True))
  op.add_column("transactions", sa.Column("stripe_customer_id", sa.String(), nullable=True))
  op.add_column("transactions", sa.Column("stripe_invoice_id", sa.String(), nullable=True))
  op.add_column("transactions", sa.Column("razorpay_invoice_id", sa.String(), nullable=True))
  op.add_column("transactions", sa.Column("razorpay_payment_id", sa.String(), nullable=True))
  op.add_column("transactions", sa.Column("razorpay_customer_id", sa.String(), nullable=True))

  # Restore the original transaction_usage_stats view
  op.execute("""
    CREATE OR REPLACE VIEW transaction_usage_stats AS
    SELECT
    t.id,
    t.user_id,
    t.organization_id AS organization_id,
    u.email AS user_email,
    u.first_name AS user_first_name,
    u.last_name AS user_last_name,
    o.name AS organization_name,
    t.type,
    t.status,
    t.credits,
    t.description,
    t.created_at,
    t.transaction_metadata,
    t.amount AS amount_usd,
    -- Extract date parts for grouping
    DATE(t.created_at) AS usage_date,
    DATE_TRUNC('month', t.created_at)::date AS usage_month,
    -- Extract service from metadata
    COALESCE(t.transaction_metadata->>'service', 'Unknown Service') AS service,
    -- Extra metadata fields that are commonly used in UI
    COALESCE(t.transaction_metadata->>'action', '') AS action,
    COALESCE(t.transaction_metadata->>'charge_name', '') AS charge_name,
    -- Calculate cost in USD based on credits
    ROUND((t.credits::float / 1000)::numeric, 2) AS cost_usd
    FROM
    transactions t
    LEFT JOIN
    users u ON t.user_id = u.id
    LEFT JOIN
    organizations o ON t.organization_id = o.id
    LEFT JOIN
    transaction_types tt ON t.type = tt.type_name
    WHERE
    tt.is_tracked = TRUE
    AND t.stripe_payment_intent_id IS NULL;
  """)

  # Drop payment_metadata column
  op.drop_column("transactions", "payment_metadata")

  # Drop trigger
  op.execute("DROP TRIGGER IF EXISTS update_customer_updated_at ON customer")

  # Drop indexes
  op.drop_index("idx_customer_user_provider_active", "customer")
  op.drop_index("idx_customer_active", "customer")
  op.drop_index("idx_customer_external_id", "customer")
  op.drop_index("idx_customer_payment_provider", "customer")
  op.drop_index("idx_customer_user_id", "customer")

  # Drop unique constraints
  op.drop_constraint("uq_customer_external_provider", "customer", type_="unique")
  op.drop_constraint("uq_customer_user_provider", "customer", type_="unique")

  # Drop table
  op.drop_table("customer")
