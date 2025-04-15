"""012_add_get_usage_history_view

Revision ID: aa9d2cf45c4d
Revises: e375ec5b6bdb
Create Date: 2025-04-03 01:05:10.878194

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "aa9d2cf45c4d"
down_revision: Union[str, None] = "e375ec5b6bdb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # First create a transaction types table if it doesn't exist
  op.execute("""
    CREATE TABLE IF NOT EXISTS transaction_types (
      type_name VARCHAR(50) PRIMARY KEY,
      is_tracked BOOLEAN NOT NULL DEFAULT TRUE,
      description TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  """)

  # Insert the current transaction types (if they don't exist)
  op.execute("""
    INSERT INTO transaction_types (type_name, is_tracked, description)
    VALUES
      ('DEBIT', TRUE, 'Standard debit transaction'),
      ('HOLD', TRUE, 'Hold on credits'),
      ('RELEASE', TRUE, 'Release held credits')
    ON CONFLICT (type_name) DO NOTHING;
  """)

  # Create view for transaction usage history using the types table
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
    t.amount_usd,
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


def downgrade():
  # Drop view first
  op.execute("DROP VIEW IF EXISTS transaction_usage_stats;")

  # We don't drop the transaction_types table as it might be used elsewhere
  # If needed in the future, add: op.execute("DROP TABLE IF EXISTS transaction_types;")
