"""009_add_get_usage_history_view

Revision ID: bc44d043fa25
Revises: 4dd488053d15
Create Date: 2025-03-31 18:32:38.134285

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bc44d043fa25"
down_revision: Union[str, None] = "4dd488053d15"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create view for transaction usage history
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
    WHERE
    t.type IN ('DEBIT', 'HOLD', 'RELEASE')
    AND t.stripe_payment_intent_id IS NULL;
  """)


def downgrade():
  # Drop view first
  op.execute("DROP VIEW IF EXISTS transaction_usage_stats;")
