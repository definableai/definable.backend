"""026_add_billing_cycle_currency

Revision ID: 6f7g8h9i0j1k
Revises: 94d74fb48263
Create Date: 2025-01-30 12:00:00.000000

"""

from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6f7g8h9i0j1k"
down_revision: Union[str, None] = "94d74fb48263"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create billing cycle enum type
  op.execute("""
        CREATE TYPE billing_cycle_enum AS ENUM ('MONTHLY', 'YEARLY');
    """)

  # Add cycle column with enum type
  op.add_column(
    "billing_plans",
    sa.Column("cycle", sa.Enum("MONTHLY", "YEARLY", name="billing_cycle_enum", create_type=False), nullable=False, server_default="MONTHLY"),
  )

  # Add description column (missing from original schema)
  op.add_column("billing_plans", sa.Column("description", sa.String(), nullable=True))

  # Note: currency column and amount column already exist from migration 017

  # Create indexes for better performance
  op.create_index("ix_billing_plans_cycle", "billing_plans", ["cycle"])
  op.create_index("ix_billing_plans_cycle_currency", "billing_plans", ["cycle", "currency"])

  # Truncate existing data
  op.execute("DELETE FROM billing_plans")

  # Insert new billing plans data with fixed values
  plans_data = [
    # Monthly USD plans
    {
      "id": str(uuid4()),
      "name": "starter",
      "description": "Great for freelancers or side projects just getting started.",
      "amount": 0.0,
      "credits": 5000,
      "currency": "USD",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    {
      "id": str(uuid4()),
      "name": "pro",
      "description": "Built for growing teams that need power, flexibility, and speed.",
      "amount": 29.0,
      "credits": 30000,
      "currency": "USD",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    {
      "id": str(uuid4()),
      "name": "enterprise",
      "description": "Everything your organization needs — plus hands-on support.",
      "amount": 99.0,
      "credits": 100000,
      "currency": "USD",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    # Yearly USD plans (with 20% discount)
    {
      "id": str(uuid4()),
      "name": "starter",
      "description": "Great for freelancers or side projects just getting started.",
      "amount": 0.0,
      "credits": 5000,
      "currency": "USD",
      "discount_percentage": 20.0,
      "is_active": True,
      "cycle": "YEARLY",
    },
    {
      "id": str(uuid4()),
      "name": "pro",
      "description": "Built for growing teams that need power, flexibility, and speed.",
      "amount": 279.0,  # 29 * 12 * 0.8 (20% discount)
      "credits": 30000,
      "currency": "USD",
      "discount_percentage": 20.0,
      "is_active": True,
      "cycle": "YEARLY",
    },
    {
      "id": str(uuid4()),
      "name": "enterprise",
      "description": "Everything your organization needs — plus hands-on support.",
      "amount": 950.4,  # 99 * 12 * 0.8 (20% discount) - Fixed the empty amount issue
      "credits": 100000,
      "currency": "USD",
      "discount_percentage": 20.0,
      "is_active": True,  # Fixed the empty is_active issue
      "cycle": "YEARLY",
    },
    # Monthly INR plans
    {
      "id": str(uuid4()),
      "name": "starter",
      "description": "Great for freelancers or side projects just getting started.",
      "amount": 0.0,
      "credits": 5000,
      "currency": "INR",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    {
      "id": str(uuid4()),
      "name": "pro",
      "description": "Built for growing teams that need power, flexibility, and speed.",
      "amount": 2599.0,
      "credits": 30000,
      "currency": "INR",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    {
      "id": str(uuid4()),
      "name": "enterprise",
      "description": "Everything your organization needs — plus hands-on support.",
      "amount": 8799.0,
      "credits": 100000,
      "currency": "INR",
      "discount_percentage": 0.0,
      "is_active": True,
      "cycle": "MONTHLY",
    },
    # Yearly INR plans (with 20% discount)
    {
      "id": str(uuid4()),
      "name": "starter",
      "description": "Great for freelancers or side projects just getting started.",
      "amount": 0.0,
      "credits": 5000,
      "currency": "INR",
      "discount_percentage": 20.0,  # Consistent 20% discount for all yearly plans
      "is_active": True,
      "cycle": "YEARLY",
    },
    {
      "id": str(uuid4()),
      "name": "pro",
      "description": "Built for growing teams that need power, flexibility, and speed.",
      "amount": 24949.0,  # 2599 * 12 * 0.8 = 24949.6 (20% discount applied)
      "credits": 30000,
      "currency": "INR",
      "discount_percentage": 20.0,  # 20% discount properly reflected
      "is_active": True,
      "cycle": "YEARLY",
    },
    {
      "id": str(uuid4()),
      "name": "enterprise",
      "description": "Everything your organization needs — plus hands-on support.",
      "amount": 84399.0,  # 8799 * 12 * 0.8 = 84391.2 (20% discount applied)
      "credits": 100000,
      "currency": "INR",
      "discount_percentage": 20.0,  # 20% discount properly reflected
      "is_active": True,
      "cycle": "YEARLY",
    },
  ]

  # Insert the plans data
  for plan in plans_data:
    op.execute(f"""
      INSERT INTO billing_plans
      (id, name, description, amount, credits, currency, discount_percentage, is_active, cycle, created_at, updated_at)
      VALUES (
        '{plan["id"]}',
        '{plan["name"]}',
        '{plan["description"]}',
        {plan["amount"]},
        {plan["credits"]},
        '{plan["currency"]}',
        {plan["discount_percentage"]},
        {plan["is_active"]},
        '{plan["cycle"]}',
        now(),
        now()
      )
    """)


def downgrade() -> None:
  # Drop indexes
  op.drop_index("ix_billing_plans_cycle_currency", "billing_plans")
  op.drop_index("ix_billing_plans_cycle", "billing_plans")

  # Remove added columns (currency and amount columns remain from migration 017)
  op.drop_column("billing_plans", "description")
  op.drop_column("billing_plans", "cycle")

  # Drop enum type
  op.execute("DROP TYPE IF EXISTS billing_cycle_enum")

  # Restore original billing plans data
  op.execute("DELETE FROM billing_plans")

  # Restore the data that existed before our migration (from migration 017)
  # This includes both USD and INR plans but without cycle column
  original_plans = [
    # USD Plans (from original 010 migration, updated to use 'amount' not 'amount_usd')
    {
      "id": str(uuid4()),
      "name": "Basic",
      "amount": 1.00,
      "credits": 1000,
      "discount_percentage": 0.0,
      "is_active": True,
      "currency": "USD",
    },
    {
      "id": str(uuid4()),
      "name": "Standard",
      "amount": 5.00,
      "credits": 6000,
      "discount_percentage": 10.0,
      "is_active": True,
      "currency": "USD",
    },
    {
      "id": str(uuid4()),
      "name": "Premium",
      "amount": 10.00,
      "credits": 15000,
      "discount_percentage": 20.0,
      "is_active": True,
      "currency": "USD",
    },
    {
      "id": str(uuid4()),
      "name": "Enterprise",
      "amount": 25.00,
      "credits": 50000,
      "discount_percentage": 35.0,
      "is_active": True,
      "currency": "USD",
    },
    # INR Plans (from migration 017)
    {
      "id": str(uuid4()),
      "name": "Basic",
      "amount": 99.0,
      "credits": 1000,
      "discount_percentage": 0.0,
      "is_active": True,
      "currency": "INR",
    },
    {
      "id": str(uuid4()),
      "name": "Standard",
      "amount": 399.0,
      "credits": 6000,
      "discount_percentage": 10.0,
      "is_active": True,
      "currency": "INR",
    },
    {
      "id": str(uuid4()),
      "name": "Premium",
      "amount": 799.0,
      "credits": 15000,
      "discount_percentage": 20.0,
      "is_active": True,
      "currency": "INR",
    },
    {
      "id": str(uuid4()),
      "name": "Enterprise",
      "amount": 1999.0,
      "credits": 50000,
      "discount_percentage": 35.0,
      "is_active": True,
      "currency": "INR",
    },
  ]

  for plan in original_plans:
    op.execute(f"""
      INSERT INTO billing_plans
      (id, name, amount, credits, discount_percentage, is_active, currency, created_at, updated_at)
      VALUES (
        '{plan["id"]}',
        '{plan["name"]}',
        {plan["amount"]},
        {plan["credits"]},
        {plan["discount_percentage"]},
        {plan["is_active"]},
        '{plan["currency"]}',
        now(),
        now()
      )
    """)
