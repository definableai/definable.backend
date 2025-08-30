"""022_add_plan_id_to_billing_plans

Revision ID: 4d5e6f7g8h9i
Revises: 3c4d5e6f7g8h
Create Date: 2025-01-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision: str = "4d5e6f7g8h9i"
down_revision: Union[str, None] = "3c4d5e6f7g8h"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Add plan_id column to billing_plans table
  op.add_column("billing_plans", sa.Column("plan_id", sa.String(), nullable=True))

  # Create index for plan_id for faster lookups
  op.create_index("idx_billing_plans_plan_id", "billing_plans", ["plan_id"])

  # Create unique constraint to ensure one-to-one mapping between billing plan and Razorpay plan
  op.create_unique_constraint("uq_billing_plans_plan_id", "billing_plans", ["plan_id"])


def downgrade() -> None:
  # Drop unique constraint
  op.drop_constraint("uq_billing_plans_plan_id", "billing_plans")

  # Drop index
  op.drop_index("idx_billing_plans_plan_id", "billing_plans")

  # Drop column
  op.drop_column("billing_plans", "plan_id")
