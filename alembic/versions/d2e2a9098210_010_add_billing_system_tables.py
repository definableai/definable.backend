"""010_add_billing_system_tables

Revision ID: d2e2a9098210
Revises: e375ec5b6bdb
Create Date: 2025-04-03 01:02:46.564111

"""

from datetime import datetime
from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2e2a9098210"
down_revision: Union[str, None] = "ba6775663b8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create charges table
  op.create_table(
    "charges",
    sa.Column("id", UUID(as_uuid=True), nullable=True),
    sa.Column("name", sa.String(), primary_key=True),
    sa.Column("amount", sa.Integer(), nullable=False),
    sa.Column("unit", sa.String(), nullable=False, server_default="credit"),
    sa.Column("measure", sa.String(), nullable=False),
    sa.Column("service", sa.String(), nullable=False),
    sa.Column("action", sa.String(), nullable=False),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.PrimaryKeyConstraint("name"),
  )

  # Create wallets table
  op.create_table(
    "wallets",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
    sa.Column("balance", sa.Integer(), default=0),
    sa.Column("hold", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("credits_spent", sa.Integer(), server_default="0", nullable=False),
    sa.Column("last_reset_date", sa.DateTime(), nullable=True),
    sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
  )

  # Create transactions table
  op.create_table(
    "transactions",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
    sa.Column("organization_id", sa.UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=True),
    sa.Column("type", sa.String(), nullable=False),
    sa.Column("status", sa.String(), nullable=False),
    sa.Column("amount_usd", sa.Float(), nullable=True),
    sa.Column("credits", sa.Integer(), nullable=False),
    sa.Column("stripe_payment_intent_id", sa.String(), nullable=True),
    sa.Column("stripe_customer_id", sa.String(), nullable=True),
    sa.Column("stripe_invoice_id", sa.String(), nullable=True),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("transaction_metadata", JSON, nullable=True),
    sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
  )

  # Create billing_plans table
  op.create_table(
    "billing_plans",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("amount_usd", sa.Float(), nullable=False),
    sa.Column("credits", sa.Integer(), nullable=False),
    sa.Column("discount_percentage", sa.Float(), nullable=False, default=0.0),
    sa.Column("is_active", sa.Boolean(), default=True),
    sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    sa.Column(
      "updated_at",
      sa.DateTime(),
      default=sa.func.now(),
      onupdate=sa.func.now(),
    ),
  )

  # Insert default billing plans matching the UI
  op.bulk_insert(
    sa.table(
      "billing_plans",
      sa.Column("id", UUID(as_uuid=True)),
      sa.Column("name", sa.String()),
      sa.Column("description", sa.String()),
      sa.Column("amount_usd", sa.Float()),
      sa.Column("credits", sa.Integer()),
      sa.Column("discount_percentage", sa.Float()),
      sa.Column("is_active", sa.Boolean()),
      sa.Column("created_at", sa.DateTime()),
      sa.Column("updated_at", sa.DateTime()),
    ),
    [
      # 100 Credits plan
      {
        "id": uuid4(),
        "name": "Basic",
        "amount_usd": 1.00,
        "credits": 1000,
        "discount_percentage": 0.0,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
      },
      # 600 Credits plan
      {
        "id": uuid4(),
        "name": "Standard",
        "amount_usd": 5.00,
        "credits": 6000,
        "discount_percentage": 10.0,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
      },
      # 1,500 Credits plan (Popular)
      {
        "id": uuid4(),
        "name": "Premium",
        "amount_usd": 10.00,
        "credits": 15000,
        "discount_percentage": 20.0,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
      },
      # 5,000 Credits plan
      {
        "id": uuid4(),
        "name": "Enterprise",
        "amount_usd": 25.00,
        "credits": 50000,
        "discount_percentage": 35.0,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
      },
    ],
  )

  # Insert initial charges data
  op.execute(f"""
    INSERT INTO charges (id, name, amount, unit, measure, service, action, description) VALUES
    ('{str(uuid4())}', 'pdf_extraction', 5, 'credit', 'page', 'KnowledgeBase', 'extraction', 'Extract text from PDF documents'),
    ('{str(uuid4())}', 'excel_ext', 4, 'credit', 'sheet', 'KnowledgeBase', 'extraction', 'Extract data from Excel spreadsheets'),
    ('{str(uuid4())}', 'o3-mini', 3, 'credit', 'token', 'Conversation', 'chat', 'Claude 3 Opus Mini model usage'),
    ('{str(uuid4())}', 'gpt-4o', 3, 'credit', 'token', 'Conversation', 'chat', 'GPT-4o model usage'),
    ('{str(uuid4())}', 'o1-small-text-indexing', 3, 'credit', 'token', 'KnowledgeBase', 'indexing', 'Claude Opus 1 Small Text indexing'),
    ('{str(uuid4())}', 'o1-small-text-retrieval', 3, 'credit', 'token', 'KnowledgeBase', 'retrieval', 'Claude Opus 1 Small Text retrieval')
    """)

  # Create indexes for better performance
  op.create_index("ix_transactions_type", "transactions", ["type"])
  op.create_index("ix_transactions_user_id_type_status", "transactions", ["user_id", "type", "status"])
  op.create_index("ix_transactions_organization_id", "transactions", ["organization_id"])
  op.create_index("ix_transactions_user_id_organization_id", "transactions", ["user_id", "organization_id"])

  # Create indexes
  op.create_index("ix_wallets_organization_id", "wallets", ["organization_id"])
  op.create_index("ix_transactions_user_id", "transactions", ["user_id"])
  op.create_index(
    "ix_transactions_stripe_payment_intent_id",
    "transactions",
    ["stripe_payment_intent_id"],
  )
  op.create_index("ix_billing_plans_is_active", "billing_plans", ["is_active"], unique=False)

  # Create triggers for updated_at columns
  for table in ["charges", "wallets", "transactions", "billing_plans"]:
    op.execute(f"""
      CREATE TRIGGER update_{table}_updated_at
          BEFORE UPDATE ON {table}
          FOR EACH ROW
          EXECUTE PROCEDURE update_updated_at_column();
    """)


def downgrade():
  # Drop triggers first
  for table in ["charges", "wallets", "transactions", "billing_plans"]:
    op.execute(f"""
      DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
    """)

  # Drop indexes
  op.drop_index("ix_billing_plans_is_active", table_name="billing_plans")
  op.drop_index("ix_transactions_stripe_payment_intent_id", table_name="transactions")
  op.drop_index("ix_transactions_user_id", table_name="transactions")
  op.drop_index("ix_transactions_organization_id", table_name="transactions")
  op.drop_index("ix_transactions_user_id_organization_id", table_name="transactions")
  op.drop_index("ix_wallets_organization_id", table_name="wallets")

  # Drop tables in reverse order
  op.drop_table("billing_plans")
  op.drop_table("transactions")
  op.drop_table("wallets")
  op.drop_table("charges")
