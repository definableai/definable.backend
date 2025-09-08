"""028_create_transaction_logs_and_status_codes

Revision ID: 8h9i0j1k2l3
Revises: 7g8h9i0j1k2
Create Date: 2025-01-31 10:00:00.000000

"""

from datetime import datetime, timezone
from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8h9i0j1k2l3"
down_revision: Union[str, None] = "7g8h9i0j1k2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create status_codes table
  op.create_table(
    "status_codes",
    sa.Column("code", sa.String(10), primary_key=True),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("category", sa.String(20), nullable=False),
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
  )

  # Create transaction_logs table
  op.create_table(
    "transaction_logs",
    sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
    sa.Column("event_id", sa.String(255), nullable=True),  # Provider's webhook event ID
    sa.Column("provider_id", UUID(as_uuid=True), sa.ForeignKey("payment_providers.id"), nullable=False),  # Reference to payment provider
    sa.Column("event_type", sa.String(100), nullable=False),  # Raw event name from provider
    sa.Column("status_code", sa.String(10), sa.ForeignKey("status_codes.code"), nullable=True),  # Generic status code
    sa.Column("entity_type", sa.String(50), nullable=True),  # subscription, payment, order, invoice
    sa.Column("entity_id", sa.String(255), nullable=True),  # Provider's entity ID
    sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True),
    sa.Column("customer_id", sa.String(255), nullable=True),  # Provider's customer ID
    sa.Column("amount", sa.Numeric(10, 2), nullable=True),  # Amount in original currency
    sa.Column("currency", sa.String(3), nullable=True),  # Currency code (USD, INR, etc.)
    sa.Column(
      "processing_status",
      sa.Enum("pending", "processed", "failed", "skipped", "duplicate", name="processing_status_enum"),
      nullable=False,
      server_default="pending",
    ),
    sa.Column("processed_at", sa.DateTime(), nullable=True),
    sa.Column("error_message", sa.Text(), nullable=True),
    sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("payload", JSON, nullable=False),  # Full webhook payload
    sa.Column("extracted_metadata", JSON, nullable=True),  # Extracted/processed metadata
    sa.Column("signature", sa.String(500), nullable=True),  # Webhook signature
    sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
  )

  # Create indexes for transaction_logs
  op.create_unique_constraint("uq_transaction_logs_event_provider", "transaction_logs", ["event_id", "provider_id"])
  op.create_index("idx_transaction_logs_provider_id", "transaction_logs", ["provider_id"])
  op.create_index("idx_transaction_logs_event_type", "transaction_logs", ["event_type"])
  op.create_index("idx_transaction_logs_status_code", "transaction_logs", ["status_code"])
  op.create_index("idx_transaction_logs_processing_status", "transaction_logs", ["processing_status"])
  op.create_index("idx_transaction_logs_organization_id", "transaction_logs", ["organization_id"])
  op.create_index("idx_transaction_logs_user_id", "transaction_logs", ["user_id"])
  op.create_index("idx_transaction_logs_entity", "transaction_logs", ["entity_type", "entity_id"])
  op.create_index("idx_transaction_logs_created_at", "transaction_logs", ["created_at"])
  op.create_index("idx_transaction_logs_org_created", "transaction_logs", ["organization_id", "created_at"])

  # Create indexes for status_codes
  op.create_index("idx_status_codes_category", "status_codes", ["category"])
  op.create_index("idx_status_codes_name", "status_codes", ["name"])

  # Insert status codes from the JSON structure
  op.bulk_insert(
    sa.table(
      "status_codes",
      sa.Column("code", sa.String()),
      sa.Column("name", sa.String()),
      sa.Column("category", sa.String()),
      sa.Column("created_at", sa.DateTime()),
      sa.Column("updated_at", sa.DateTime()),
    ),
    [
      # Subscription status codes
      {
        "code": "DFS001",
        "name": "subscription charged",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS002",
        "name": "subscription activated",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS003",
        "name": "subscription authenticated",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS004",
        "name": "subscription paused",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS005",
        "name": "subscription resumed",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS006",
        "name": "subscription cancelled",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS100",
        "name": "subscription error",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFS200",
        "name": "subscription pending",
        "category": "subscription",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      # Payment status codes
      {
        "code": "DFP001",
        "name": "payment authenticated",
        "category": "payment",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFP002",
        "name": "payment captured",
        "category": "payment",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFP100",
        "name": "payment error",
        "category": "payment",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      # Order status codes
      {
        "code": "DFO001",
        "name": "order paid",
        "category": "order",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFO100",
        "name": "order error",
        "category": "order",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      # Invoice status codes
      {
        "code": "DFI001",
        "name": "invoice paid",
        "category": "invoice",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
      {
        "code": "DFI100",
        "name": "invoice error",
        "category": "invoice",
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
      },
    ],
  )

  # Create triggers for updated_at columns
  for table in ["status_codes", "transaction_logs"]:
    op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE PROCEDURE update_updated_at_column();
        """)


def downgrade() -> None:
  # Drop triggers first
  for table in ["status_codes", "transaction_logs"]:
    op.execute(f"""
            DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
        """)

  # Drop indexes for transaction_logs
  op.drop_index("idx_transaction_logs_org_created", "transaction_logs")
  op.drop_index("idx_transaction_logs_created_at", "transaction_logs")
  op.drop_index("idx_transaction_logs_entity", "transaction_logs")
  op.drop_index("idx_transaction_logs_user_id", "transaction_logs")
  op.drop_index("idx_transaction_logs_organization_id", "transaction_logs")
  op.drop_index("idx_transaction_logs_processing_status", "transaction_logs")
  op.drop_index("idx_transaction_logs_status_code", "transaction_logs")
  op.drop_index("idx_transaction_logs_event_type", "transaction_logs")
  op.drop_index("idx_transaction_logs_provider_id", "transaction_logs")

  # Drop unique constraint
  op.drop_constraint("uq_transaction_logs_event_provider", "transaction_logs", type_="unique")

  # Drop indexes for status_codes
  op.drop_index("idx_status_codes_name", "status_codes")
  op.drop_index("idx_status_codes_category", "status_codes")

  # Drop tables in reverse order (must drop transaction_logs first before dropping the enum it depends on)
  op.drop_table("transaction_logs")
  op.drop_table("status_codes")

  # Drop enum type after the table that uses it is dropped
  op.execute("DROP TYPE IF EXISTS processing_status_enum")
