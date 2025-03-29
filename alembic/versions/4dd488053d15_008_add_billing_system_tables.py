"""008 Add billing system tables

Revision ID: 4dd488053d15
Revises: fb7256243eb1
Create Date: 2025-03-11 01:07:12.125677

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID, JSON

# revision identifiers, used by Alembic.
revision: str = "4dd488053d15"
down_revision: Union[str, None] = "fb7256243eb1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create credit_balances table
    op.create_table(
        "credit_balances",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("balance", sa.Integer(), default=0),
        sa.Column("credits_spent", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_reset_date", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()
        ),
    )

    # Create transactions table
    op.create_table(
        "transactions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("amount_usd", sa.Float(), nullable=False),
        sa.Column("credits", sa.Integer(), nullable=False),
        sa.Column("stripe_payment_intent_id", sa.String(), nullable=True),
        sa.Column("stripe_customer_id", sa.String(), nullable=True),
        sa.Column("stripe_invoice_id", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("transaction_metadata", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()
        ),
    )

    # Create billing_plans table
    op.create_table(
        "billing_plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
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

    # Create llm_model_pricing table (in model.py but not in previous migrations)
    op.create_table(
        "llm_model_pricing",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("model_name", sa.String(), unique=True, nullable=False),
        sa.Column("credits_per_token", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()
        ),
    )

    # Create billing_services table
    op.create_table(
        "billing_services",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("credit_cost", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            onupdate=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    # Create billing_user_usage table
    op.create_table(
        "billing_user_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "service_id",
            UUID(as_uuid=True),
            sa.ForeignKey("billing_services.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("month", sa.DateTime(), nullable=False),
        sa.Column("total_requests", sa.Integer(), server_default="0"),
        sa.Column("total_credits", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
        ),
    )

    # Create billing_api_requests table
    op.create_table(
        "billing_api_requests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "service_id",
            UUID(as_uuid=True),
            sa.ForeignKey("billing_services.id"),
            nullable=False,
        ),
        sa.Column("credits_used", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("request_time", sa.DateTime(), nullable=False),
        sa.Column("response_time", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")
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
                "id": uuid.uuid4(),
                "name": "Basic",
                "amount_usd": 1.00,
                "credits": 100,
                "discount_percentage": 0.0,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
            # 600 Credits plan
            {
                "id": uuid.uuid4(),
                "name": "Standard",
                "amount_usd": 5.00,
                "credits": 600,
                "discount_percentage": 10.0,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
            # 1,500 Credits plan (Popular)
            {
                "id": uuid.uuid4(),
                "name": "Premium",
                "amount_usd": 10.00,
                "credits": 1500,
                "discount_percentage": 20.0,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
            # 5,000 Credits plan
            {
                "id": uuid.uuid4(),
                "name": "Enterprise",
                "amount_usd": 25.00,
                "credits": 5000,
                "discount_percentage": 35.0,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        ],
    )

    # Create indexes
    op.create_index("ix_credit_balances_user_id", "credit_balances", ["user_id"])
    op.create_index("ix_transactions_user_id", "transactions", ["user_id"])
    op.create_index(
        "ix_transactions_stripe_payment_intent_id",
        "transactions",
        ["stripe_payment_intent_id"],
    )
    op.create_index(
        "ix_billing_plans_is_active", "billing_plans", ["is_active"], unique=False
    )
    op.create_index(
        "ix_billing_user_usage_user_service_month",
        "billing_user_usage",
        ["user_id", "service_id", "month"],
    )
    op.create_index(
        "ix_billing_api_requests_user_org",
        "billing_api_requests",
        ["user_id", "org_id"],
    )
    op.create_index(
        "ix_billing_services_name_org", "billing_services", ["name", "org_id"]
    )


def downgrade():
    # Drop tables in reverse order
    op.drop_index("ix_billing_api_requests_user_org", table_name="billing_api_requests")
    op.drop_index(
        "ix_billing_user_usage_user_service_month", table_name="billing_user_usage"
    )
    op.drop_index("ix_billing_services_name_org", table_name="billing_services")
    op.drop_index("ix_billing_plans_is_active", table_name="billing_plans")
    op.drop_index("ix_transactions_stripe_payment_intent_id", table_name="transactions")
    op.drop_index("ix_transactions_user_id", table_name="transactions")
    op.drop_index("ix_credit_balances_user_id", table_name="credit_balances")

    op.drop_table("billing_api_requests")
    op.drop_table("billing_user_usage")
    op.drop_table("billing_services")
    op.drop_table("llm_model_pricing")
    op.drop_table("billing_plans")
    op.drop_table("transactions")
    op.drop_table("credit_balances")
