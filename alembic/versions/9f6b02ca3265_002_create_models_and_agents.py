"""002_create_models_and_agents

Revision ID: 9f6b02ca3265
Revises: d9ffb8ed17ea
Create Date: 2025-01-29 09:18:09.544868

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9f6b02ca3265"
down_revision: Union[str, None] = "d9ffb8ed17ea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Create models table
  op.create_table(
    "models",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("provider", sa.String(100), nullable=False),
    sa.Column("version", sa.String(50), nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("config", postgresql.JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name", "provider", "version"),
  )
  op.create_index("ix_models_name_provider_version", "models", ["name", "provider", "version"], unique=True)

  # Create agents table
  op.create_table(
    "agents",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("description", sa.Text(), nullable=False),
    sa.Column("model_id", postgresql.UUID(), nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("settings", postgresql.JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("name"),
  )
  op.create_index("ix_agents_name", "agents", ["name"], unique=True)
  op.create_index("ix_agents_model_id", "agents", ["model_id"])

  # Create updated_at triggers
  for table in ["models", "agents"]:
    op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE PROCEDURE update_updated_at_column();
        """)


def downgrade():
  for table in ["models", "agents"]:
    op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
  op.drop_table("agents")
  op.drop_table("models")
