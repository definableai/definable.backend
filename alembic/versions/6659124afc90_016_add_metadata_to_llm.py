"""016_add_metadata_to_llm

Revision ID: bf5e71d045a3
Revises: ae5e495126f0
Create Date: 2025-04-30 10:15:20.123456

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bf5e71d045a3"
down_revision: Union[str, None] = "ae5e495126f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Add the column with a non-reserved name
  op.add_column("models", sa.Column("model_metadata", sa.JSON, nullable=True, server_default="{}"))

  # Update models with pricing info
  op.execute("""
    UPDATE models
    SET model_metadata = jsonb_build_object('credits_per_1000_tokens', jsonb_build_object('input', 1.5, 'output', 2.0))
    WHERE provider = 'openai'
    """)

  op.execute("""
    UPDATE models
    SET model_metadata = jsonb_build_object('credits_per_1000_tokens', jsonb_build_object('input', 2.0, 'output', 3.0))
    WHERE provider = 'anthropic'
    """)


def downgrade() -> None:
  op.drop_column("models", "model_metadata")
