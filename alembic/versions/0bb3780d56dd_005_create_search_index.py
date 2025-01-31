"""005_create_search_index

Revision ID: 0bb3780d56dd
Revises: e09e0e697964
Create Date: 2025-01-29 09:20:25.767769

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0bb3780d56dd"
down_revision: Union[str, None] = "e09e0e697964"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  op.create_table(
    "search_index",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("entity_id", postgresql.UUID(), nullable=False),
    sa.Column("entity_type", sa.Enum("message", "conversation", name="search_entity_type"), nullable=False),
    sa.Column("search_vector", postgresql.TSVECTOR(), nullable=False),
    sa.Column("last_updated", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("entity_id", "entity_type"),
  )

  # Create two separate foreign key constraints with ON DELETE CASCADE
  op.create_foreign_key("fk_search_index_message", "search_index", "messages", ["entity_id"], ["id"], ondelete="CASCADE")

  op.create_foreign_key("fk_search_index_conversation", "search_index", "conversations", ["entity_id"], ["id"], ondelete="CASCADE")

  # Create GIN index for better full-text search performance
  op.create_index("ix_search_index_vector", "search_index", ["search_vector"], postgresql_using="gin")


def downgrade():
  op.drop_constraint("fk_search_index_conversation", "search_index")
  op.drop_constraint("fk_search_index_message", "search_index")
  op.drop_index("ix_search_index_vector")
  op.drop_table("search_index")
  op.execute("DROP TYPE search_entity_type")
