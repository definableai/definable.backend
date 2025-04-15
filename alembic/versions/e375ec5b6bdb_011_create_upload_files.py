"""011_create_upload_files

Revision ID: e375ec5b6bdb
Revises: ba6775663b8a
Create Date: 2025-04-02 16:41:51.207082

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e375ec5b6bdb"
down_revision: Union[str, None] = "d2e2a9098210"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  op.create_table(
    "public_uploads",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("filename", sa.String(100), nullable=False),
    sa.Column("content_type", sa.String(50), nullable=False),
    sa.Column("url", sa.String(255), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), nullable=True),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("id"),
  )

  # Create index on filename
  op.create_index("idx_public_uploads_filename", "public_uploads", ["filename"])

  op.execute("""
      CREATE TRIGGER update_public_uploads_updated_at
          BEFORE UPDATE ON public_uploads
          FOR EACH ROW
          EXECUTE PROCEDURE update_updated_at_column();
      """)


def downgrade():
  # Drop the index first
  op.drop_index("idx_public_uploads_filename", table_name="public_uploads")

  # Drop the table
  op.drop_table("public_uploads")
