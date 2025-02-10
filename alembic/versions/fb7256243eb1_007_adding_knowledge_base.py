"""007_adding_knowledge_base

Revision ID: fb7256243eb1
Revises: e080a53fbcbd
Create Date: 2025-02-09 15:30:13.955455

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fb7256243eb1"
down_revision: Union[str, None] = "e080a53fbcbd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# TODO: add ENUM types for doc_type and processing_status
def upgrade() -> None:
  # Create knowledge_base table
  op.create_table(
    "knowledge_base",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("collection_id", postgresql.UUID(), nullable=False),
    sa.Column("organization_id", postgresql.UUID(), nullable=False),
    sa.Column("user_id", postgresql.UUID(), nullable=False),
    sa.Column("embedding_model", sa.String(50), nullable=False),
    sa.Column("settings", postgresql.JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    sa.UniqueConstraint("collection_id"),
  )
  # Create kb_documents table
  op.create_table(
    "kb_documents",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("title", sa.String(200), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("doc_type", sa.SmallInteger(), nullable=False),
    sa.Column("s3_key", sa.String(500), nullable=True),
    sa.Column("original_filename", sa.String(255), nullable=True),
    sa.Column("file_type", sa.String(20), nullable=False),
    sa.Column("file_size", sa.BigInteger(), nullable=True),
    sa.Column("kb_id", postgresql.UUID(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("last_processed_at", sa.TIMESTAMP(), nullable=True),
    sa.Column("processing_status", sa.SmallInteger(), server_default="0", nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"),
  )
  # Create indexes
  op.create_index("ix_kb_documents_kb_id_doc_type", "kb_documents", ["kb_id", "doc_type"])
  op.create_index("ix_kb_documents_processing_status", "kb_documents", ["processing_status"])
  op.create_index("ix_kb_documents_file_type", "kb_documents", ["file_type"])

  # Create updated_at triggers
  for table in ["knowledge_base", "kb_documents"]:
    op.execute(f"""
          CREATE TRIGGER update_{table}_updated_at
              BEFORE UPDATE ON {table}
              FOR EACH ROW
              EXECUTE PROCEDURE update_updated_at_column();
      """)


def downgrade() -> None:
  # Drop triggers
  for table in ["knowledge_base", "kb_documents"]:
    op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

  # Drop tables and indexes
  op.drop_index("ix_kb_documents_file_type", table_name="kb_documents")
  op.drop_index("ix_kb_documents_processing_status", table_name="kb_documents")
  op.drop_index("ix_kb_documents_kb_id_doc_type", table_name="kb_documents")
  op.drop_table("kb_documents")
  op.drop_table("knowledge_base")
