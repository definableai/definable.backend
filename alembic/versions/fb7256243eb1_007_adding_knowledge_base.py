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


# TODO: add ENUM types for processing_status
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

  op.create_table(
    "source_types",
    sa.Column("id", sa.SmallInteger(), primary_key=True, nullable=False),
    sa.Column("name", sa.String(50), nullable=False, unique=True),
    sa.Column("handler_class", sa.String(100), nullable=False),
    sa.Column("config_schema", postgresql.JSONB(), nullable=False),
    sa.Column("metadata_schema", postgresql.JSONB(), nullable=False),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
  )
  op.create_index("ix_source_types_name", "source_types", ["name"], unique=True)

  # Create kb_folders table
  op.create_table(
    "kb_folders",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("kb_id", postgresql.UUID(), nullable=False),
    sa.Column("name", sa.String(100), nullable=False),
    sa.Column("parent_id", postgresql.UUID(), nullable=True),
    sa.Column("folder_info", postgresql.JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["parent_id"], ["kb_folders.id"], ondelete="CASCADE"),
  )

  # Create index on kb_id and name
  op.create_index("ix_kb_folders_kb_id_name", "kb_folders", ["kb_id", "name"])
  op.create_index("ix_kb_folders_parent_id", "kb_folders", ["parent_id"])

  # Create kb_documents table
  op.create_table(
    "kb_documents",
    sa.Column("id", postgresql.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
    sa.Column("title", sa.String(200), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("kb_id", postgresql.UUID(), nullable=False),
    sa.Column("folder_id", postgresql.UUID(), nullable=True),
    sa.Column("source_type_id", sa.SmallInteger(), nullable=False),
    sa.Column("source_metadata", postgresql.JSONB(), nullable=False),
    sa.Column("content", sa.Text(), nullable=True),
    sa.Column("extraction_status", sa.SmallInteger(), server_default="0", nullable=False),
    sa.Column("indexing_status", sa.SmallInteger(), server_default="0", nullable=False),
    sa.Column("error_message", sa.Text(), nullable=True),
    sa.Column("extraction_completed_at", sa.TIMESTAMP(), nullable=True),
    sa.Column("indexing_completed_at", sa.TIMESTAMP(), nullable=True),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["folder_id"], ["kb_folders.id"], ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["source_type_id"], ["source_types.id"], ondelete="CASCADE"),
  )
  # Create indexes
  op.create_index("ix_kb_documents_source_type", "kb_documents", ["kb_id", "source_type_id"])
  op.create_index("ix_kb_documents_extraction_status", "kb_documents", ["extraction_status"])
  op.create_index("ix_kb_documents_indexing_status", "kb_documents", ["indexing_status"])

  # Create updated_at triggers
  for table in ["knowledge_base", "kb_documents", "kb_folders"]:
    op.execute(f"""
      CREATE TRIGGER update_{table}_updated_at
          BEFORE UPDATE ON {table}
          FOR EACH ROW
          EXECUTE PROCEDURE update_updated_at_column();
    """)

  # add initial source types
  op.execute("""
    INSERT INTO source_types (id, name, handler_class, config_schema, metadata_schema, is_active)
    VALUES
    (1, 'file', 'FileSourceHandler',
    '{
        "max_file_size": 10485760,
        "allowed_extensions": ["pdf", "docx", "txt", "md", "csv", "html", "eml", "msg", "xlsx"],
        "storage": {
            "bucket": "documents",
            "path": "uploads"
        }
    }'::jsonb,
    '{
        "type": "object",
        "required": ["file_type", "original_filename", "size", "mime_type"],
        "properties": {
            "file_type": {"type": "string"},
            "original_filename": {"type": "string"},
            "size": {"type": "integer"},
            "mime_type": {"type": "string"},
            "s3_key": {"type": "string"}
        }
    }'::jsonb,
    true),
    (2, 'url', 'URLSourceHandler',
    '{
        "max_urls": 100,
        "timeout": 30,
        "user_agent": "Mozilla/5.0",
        "follow_redirects": true,
        "verify_ssl": true
    }'::jsonb,
    '{
        "type": "object",
        "required": ["urls"],
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string", "format": "uri"},
                "minItems": 1
            },
            "crawl_config": {
                "type": "object",
                "properties": {
                    "max_depth": {"type": "integer", "minimum": 1},
                    "exclude_paths": {"type": "array", "items": {"type": "string"}},
                    "include_paths": {"type": "array", "items": {"type": "string"}},
                    "follow_links": {"type": "boolean"}
                }
            }
        }
    }'::jsonb,
    true)
  """)


def downgrade() -> None:
  # Drop triggers
  for table in ["knowledge_base", "kb_documents", "kb_folders"]:
    op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")

  # Drop tables and indexes
  op.drop_index("ix_kb_documents_source_type", table_name="kb_documents")
  op.drop_index("ix_kb_documents_extraction_status", table_name="kb_documents")
  op.drop_index("ix_kb_documents_indexing_status", table_name="kb_documents")
  op.drop_table("kb_documents")
  op.drop_table("kb_folders")
  op.drop_table("knowledge_base")
  op.drop_table("source_types")
