"""003_create_chats_and_messages

Revision ID: a7c65f3cd908
Revises: 9f6b02ca3265
Create Date: 2025-01-29 09:18:59.049322

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c65f3cd908"
down_revision: Union[str, None] = "9f6b02ca3265"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  op.create_table(
    "chats",
    sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("org_id", sa.UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
    sa.Column("user_id", sa.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("status", sa.Enum("ACTIVE", "ARCHIVED", "DELETED", name="chat_status"), nullable=False, server_default="ACTIVE"),
    sa.Column("metadata", postgresql.JSONB, nullable=True, default={}),
    sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp()),
    sa.Column("updated_at", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp(), onupdate=sa.func.current_timestamp()),
  )
  # Create messages table
  op.create_table(
    "messages",
    sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("chat_session_id", sa.UUID(as_uuid=True), sa.ForeignKey("chats.id"), nullable=False),
    sa.Column("parent_message_id", sa.UUID(as_uuid=True), sa.ForeignKey("messages.id"), nullable=True),
    sa.Column("role", sa.Enum("USER", "AGENT", "MODEL", name="message_role"), nullable=False),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("model_id", sa.UUID(as_uuid=True), sa.ForeignKey("models.id"), nullable=True),
    sa.Column("agent_id", sa.UUID(as_uuid=True), sa.ForeignKey("agents.id"), nullable=True),
    sa.Column("metadata", postgresql.JSONB, nullable=True, default={}),
    sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp()),
  )
  # Create indexes for messages
  op.create_index("idx_messages_chat_session_id", "messages", ["chat_session_id"])
  op.create_index("idx_messages_created_at", "messages", ["created_at"])
  op.create_index("idx_messages_chat_created", "messages", ["chat_session_id", "created_at"])
  op.create_index("idx_messages_parent_id", "messages", ["parent_message_id"])

  # Create chat_uploads table
  op.create_table(
    "chat_uploads",
    sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
    sa.Column("message_id", sa.UUID(as_uuid=True), sa.ForeignKey("messages.id"), nullable=True),
    sa.Column("filename", sa.String(255), nullable=False),
    sa.Column("content_type", sa.String(150), nullable=False),
    sa.Column("file_size", sa.BigInteger, nullable=False),
    sa.Column("url", sa.String(500), nullable=False),
    sa.Column("metadata", postgresql.JSONB, nullable=True, default={}),
    sa.Column("created_at", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp()),
    sa.Column("updated_at", sa.TIMESTAMP, nullable=False, server_default=sa.func.current_timestamp(), onupdate=sa.func.current_timestamp()),
  )

  # Create indexes for chat_uploads
  op.create_index("idx_chat_uploads_filename", "chat_uploads", ["filename"])
  op.create_index("idx_chat_uploads_message_id", "chat_uploads", ["message_id"])


def downgrade():
  # Drop tables
  op.drop_table("chat_uploads")
  op.drop_table("messages")
  op.drop_table("chats")

  # Drop enum types
  op.execute("DROP TYPE IF EXISTS message_status")
  op.execute("DROP TYPE IF EXISTS message_role")
  op.execute("DROP TYPE IF EXISTS chat_status")
