"""014_add_agents_category

Revision ID: f0c5388afe0a
Revises: 7364319dc095
Create Date: 2025-05-11 21:09:09.160217

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f0c5388afe0a"
down_revision: Union[str, None] = "7364319dc095"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Create categories table
  op.create_table(
    "agents_category",
    sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
    sa.Column("name", sa.String(length=100), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("current_timestamp"), nullable=False),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("name"),
  )

  # Add category_id column to agents table (one-to-many relationship)
  op.add_column("agents", sa.Column("category_id", postgresql.UUID(as_uuid=True), nullable=True))
  op.add_column("agents", sa.Column("properties", postgresql.JSONB(), nullable=True, server_default="{}"))

  # Create foreign key constraint
  op.create_foreign_key("fk_agents_category_id", "agents", "agents_category", ["category_id"], ["id"], ondelete="SET NULL")

  # Create index for better performance
  op.create_index("idx_agents_category_id", "agents", ["category_id"])
  op.create_index("idx_agents_properties", "agents", ["properties"], postgresql_using="gin")

  # Insert initial categories
  categories_table = sa.table("agents_category", sa.column("name", sa.String), sa.column("description", sa.Text))

  op.bulk_insert(
    categories_table,
    [
      {"name": "Coding", "description": "Agents specialized in programming and development tasks"},
      {"name": "Writing", "description": "Agents focused on content creation and writing"},
      {"name": "Analysis", "description": "Agents designed for data analysis and processing"},
      {"name": "Creative", "description": "Agents for creative tasks and ideation"},
      {"name": "Research", "description": "Agents specialized in information gathering and research"},
      {"name": "Productivity", "description": "Agents focused on task management and productivity"},
    ],
  )

  # Create full-text search index (added for efficient search)
  op.execute("""
      CREATE INDEX idx_prompts_search ON prompts
      USING gin(to_tsvector('english', title || ' ' || content || ' ' || COALESCE(description, '')))
  """)


def downgrade() -> None:
  op.drop_constraint("fk_agents_category_id", "agents", type_="foreignkey")
  op.drop_index("idx_agents_category_id", table_name="agents")
  op.drop_index("idx_prompts_search", table_name="prompts")
  op.drop_column("agents", "category_id")
  op.drop_column("agents", "properties")
  op.drop_table("agents_category")
