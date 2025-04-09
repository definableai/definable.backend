"""013_alter_agents_model

Revision ID: 646684fd97f6
Revises: aa9d2cf45c4d
Create Date: 2025-04-07 14:33:05.274754

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "646684fd97f6"
down_revision: Union[str, None] = "aa9d2cf45c4d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
  # Drop the existing unique constraint and index on agents.name
  op.drop_index("ix_agents_name", table_name="agents")
  op.drop_constraint("agents_name_key", table_name="agents")

  # Add a new unique constraint on the combination of name, version, and organization_id
  op.create_unique_constraint("uq_agents_name_version_org", "agents", ["name", "version", "organization_id"])

  # Create a new index for this combination
  op.create_index("ix_agents_name_version_org", "agents", ["name", "version", "organization_id"], unique=True)


def downgrade():
  # Drop the new constraint and index
  op.drop_index("ix_agents_name_version_org", table_name="agents")
  op.drop_constraint("uq_agents_name_version_org", table_name="agents")

  # Recreate the original constraint and index
  op.create_unique_constraint("agents_name_key", "agents", ["name"])
  op.create_index("ix_agents_name", "agents", ["name"], unique=True)
