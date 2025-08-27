"""021_add_url_extraction_charge

Revision ID: 3c4d5e6f7g8h
Revises: 2b3c4d5e6f7g
Create Date: 2025-08-16 04:30:00.000000

"""

from typing import Sequence, Union
from uuid import uuid4

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7g8h"
down_revision: Union[str, None] = "2b3c4d5e6f7g"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Insert url_extraction charge
  op.execute(f"""
        INSERT INTO charges (id, name, amount, unit, measure, service, action, description, is_active)
        VALUES ('{str(uuid4())}', 'url_extraction', 2, 'credit', 'page', 'KnowledgeBase', 'extraction', 'Extract content from web URLs', true)
        ON CONFLICT (name) DO UPDATE
        SET amount = EXCLUDED.amount,
            unit = EXCLUDED.unit,
            measure = EXCLUDED.measure,
            service = EXCLUDED.service,
            action = EXCLUDED.action,
            description = EXCLUDED.description,
            is_active = EXCLUDED.is_active;
    """)


def downgrade() -> None:
  # Remove the url_extraction charge
  op.execute("""
        DELETE FROM charges WHERE name = 'url_extraction';
    """)
