"""022_backfill_null_ids_in_charges

Revision ID: 6a4713935734
Revises: 3c4d5e6f7g8h
Create Date: 2025-08-28 14:20:03.321547

"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6a4713935734'
down_revision: Union[str, None] = '3c4d5e6f7g8h'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """ Find charges with NULL IDs and assign them a new UUID."""
    conn = op.get_bind()

    # Define a helper for the charges table - must match the actual database schema
    charges_table = sa.Table(
        'charges',
        sa.MetaData(),
        sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.String(), primary_key=True),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('unit', sa.String(), nullable=False),
        sa.Column('measure', sa.String(), nullable=False),
        sa.Column('service', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Find all rows where the id is NULL
    results = conn.execute(sa.select(charges_table).where(charges_table.c.id.is_(None))).fetchall()

    print(f"Found {len(results)} charges with NULL IDs. Backfilling now...")

    for row in results:
        new_id = uuid4()
        # Create an UPDATE statement to set the new ID
        update_statement = (
            sa.update(charges_table)
            .where(charges_table.c.name == row.name) # Use a unique column to target the row
            .values(id=new_id)
        )
        conn.execute(update_statement)
        print(f"  -> Assigned ID {new_id} to charge '{row.name}'")

    print("Finished backfilling NULL IDs.")

# def upgrade() -> None:
#     pass

def downgrade() -> None:
    pass