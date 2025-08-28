"""backfill_null_ids_in_charges

Revision ID: 3643ed7fee67
Revises: 3c4d5e6f7g8h
Create Date: 2025-08-28 13:29:51.476481

"""
from typing import Sequence, Union
import uuid

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3643ed7fee67'
down_revision: Union[str, None] = '3c4d5e6f7g8h'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """ Find charges with NULL IDs and assign them a new UUID."""
    conn = op.get_bind()
    
    # Define a helper for the charges table
    charges_table = sa.Table(
        'charges',
        sa.MetaData(),
        sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True)),
        sa.Column('name', sa.String())
    )
    
    # Find all rows where the id is NULL
    results = conn.execute(sa.select(charges_table).where(charges_table.c.id == None)).fetchall()
    
    print(f"Found {len(results)} charges with NULL IDs. Backfilling now...")

    for row in results:
        new_id = uuid.uuid4()
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