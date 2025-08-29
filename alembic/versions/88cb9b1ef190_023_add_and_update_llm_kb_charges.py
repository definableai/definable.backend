"""023_add_and_update_llm_kb_charges

Revision ID: 88cb9b1ef190
Revises: 6a4713935734
Create Date: 2025-08-28 14:23:24.858076

"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '88cb9b1ef190'
down_revision: Union[str, None] = '6a4713935734'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """ Adds new AI model charges and updates existing ones."""
    conn = op.get_bind()

    # Define a helper for the charges table
    charges_table = sa.Table(
        'charges',
        sa.MetaData(),
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column('name', sa.String, unique=True, nullable=False),
        sa.Column('amount', sa.Integer, nullable=False),
        sa.Column('unit', sa.String, nullable=False, default="credit"),
        sa.Column('measure', sa.String, nullable=False),
        sa.Column('service', sa.String, nullable=False),
        sa.Column('action', sa.String, nullable=False),
        sa.Column('description', sa.String, nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
    )

    # --- List of charges ---
    desired_charges = [
        # ------------------------
        # OpenAI LLM Models
        # ------------------------
        {
          "name": "gpt-4.1", "amount": 6, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "GPT-4.1 model usage charge",
        },
        {
          "name": "gpt-4o", "amount": 3, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "GPT-4o model usage charge",
        },
        {
          "name": "gpt-4o-mini", "amount": 2, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "GPT-4o Mini model usage charge",
        },
        {
          "name": "gpt-3.5-turbo", "amount": 1, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "GPT-3.5 Turbo model usage charge",
        },
        # ------------------------
        # Anthropic LLM Models
        # ------------------------
        {
          "name": "claude-3.7-sonnet", "amount": 6, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "Claude 3.7 Sonnet model usage charge",
        },
        {
          "name": "claude-3.5-sonnet", "amount": 4, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "Claude 3.5 Sonnet model usage charge",
        },
        {
          "name": "claude-3.5-haiku", "amount": 1, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "Claude 3 Haiku model usage charge",
        },
        # ------------------------
        # DeepSeek LLM Models
        # ------------------------
        {
          "name": "deepseek-chat", "amount": 2, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "DeepSeek Chat model usage charge",
        },
        {
          "name": "deepseek-reason", "amount": 4, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "DeepSeek Reason model usage charge",
        },
        # ------------------------
        # Other OpenAI O-Series Models
        # ------------------------
        {
          "name": "o4-mini", "amount": 2, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "O4 Mini model usage charge",
        },
        {
          "name": "o1-preview", "amount": 7, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "O1 Preview model usage charge",
        },
        {
          "name": "o1", "amount": 4, "service": "llm", "action": "generate",
          "unit": "credit", "measure": "token", "description": "O1 model usage charge",
        },
        # ------------------------
        # Knowledge Base Services
        # ------------------------
        {
          "name": "o1-small-text-indexing", "amount": 3, "service": "kb", "action": "index",
          "unit": "credit", "measure": "token", "description": "Text indexing with OpenAI Ada embedding model",
        },
        {
          "name": "o1-small-text-retrieval", "amount": 3, "service": "kb", "action": "retrieval",
          "unit": "credit", "measure": "token", "description": "Text retrieval with OpenAI Ada embedding model",
        },
        # {
        #   "name": "pdf-extraction", "amount": 5, "service": "kb", "action": "extract",
        #   "unit": "credit", "measure": "page", "description": "PDF text extraction per page",
        # },
        {
          "name": "excel-ext", "amount": 4, "service": "kb", "action": "extraction",
          "unit": "credit", "measure": "sheet", "description": "Extract data from Excel spreadsheets",
        },
    ]

    for charge_data in desired_charges:
        # Check if a charge with this name already exists
        result = conn.execute(
            sa.select(charges_table.c.id).where(charges_table.c.name == charge_data['name'])
        ).fetchone()

        if result:
            # Update existing charge
            print(f"Updating existing charge: '{charge_data['name']}'")
            update_statement = (
                sa.update(charges_table)
                .where(charges_table.c.name == charge_data['name'])
                .values(
                    amount=charge_data.get('amount'),
                    unit=charge_data.get('unit'),
                    measure=charge_data.get('measure'),
                    service=charge_data.get('service'),
                    action=charge_data.get('action'),
                    description=charge_data.get('description'),
                    is_active=charge_data.get('is_active', True)
                )
            )
            conn.execute(update_statement)
        else:
            # Insert new charge
            print(f"Inserting new charge: '{charge_data['name']}'")
            insert_statement = sa.insert(charges_table).values(**charge_data)
            conn.execute(insert_statement)


def downgrade() -> None:
    """This downgrade will remove the charges if they were added by this migration."""
    charge_names = [
        "gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
        "claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3-haiku",
        "deepseek-chat", "deepseek-reasoner", "o4-mini", "o1-preview", "o1",
        "o1-small-text-indexing", "o1-small-text-retrieval", "pdf-extraction", "excel-ext"
    ]

    # Create a comma-separated string of single-quoted names for the SQL IN clause.
    names_for_sql = ", ".join([f"'{name}'" for name in charge_names])
    op.execute(f"DELETE FROM charges WHERE name IN ({names_for_sql})")
