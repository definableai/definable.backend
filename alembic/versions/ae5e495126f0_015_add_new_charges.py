"""015_add_new_charges

Revision ID: ae5e495126f0
Revises: aa9d2cf45c4d
Create Date: 2025-04-29 02:16:50.872039

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ae5e495126f0"
down_revision: Union[str, None] = "f0c5388afe0a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Insert new charges for different operations
  op.execute("""
    INSERT INTO charges (name, amount, service, action, unit, measure, description)
    VALUES
    ('gpt-4o', 3, 'llm', 'generate', 'credit', 'token', 'GPT-4o model usage charge'),
    ('claude-3-opus', 5, 'llm', 'generate', 'credit', 'token', 'Claude 3 Opus model usage charge'),
    ('claude-3-sonnet', 3, 'llm', 'generate', 'credit', 'token', 'Claude 3 Sonnet model usage charge'),
    ('claude-3-haiku', 1, 'llm', 'generate', 'credit', 'token', 'Claude 3 Haiku model usage charge'),
    ('o1-small-text-indexing', 3, 'kb', 'index', 'credit', 'token', 'Text indexing with OpenAI Ada embedding model'),
    ('pdf-extraction', 5, 'kb', 'extract', 'credit', 'page', 'PDF text extraction per page')
    ON CONFLICT (name) DO UPDATE
    SET amount = EXCLUDED.amount,
        service = EXCLUDED.service,
        action = EXCLUDED.action,
        unit = EXCLUDED.unit,
        measure = EXCLUDED.measure,
        description = EXCLUDED.description;
    """)


def downgrade() -> None:
  # Remove the added charges
  op.execute("""
    DELETE FROM charges
    WHERE name IN (
        'gpt-4o',
        'claude-3-opus',
        'claude-3-sonnet',
        'claude-3-haiku',
        'o1-small-text-indexing',
        'pdf-extraction'
    );
    """)
