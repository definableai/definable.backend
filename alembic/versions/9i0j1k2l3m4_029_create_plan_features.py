"""029_create_plan_features

Revision ID: 9i0j1k2l3m4
Revises: 8h9i0j1k2l3
Create Date: 2025-01-30 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9i0j1k2l3m4"
down_revision: Union[str, None] = "8h9i0j1k2l3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  # Clean slate approach - drop everything first, then recreate
  op.execute("DROP TABLE IF EXISTS plan_feature_limits CASCADE")
  op.execute("DROP TABLE IF EXISTS plan_features CASCADE")
  op.execute("DROP TABLE IF EXISTS plan_feature_categories CASCADE")
  op.execute("DROP TYPE IF EXISTS reset_period_enum CASCADE")
  op.execute("DROP TYPE IF EXISTS feature_type_enum CASCADE")

  # Create enum types
  op.execute("""
        CREATE TYPE feature_type_enum AS ENUM (
            'model_access',
            'usage_limit',
            'storage_limit',
            'feature_toggle',
            'file_limit',
            'time_limit'
        )
    """)

  op.execute("CREATE TYPE reset_period_enum AS ENUM ('daily', 'monthly', 'never')")

  # Create plan_feature_categories table
  op.execute("""
        CREATE TABLE plan_feature_categories (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(50) NOT NULL UNIQUE,
            display_name VARCHAR(100) NOT NULL,
            description TEXT,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            updated_at TIMESTAMP NOT NULL DEFAULT now()
        )
    """)

  # Create plan_features table
  op.execute("""
        CREATE TABLE plan_features (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category_id UUID NOT NULL REFERENCES plan_feature_categories(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL UNIQUE,
            display_name VARCHAR(200) NOT NULL,
            feature_type feature_type_enum NOT NULL,
            measurement_unit VARCHAR(50),
            description TEXT,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            updated_at TIMESTAMP NOT NULL DEFAULT now()
        )
    """)

  # Create plan_feature_limits table
  op.execute("""
        CREATE TABLE plan_feature_limits (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            billing_plan_id UUID NOT NULL REFERENCES billing_plans(id) ON DELETE CASCADE,
            feature_id UUID NOT NULL REFERENCES plan_features(id) ON DELETE CASCADE,
            is_available BOOLEAN NOT NULL DEFAULT false,
            limit_value BIGINT,
            limit_metadata JSONB,
            reset_period reset_period_enum,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            updated_at TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE(billing_plan_id, feature_id)
        )
    """)

  # Create indexes
  op.execute("CREATE INDEX ix_plan_feature_categories_name ON plan_feature_categories(name)")
  op.execute("CREATE INDEX ix_plan_feature_categories_sort_order ON plan_feature_categories(sort_order)")
  op.execute("CREATE INDEX ix_plan_features_name ON plan_features(name)")
  op.execute("CREATE INDEX ix_plan_features_category_id ON plan_features(category_id)")
  op.execute("CREATE INDEX ix_plan_features_feature_type ON plan_features(feature_type)")
  op.execute("CREATE INDEX ix_plan_features_sort_order ON plan_features(sort_order)")
  op.execute("CREATE INDEX ix_plan_feature_limits_billing_plan_id ON plan_feature_limits(billing_plan_id)")
  op.execute("CREATE INDEX ix_plan_feature_limits_feature_id ON plan_feature_limits(feature_id)")
  op.execute("CREATE INDEX ix_plan_feature_limits_plan_feature ON plan_feature_limits(billing_plan_id, feature_id)")
  op.execute("CREATE INDEX ix_plan_feature_limits_is_available ON plan_feature_limits(is_available)")

  # Create triggers
  op.execute("""
        CREATE TRIGGER update_plan_feature_categories_updated_at
            BEFORE UPDATE ON plan_feature_categories
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column()
    """)

  op.execute("""
        CREATE TRIGGER update_plan_features_updated_at
            BEFORE UPDATE ON plan_features
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column()
    """)

  op.execute("""
        CREATE TRIGGER update_plan_feature_limits_updated_at
            BEFORE UPDATE ON plan_feature_limits
            FOR EACH ROW
            EXECUTE PROCEDURE update_updated_at_column()
    """)


def downgrade() -> None:
  # Drop everything in reverse order
  op.execute("DROP TRIGGER IF EXISTS update_plan_feature_limits_updated_at ON plan_feature_limits")
  op.execute("DROP TRIGGER IF EXISTS update_plan_features_updated_at ON plan_features")
  op.execute("DROP TRIGGER IF EXISTS update_plan_feature_categories_updated_at ON plan_feature_categories")

  op.execute("DROP TABLE IF EXISTS plan_feature_limits CASCADE")
  op.execute("DROP TABLE IF EXISTS plan_features CASCADE")
  op.execute("DROP TABLE IF EXISTS plan_feature_categories CASCADE")

  op.execute("DROP TYPE IF EXISTS reset_period_enum CASCADE")
  op.execute("DROP TYPE IF EXISTS feature_type_enum CASCADE")
