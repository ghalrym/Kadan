"""Allow independent subscription selection for Vision.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        "capability IN ('language', 'vision')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM capability_providers WHERE capability = 'vision'")
    op.drop_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        "capability IN ('language')",
    )
