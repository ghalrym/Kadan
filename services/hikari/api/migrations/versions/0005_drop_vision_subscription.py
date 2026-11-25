"""Drop the retired Vision subscription capability.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM capability_providers WHERE capability = 'vision'")
    op.drop_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        "capability IN ('language', 'image')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_capability_providers_capability"),
        "capability_providers",
        "capability IN ('language', 'image', 'vision')",
    )
