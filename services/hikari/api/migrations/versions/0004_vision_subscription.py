"""Allow independent subscription selection for Vision.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language', 'image', 'vision')",
        )


def downgrade() -> None:
    op.execute("DELETE FROM capability_providers WHERE capability = 'vision'")
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language', 'image')",
        )
