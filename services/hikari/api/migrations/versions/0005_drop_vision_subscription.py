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
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language', 'image')",
        )


def downgrade() -> None:
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"),
            type_="check",
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language', 'image', 'vision')",
        )
