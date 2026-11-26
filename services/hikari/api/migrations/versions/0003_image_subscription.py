"""Allow ChatGPT to serve image generation independently of language."""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language', 'image')",
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_image_provider"),
            "capability != 'image' OR provider IN ('local', 'codex')",
        )


def downgrade() -> None:
    op.execute("DELETE FROM capability_providers WHERE capability = 'image'")
    with op.batch_alter_table("capability_providers") as batch_op:
        batch_op.drop_constraint(
            op.f("ck_capability_providers_image_provider"), type_="check"
        )
        batch_op.drop_constraint(
            op.f("ck_capability_providers_capability"), type_="check"
        )
        batch_op.create_check_constraint(
            op.f("ck_capability_providers_capability"),
            "capability IN ('language')",
        )
