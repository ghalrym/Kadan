"""Allow ChatGPT to serve image generation independently of language."""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_capability_providers_capability", "capability_providers")
    op.create_check_constraint(
        "capability", "capability_providers", "capability IN ('language', 'image')"
    )
    op.create_check_constraint(
        "image_provider",
        "capability_providers",
        "capability != 'image' OR provider IN ('local', 'codex')",
    )


def downgrade() -> None:
    op.execute("DELETE FROM capability_providers WHERE capability = 'image'")
    op.drop_constraint("ck_capability_providers_image_provider", "capability_providers")
    op.drop_constraint("ck_capability_providers_capability", "capability_providers")
    op.create_check_constraint(
        "capability", "capability_providers", "capability IN ('language')"
    )
