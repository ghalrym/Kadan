"""Store subscription sign-ins and per-capability providers.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subscriptions",
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("credential", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "provider IN ('claude', 'codex')", name=op.f("ck_subscriptions_provider")
        ),
        sa.CheckConstraint(
            "length(credential) > 0",
            name=op.f("ck_subscriptions_credential_not_empty"),
        ),
        sa.PrimaryKeyConstraint("provider", name=op.f("pk_subscriptions")),
    )
    op.create_table(
        "capability_providers",
        sa.Column("capability", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "capability IN ('language')",
            name=op.f("ck_capability_providers_capability"),
        ),
        sa.CheckConstraint(
            "provider IN ('local', 'claude', 'codex')",
            name=op.f("ck_capability_providers_provider"),
        ),
        sa.PrimaryKeyConstraint("capability", name=op.f("pk_capability_providers")),
    )


def downgrade() -> None:
    op.drop_table("capability_providers")
    op.drop_table("subscriptions")
