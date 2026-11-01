from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base


class Subscription(Base):
    """A user's own Claude or Codex sign-in, kept for its CLI."""

    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint("provider IN ('claude', 'codex')", name="provider"),
        CheckConstraint("length(credential) > 0", name="credential_not_empty"),
    )

    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    credential: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CapabilityProvider(Base):
    """Which provider serves a capability; capabilities without a row stay local."""

    __tablename__ = "capability_providers"
    __table_args__ = (
        CheckConstraint("capability IN ('language', 'vision')", name="capability"),
        CheckConstraint("provider IN ('local', 'claude', 'codex')", name="provider"),
    )

    capability: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text)
