from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('system', 'user', 'assistant')", name="role"),
        CheckConstraint("length(text) > 0", name="text_not_empty"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    role: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[str | None] = mapped_column(Text)
