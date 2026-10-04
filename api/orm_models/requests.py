from sqlalchemy import CheckConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base


class RequestRecord(Base):
    __tablename__ = "request_records"
    __table_args__ = (
        CheckConstraint("type IN ('LLM', 'Image', 'Video', 'TTS', 'STT', 'Decision')", name="type"),
        CheckConstraint("status IN (200, 202, 429, 500)", name="status"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    time: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    status: Mapped[int]
    latency: Mapped[str] = mapped_column(Text)
    ttft: Mapped[str | None] = mapped_column(Text)
    tokens_per_second: Mapped[int | None]
    endpoint: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text)
