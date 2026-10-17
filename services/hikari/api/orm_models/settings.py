from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base


class ModelSetting(Base):
    __tablename__ = "model_settings"
    __table_args__ = (
        CheckConstraint("type IN ('LLM', 'Image', 'Video', 'TTS', 'STT')", name="type"),
        CheckConstraint("selected = ANY(options)", name="selected_available"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    label: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)
    selected: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(ARRAY(Text))
