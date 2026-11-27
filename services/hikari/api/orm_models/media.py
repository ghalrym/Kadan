from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base, PortableBigIntArray


class ImageSet(Base):
    __tablename__ = "image_sets"
    __table_args__ = (
        CheckConstraint("mode IN ('Generate', 'Edit')", name="mode"),
        CheckConstraint(
            "aspect IN ('square', 'landscape', 'portrait', 'wide')", name="aspect"
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    mode: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    aspect: Mapped[str] = mapped_column(Text)
    seeds: Mapped[list[int]] = mapped_column(PortableBigIntArray)
    meta: Mapped[str] = mapped_column(Text)


class GeneratedSpeech(Base):
    __tablename__ = "generated_speech"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    voice: Mapped[str] = mapped_column(Text)
    meta: Mapped[str] = mapped_column(Text)
    script: Mapped[str] = mapped_column(Text)
    time: Mapped[str] = mapped_column(Text)
