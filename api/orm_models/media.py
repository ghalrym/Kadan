from uuid import UUID, uuid4

from sqlalchemy import BigInteger, CheckConstraint, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from api.orm_models.base import Base


class ImageSet(Base):
    __tablename__ = "image_sets"
    __table_args__ = (
        CheckConstraint("mode IN ('Generate', 'Edit')", name="mode"),
        CheckConstraint("aspect IN ('square', 'landscape', 'portrait', 'wide')", name="aspect"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    mode: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    aspect: Mapped[str] = mapped_column(Text)
    seeds: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    meta: Mapped[str] = mapped_column(Text)


class VideoJob(Base):
    __tablename__ = "video_jobs"
    __table_args__ = (
        CheckConstraint("aspect IN ('wide', 'portrait', 'square')", name="aspect"),
        CheckConstraint("progress BETWEEN 0 AND 100", name="progress_range"),
        CheckConstraint("status IN ('Rendering', 'Queued', 'Done')", name="status"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text)
    duration: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(Text)
    aspect: Mapped[str] = mapped_column(Text)
    fps: Mapped[str] = mapped_column(Text)
    progress: Mapped[int]
    time: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    thumbnail: Mapped[str] = mapped_column(Text)
    progress_text: Mapped[str] = mapped_column(Text)


class GeneratedSpeech(Base):
    __tablename__ = "generated_speech"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    voice: Mapped[str] = mapped_column(Text)
    meta: Mapped[str] = mapped_column(Text)
    script: Mapped[str] = mapped_column(Text)
    time: Mapped[str] = mapped_column(Text)
