from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.orm_models.base import Base


class ChoiceQuestion(Base):
    __tablename__ = "choice_questions"
    __table_args__ = (
        CheckConstraint("length(key) > 0", name="key_not_empty"),
        CheckConstraint("length(instructions) > 0", name="instructions_not_empty"),
        CheckConstraint("type = 'Choice'", name="type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(Text)
    instructions: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, default="Choice")
    options: Mapped[list["ChoiceOption"]] = relationship(
        back_populates="question", cascade="all, delete-orphan",
        order_by="ChoiceOption.position", collection_class=ordering_list("position"),
        passive_deletes=True,
    )


class ChoiceOption(Base):
    __tablename__ = "choice_options"
    __table_args__ = (
        CheckConstraint("length(key) > 0", name="key_not_empty"),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        UniqueConstraint("question_id", "position", deferrable=True, initially="DEFERRED"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    question_id: Mapped[UUID] = mapped_column(
        ForeignKey("choice_questions.id", ondelete="CASCADE"),
    )
    position: Mapped[int]
    key: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    question: Mapped[ChoiceQuestion] = relationship(back_populates="options")


class ScoreQuestion(Base):
    __tablename__ = "score_questions"
    __table_args__ = (
        CheckConstraint("length(key) > 0", name="key_not_empty"),
        CheckConstraint("length(instructions) > 0", name="instructions_not_empty"),
        CheckConstraint("type = 'Score'", name="type"),
        CheckConstraint("cardinality(levels) > 0", name="levels_not_empty"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(Text)
    instructions: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, default="Score")
    levels: Mapped[list[str]] = mapped_column(ARRAY(Text))


class NoulQuestion(Base):
    __tablename__ = "noul_questions"
    __table_args__ = (
        CheckConstraint("length(key) > 0", name="key_not_empty"),
        CheckConstraint("length(instructions) > 0", name="instructions_not_empty"),
        CheckConstraint("type = 'Noul'", name="type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(Text)
    instructions: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, default="Noul")
    true_when: Mapped[str] = mapped_column(Text, default="")
    false_when: Mapped[str] = mapped_column(Text, default="")


class DecisionAnswer(Base):
    __tablename__ = "decision_answers"
    __table_args__ = (
        CheckConstraint("type IN ('Choice', 'Score', 'Noul')", name="type"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence_range"),
        CheckConstraint("jsonb_typeof(value) IN ('string', 'number', 'boolean')", name="value_type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)
    value: Mapped[str | float | bool] = mapped_column(JSONB)
    confidence: Mapped[float | None]
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSONB(none_as_null=True))
