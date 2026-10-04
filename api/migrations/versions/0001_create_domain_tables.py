"""Create empty domain tables.

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('chat_messages',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('meta', sa.Text(), nullable=True),
        sa.CheckConstraint("role IN ('system', 'user', 'assistant')", name=op.f('ck_chat_messages_role')),
        sa.CheckConstraint('length(text) > 0', name=op.f('ck_chat_messages_text_not_empty')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_messages'))
    )
    op.create_table('choice_questions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('instructions', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.CheckConstraint("type = 'Choice'", name=op.f('ck_choice_questions_type')),
        sa.CheckConstraint('length(instructions) > 0', name=op.f('ck_choice_questions_instructions_not_empty')),
        sa.CheckConstraint('length(key) > 0', name=op.f('ck_choice_questions_key_not_empty')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_choice_questions'))
    )
    op.create_table('decision_answers',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence', sa.Double(), nullable=True),
        sa.Column('probabilities', postgresql.JSONB(none_as_null=True, astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("jsonb_typeof(value) IN ('string', 'number', 'boolean')", name=op.f('ck_decision_answers_value_type')),
        sa.CheckConstraint("type IN ('Choice', 'Score', 'Noul')", name=op.f('ck_decision_answers_type')),
        sa.CheckConstraint('confidence BETWEEN 0 AND 1', name=op.f('ck_decision_answers_confidence_range')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_decision_answers'))
    )
    op.create_table('generated_speech',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('voice', sa.Text(), nullable=False),
        sa.Column('meta', sa.Text(), nullable=False),
        sa.Column('script', sa.Text(), nullable=False),
        sa.Column('time', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_generated_speech'))
    )
    op.create_table('image_sets',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('mode', sa.Text(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('aspect', sa.Text(), nullable=False),
        sa.Column('seeds', postgresql.ARRAY(sa.BigInteger()), nullable=False),
        sa.Column('meta', sa.Text(), nullable=False),
        sa.CheckConstraint("aspect IN ('square', 'landscape', 'portrait', 'wide')", name=op.f('ck_image_sets_aspect')),
        sa.CheckConstraint("mode IN ('Generate', 'Edit')", name=op.f('ck_image_sets_mode')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_image_sets'))
    )
    op.create_table('model_settings',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('label', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('selected', sa.Text(), nullable=False),
        sa.Column('options', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.CheckConstraint("type IN ('LLM', 'Image', 'Video', 'TTS', 'STT')", name=op.f('ck_model_settings_type')),
        sa.CheckConstraint('selected = ANY(options)', name=op.f('ck_model_settings_selected_available')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_model_settings'))
    )
    op.create_table('noul_questions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('instructions', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('true_when', sa.Text(), nullable=False),
        sa.Column('false_when', sa.Text(), nullable=False),
        sa.CheckConstraint("type = 'Noul'", name=op.f('ck_noul_questions_type')),
        sa.CheckConstraint('length(instructions) > 0', name=op.f('ck_noul_questions_instructions_not_empty')),
        sa.CheckConstraint('length(key) > 0', name=op.f('ck_noul_questions_key_not_empty')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_noul_questions'))
    )
    op.create_table('request_records',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('time', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('model', sa.Text(), nullable=False),
        sa.Column('status', sa.Integer(), nullable=False),
        sa.Column('latency', sa.Text(), nullable=False),
        sa.Column('ttft', sa.Text(), nullable=True),
        sa.Column('tokens_per_second', sa.Integer(), nullable=True),
        sa.Column('endpoint', sa.Text(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('output', sa.Text(), nullable=False),
        sa.CheckConstraint("type IN ('LLM', 'Image', 'Video', 'TTS', 'STT', 'Decision')", name=op.f('ck_request_records_type')),
        sa.CheckConstraint('status IN (200, 202, 429, 500)', name=op.f('ck_request_records_status')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_request_records'))
    )
    op.create_table('score_questions',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('instructions', sa.Text(), nullable=False),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('levels', postgresql.ARRAY(sa.Text()), nullable=False),
        sa.CheckConstraint("type = 'Score'", name=op.f('ck_score_questions_type')),
        sa.CheckConstraint('cardinality(levels) > 0', name=op.f('ck_score_questions_levels_not_empty')),
        sa.CheckConstraint('length(instructions) > 0', name=op.f('ck_score_questions_instructions_not_empty')),
        sa.CheckConstraint('length(key) > 0', name=op.f('ck_score_questions_key_not_empty')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_score_questions'))
    )
    op.create_table('video_jobs',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('duration', sa.Text(), nullable=False),
        sa.Column('resolution', sa.Text(), nullable=False),
        sa.Column('aspect', sa.Text(), nullable=False),
        sa.Column('fps', sa.Text(), nullable=False),
        sa.Column('progress', sa.Integer(), nullable=False),
        sa.Column('time', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('thumbnail', sa.Text(), nullable=False),
        sa.Column('progress_text', sa.Text(), nullable=False),
        sa.CheckConstraint("aspect IN ('wide', 'portrait', 'square')", name=op.f('ck_video_jobs_aspect')),
        sa.CheckConstraint("status IN ('Rendering', 'Queued', 'Done')", name=op.f('ck_video_jobs_status')),
        sa.CheckConstraint('progress BETWEEN 0 AND 100', name=op.f('ck_video_jobs_progress_range')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_video_jobs'))
    )
    op.create_table('choice_options',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('question_id', sa.Uuid(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.CheckConstraint('length(key) > 0', name=op.f('ck_choice_options_key_not_empty')),
        sa.CheckConstraint('position >= 0', name=op.f('ck_choice_options_position_nonnegative')),
        sa.ForeignKeyConstraint(['question_id'], ['choice_questions.id'], name=op.f('fk_choice_options_question_id_choice_questions'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_choice_options')),
        sa.UniqueConstraint('question_id', 'position', deferrable=True, initially='DEFERRED', name=op.f('uq_choice_options_question_id'))
    )


def downgrade() -> None:
    op.drop_table('choice_options')
    op.drop_table('video_jobs')
    op.drop_table('score_questions')
    op.drop_table('request_records')
    op.drop_table('noul_questions')
    op.drop_table('model_settings')
    op.drop_table('image_sets')
    op.drop_table('generated_speech')
    op.drop_table('decision_answers')
    op.drop_table('choice_questions')
    op.drop_table('chat_messages')
