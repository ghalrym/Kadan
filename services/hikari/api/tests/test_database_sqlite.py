import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sqlalchemy.orm import Session

from api.database import build_engine, ensure_database_ready, get_engine
from api.orm_models.decisions import DecisionAnswer, ScoreQuestion
from api.orm_models.media import ImageSet
from api.orm_models.settings import ModelSetting
from api.repositories import subscriptions


class HikariSqliteDatabaseTests(unittest.TestCase):
    def test_alembic_migrations_and_orm_roundtrip_on_sqlite(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "hikari.db"
            url = f"sqlite:///{db_path}"
            get_engine.cache_clear()
            try:
                with patch.dict(
                    os.environ,
                    {"HIKARI_DATABASE_URL": url, "HIKARI_AUTO_MIGRATE": "true"},
                    clear=False,
                ):
                    ensure_database_ready()
                    self.assertTrue(db_path.is_file())
                    engine = build_engine(url)
                    try:
                        with Session(engine) as session:
                            setting = ModelSetting(
                                label="Language",
                                type="LLM",
                                selected="lfm3-1.7b-bf16",
                                options=["lfm3-1.7b-bf16"],
                            )
                            score = ScoreQuestion(
                                key="quality",
                                instructions="Rate quality",
                                levels=["low", "high"],
                            )
                            answer = DecisionAnswer(
                                key="quality",
                                type="Score",
                                value="high",
                                confidence=0.95,
                                probabilities={"low": 0.05, "high": 0.95},
                            )
                            images = ImageSet(
                                id="img-1",
                                mode="Generate",
                                prompt="sunset",
                                aspect="square",
                                seeds=[1, 2],
                                meta="{}",
                            )
                            session.add_all([setting, score, answer, images])
                            session.commit()

                            subscriptions.connect(session, "claude", "tok_test_123")
                            subscriptions.choose(session, "language", "claude")
                            snap = subscriptions.snapshot(session)
                            self.assertEqual(snap.capabilities.language, "claude")

                            loaded_setting = session.get(ModelSetting, setting.id)
                            self.assertIsNotNone(loaded_setting)
                            assert loaded_setting is not None
                            self.assertEqual(loaded_setting.options, ["lfm3-1.7b-bf16"])

                            loaded_answer = session.get(DecisionAnswer, answer.id)
                            self.assertIsNotNone(loaded_answer)
                            assert loaded_answer is not None
                            self.assertEqual(loaded_answer.probabilities, {"low": 0.05, "high": 0.95})
                    finally:
                        engine.dispose()
            finally:
                get_engine.cache_clear()
