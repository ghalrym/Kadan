import unittest
from unittest.mock import MagicMock, patch

from api.database import DatabaseSettings, get_session


class DatabaseTests(unittest.TestCase):
    def test_url_preserves_special_characters_in_credentials(self):
        settings = DatabaseSettings(
            _env_file=None, user="user@name", password="p@ss:/%word",
        )
        self.assertEqual(settings.url.username, "user@name")
        self.assertEqual(settings.url.password, "p@ss:/%word")
        self.assertNotIn("p@ss:/%word", repr(settings))

    @patch("api.database.get_engine")
    @patch("api.database.Session")
    def test_session_closes_without_implicitly_committing(self, session_class, get_engine):
        session = MagicMock()
        session_class.return_value.__enter__.return_value = session
        dependency = get_session()
        self.assertIs(next(dependency), session)
        with self.assertRaises(StopIteration):
            next(dependency)
        session.commit.assert_not_called()
        session_class.return_value.__exit__.assert_called_once_with(None, None, None)

    @patch("api.database.get_engine")
    @patch("api.database.Session")
    def test_session_closes_when_handler_fails(self, session_class, get_engine):
        session_class.return_value.__exit__.return_value = False
        dependency = get_session()
        next(dependency)
        error = ValueError("write failed")
        with self.assertRaisesRegex(ValueError, "write failed"):
            dependency.throw(error)
        self.assertIs(session_class.return_value.__exit__.call_args.args[1], error)
